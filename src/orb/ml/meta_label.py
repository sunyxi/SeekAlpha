"""Meta-labeling filter for ORB signals.

Uses LightGBM with purged walk-forward time-series CV to estimate P(win) for
each ORB trade.  All constants are pre-declared (frozen) and must not be tuned
on outer-test data (AGENT.md Rule 1 & 2).

Public API
----------
MetaLabelModel           — LightGBM binary classifier with purged CV
build_features_from_trades(trades) -> ndarray (n, 4)
    Build ORB feature matrix from raw trade dicts.
train_and_filter(train_trades, test_trades, baseline_bps) -> list[dict]
    Convenience wrapper: fit model on train_trades, return filtered test_trades.

Constants (frozen — match search_space.json)
--------------------------------------------
PURGE_DAYS      = 20   — label horizon; training samples within this window
                         of the validation start are purged.
EMBARGO_DAYS    = 5    — gap added between training end and validation start.
PROB_THRESHOLD  = 0.55 — P(win) gate declared before any outer-test is seen.
RANDOM_SEED     = 42
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Generator

import numpy as np

# ---------------------------------------------------------------------------
# Pre-declared frozen constants (must match search_space.json)

PURGE_DAYS:     int   = 20
EMBARGO_DAYS:   int   = 5
PROB_THRESHOLD: float = 0.55
RANDOM_SEED:    int   = 42
_N_CV_SPLITS:   int   = 5
_NUM_BOOST_ROUND: int = 200
_EARLY_STOP:    int   = 20


# ---------------------------------------------------------------------------
# Internal helpers

def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Binary log-loss without scipy."""
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(
        y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)
    ))


def _net_pnl(trade: dict, bps_per_side: float) -> float:
    cost = (trade["entry_notional"] + trade["exit_notional"]) * bps_per_side / 10_000.0
    return trade["gross_pnl"] - cost


def _parse_entry_date(trade: dict) -> date:
    return date.fromisoformat(trade["entry_time"][:10])


# ---------------------------------------------------------------------------
# Purged walk-forward CV

def _purged_wf_splits(
    dates: list[date],
    n_splits: int = _N_CV_SPLITS,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits with purging and embargo.

    For each split k the validation period is the (k+1)-th block of unique
    dates.  Training samples whose label window (date + purge_days) overlaps
    the validation start are purged.  An embargo of embargo_days is added
    so the purge boundary = val_start - purge_days - embargo_days.

    Returns list of (train_idx, val_idx) index arrays.  Both arrays index
    into the input ``dates`` list.  Returns [] if there are insufficient dates.
    """
    if not dates:
        return []

    dates_arr = np.array(dates)
    unique = sorted(set(dates))
    n_unique = len(unique)

    # Need at least (n_splits + 1) blocks
    if n_unique < n_splits + 2:
        return []

    block = n_unique // (n_splits + 1)
    if block < 1:
        return []

    results = []
    for k in range(n_splits):
        i0 = (k + 1) * block
        i1 = (k + 2) * block if k < n_splits - 1 else n_unique
        val_start = unique[i0]
        val_end   = unique[i1 - 1]

        # Purge + embargo boundary: samples after this date are excluded from training
        cutoff = val_start - timedelta(days=purge_days + embargo_days)

        train_mask = dates_arr < cutoff
        val_mask   = (dates_arr >= val_start) & (dates_arr <= val_end)

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(train_idx) >= 10 and len(val_idx) >= 5:
            results.append((train_idx, val_idx))

    return results


# ---------------------------------------------------------------------------
# ORB feature engineering

def build_features_from_trades(trades: list[dict]) -> np.ndarray:
    """Build (n_trades, 4) feature matrix from ORB trade attributes.

    Features (all point-in-time at trade entry):
      0  atr_frac    — (entry - stop) / entry  [normalized ATR]
      1  target_frac — (target - entry) / entry [normalized distance to target]
      2  rr          — target_frac / atr_frac   [risk/reward ratio]
      3  session_frac — minutes from open / 390  [time in session, 0=open 1=close]
    """
    if not trades:
        return np.empty((0, 4), dtype=float)

    rows = []
    for t in trades:
        entry  = float(t["entry_price"])
        stop   = float(t["stop_price"])
        target = float(t["target_price"])

        atr_frac    = (entry - stop)   / entry if entry > 0 else 0.0
        target_frac = (target - entry) / entry if entry > 0 else 0.0
        rr          = target_frac / atr_frac if atr_frac > 1e-9 else 0.0

        # session fraction: entry_time in "YYYY-MM-DDTHH:MM:SS" or with space
        et = t["entry_time"].replace("T", " ")
        hh, mm = int(et[11:13]), int(et[14:16])
        mins_from_open = max(0, hh * 60 + mm - 9 * 60 - 30)  # 9:30 open
        session_frac = min(1.0, mins_from_open / 390.0)       # 390 min in RTH

        rows.append([atr_frac, target_frac, rr, session_frac])

    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# MetaLabelModel

class MetaLabelModel:
    """LightGBM meta-labeling classifier with purged walk-forward inner CV.

    Attributes
    ----------
    PURGE_DAYS, EMBARGO_DAYS, PROB_THRESHOLD, RANDOM_SEED — frozen module constants.

    Parameters
    ----------
    params : dict, optional
        LightGBM training parameters.  Defaults to search_space.json defaults.
        M4-3 will pass Optuna-optimised params here.
    """

    # Surface frozen constants as class attributes for external access
    PURGE_DAYS:     int   = PURGE_DAYS
    EMBARGO_DAYS:   int   = EMBARGO_DAYS
    PROB_THRESHOLD: float = PROB_THRESHOLD
    RANDOM_SEED:    int   = RANDOM_SEED

    def __init__(self, params: dict | None = None) -> None:
        self._params = params or self._default_lgb_params()
        self._booster = None   # lgb.Booster after fit()
        self._cv_score: float | None = None

    @staticmethod
    def _default_lgb_params() -> dict:
        return {
            "objective":         "binary",
            "metric":            "binary_logloss",
            "verbosity":         -1,
            "num_leaves":        31,
            "learning_rate":     0.05,
            "min_child_samples": 20,
            "feature_fraction":  1.0,
            "bagging_fraction":  1.0,
            "bagging_freq":      0,
        }

    # ------------------------------------------------------------------ public

    def fit(
        self,
        X: np.ndarray,           # (n_samples, n_features)
        y: np.ndarray,           # (n_samples,) — 1=win, 0=loss
        trade_dates: list[date], # (n_samples,)
    ) -> "MetaLabelModel":
        """Train on (X, y) with purged walk-forward inner CV.

        Sets self._cv_score (mean inner-CV log-loss) and self._booster
        (final model trained on the full dataset).
        """
        import lightgbm as lgb  # lazy import — only needed when ML is active

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        params = dict(self._params)
        if params.get("scale_pos_weight") in (None, "auto"):
            params["scale_pos_weight"] = float(n_neg) / max(n_pos, 1)
        params.setdefault("seed", RANDOM_SEED)

        # Inner purged CV to estimate generalisation
        cv_scores: list[float] = []
        for train_idx, val_idx in _purged_wf_splits(trade_dates, _N_CV_SPLITS,
                                                     PURGE_DAYS, EMBARGO_DAYS):
            dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
            dval   = lgb.Dataset(X[val_idx],   label=y[val_idx], reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=_NUM_BOOST_ROUND,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(_EARLY_STOP, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            pred = booster.predict(X[val_idx])
            cv_scores.append(_log_loss(y[val_idx], pred))

        self._cv_score = float(np.mean(cv_scores)) if cv_scores else None

        # Final model on full data
        dtrain_full = lgb.Dataset(X, label=y)
        self._booster = lgb.train(
            params,
            dtrain_full,
            num_boost_round=_NUM_BOOST_ROUND,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(win) for each row of X.  Raises RuntimeError if not fitted."""
        if self._booster is None:
            raise RuntimeError("model not fitted; call fit() first")
        return self._booster.predict(np.asarray(X, dtype=float))

    def filter_signals(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: True where P(win) >= PROB_THRESHOLD."""
        return self.predict_proba(X) >= PROB_THRESHOLD

    def is_fit(self) -> bool:
        return self._booster is not None

    @property
    def cv_score(self) -> float | None:
        """Mean inner-CV binary log-loss (None if no valid splits existed)."""
        return self._cv_score


# ---------------------------------------------------------------------------
# Convenience wrapper for wf_select.py integration

def train_and_filter(
    train_trades: list[dict],
    test_trades: list[dict],
    baseline_bps: float = 2.5,
    params: dict | None = None,
) -> tuple[list[dict], float | None]:
    """Fit a MetaLabelModel on train_trades; filter test_trades by P(win).

    Returns (filtered_test_trades, cv_score).
    Falls back to unfiltered test_trades if training data is too small.
    """
    if len(train_trades) < 20:
        return test_trades, None

    X_train = build_features_from_trades(train_trades)
    y_train = np.array(
        [1.0 if _net_pnl(t, baseline_bps) > 0 else 0.0 for t in train_trades]
    )
    dates = [_parse_entry_date(t) for t in train_trades]

    model = MetaLabelModel(params=params)
    model.fit(X_train, y_train, dates)

    if not test_trades:
        return [], model.cv_score

    X_test = build_features_from_trades(test_trades)
    mask = model.filter_signals(X_test)
    filtered = [t for t, keep in zip(test_trades, mask) if keep]
    return filtered, model.cv_score
