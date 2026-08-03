"""Bayesian hyperparameter search for the LightGBM meta-label classifier.

Uses Optuna TPE sampler (seed-deterministic) with MedianPruner.
All search-space bounds are loaded from the frozen search_space.json —
declared before any outer-test data is seen (Rule 2 in AGENT.md).

Public API
----------
N_TRIALS_MAX : int  — frozen budget (50); matches search_space.json
RANDOM_SEED  : int  — frozen seed (42); matches search_space.json

load_search_space() -> dict
    Return the parsed search_space.json.

_suggest_params(trial, param_ranges) -> dict
    Translate JSON "param_ranges" into {name: value} using trial.suggest_*.

optimize_hyperparams(X, y, trade_dates, n_trials=None, seed=RANDOM_SEED) -> dict
    Run Optuna TPE study.  Returns best LightGBM params dict (including fixed
    keys: objective, metric, verbosity) ready to pass to MetaLabelModel(params=...).
    Requires LightGBM and Optuna to be installed (ml optional group).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from orb.ml.meta_label import (
    EMBARGO_DAYS,
    PURGE_DAYS,
    _N_CV_SPLITS,
    _NUM_BOOST_ROUND,
    _EARLY_STOP,
    _log_loss,
    _purged_wf_splits,
)

# ---------------------------------------------------------------------------
# Frozen constants — must match search_space.json

N_TRIALS_MAX: int = 50
RANDOM_SEED:  int = 42

_SEARCH_SPACE_PATH = Path(__file__).parent / "search_space.json"

_FIXED_LGB_PARAMS: dict = {
    "objective": "binary",
    "metric":    "binary_logloss",
    "verbosity": -1,
}


# ---------------------------------------------------------------------------
# Helpers

def load_search_space() -> dict:
    """Return the parsed search_space.json dict (cached at import time)."""
    return _CACHED_SS


def _suggest_params(trial: object, param_ranges: dict) -> dict:
    """Translate param_ranges JSON spec into Optuna trial suggestions."""
    params: dict = {}
    for name, spec in param_ranges.items():
        typ = spec["type"]
        low, high = spec["low"], spec["high"]
        if typ == "int":
            params[name] = trial.suggest_int(name, low, high)
        elif typ == "float":
            log = spec.get("log", False)
            params[name] = trial.suggest_float(name, low, high, log=log)
        else:
            raise ValueError(f"unknown param type '{typ}' for '{name}'")
    return params


def _build_objective(X: np.ndarray, y: np.ndarray, trade_dates: list[date],
                     param_ranges: dict):
    """Return an Optuna objective function closed over (X, y, dates)."""
    import lightgbm as lgb
    import optuna

    splits = _purged_wf_splits(trade_dates, _N_CV_SPLITS, PURGE_DAYS, EMBARGO_DAYS)
    if not splits:
        raise ValueError("No valid purged CV splits — provide more training data.")

    def objective(trial) -> float:
        tunable = _suggest_params(trial, param_ranges)
        params = {**_FIXED_LGB_PARAMS, **tunable, "seed": RANDOM_SEED}

        fold_scores: list[float] = []
        for step, (train_idx, val_idx) in enumerate(splits):
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
            score = _log_loss(y[val_idx], booster.predict(X[val_idx]))
            fold_scores.append(score)

            # Report intermediate score for MedianPruner
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Public entry point

def optimize_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    trade_dates: list[date],
    n_trials: int | None = None,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run Optuna TPE Bayesian search; return best LightGBM params dict.

    The returned dict merges the fixed LightGBM params (objective, metric,
    verbosity) with the Optuna-chosen tuneable params so it can be passed
    directly to ``MetaLabelModel(params=<result>).fit(...)``.

    Parameters
    ----------
    X, y        : training features and labels
    trade_dates : entry dates (same length as X/y rows)
    n_trials    : number of Optuna trials; capped at N_TRIALS_MAX if None
    seed        : RNG seed for TPESampler; must equal RANDOM_SEED for
                  reproducibility between runs (Rule 6 in AGENT.md)
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    budget = min(n_trials or N_TRIALS_MAX, N_TRIALS_MAX)
    ss = load_search_space()

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    objective = _build_objective(X_arr, y_arr, trade_dates, ss["param_ranges"])

    study.optimize(objective, n_trials=budget)

    best_tunable = study.best_trial.params
    return {**_FIXED_LGB_PARAMS, **best_tunable, "seed": seed}


# ---------------------------------------------------------------------------
# Convenience wrapper — keeps numpy out of wf_select.py (ADR-001/003)

def optimize_from_trades(
    train_trades: list[dict],
    baseline_bps: float = 2.5,
    n_trials: int | None = None,
    seed: int = RANDOM_SEED,
) -> dict | None:
    """Build features from raw trade dicts and run Optuna search.

    Returns best LightGBM params dict, or None if training data too small.
    All numpy operations are isolated inside this module (ADR-003).
    """
    from orb.ml.meta_label import build_features_from_trades, _net_pnl, _parse_entry_date

    if len(train_trades) < 20:
        return None

    X      = build_features_from_trades(train_trades)
    y      = np.array([1.0 if _net_pnl(t, baseline_bps) > 0 else 0.0
                       for t in train_trades])
    dates  = [_parse_entry_date(t) for t in train_trades]
    return optimize_hyperparams(X, y, dates, n_trials=n_trials, seed=seed)


# ---------------------------------------------------------------------------
# Module-level cache (parse JSON once at import)

with open(_SEARCH_SPACE_PATH) as _f:
    _CACHED_SS: dict = json.load(_f)
