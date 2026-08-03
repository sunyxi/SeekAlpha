"""Tests for src/orb/ml/meta_label.py.

Tests that do NOT require LightGBM:
  - TestPurgedSplits: correctness of the CV split generator
  - TestNoLeakage: no future dates appear in any training set
  - TestBuildFeatures: ORB feature engineering from trade dicts
  - TestSearchSpace: frozen search-space JSON is valid and contains required keys

Tests that require LightGBM (skipped when not installed):
  - TestMetaLabelFit: fit() + predict_proba() end-to-end
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from orb.ml.meta_label import (
    EMBARGO_DAYS,
    PROB_THRESHOLD,
    PURGE_DAYS,
    RANDOM_SEED,
    _purged_wf_splits,
    build_features_from_trades,
)

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None

_SEARCH_SPACE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "orb" / "ml" / "search_space.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int, start: date = date(2021, 1, 4)) -> list[date]:
    """n consecutive calendar dates (includes weekends, good enough for tests)."""
    return [start + timedelta(days=i) for i in range(n)]


def _make_trade(entry_date: date, win: bool = True) -> dict:
    entry_price = 100.0
    stop_price  = 98.0
    target_price = 104.0
    exit_price   = target_price if win else stop_price
    return {
        "candidate_id": "test",
        "symbol": "SPY",
        "signal_time":   entry_date.isoformat() + "T09:35:00",
        "entry_time":    entry_date.isoformat() + "T09:35:00",
        "exit_time":     entry_date.isoformat() + "T10:00:00",
        "entry_price":   entry_price,
        "stop_price":    stop_price,
        "target_price":  target_price,
        "exit_price":    exit_price,
        "exit_reason":   "target" if win else "stop",
        "quantity":      10,
        "gross_pnl":     (exit_price - entry_price) * 10,
        "entry_notional": entry_price * 10,
        "exit_notional":  exit_price  * 10,
    }


# ---------------------------------------------------------------------------
# TestPurgedSplits
# ---------------------------------------------------------------------------

class TestPurgedSplits(unittest.TestCase):

    def _run_splits(self, n_dates: int, n_splits: int = 5) -> list:
        dates = _make_dates(n_dates)
        return list(_purged_wf_splits(dates, n_splits=n_splits,
                                      purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS))

    def test_returns_list_of_tuples(self):
        splits = self._run_splits(200)
        self.assertIsInstance(splits, list)
        for train_idx, val_idx in splits:
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(val_idx, np.ndarray)

    def test_at_most_n_splits(self):
        splits = self._run_splits(200, n_splits=5)
        self.assertLessEqual(len(splits), 5)

    def test_empty_on_too_few_dates(self):
        splits = self._run_splits(10, n_splits=5)
        self.assertEqual(splits, [])

    def test_train_and_val_disjoint(self):
        splits = self._run_splits(200)
        for train_idx, val_idx in splits:
            self.assertEqual(len(set(train_idx) & set(val_idx)), 0,
                             "train and val indices must be disjoint")

    def test_val_indices_nonempty(self):
        splits = self._run_splits(200)
        self.assertGreater(len(splits), 0)
        for _, val_idx in splits:
            self.assertGreater(len(val_idx), 0)

    def test_train_indices_nonempty(self):
        for train_idx, _ in self._run_splits(200):
            self.assertGreater(len(train_idx), 0)


# ---------------------------------------------------------------------------
# TestNoLeakage
# ---------------------------------------------------------------------------

class TestNoLeakage(unittest.TestCase):
    """Purged CV must not allow any training sample to see into the validation period."""

    def setUp(self):
        n = 300
        self.all_dates = _make_dates(n)

    def test_all_train_dates_before_val_dates(self):
        for train_idx, val_idx in _purged_wf_splits(
            self.all_dates, n_splits=5, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS
        ):
            train_dates = [self.all_dates[i] for i in train_idx]
            val_dates   = [self.all_dates[i] for i in val_idx]
            max_train = max(train_dates)
            min_val   = min(val_dates)
            self.assertLess(
                max_train, min_val,
                f"Training date {max_train} is not before val start {min_val}",
            )

    def test_purge_gap_respected(self):
        """max(train_date) + purge_days must be strictly before min(val_date)."""
        for train_idx, val_idx in _purged_wf_splits(
            self.all_dates, n_splits=5, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS
        ):
            train_dates = [self.all_dates[i] for i in train_idx]
            val_dates   = [self.all_dates[i] for i in val_idx]
            max_train = max(train_dates)
            min_val   = min(val_dates)
            # The label (forward return) for max_train extends max_train + purge_days.
            # That must not reach min_val.
            self.assertLess(
                max_train + timedelta(days=PURGE_DAYS),
                min_val,
                f"Purge gap violated: max_train={max_train} "
                f"purge_days={PURGE_DAYS} min_val={min_val}",
            )


# ---------------------------------------------------------------------------
# TestBuildFeatures
# ---------------------------------------------------------------------------

class TestBuildFeatures(unittest.TestCase):

    def _make_trades(self, n: int) -> list[dict]:
        dates = _make_dates(n)
        return [_make_trade(d, win=(i % 2 == 0)) for i, d in enumerate(dates)]

    def test_shape(self):
        trades = self._make_trades(10)
        X = build_features_from_trades(trades)
        self.assertEqual(X.shape, (10, 4))

    def test_dtype_float(self):
        trades = self._make_trades(5)
        X = build_features_from_trades(trades)
        self.assertEqual(X.dtype, np.float64)

    def test_no_nan_in_output(self):
        trades = self._make_trades(20)
        X = build_features_from_trades(trades)
        self.assertFalse(np.any(np.isnan(X)),
                         "feature matrix must not contain NaN")

    def test_atr_frac_positive(self):
        """ATR fraction = (entry - stop) / entry > 0 for long trades."""
        trades = self._make_trades(5)
        X = build_features_from_trades(trades)
        self.assertTrue(np.all(X[:, 0] > 0), "ATR fraction must be positive")

    def test_session_frac_in_range(self):
        """Session fraction must be in [0, 1]."""
        trades = self._make_trades(10)
        X = build_features_from_trades(trades)
        self.assertTrue(np.all(X[:, 3] >= 0.0) and np.all(X[:, 3] <= 1.0))

    def test_empty_returns_empty(self):
        X = build_features_from_trades([])
        self.assertEqual(X.shape, (0, 4))

    def test_rr_ratio_positive(self):
        """Risk/reward ratio must be > 0 (target above entry, stop below entry)."""
        trades = self._make_trades(5)
        X = build_features_from_trades(trades)
        self.assertTrue(np.all(X[:, 2] > 0), "R/R ratio must be positive")


# ---------------------------------------------------------------------------
# TestSearchSpace
# ---------------------------------------------------------------------------

class TestSearchSpace(unittest.TestCase):

    def setUp(self):
        self.assertTrue(_SEARCH_SPACE_PATH.exists(),
                        f"search_space.json not found at {_SEARCH_SPACE_PATH}")
        with open(_SEARCH_SPACE_PATH) as f:
            self.ss = json.load(f)

    def test_prob_threshold_matches_constant(self):
        self.assertAlmostEqual(self.ss["prob_threshold"], PROB_THRESHOLD)

    def test_purge_days_matches_constant(self):
        self.assertEqual(self.ss["purge_days"], PURGE_DAYS)

    def test_embargo_days_matches_constant(self):
        self.assertEqual(self.ss["embargo_days"], EMBARGO_DAYS)

    def test_random_seed_matches_constant(self):
        self.assertEqual(self.ss["random_seed"], RANDOM_SEED)

    def test_param_ranges_present(self):
        self.assertIn("param_ranges", self.ss)
        required = {"num_leaves", "learning_rate", "min_child_samples", "scale_pos_weight"}
        self.assertTrue(required <= set(self.ss["param_ranges"]),
                        f"missing param ranges: {required - set(self.ss['param_ranges'])}")

    def test_default_params_present(self):
        self.assertIn("default_params", self.ss)


# ---------------------------------------------------------------------------
# TestMetaLabelFit (requires LightGBM)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LGB, "lightgbm not installed — skipping fit tests")
class TestMetaLabelFit(unittest.TestCase):

    def setUp(self):
        from orb.ml.meta_label import MetaLabelModel
        rng = np.random.RandomState(0)
        n = 200
        self.dates = _make_dates(n)
        self.X = rng.randn(n, 4).astype(float)
        # 40% wins
        self.y = (rng.rand(n) > 0.6).astype(float)
        self.model = MetaLabelModel()

    def test_fit_runs_without_error(self):
        self.model.fit(self.X, self.y, self.dates)
        self.assertTrue(self.model.is_fit())

    def test_predict_proba_shape(self):
        self.model.fit(self.X, self.y, self.dates)
        proba = self.model.predict_proba(self.X[:10])
        self.assertEqual(proba.shape, (10,))

    def test_predict_proba_in_range(self):
        self.model.fit(self.X, self.y, self.dates)
        proba = self.model.predict_proba(self.X)
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_predict_before_fit_raises(self):
        from orb.ml.meta_label import MetaLabelModel
        m = MetaLabelModel()
        with self.assertRaises(RuntimeError):
            m.predict_proba(self.X)

    def test_filter_signals_is_bool_array(self):
        self.model.fit(self.X, self.y, self.dates)
        mask = self.model.filter_signals(self.X)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (len(self.X),))

    def test_cv_score_is_finite_after_fit(self):
        self.model.fit(self.X, self.y, self.dates)
        # cv_score may be None if no valid splits (possible with 200 dates),
        # but if it exists it must be a finite float
        if self.model.cv_score is not None:
            self.assertTrue(np.isfinite(self.model.cv_score))


if __name__ == "__main__":
    unittest.main()
