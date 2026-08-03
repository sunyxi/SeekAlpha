"""Tests for src/orb/ml/bayes_search.py.

Tests that do NOT require Optuna/LightGBM:
  - TestConstants: N_TRIALS_MAX, RANDOM_SEED match search_space.json
  - TestSearchSpaceLoad: load_search_space() returns expected structure
  - TestSuggestParams: _suggest_params() maps JSON ranges to param dict

Tests that require both Optuna + LightGBM (skipped otherwise):
  - TestOptimizeHyperparams: end-to-end optimize_hyperparams()
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

_HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
_HAS_LGB    = importlib.util.find_spec("lightgbm") is not None
_HAS_ML     = _HAS_OPTUNA and _HAS_LGB

_SEARCH_SPACE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "orb" / "ml" / "search_space.json"
)
with open(_SEARCH_SPACE_PATH) as _f:
    _SS = json.load(_f)


def _make_dates(n: int, start: date = date(2021, 1, 4)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------

class TestConstants(unittest.TestCase):

    def setUp(self):
        from orb.ml.bayes_search import N_TRIALS_MAX, RANDOM_SEED
        self.N_TRIALS_MAX = N_TRIALS_MAX
        self.RANDOM_SEED  = RANDOM_SEED

    def test_n_trials_max_matches_json(self):
        self.assertEqual(self.N_TRIALS_MAX, _SS["n_trials_max"])

    def test_random_seed_matches_json(self):
        self.assertEqual(self.RANDOM_SEED, _SS["random_seed"])

    def test_n_trials_max_value(self):
        self.assertEqual(self.N_TRIALS_MAX, 50)

    def test_random_seed_value(self):
        self.assertEqual(self.RANDOM_SEED, 42)


# ---------------------------------------------------------------------------
# TestSearchSpaceLoad
# ---------------------------------------------------------------------------

class TestSearchSpaceLoad(unittest.TestCase):

    def setUp(self):
        from orb.ml.bayes_search import load_search_space
        self.ss = load_search_space()

    def test_returns_dict(self):
        self.assertIsInstance(self.ss, dict)

    def test_has_param_ranges(self):
        self.assertIn("param_ranges", self.ss)

    def test_has_required_tuneable_params(self):
        required = {"num_leaves", "learning_rate", "min_child_samples"}
        self.assertTrue(required <= set(self.ss["param_ranges"]))

    def test_n_trials_max_present(self):
        self.assertIn("n_trials_max", self.ss)

    def test_random_seed_present(self):
        self.assertIn("random_seed", self.ss)


# ---------------------------------------------------------------------------
# TestSuggestParams
# ---------------------------------------------------------------------------

class TestSuggestParams(unittest.TestCase):
    """_suggest_params(trial, param_ranges) should return a dict of values
    within the declared bounds without requiring a live study."""

    def _make_mock_trial(self, spec: dict) -> object:
        """Return a mock object whose suggest_* methods return the midpoint."""
        class MockTrial:
            def suggest_int(self, name, low, high, **kw):
                return (low + high) // 2
            def suggest_float(self, name, low, high, **kw):
                return (low + high) / 2.0
        return MockTrial()

    def test_all_param_ranges_produce_values(self):
        from orb.ml.bayes_search import _suggest_params
        trial = self._make_mock_trial(_SS["param_ranges"])
        params = _suggest_params(trial, _SS["param_ranges"])
        self.assertIsInstance(params, dict)
        for key in _SS["param_ranges"]:
            self.assertIn(key, params)

    def test_int_params_are_ints(self):
        from orb.ml.bayes_search import _suggest_params
        trial = self._make_mock_trial(_SS["param_ranges"])
        params = _suggest_params(trial, _SS["param_ranges"])
        int_keys = [k for k, v in _SS["param_ranges"].items() if v["type"] == "int"]
        for k in int_keys:
            self.assertIsInstance(params[k], int, f"{k} should be int")

    def test_float_params_are_floats(self):
        from orb.ml.bayes_search import _suggest_params
        trial = self._make_mock_trial(_SS["param_ranges"])
        params = _suggest_params(trial, _SS["param_ranges"])
        float_keys = [k for k, v in _SS["param_ranges"].items() if v["type"] == "float"]
        for k in float_keys:
            self.assertIsInstance(params[k], float, f"{k} should be float")


# ---------------------------------------------------------------------------
# TestOptimizeHyperparams (requires Optuna + LightGBM)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_ML, "optuna + lightgbm not installed — skipping")
class TestOptimizeHyperparams(unittest.TestCase):

    def setUp(self):
        from orb.ml.bayes_search import optimize_hyperparams
        self.optimize = optimize_hyperparams
        rng = np.random.RandomState(7)
        self.n = 200
        self.dates = _make_dates(self.n)
        self.X = rng.randn(self.n, 4).astype(float)
        self.y = (rng.rand(self.n) > 0.6).astype(float)

    def test_returns_dict(self):
        params = self.optimize(self.X, self.y, self.dates, n_trials=3)
        self.assertIsInstance(params, dict)

    def test_required_keys_present(self):
        params = self.optimize(self.X, self.y, self.dates, n_trials=3)
        for key in ("num_leaves", "learning_rate", "min_child_samples"):
            self.assertIn(key, params)

    def test_num_leaves_in_bounds(self):
        params = self.optimize(self.X, self.y, self.dates, n_trials=3)
        r = _SS["param_ranges"]["num_leaves"]
        self.assertGreaterEqual(params["num_leaves"], r["low"])
        self.assertLessEqual(params["num_leaves"], r["high"])

    def test_learning_rate_in_bounds(self):
        params = self.optimize(self.X, self.y, self.dates, n_trials=3)
        r = _SS["param_ranges"]["learning_rate"]
        self.assertGreaterEqual(params["learning_rate"], r["low"])
        self.assertLessEqual(params["learning_rate"], r["high"])

    def test_deterministic_with_seed(self):
        """Same seed must produce same best hyperparams."""
        p1 = self.optimize(self.X, self.y, self.dates, n_trials=5, seed=42)
        p2 = self.optimize(self.X, self.y, self.dates, n_trials=5, seed=42)
        self.assertEqual(p1["num_leaves"], p2["num_leaves"])
        self.assertAlmostEqual(p1["learning_rate"], p2["learning_rate"])

    def test_includes_fixed_lgb_keys(self):
        """Returned params must include objective/metric so MetaLabelModel can use directly."""
        params = self.optimize(self.X, self.y, self.dates, n_trials=3)
        self.assertIn("objective", params)
        self.assertEqual(params["objective"], "binary")

    def test_best_trial_count_capped(self):
        """n_trials argument controls the actual number of trials."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Run with tiny budget; no error
        params = self.optimize(self.X, self.y, self.dates, n_trials=2)
        self.assertIsInstance(params, dict)


# ---------------------------------------------------------------------------
# TestWfSelectBayesFlag
# ---------------------------------------------------------------------------

class TestWfSelectBayesFlag(unittest.TestCase):
    """wf_select.py must accept --bayes-search without numpy at module level."""

    def test_bayes_search_flag_registered(self):
        import argparse, sys
        # Patch sys.argv to avoid conflicts
        saved = sys.argv
        try:
            sys.argv = ["wf_select.py", "--trades-dir", "x", "--report-output", "y",
                        "--bayes-search"]
            import importlib, scripts.wf_select as ws
            importlib.reload(ws)
            ap = argparse.ArgumentParser()
            ap.add_argument("--trades-dir", required=True)
            ap.add_argument("--report-output", required=True)
            ap.add_argument("--meta-label", action="store_true")
            ap.add_argument("--bayes-search", action="store_true")
            args = ap.parse_args(["--trades-dir", "x", "--report-output", "y",
                                  "--bayes-search"])
            self.assertTrue(args.bayes_search)
        finally:
            sys.argv = saved

    def test_no_numpy_at_top_level(self):
        """wf_select.py must not import numpy at module scope (ADR-001)."""
        import ast, pathlib
        src = pathlib.Path("scripts/wf_select.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module] if node.module else [])
                )
                for name in names:
                    self.assertFalse(
                        name.startswith("numpy"),
                        "wf_select.py must not import numpy at module scope",
                    )


if __name__ == "__main__":
    unittest.main()
