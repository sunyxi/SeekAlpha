"""Tests for scripts/compare_runs.py — paired ML vs baseline comparison.

All tests use synthetic wf_select.py report dicts; no real data needed.
No ML dependencies required — compare_runs.py is stdlib-only.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic wf_select reports
# ---------------------------------------------------------------------------

def _test_metrics(sharpe: float = 1.0, pf: float = 1.2, bps: float = 3.0,
                  trades: int = 40, win_rate: float = 0.55,
                  total_pnl: float = 500.0) -> dict:
    return {
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "sharpe": sharpe,
        "mean_net_bps": bps,
        "total_net_pnl": total_pnl,
        "max_drawdown_frac": 0.05,
    }


def _fold(k: int, selected: bool = True, sharpe: float = 1.0,
          pf: float = 1.2, bps: float = 3.0) -> dict:
    rep: dict = {
        "fold": k,
        "train": [f"2021-0{1+k}-01", f"2021-0{1+k}-28"],
        "test":  [f"2021-0{2+k}-01", f"2021-0{2+k}-28"],
        "rejection_reasons": {},
    }
    if selected:
        rep["selected"] = "cand_abc"
        rep["test_metrics"] = _test_metrics(sharpe=sharpe, pf=pf, bps=bps)
    else:
        rep["selected"] = None
    return rep


def _report(folds: list[dict], meta_label: bool = False) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-08-03T00:00:00+00:00",
        "folds": folds,
        "outer_test_metrics_by_cost": {
            "baseline": _test_metrics()
        },
        "decision": "Candidate",
        "decision_reasons": ["all pre-declared gates passed"],
        "meta_label_enabled": meta_label,
        "bayes_search_enabled": False,
    }


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))


class _ImportMixin:
    """Lazy import so tests fail on ImportError rather than collection error."""

    @classmethod
    def _import(cls):
        import compare_runs as cr
        return cr


# ---------------------------------------------------------------------------
# TestPairedTtest
# ---------------------------------------------------------------------------

class TestPairedTtest(_ImportMixin, unittest.TestCase):

    def setUp(self):
        self.cr = self._import()

    def test_zero_diffs_raises_or_nan(self):
        """All diffs identical → std=0; function must not crash."""
        diffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = self.cr.paired_ttest(diffs)
        self.assertIn("t_stat", result)
        # t_stat should be nan or inf when std is 0
        self.assertTrue(not math.isfinite(result["t_stat"]) or result["t_stat"] == 0.0)

    def test_known_diffs_t_stat(self):
        """Verify t_stat against hand-calculated value."""
        diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.cr.paired_ttest(diffs)
        mean_d = statistics.mean(diffs)          # 3.0
        std_d  = statistics.stdev(diffs)         # ~1.581
        n      = len(diffs)
        expected_t = mean_d / (std_d / math.sqrt(n))
        self.assertAlmostEqual(result["t_stat"], expected_t, places=6)

    def test_p_value_in_range(self):
        diffs = [0.5, 1.0, 0.8, 1.2, 0.6, 0.9, 1.1]
        result = self.cr.paired_ttest(diffs)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_negative_mean_positive_p(self):
        """Negative mean diff → t_stat negative; p_value still in [0,1]."""
        diffs = [-1.0, -0.5, -0.8, -1.2, -0.6]
        result = self.cr.paired_ttest(diffs)
        self.assertLess(result["t_stat"], 0)
        self.assertGreaterEqual(result["p_value"], 0.0)

    def test_single_diff_raises(self):
        """Need at least 2 observations for a paired t-test."""
        with self.assertRaises((ValueError, ZeroDivisionError, statistics.StatisticsError)):
            self.cr.paired_ttest([1.0])

    def test_returns_mean_diff(self):
        diffs = [1.0, 3.0, 2.0]
        result = self.cr.paired_ttest(diffs)
        self.assertAlmostEqual(result["mean_diff"], 2.0)

    def test_returns_n(self):
        diffs = [0.1, 0.2, 0.3]
        result = self.cr.paired_ttest(diffs)
        self.assertEqual(result["n"], 3)


# ---------------------------------------------------------------------------
# TestExtractFoldMetrics
# ---------------------------------------------------------------------------

class TestExtractFoldMetrics(_ImportMixin, unittest.TestCase):

    def setUp(self):
        self.cr = self._import()

    def test_skips_no_selection_folds(self):
        folds = [_fold(0, selected=True), _fold(1, selected=False), _fold(2, selected=True)]
        m = self.cr.fold_metrics(folds)
        self.assertIn(0, m)
        self.assertNotIn(1, m)
        self.assertIn(2, m)

    def test_metrics_contain_sharpe(self):
        folds = [_fold(0, sharpe=1.5)]
        m = self.cr.fold_metrics(folds)
        self.assertAlmostEqual(m[0]["sharpe"], 1.5)

    def test_empty_folds(self):
        m = self.cr.fold_metrics([])
        self.assertEqual(m, {})

    def test_all_no_selection(self):
        folds = [_fold(k, selected=False) for k in range(5)]
        m = self.cr.fold_metrics(folds)
        self.assertEqual(m, {})


# ---------------------------------------------------------------------------
# TestCompare
# ---------------------------------------------------------------------------

class TestCompare(_ImportMixin, unittest.TestCase):

    def setUp(self):
        self.cr = self._import()

    def _run(self, base_folds, ml_folds):
        base = _report(base_folds, meta_label=False)
        ml   = _report(ml_folds,   meta_label=True)
        return self.cr.compare(base, ml)

    def test_common_folds_counted(self):
        base = [_fold(k) for k in range(5)]
        ml   = [_fold(k) for k in range(5)]
        result = self._run(base, ml)
        self.assertEqual(result["common_folds"], 5)

    def test_no_common_folds_returns_gracefully(self):
        """When baseline has folds 0-2 selected and ML has none, common=0."""
        base = [_fold(k) for k in range(3)]
        ml   = [_fold(k, selected=False) for k in range(3)]
        result = self._run(base, ml)
        self.assertEqual(result["common_folds"], 0)
        self.assertIsNone(result.get("paired_comparison"))

    def test_mean_diff_direction(self):
        """ML sharpe consistently higher → positive mean diff in sharpe."""
        base = [_fold(k, sharpe=0.5) for k in range(5)]
        ml   = [_fold(k, sharpe=1.5) for k in range(5)]
        result = self._run(base, ml)
        cmp = result["paired_comparison"]
        self.assertGreater(cmp["sharpe"]["mean_diff"], 0)

    def test_metrics_compared_list(self):
        base = [_fold(k) for k in range(3)]
        ml   = [_fold(k) for k in range(3)]
        result = self._run(base, ml)
        self.assertIn("metrics_compared", result)
        self.assertIn("sharpe", result["metrics_compared"])
        self.assertIn("mean_net_bps", result["metrics_compared"])

    def test_verdict_present(self):
        base = [_fold(k, sharpe=0.5) for k in range(8)]
        ml   = [_fold(k, sharpe=2.0) for k in range(8)]
        result = self._run(base, ml)
        self.assertIn("verdict", result)
        self.assertIn(result["verdict"],
                      ("ml_significantly_better", "no_significant_improvement",
                       "ml_significantly_worse", "insufficient_data"))

    def test_schema_version_in_output(self):
        result = self._run([_fold(0)], [_fold(0)])
        self.assertEqual(result["schema_version"], 1)


# ---------------------------------------------------------------------------
# TestCreateOnly
# ---------------------------------------------------------------------------

class TestCreateOnly(_ImportMixin, unittest.TestCase):

    def setUp(self):
        self.cr = self._import()

    def test_refuses_to_overwrite(self):
        base = _report([_fold(0)])
        ml   = _report([_fold(0)], meta_label=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                self.cr.write_report(base, ml, path)
        finally:
            os.unlink(path)

    def test_writes_valid_json(self):
        base = _report([_fold(0)])
        ml   = _report([_fold(0)], meta_label=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            self.cr.write_report(base, ml, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["schema_version"], 1)


# ---------------------------------------------------------------------------
# TestStdlibOnly
# ---------------------------------------------------------------------------

class TestStdlibOnly(unittest.TestCase):
    """compare_runs.py must not import numpy, scipy, lightgbm, or optuna."""

    _BANNED = {"numpy", "scipy", "lightgbm", "optuna", "pandas"}

    def test_no_banned_imports(self):
        src_path = pathlib.Path(__file__).parent.parent / "scripts" / "compare_runs.py"
        self.assertTrue(src_path.exists(), f"compare_runs.py not found at {src_path}")
        tree = ast.parse(src_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    self.assertNotIn(root, self._BANNED,
                                     f"banned import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    self.assertNotIn(root, self._BANNED,
                                     f"banned import: from {node.module}")


if __name__ == "__main__":
    unittest.main()
