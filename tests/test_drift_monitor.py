"""Tests for scripts/drift_monitor.py — per-metric fold drift detection.

All tests use synthetic wf_select.py report dicts; no real data needed.
drift_monitor.py is stdlib-only.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _test_metrics(sharpe: float = 0.8, pf: float = 1.15, bps: float = 4.0,
                  trades: int = 40, win_rate: float = 0.55,
                  total_pnl: float = 500.0) -> dict:
    return {
        "trades": trades, "win_rate": win_rate,
        "profit_factor": pf, "sharpe": sharpe,
        "mean_net_bps": bps, "total_net_pnl": total_pnl,
        "max_drawdown_frac": 0.05,
    }


def _fold(k: int, selected: bool = True, **metric_kw) -> dict:
    rep: dict = {"fold": k,
                 "train": [f"2021-0{1+k%8}-01", f"2021-0{1+k%8}-28"],
                 "test":  [f"2021-0{2+k%8}-01", f"2021-0{2+k%8}-15"],
                 "rejection_reasons": {}}
    if selected:
        rep["selected"] = "cand_abc"
        rep["test_metrics"] = _test_metrics(**metric_kw)
    else:
        rep["selected"] = None
    return rep


def _report(folds: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-08-03T00:00:00+00:00",
        "folds": folds,
        "outer_test_metrics_by_cost": {"baseline": _test_metrics()},
        "decision": "Candidate",
        "decision_reasons": ["all pre-declared gates passed"],
        "meta_label_enabled": False,
        "bayes_search_enabled": False,
    }


def _stable_report(n: int = 10) -> dict:
    """All folds with nearly identical metrics — no drift."""
    return _report([_fold(k) for k in range(n)])


def _drifting_report(n: int = 10, recent_sharpe: float = -0.5) -> dict:
    """Reference folds with varied good sharpe; last 30% have low sharpe."""
    split = int(n * 0.7)
    # Vary reference sharpes so std > 0
    ref_sharpes = [0.6 + 0.2 * (k % 4) for k in range(split)]
    folds = [_fold(k, sharpe=s) for k, s in enumerate(ref_sharpes)]
    folds += [_fold(k + split, sharpe=recent_sharpe) for k in range(n - split)]
    return _report(folds)


# ---------------------------------------------------------------------------
# TestSplitFolds
# ---------------------------------------------------------------------------

class TestSplitFolds(unittest.TestCase):
    def setUp(self):
        import drift_monitor as dm
        self.dm = dm

    def test_returns_two_lists(self):
        folds = [_fold(k) for k in range(10)]
        ref, rec = self.dm.split_folds(folds, recent_folds=3)
        self.assertIsInstance(ref, list)
        self.assertIsInstance(rec, list)

    def test_counts_add_up_to_selected(self):
        folds = ([_fold(k) for k in range(8)]
                 + [_fold(k + 8, selected=False) for k in range(2)])
        ref, rec = self.dm.split_folds(folds, recent_folds=2)
        self.assertEqual(len(ref) + len(rec), 8)

    def test_recent_folds_at_end(self):
        folds = [_fold(k) for k in range(10)]
        ref, rec = self.dm.split_folds(folds, recent_folds=3)
        self.assertEqual(len(rec), 3)
        ref_idx = [f["fold"] for f in ref]
        rec_idx = [f["fold"] for f in rec]
        self.assertTrue(max(ref_idx) < min(rec_idx))

    def test_single_fold_recent(self):
        folds = [_fold(k) for k in range(5)]
        ref, rec = self.dm.split_folds(folds, recent_folds=1)
        self.assertEqual(len(rec), 1)
        self.assertEqual(len(ref), 4)

    def test_insufficient_reference_returns_empty(self):
        folds = [_fold(k) for k in range(2)]
        ref, rec = self.dm.split_folds(folds, recent_folds=2)
        self.assertEqual(ref, [])

    def test_skips_unselected_folds(self):
        folds = ([_fold(k) for k in range(5)]
                 + [_fold(5, selected=False)])
        ref, rec = self.dm.split_folds(folds, recent_folds=2)
        self.assertEqual(len(ref) + len(rec), 5)


# ---------------------------------------------------------------------------
# TestZScore
# ---------------------------------------------------------------------------

class TestZScore(unittest.TestCase):
    def setUp(self):
        import drift_monitor as dm
        self.dm = dm

    def test_zero_std_returns_nan(self):
        result = self.dm.metric_zscore([1.0, 1.0, 1.0], [1.5, 1.5])
        self.assertTrue(math.isnan(result))

    def test_known_zscore(self):
        import statistics
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        rec = [0.0, 0.5]
        mean_ref = statistics.mean(ref)
        std_ref  = statistics.stdev(ref)
        mean_rec = statistics.mean(rec)
        expected = (mean_rec - mean_ref) / std_ref
        result = self.dm.metric_zscore(ref, rec)
        self.assertAlmostEqual(result, expected, places=6)

    def test_positive_for_improvement(self):
        result = self.dm.metric_zscore([1.0, 2.0, 1.5, 2.5], [5.0, 6.0])
        self.assertGreater(result, 0)

    def test_negative_for_degradation(self):
        result = self.dm.metric_zscore([4.0, 5.0, 6.0, 5.0, 4.5], [0.0, 0.5])
        self.assertLess(result, 0)

    def test_insufficient_reference_returns_nan(self):
        result = self.dm.metric_zscore([1.0], [2.0])
        self.assertTrue(math.isnan(result))


# ---------------------------------------------------------------------------
# TestMonitorReport
# ---------------------------------------------------------------------------

class TestMonitorReport(unittest.TestCase):
    def setUp(self):
        import drift_monitor as dm
        self.dm = dm

    def test_stable_verdict(self):
        report = _stable_report(n=12)
        result = self.dm.monitor(report, recent_folds=3)
        self.assertIn(result["verdict"], ("stable", "moderate_drift"))

    def test_significant_drift_detected(self):
        report = _drifting_report(n=12, recent_sharpe=-2.0)
        result = self.dm.monitor(report, recent_folds=4)
        self.assertIn(result["verdict"],
                      ("significant_drift", "moderate_drift"),
                      "expected drift to be detected")

    def test_insufficient_data_verdict(self):
        report = _report([_fold(0), _fold(1)])
        result = self.dm.monitor(report, recent_folds=2)
        self.assertEqual(result["verdict"], "insufficient_data")

    def test_schema_version(self):
        result = self.dm.monitor(_stable_report(10), recent_folds=3)
        self.assertEqual(result["schema_version"], 1)

    def test_metrics_listed(self):
        result = self.dm.monitor(_stable_report(10), recent_folds=3)
        self.assertIn("metrics", result)
        self.assertIn("sharpe", result["metrics"])
        self.assertIn("mean_net_bps", result["metrics"])

    def test_each_metric_has_zscore_and_verdict(self):
        result = self.dm.monitor(_stable_report(10), recent_folds=3)
        for name, m in result["metrics"].items():
            self.assertIn("z_score",  m, f"missing z_score for {name}")
            self.assertIn("verdict",  m, f"missing verdict for {name}")
            self.assertIn("ref_mean", m, f"missing ref_mean for {name}")
            self.assertIn("rec_mean", m, f"missing rec_mean for {name}")

    def test_reference_and_recent_fold_counts(self):
        result = self.dm.monitor(_stable_report(10), recent_folds=3)
        self.assertEqual(result["recent_folds"], 3)
        self.assertEqual(result["reference_folds"], 7)

    def test_no_selected_folds(self):
        report = _report([_fold(k, selected=False) for k in range(8)])
        result = self.dm.monitor(report, recent_folds=2)
        self.assertEqual(result["verdict"], "insufficient_data")


# ---------------------------------------------------------------------------
# TestWriteReport (create-only)
# ---------------------------------------------------------------------------

class TestWriteReport(unittest.TestCase):
    def setUp(self):
        import drift_monitor as dm
        self.dm = dm

    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "drift.json")
            self.dm.write_report(_stable_report(10), path, recent_folds=3)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["schema_version"], 1)

    def test_refuses_to_overwrite(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                self.dm.write_report(_stable_report(10), path, recent_folds=3)
        finally:
            os.unlink(path)

    def test_no_tmp_file_left(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "drift.json")
            self.dm.write_report(_stable_report(10), path, recent_folds=3)
            tmps = [f for f in os.listdir(d) if f.endswith(".tmp")]
            self.assertEqual(tmps, [])


# ---------------------------------------------------------------------------
# TestStdlibOnly
# ---------------------------------------------------------------------------

class TestStdlibOnly(unittest.TestCase):
    _BANNED = {"numpy", "scipy", "lightgbm", "optuna", "pandas"}

    def test_no_banned_imports(self):
        src = pathlib.Path(__file__).parent.parent / "scripts" / "drift_monitor.py"
        self.assertTrue(src.exists(), f"drift_monitor.py not found")
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    continue
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else ([node.module] if node.module else []))
                for name in names:
                    root = (name or "").split(".")[0]
                    self.assertNotIn(root, self._BANNED,
                                     f"banned import in drift_monitor.py: {name}")


if __name__ == "__main__":
    unittest.main()
