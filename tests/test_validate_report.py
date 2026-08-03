"""Tests for scripts/validate_report.py.

No file I/O beyond what validate() itself does — synthetic dicts used
for all positive and negative cases.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_report import (
    _FROZEN_GRID_HASH,
    _SCHEMA_VERSION,
    validate,
    validate_file,
)


def _minimal_valid() -> dict:
    """Return a minimal fully-valid report dict."""
    cost_scenario = {
        "trades": 100,
        "win_rate": 0.5,
        "profit_factor": 1.1,
        "sharpe": 0.6,
        "mean_net_bps": 2.0,
        "total_net_pnl": 500.0,
        "max_drawdown_frac": 0.05,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:00:00",
        "input_meta": [
            {
                "grid_spec_hash": _FROZEN_GRID_HASH,
                "candidate_shard": 0,
                "candidate_shards": 1,
            }
        ],
        "fold_definition": {
            "train_days": 252,
            "test_days": 63,
            "step_days": 63,
        },
        "selection_gates": {
            "min_trades": 30,
            "min_train_sharpe": 0.0,
            "min_train_pf": 1.0,
            "validation": "last 20% of train, net pnl > 0",
        },
        "decision_gates": {
            "min_outer_trades": 100,
            "min_outer_sharpe": 0.5,
            "min_outer_pf": 1.1,
        },
        "folds": [{"fold": 0, "selected_candidate": None}],
        "outer_test_metrics_by_cost": {
            "zero": dict(cost_scenario),
            "baseline": dict(cost_scenario),
            "double": dict(cost_scenario),
        },
        "symbol_attribution_baseline": {"SPY": 100.0, "QQQ": -50.0},
        "decision": "No-Go",
        "decision_reasons": ["outer sharpe -0.03 <= 0.5"],
    }


class TestValidateHappyPath(unittest.TestCase):

    def test_minimal_valid_has_no_errors(self):
        errors = validate(_minimal_valid())
        self.assertEqual(errors, [])

    def test_candidate_decision_accepted(self):
        d = _minimal_valid()
        d["decision"] = "Candidate"
        d["decision_reasons"] = []
        errors = validate(d)
        self.assertEqual(errors, [])

    def test_extra_fields_tolerated(self):
        d = _minimal_valid()
        d["extra_future_field"] = "anything"
        errors = validate(d)
        self.assertEqual(errors, [])


class TestSchemaVersion(unittest.TestCase):

    def test_wrong_schema_version(self):
        d = _minimal_valid()
        d["schema_version"] = 2
        errors = validate(d)
        self.assertTrue(any("schema_version" in e for e in errors))

    def test_missing_schema_version(self):
        d = _minimal_valid()
        del d["schema_version"]
        errors = validate(d)
        # Either missing-field error or version-mismatch error
        self.assertTrue(any("schema_version" in e for e in errors))


class TestRequiredTopLevelFields(unittest.TestCase):

    def _check_missing(self, field: str) -> None:
        d = _minimal_valid()
        del d[field]
        errors = validate(d)
        self.assertTrue(
            any(field in e for e in errors),
            f"expected error mentioning {field!r}; got {errors}",
        )

    def test_missing_generated_at(self):
        self._check_missing("generated_at")

    def test_missing_input_meta(self):
        self._check_missing("input_meta")

    def test_missing_fold_definition(self):
        self._check_missing("fold_definition")

    def test_missing_selection_gates(self):
        self._check_missing("selection_gates")

    def test_missing_decision_gates(self):
        self._check_missing("decision_gates")

    def test_missing_folds(self):
        self._check_missing("folds")

    def test_missing_outer_test_metrics(self):
        self._check_missing("outer_test_metrics_by_cost")

    def test_missing_symbol_attribution(self):
        self._check_missing("symbol_attribution_baseline")

    def test_missing_decision(self):
        self._check_missing("decision")

    def test_missing_decision_reasons(self):
        self._check_missing("decision_reasons")


class TestSubFieldValidation(unittest.TestCase):

    def test_fold_definition_missing_sub_field(self):
        d = _minimal_valid()
        del d["fold_definition"]["train_days"]
        errors = validate(d)
        self.assertTrue(any("fold_definition" in e and "train_days" in e
                            for e in errors))

    def test_selection_gates_missing_sub_field(self):
        d = _minimal_valid()
        del d["selection_gates"]["min_trades"]
        errors = validate(d)
        self.assertTrue(any("selection_gates" in e and "min_trades" in e
                            for e in errors))

    def test_decision_gates_missing_sub_field(self):
        d = _minimal_valid()
        del d["decision_gates"]["min_outer_sharpe"]
        errors = validate(d)
        self.assertTrue(any("decision_gates" in e and "min_outer_sharpe" in e
                            for e in errors))

    def test_missing_cost_scenario(self):
        d = _minimal_valid()
        del d["outer_test_metrics_by_cost"]["baseline"]
        errors = validate(d)
        self.assertTrue(any("baseline" in e for e in errors))

    def test_cost_scenario_missing_field(self):
        d = _minimal_valid()
        del d["outer_test_metrics_by_cost"]["zero"]["profit_factor"]
        errors = validate(d)
        self.assertTrue(any("profit_factor" in e and "zero" in e for e in errors))


class TestDecisionValidation(unittest.TestCase):

    def test_invalid_decision_string(self):
        d = _minimal_valid()
        d["decision"] = "Maybe"
        errors = validate(d)
        self.assertTrue(any("decision" in e for e in errors))

    def test_none_decision(self):
        d = _minimal_valid()
        d["decision"] = None
        errors = validate(d)
        self.assertTrue(any("decision" in e for e in errors))

    def test_decision_reasons_not_a_list(self):
        d = _minimal_valid()
        d["decision_reasons"] = "single string"
        errors = validate(d)
        self.assertTrue(any("decision_reasons" in e for e in errors))


class TestGridHashValidation(unittest.TestCase):

    def test_wrong_hash_flagged(self):
        d = _minimal_valid()
        d["input_meta"][0]["grid_spec_hash"] = "deadbeef12345678"
        errors = validate(d)
        self.assertTrue(any("grid_spec_hash" in e for e in errors))

    def test_correct_hash_passes(self):
        d = _minimal_valid()
        d["input_meta"][0]["grid_spec_hash"] = _FROZEN_GRID_HASH
        errors = validate(d)
        self.assertEqual(errors, [])

    def test_empty_input_meta_flagged(self):
        d = _minimal_valid()
        d["input_meta"] = []
        errors = validate(d)
        self.assertTrue(any("input_meta" in e for e in errors))

    def test_symbol_attribution_empty_dict_flagged(self):
        d = _minimal_valid()
        d["symbol_attribution_baseline"] = {}
        errors = validate(d)
        self.assertTrue(any("symbol_attribution" in e for e in errors))


class TestValidateFile(unittest.TestCase):

    def test_valid_json_file_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            p.write_text(json.dumps(_minimal_valid()))
            errors = validate_file(p)
        self.assertEqual(errors, [])

    def test_invalid_json_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            p.write_text("{not valid json")
            errors = validate_file(p)
        self.assertTrue(any("JSON" in e for e in errors))

    def test_missing_file_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nonexistent.json"
            errors = validate_file(p)
        self.assertTrue(errors)

    def test_invalid_report_returns_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            bad = {"schema_version": 99}
            p.write_text(json.dumps(bad))
            errors = validate_file(p)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
