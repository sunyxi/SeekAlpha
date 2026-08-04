import json
import tempfile
import unittest
from pathlib import Path

from orb.research_protocol import ResearchProtocol
from orb.strategy_contract import (
    DecisionReport,
    StrategySpec,
    create_only_write_report,
    render_decision_summary,
)


def valid_spec() -> dict:
    return {
        "schema_version": 1,
        "spec_id": "residual-momentum-v1",
        "strategy_family": "residual_momentum",
        "hypothesis": "Stocks with persistent sector-relative strength may continue to outperform over the declared horizon.",
        "universe": {
            "market": "US_EQUITIES",
            "membership_source": "point_in_time_constituents_v1",
            "point_in_time": True,
            "exclusions": ["earnings_event", "halted", "insufficient_liquidity"],
        },
        "features": [
            {"name": "sector_residual_return", "source": "daily_panel_v1", "as_of": "close", "transform": "cross_sectional_rank"}
        ],
        "label": {
            "name": "forward_net_return_5d",
            "horizon_days": 5,
            "definition": "forward_return_minus_baseline_cost",
            "cost_included": True,
        },
        "holding_period": {"min_days": 1, "max_days": 5, "overnight_allowed": True},
        "parameters": {
            "search_space_hash": "a" * 64,
            "declared": {"lookback_days": {"type": "int", "low": 20, "high": 60}},
        },
        "costs": {
            "scenarios": {"zero_bps_per_side": 0.0, "baseline_bps_per_side": 2.5, "double_bps_per_side": 5.0},
            "commission_model": "declared_broker_schedule_v1",
        },
        "protocol": {"protocol_id": "seekalpha_strategy_discovery_v1", "protocol_hash": "b" * 64},
        "data": {"manifest_hash": "c" * 64, "snapshot_id": "daily-panel-2026-06-30"},
        "experiment_budget": {"max_parameter_trials": 192, "max_model_trials": 50},
        "invalidation": ["point_in_time_membership_unavailable", "protocol_hash_changed", "posthoc_gate_change"],
    }


def valid_report(spec: StrategySpec) -> dict:
    return {
        "schema_version": 1,
        "report_id": "residual-momentum-v1-exp-001",
        "spec_id": spec.spec_id,
        "strategy_family": spec.strategy_family,
        "spec_hash": spec.spec_hash,
        "protocol_hash": spec.protocol_hash,
        "data_manifest_hash": spec.data_manifest_hash,
        "code_commit": "d" * 40,
        "experiment_budget": spec.experiment_budget,
        "decision": {
            "state": "No-Go",
            "reasons": ["baseline cost gate not met"],
            "gates": [{"name": "min_outer_sharpe", "threshold": 1.2, "observed": 0.4, "status": "failed"}],
        },
        "evidence": {"outer_test": {"metrics_by_cost": {"baseline": {"sharpe": 0.4}}}},
    }


class StrategyContractTests(unittest.TestCase):
    def test_spec_fixture_is_deterministic_and_contains_predeclared_fields(self):
        spec = StrategySpec.from_mapping(valid_spec())
        self.assertEqual(spec.spec_hash, StrategySpec.from_mapping(spec.to_mapping()).spec_hash)
        self.assertEqual(spec.experiment_budget["max_parameter_trials"], 192)
        self.assertTrue(spec.universe["point_in_time"])

    def test_missing_predeclared_field_is_rejected(self):
        raw = valid_spec()
        del raw["label"]
        with self.assertRaises(ValueError):
            StrategySpec.from_mapping(raw)

    def test_report_rejects_modified_protocol_or_data_hash(self):
        spec = StrategySpec.from_mapping(valid_spec())
        report = valid_report(spec)
        report["protocol_hash"] = "e" * 64
        with self.assertRaises(ValueError):
            DecisionReport.validate(report, spec)

        report = valid_report(spec)
        report["data_manifest_hash"] = "f" * 64
        with self.assertRaises(ValueError):
            DecisionReport.validate(report, spec)

    def test_report_state_and_gate_schema_are_validated(self):
        spec = StrategySpec.from_mapping(valid_spec())
        report = valid_report(spec)
        validated = DecisionReport.validate(report, spec)
        self.assertEqual(validated.decision_state, "No-Go")
        report["decision"]["state"] = "Candidate"
        report["decision"]["gates"][0]["status"] = "unknown"
        with self.assertRaises(ValueError):
            DecisionReport.validate(report, spec)

    def test_create_only_report_and_summary_are_reproducible(self):
        spec = StrategySpec.from_mapping(valid_spec())
        report = DecisionReport.validate(valid_report(spec), spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.json"
            create_only_write_report(path, report.to_mapping())
            with self.assertRaises(FileExistsError):
                create_only_write_report(path, report.to_mapping())
            summary = render_decision_summary(spec, report)
            self.assertIn("residual-momentum-v1-exp-001", summary)
            self.assertIn("No-Go", summary)
            self.assertEqual(json.loads(path.read_text())["spec_hash"], spec.spec_hash)


if __name__ == "__main__":
    unittest.main()
