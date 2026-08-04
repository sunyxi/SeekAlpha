import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from orb.research_protocol import (
    RetentionAlreadyConsumedError,
    RetentionNotAvailableError,
    RetentionLedger,
    ResearchProtocol,
)


class ResearchProtocolTests(unittest.TestCase):
    def test_default_protocol_is_disjoint_and_has_frozen_controls(self):
        protocol = ResearchProtocol.load(
            Path("src/orb/research_protocol.json")
        )

        self.assertLess(protocol.windows.development.end, protocol.windows.outer_test.start)
        self.assertLess(protocol.windows.outer_test.end, protocol.windows.retention.start)
        self.assertGreater(protocol.budget.max_experiments_per_strategy_family, 0)
        self.assertGreater(protocol.budget.max_parameter_trials_per_experiment, 0)
        self.assertEqual(protocol.validation.purge_days, 20)
        self.assertEqual(protocol.validation.embargo_days, 5)
        self.assertEqual(protocol.costs["baseline_bps_per_side"], 2.5)
        self.assertTrue(protocol.gates)
        self.assertTrue(protocol.random_seeds)

    def test_rejects_overlapping_windows(self):
        raw = {
            "schema_version": 1,
            "protocol_id": "test",
            "date_windows": {
                "development": {"start": "2021-01-01", "end": "2022-12-31"},
                "outer_test": {"start": "2022-12-01", "end": "2023-12-31"},
                "retention": {"start": "2024-01-01", "end": "2024-12-31"},
            },
            "experiment_budget": {
                "max_experiments_per_strategy_family": 3,
                "max_total_experiments": 12,
                "max_parameter_trials_per_experiment": 10,
                "max_model_trials_per_experiment": 5,
            },
            "validation": {
                "train_days": 252,
                "validation_days": 63,
                "outer_test_days": 63,
                "step_days": 63,
                "purge_days": 20,
                "embargo_days": 5,
            },
            "costs": {
                "zero_bps_per_side": 0,
                "baseline_bps_per_side": 2.5,
                "double_bps_per_side": 5,
            },
            "random_seeds": {"default": 42},
            "gates": {"min_outer_trades": 1},
        }

        with self.assertRaises(ValueError, msg="overlapping windows must be rejected"):
            ResearchProtocol.from_mapping(raw)

    def test_retention_can_be_read_once_across_ledger_instances(self):
        protocol = ResearchProtocol.load(Path("src/orb/research_protocol.json"))
        with tempfile.TemporaryDirectory() as tmp:
            first = RetentionLedger(tmp, protocol, as_of=protocol.retention_available_after)
            calls = []

            result = first.read_once(
                "residual-momentum-001",
                lambda window: calls.append(window.start) or "retention-result",
                reader="test",
            )

            self.assertEqual(result, "retention-result")
            self.assertEqual(calls, [protocol.windows.retention.start])

            second = RetentionLedger(tmp, protocol, as_of=protocol.retention_available_after)
            with self.assertRaises(RetentionAlreadyConsumedError):
                second.read_once(
                    "residual-momentum-001",
                    lambda _window: self.fail("loader must not run twice"),
                    reader="test-again",
                )

            receipt = next(Path(tmp).glob("*.json"))
            audit = json.loads(receipt.read_text())
            self.assertEqual(audit["protocol_hash"], protocol.protocol_hash)
            self.assertEqual(audit["experiment_id"], "residual-momentum-001")

    def test_failed_retention_loader_still_consumes_access(self):
        protocol = ResearchProtocol.load(Path("src/orb/research_protocol.json"))
        with tempfile.TemporaryDirectory() as tmp:
            ledger = RetentionLedger(tmp, protocol, as_of=protocol.retention_available_after)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                ledger.read_once(
                    "failed-read-001",
                    lambda _window: (_ for _ in ()).throw(RuntimeError("fixture failure")),
                )

            with self.assertRaises(RetentionAlreadyConsumedError):
                ledger.read_once("failed-read-001", lambda _window: "must not read")

    def test_retention_is_unavailable_before_declared_release_date(self):
        protocol = ResearchProtocol.load(Path("src/orb/research_protocol.json"))
        with tempfile.TemporaryDirectory() as tmp:
            ledger = RetentionLedger(tmp, protocol, as_of=date(2026, 8, 4))
            with self.assertRaises(RetentionNotAvailableError):
                ledger.read_once("too-early-001", lambda _window: "must not read")

    def test_protocol_hash_is_stable_for_same_mapping(self):
        protocol = ResearchProtocol.load(Path("src/orb/research_protocol.json"))
        clone = ResearchProtocol.from_mapping(protocol.to_mapping())
        self.assertEqual(protocol.protocol_hash, clone.protocol_hash)
        self.assertIsInstance(protocol.windows.retention.start, date)


if __name__ == "__main__":
    unittest.main()
