import unittest
from datetime import date
from pathlib import Path

from orb.data_source_policy import (
    DataSourcePolicy,
    PointInTimeUniverse,
)


class DataSourcePolicyTests(unittest.TestCase):
    def test_policy_selects_primary_and_fallback_without_credentials(self):
        policy = DataSourcePolicy.load(Path("src/orb/data_source_policy.json"))
        self.assertEqual(policy.decision, "selected_with_blockers")
        self.assertEqual(policy.primary.provider, "databento")
        self.assertEqual(policy.fallback.provider, "alpaca")
        self.assertIn("historical_minute_ohlcv", policy.primary.capabilities)
        self.assertIn("bid_ask", policy.primary.capabilities)
        self.assertTrue(policy.licensing["credentials_from_environment_only"])
        self.assertTrue(policy.blockers)

    def test_universe_reconstruction_uses_listing_intervals_not_current_constituents(self):
        universe = PointInTimeUniverse.from_records([
            {"instrument_id": "A", "symbol": "AAA", "asset_type": "common_stock", "listed_from": "2020-01-01", "listed_to": "2023-06-30"},
            {"instrument_id": "B", "symbol": "BBB", "asset_type": "common_stock", "listed_from": "2022-01-01", "listed_to": None},
            {"instrument_id": "ETF", "symbol": "ETF", "asset_type": "etf", "listed_from": "2010-01-01", "listed_to": None},
        ])
        self.assertEqual(universe.members_at(date(2022, 6, 1)), ("A", "B"))
        self.assertEqual(universe.members_at(date(2024, 1, 1)), ("B",))
        self.assertEqual(universe.resolve_symbol("AAA", date(2022, 6, 1)), "A")
        with self.assertRaises(ValueError):
            universe.resolve_symbol("AAA", date(2024, 1, 1))

    def test_universe_rejects_overlapping_identity_intervals_and_invalid_symbols(self):
        with self.assertRaises(ValueError):
            PointInTimeUniverse.from_records([
                {"instrument_id": "A", "symbol": "AAA", "asset_type": "common_stock", "listed_from": "2020-01-01", "listed_to": "2023-01-01"},
                {"instrument_id": "A", "symbol": "AAA2", "asset_type": "common_stock", "listed_from": "2022-01-01", "listed_to": None},
            ])
        with self.assertRaises(ValueError):
            PointInTimeUniverse.from_records([
                {"instrument_id": "A", "symbol": "AAA", "asset_type": "common_stock", "listed_from": "2020-01-01", "listed_to": "2019-01-01"},
            ])

    def test_policy_requires_minimum_history_and_regime_coverage(self):
        policy = DataSourcePolicy.load(Path("src/orb/data_source_policy.json"))
        self.assertLessEqual(policy.universe.minimum_history_start, date(2021, 1, 4))
        self.assertGreaterEqual(policy.universe.minimum_daily_universe_size, 500)
        self.assertGreaterEqual(policy.universe.minimum_regime_years, 5)
        self.assertEqual(policy.universe.interval_end_semantics, "exclusive")


if __name__ == "__main__":
    unittest.main()
