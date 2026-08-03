"""Tests for src/orb/features/panel_builder.py."""

from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from orb.features.panel_builder import (
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    VOLUME,
    _aggregate_symbol,
    _rth_et_time,
    build_panel,
)

# Two consecutive NYSE trading days in January 2021 (EST, UTC-5).
# RTH = 14:30–20:59 UTC (09:30–15:59 ET).
_D1 = date(2021, 1, 4)   # Monday
_D2 = date(2021, 1, 5)   # Tuesday

_PRE_MARKET  = "2021-01-04T13:00:00+00:00"   # 08:00 ET — before open
_POST_MARKET = "2021-01-04T21:01:00+00:00"   # 16:01 ET — after close


def _write_csv(path: Path, rows: list[tuple]) -> None:
    """Write a gzip 1-min CSV with the standard header."""
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# RTH filter
# ---------------------------------------------------------------------------

class TestRthFilter(unittest.TestCase):

    def test_open_bar_included(self):
        result = _rth_et_time("2021-01-04T14:30:00+00:00")   # 09:30 ET
        self.assertIsNotNone(result)
        self.assertEqual(result[0], _D1)

    def test_last_rth_bar_included(self):
        result = _rth_et_time("2021-01-04T20:59:00+00:00")   # 15:59 ET
        self.assertIsNotNone(result)
        self.assertEqual(result[0], _D1)

    def test_close_at_1600_excluded(self):
        result = _rth_et_time("2021-01-04T21:00:00+00:00")   # 16:00 ET exactly
        self.assertIsNone(result)

    def test_pre_market_excluded(self):
        self.assertIsNone(_rth_et_time(_PRE_MARKET))

    def test_edt_open_bar_included(self):
        # June 1 2021 is EDT (UTC-4); RTH opens at 13:30 UTC
        result = _rth_et_time("2021-06-01T13:30:00+00:00")   # 09:30 ET
        self.assertIsNotNone(result)
        self.assertEqual(result[0], date(2021, 6, 1))

    def test_edt_post_market_excluded(self):
        # 20:00 UTC in EDT = 16:00 ET → excluded
        result = _rth_et_time("2021-06-01T20:00:00+00:00")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

class TestDailyAggregation(unittest.TestCase):
    """Verify OHLCV aggregation: open=first, H=max, L=min, close=last, vol=sum."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "AAA_2021-01-04_2021-01-05_1min.csv.gz"
        rows = [
            # pre-market Jan 4 — must be ignored
            (_PRE_MARKET, 99.0, 100.0, 98.0, 99.5, 1000),
            # three RTH bars on Jan 4
            ("2021-01-04T14:30:00+00:00", 100.0, 105.0, 98.0, 103.0, 5000),
            ("2021-01-04T15:00:00+00:00", 103.5, 107.0, 102.0, 106.0, 3000),
            ("2021-01-04T20:59:00+00:00", 106.5, 108.0, 105.0, 107.0, 2000),
            # post-market Jan 4 — must be ignored
            (_POST_MARKET, 107.0, 108.0, 106.0, 107.0, 500),
            # two RTH bars on Jan 5
            ("2021-01-05T14:30:00+00:00", 108.0, 110.0, 107.5, 109.0, 4000),
            ("2021-01-05T20:59:00+00:00", 109.5, 111.0, 109.0, 110.0, 6000),
        ]
        _write_csv(path, rows)
        self.agg = _aggregate_symbol(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dates_found(self):
        self.assertSetEqual(set(self.agg.keys()), {_D1, _D2})

    # D1 checks
    def test_d1_open_is_first_bar(self):
        self.assertAlmostEqual(self.agg[_D1][OPEN], 100.0)

    def test_d1_high_is_max(self):
        self.assertAlmostEqual(self.agg[_D1][HIGH], 108.0)

    def test_d1_low_is_min(self):
        self.assertAlmostEqual(self.agg[_D1][LOW], 98.0)

    def test_d1_close_is_last_bar(self):
        self.assertAlmostEqual(self.agg[_D1][CLOSE], 107.0)

    def test_d1_volume_excludes_premarket(self):
        self.assertAlmostEqual(self.agg[_D1][VOLUME], 10_000.0)  # 5000+3000+2000

    # D2 checks
    def test_d2_open_is_first_bar(self):
        self.assertAlmostEqual(self.agg[_D2][OPEN], 108.0)

    def test_d2_close_is_last_bar(self):
        self.assertAlmostEqual(self.agg[_D2][CLOSE], 110.0)

    def test_d2_volume_sum(self):
        self.assertAlmostEqual(self.agg[_D2][VOLUME], 10_000.0)  # 4000+6000


# ---------------------------------------------------------------------------
# build_panel integration
# ---------------------------------------------------------------------------

class TestBuildPanel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cache = Path(self.tmp.name)

        # Symbol A: data only on D1
        _write_csv(cache / "AAA_2021-01-04_2021-01-05_1min.csv.gz", [
            ("2021-01-04T14:30:00+00:00", 10.0, 11.0, 9.5, 10.5, 1000),
        ])
        # Symbol B: data on D1 and D2
        _write_csv(cache / "BBB_2021-01-04_2021-01-05_1min.csv.gz", [
            ("2021-01-04T14:30:00+00:00", 20.0, 21.0, 19.5, 20.5, 2000),
            ("2021-01-05T14:30:00+00:00", 21.0, 22.0, 20.5, 21.5, 2500),
        ])
        self.panel = build_panel(cache, start=_D1, end=_D2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_symbols_sorted_alphabetically(self):
        self.assertEqual(self.panel.symbols, ("AAA", "BBB"))

    def test_dates_ascending(self):
        self.assertEqual(list(self.panel.dates), [_D1, _D2])

    def test_panel_shape(self):
        self.assertEqual(self.panel.panel.shape, (2, 2, 5))

    def test_symbol_a_d1_values(self):
        i = self.panel.symbols.index("AAA")
        j = list(self.panel.dates).index(_D1)
        row = self.panel.panel[i, j]
        self.assertAlmostEqual(row[OPEN],   10.0)
        self.assertAlmostEqual(row[CLOSE],  10.5)
        self.assertAlmostEqual(row[VOLUME], 1000.0)

    def test_symbol_a_d2_forward_filled(self):
        """A has no D2 bar → O=H=L=C=prev_close, vol=0."""
        i = self.panel.symbols.index("AAA")
        j = list(self.panel.dates).index(_D2)
        row = self.panel.panel[i, j]
        prev_close = 10.5
        self.assertAlmostEqual(row[OPEN],   prev_close)
        self.assertAlmostEqual(row[HIGH],   prev_close)
        self.assertAlmostEqual(row[LOW],    prev_close)
        self.assertAlmostEqual(row[CLOSE],  prev_close)
        self.assertAlmostEqual(row[VOLUME], 0.0)

    def test_symbol_b_both_days(self):
        i = self.panel.symbols.index("BBB")
        self.assertAlmostEqual(self.panel.panel[i, 0, CLOSE], 20.5)
        self.assertAlmostEqual(self.panel.panel[i, 1, CLOSE], 21.5)

    def test_no_data_before_first_bar_is_nan(self):
        """First day with no prior data must stay NaN (not forward-filled)."""
        # Only A has no data before D1 — its D1 row was present, not NaN.
        # Test with a 3-day range where first day is missing:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            # D1 = Jan 4, but data only starts Jan 5
            _write_csv(cache / "ZZZ_2021-01-05_2021-01-05_1min.csv.gz", [
                ("2021-01-05T14:30:00+00:00", 50.0, 51.0, 49.0, 50.5, 3000),
            ])
            p = build_panel(cache, start=_D1, end=_D2)
        i = p.symbols.index("ZZZ")
        j = list(p.dates).index(_D1)
        self.assertTrue(np.all(np.isnan(p.panel[i, j])))


class TestToAlpha101Kwargs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cache = Path(self.tmp.name)
        _write_csv(cache / "XXX_2021-01-04_2021-01-04_1min.csv.gz", [
            ("2021-01-04T14:30:00+00:00", 10, 11, 9, 10.5, 1000),
        ])
        _write_csv(cache / "YYY_2021-01-04_2021-01-04_1min.csv.gz", [
            ("2021-01-04T14:30:00+00:00", 20, 21, 19, 20.5, 2000),
        ])
        self.panel = build_panel(cache)

    def tearDown(self):
        self.tmp.cleanup()

    def test_kwargs_keys(self):
        kwargs = self.panel.to_alpha101_kwargs()
        self.assertSetEqual(set(kwargs), {"open_", "high", "low", "close", "volume"})

    def test_shapes_are_T_N(self):
        T = len(self.panel.dates)
        N = len(self.panel.symbols)
        kwargs = self.panel.to_alpha101_kwargs()
        for key, arr in kwargs.items():
            with self.subTest(key=key):
                self.assertEqual(arr.shape, (T, N))


class TestBuildPanelErrors(unittest.TestCase):

    def test_empty_dir_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_panel(tmp)

    def test_symbol_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _write_csv(cache / "AAA_2021-01-04_2021-01-04_1min.csv.gz", [
                ("2021-01-04T14:30:00+00:00", 10, 11, 9, 10, 100),
            ])
            _write_csv(cache / "BBB_2021-01-04_2021-01-04_1min.csv.gz", [
                ("2021-01-04T14:30:00+00:00", 20, 21, 19, 20, 200),
            ])
            p = build_panel(cache, symbols=["AAA"])
        self.assertEqual(p.symbols, ("AAA",))


if __name__ == "__main__":
    unittest.main()
