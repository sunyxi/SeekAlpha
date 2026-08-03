"""Tests for per-symbol data quality statistics (src/orb/quality.py).

All tests use synthetic bar rows — no file I/O, no network calls.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from orb.quality import (
    SymbolQuality,
    _expected_bars,
    compute_quality,
    format_quality_report,
    write_quality_json,
)

NY = ZoneInfo("America/New_York")

# Four consecutive NYSE trading days used as fixtures.
# Week of 2024-01-02: Tue, Wed, Thu, Fri — all normal sessions.
_DAYS = [date(2024, 1, d) for d in (2, 3, 4, 5)]
_START = "2024-01-01"
_END = "2024-01-07"

_NORMAL_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)


def _is_trading_day_fixture(d: date) -> bool:
    """Treat only _DAYS as trading days for test isolation."""
    return d in _DAYS


def _session_end_fixture(d: date) -> time:
    return _NORMAL_CLOSE


def _make_rows(d: date, count: int, stale: int = 0) -> list:
    """Synthetic minute_rows for date d: `count` bars starting at 09:30."""
    rows = []
    for i in range(count):
        t = datetime.combine(d, time(9, 30)) + timedelta(minutes=i)
        vol = 0.0 if i < stale else 1000.0
        rows.append((t.replace(tzinfo=NY), 100.0, 101.0, 99.0, 100.0, vol))
    return rows


def _all_full(stale: int = 0) -> list:
    """390 bars for each of the four fixture days."""
    rows = []
    for d in _DAYS:
        rows.extend(_make_rows(d, 390, stale=stale))
    return rows


class TestExpectedBars(unittest.TestCase):

    def test_normal_session_is_390(self):
        self.assertEqual(_expected_bars(time(16, 0)), 390)

    def test_half_day_is_210(self):
        self.assertEqual(_expected_bars(time(13, 0)), 210)


class TestComputeQualityHappyPath(unittest.TestCase):

    def test_full_sessions_counted(self):
        q = compute_quality("SPY", _all_full(), _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.total_sessions, 4)
        self.assertEqual(q.full_bar_sessions, 4)
        self.assertEqual(q.partial_sessions, 0)
        self.assertEqual(q.zero_bar_sessions, 0)

    def test_gap_rate_zero_when_all_full(self):
        q = compute_quality("SPY", _all_full(), _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.gap_rate_pct, 0.0)
        self.assertFalse(q.has_warning())

    def test_max_consecutive_gaps_zero_when_all_full(self):
        q = compute_quality("SPY", _all_full(), _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.max_consecutive_gaps, 0)

    def test_stale_bar_count_zero_when_all_have_volume(self):
        q = compute_quality("SPY", _all_full(), _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.stale_bar_count, 0)


class TestComputeQualityGaps(unittest.TestCase):

    def test_zero_bar_session_counted(self):
        # Only provide bars for 3 of the 4 trading days
        rows = (
            _make_rows(_DAYS[0], 390)
            + _make_rows(_DAYS[1], 390)
            # _DAYS[2] missing
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.zero_bar_sessions, 1)
        self.assertEqual(q.total_sessions, 4)

    def test_gap_rate_computed_correctly(self):
        # 1 missing out of 4 → 25%
        rows = (
            _make_rows(_DAYS[0], 390)
            + _make_rows(_DAYS[1], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertAlmostEqual(q.gap_rate_pct, 25.0, places=2)
        self.assertTrue(q.has_warning())

    def test_gap_rate_below_threshold_no_warning(self):
        # gap_rate = 0 → no warning
        q = compute_quality("SPY", _all_full(), _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertFalse(q.has_warning(threshold_pct=1.0))

    def test_max_consecutive_gaps_one(self):
        rows = (
            _make_rows(_DAYS[0], 390)
            # _DAYS[1] missing — gap of 1
            + _make_rows(_DAYS[2], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.max_consecutive_gaps, 1)

    def test_max_consecutive_gaps_two(self):
        # _DAYS[1] and _DAYS[2] both missing → run of 2
        rows = (
            _make_rows(_DAYS[0], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.max_consecutive_gaps, 2)

    def test_all_sessions_missing(self):
        q = compute_quality("SPY", [], _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.zero_bar_sessions, 4)
        self.assertEqual(q.full_bar_sessions, 0)
        self.assertEqual(q.gap_rate_pct, 100.0)
        self.assertEqual(q.max_consecutive_gaps, 4)


class TestComputeQualityPartialSessions(unittest.TestCase):

    def test_partial_session_counted(self):
        rows = (
            _make_rows(_DAYS[0], 390)
            + _make_rows(_DAYS[1], 200)  # fewer than 390
            + _make_rows(_DAYS[2], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.partial_sessions, 1)
        self.assertEqual(q.full_bar_sessions, 3)

    def test_partial_session_not_a_gap(self):
        rows = (
            _make_rows(_DAYS[0], 1)  # only 1 bar — partial, not a gap
            + _make_rows(_DAYS[1], 390)
            + _make_rows(_DAYS[2], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.zero_bar_sessions, 0)
        self.assertEqual(q.partial_sessions, 1)


class TestComputeQualityStale(unittest.TestCase):

    def test_stale_bars_counted(self):
        # First 5 bars of day 0 have volume=0
        rows = (
            _make_rows(_DAYS[0], 390, stale=5)
            + _make_rows(_DAYS[1], 390)
            + _make_rows(_DAYS[2], 390)
            + _make_rows(_DAYS[3], 390)
        )
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.stale_bar_count, 5)

    def test_stale_bars_do_not_affect_gap_rate(self):
        # A session with all-zero-volume bars is still not a gap
        rows = _all_full(stale=390)  # every bar has vol=0
        q = compute_quality("SPY", rows, _START, _END,
                            _is_trading_day_fixture, _session_end_fixture)
        self.assertEqual(q.zero_bar_sessions, 0)
        self.assertEqual(q.stale_bar_count, 4 * 390)


class TestComputeQualityHalfDay(unittest.TestCase):

    def test_half_day_expected_bars_is_210(self):
        # Use Black Friday 2024 (2024-11-29) — a real half-day
        from orb.calendar import is_trading_day, session_end

        half_day = date(2024, 11, 29)
        rows = _make_rows(half_day, 210)  # exactly 210 bars (9:30–12:59)
        q = compute_quality("SPY", rows, "2024-11-29", "2024-11-29",
                            is_trading_day, session_end)
        # 210 >= expected(13:00)=210 → full session
        self.assertEqual(q.full_bar_sessions, 1)
        self.assertEqual(q.partial_sessions, 0)
        self.assertEqual(q.zero_bar_sessions, 0)

    def test_half_day_fewer_bars_is_partial(self):
        from orb.calendar import is_trading_day, session_end

        half_day = date(2024, 11, 29)
        rows = _make_rows(half_day, 100)  # only 100 bars < 210
        q = compute_quality("SPY", rows, "2024-11-29", "2024-11-29",
                            is_trading_day, session_end)
        self.assertEqual(q.partial_sessions, 1)


class TestFormatQualityReport(unittest.TestCase):

    def _make_q(self, symbol: str, gap_rate: float = 0.0,
                total: int = 252, zero: int = 0,
                full: int = 252, partial: int = 0,
                stale: int = 0, max_run: int = 0) -> SymbolQuality:
        return SymbolQuality(
            symbol=symbol,
            total_sessions=total,
            full_bar_sessions=full,
            partial_sessions=partial,
            zero_bar_sessions=zero,
            stale_bar_count=stale,
            gap_rate_pct=gap_rate,
            max_consecutive_gaps=max_run,
        )

    def test_contains_header(self):
        report = format_quality_report([self._make_q("SPY")])
        self.assertIn("Sessions", report)
        self.assertIn("Gap%", report)

    def test_warning_shown_for_high_gap_rate(self):
        q = self._make_q("BAD", gap_rate=5.0, zero=13, full=239)
        report = format_quality_report([q])
        self.assertIn("WARNING", report)
        self.assertIn("BAD", report)

    def test_no_warning_for_low_gap_rate(self):
        q = self._make_q("SPY", gap_rate=0.5, zero=1, full=251)
        report = format_quality_report([q])
        self.assertNotIn("WARNING", report)

    def test_empty_results(self):
        report = format_quality_report([])
        self.assertIn("no symbols", report)


class TestWriteQualityJson(unittest.TestCase):

    def test_json_file_written_atomically(self):
        qs = [
            SymbolQuality("SPY", 252, 250, 1, 1, 0, 0.397, 1),
            SymbolQuality("QQQ", 252, 252, 0, 0, 0, 0.0, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "quality.json"
            write_quality_json(qs, "2024-01-01", "2024-12-31", out)
            self.assertTrue(out.exists())
            # No leftover tmp file
            self.assertFalse(Path(str(out) + ".tmp").exists())
            import json
            data = json.loads(out.read_text())
        self.assertEqual(data["start"], "2024-01-01")
        self.assertEqual(len(data["symbols"]), 2)
        self.assertIn("summary", data)
        self.assertFalse(data["summary"]["any_warnings"])

    def test_json_warning_flag_set(self):
        qs = [SymbolQuality("BAD", 252, 239, 0, 13, 0, 5.16, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q.json"
            write_quality_json(qs, "2024-01-01", "2024-12-31", out)
            import json
            data = json.loads(out.read_text())
        self.assertTrue(data["summary"]["any_warnings"])
        self.assertIn("BAD", data["summary"]["warned_symbols"])


if __name__ == "__main__":
    unittest.main()
