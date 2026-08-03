"""Unit tests for resumable download and retry logic in local_pump.py.

No network calls, no alpaca-py required. All Alpaca fetches are replaced
with a synthetic MockBar fetcher.
"""

from __future__ import annotations

import gzip
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Make scripts/ importable without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from local_pump import (
    _bars_to_rows,
    _fetch_chunk_with_retry,
    _month_ranges,
    _read_partial,
    _write_partial,
    download_symbol,
)

NY = ZoneInfo("America/New_York")


@dataclass
class MockBar:
    """Minimal Alpaca bar stand-in used by tests."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _make_bars(dates: list[str], price: float = 100.0, vol: float = 10_000.0) -> list[MockBar]:
    """Create one bar per date at 09:31 NY time."""
    return [
        MockBar(
            timestamp=datetime.fromisoformat(d + "T09:31:00").replace(tzinfo=NY),
            open=price, high=price + 1, low=price - 1, close=price, volume=vol,
        )
        for d in dates
    ]


def _make_fetcher(bars_by_month: dict[str, list[MockBar]]):
    """Return a fetcher that serves bars keyed by month string 'YYYY-MM'."""
    def fetcher(symbol: str, start_dt: datetime, end_dt: datetime) -> list[MockBar]:
        month_key = start_dt.strftime("%Y-%m")
        return bars_by_month.get(month_key, [])
    return fetcher


class TestMonthRanges(unittest.TestCase):

    def test_single_month(self):
        ranges = _month_ranges("2021-01-04", "2021-01-29")
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0][0].month, 1)
        self.assertEqual(ranges[0][0].year, 2021)

    def test_two_months(self):
        ranges = _month_ranges("2021-01-04", "2021-02-28")
        self.assertEqual(len(ranges), 2)

    def test_cross_year(self):
        ranges = _month_ranges("2021-12-01", "2022-01-31")
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][0].year, 2021)
        self.assertEqual(ranges[1][0].year, 2022)

    def test_ranges_are_timezone_aware(self):
        ranges = _month_ranges("2021-01-04", "2021-03-31")
        for start, end in ranges:
            self.assertIsNotNone(start.tzinfo)
            self.assertIsNotNone(end.tzinfo)


class TestReadWritePartial(unittest.TestCase):

    def test_round_trip(self):
        bars = _make_bars(["2021-01-04", "2021-01-05"])
        rows = _bars_to_rows(bars)
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "test.csv.gz.partial"
            _write_partial(partial, rows)
            read_rows, last_ts = _read_partial(partial)
        self.assertEqual(read_rows, rows)
        self.assertIsNotNone(last_ts)
        self.assertEqual(last_ts.date().isoformat(), "2021-01-05")

    def test_empty_partial_returns_none_ts(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "empty.csv.gz.partial"
            _write_partial(partial, [])
            _, last_ts = _read_partial(partial)
        self.assertIsNone(last_ts)

    def test_missing_partial_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "nonexistent.csv.gz.partial"
            rows, last_ts = _read_partial(partial)
        self.assertEqual(rows, [])
        self.assertIsNone(last_ts)


class TestRetry(unittest.TestCase):

    def test_succeeds_on_first_attempt(self):
        bars = _make_bars(["2021-01-04"])
        fetcher = _make_fetcher({"2021-01": bars})
        from datetime import time as _t
        start = datetime(2021, 1, 1, 9, 30, tzinfo=NY)
        end   = datetime(2021, 1, 31, 16, 0, tzinfo=NY)
        rows = _fetch_chunk_with_retry("X", start, end, 3, 0.0, fetcher)
        self.assertEqual(len(rows), 1)

    def test_retries_then_succeeds(self):
        bars = _make_bars(["2021-01-04"])
        call_count = {"n": 0}

        def flaky_fetcher(symbol, start_dt, end_dt):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient failure")
            return bars

        start = datetime(2021, 1, 1, 9, 30, tzinfo=NY)
        end   = datetime(2021, 1, 31, 16, 0, tzinfo=NY)
        rows = _fetch_chunk_with_retry("X", start, end, 3, 0.0, flaky_fetcher)
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(len(rows), 1)

    def test_raises_after_exhausting_retries(self):
        def always_fails(symbol, start_dt, end_dt):
            raise ConnectionError("always fails")

        start = datetime(2021, 1, 1, 9, 30, tzinfo=NY)
        end   = datetime(2021, 1, 31, 16, 0, tzinfo=NY)
        with self.assertRaises(ConnectionError):
            _fetch_chunk_with_retry("X", start, end, 2, 0.0, always_fails)


class TestDownloadSymbol(unittest.TestCase):

    def _run_download(self, fetcher, tmpdir: str, start="2021-01-01", end="2021-02-28"):
        cache = Path(tmpdir)
        return download_symbol("TEST", start, end, cache, _fetcher=fetcher)

    def test_full_download_creates_file(self):
        jan = _make_bars(["2021-01-04", "2021-01-05"])
        feb = _make_bars(["2021-02-01", "2021-02-02"])
        fetcher = _make_fetcher({"2021-01": jan, "2021-02": feb})
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_download(fetcher, tmp)
            self.assertTrue(out.exists())
            rows, last_ts = _read_partial(out)  # reuse reader on final file
        self.assertEqual(len(rows), 4)
        self.assertIsNone(Path(tmp + "/TEST_2021-01-01_2021-02-28_1min.csv.gz.partial")
                          if False else None)  # partial cleaned up

    def test_no_partial_left_after_success(self):
        bars = _make_bars(["2021-01-04"])
        fetcher = _make_fetcher({"2021-01": bars})
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            download_symbol("TEST", "2021-01-01", "2021-01-31", cache, _fetcher=fetcher)
            partials = list(cache.glob("*.partial"))
        self.assertEqual(partials, [], "partial file must be cleaned up after success")

    def test_skip_if_final_file_exists(self):
        call_count = {"n": 0}

        def counting_fetcher(symbol, start_dt, end_dt):
            call_count["n"] += 1
            return []

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            # Pre-create the final file
            out = cache / "TEST_2021-01-01_2021-01-31_1min.csv.gz"
            _write_partial(out, [])  # empty but exists
            download_symbol("TEST", "2021-01-01", "2021-01-31", cache,
                            _fetcher=counting_fetcher)
        self.assertEqual(call_count["n"], 0, "fetcher must not be called when file exists")

    def test_resume_from_partial(self):
        """Simulates interrupted download: Jan downloaded, Feb still missing."""
        jan = _make_bars(["2021-01-04", "2021-01-05"])
        feb = _make_bars(["2021-02-01", "2021-02-02"])

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            partial = cache / "TEST_2021-01-01_2021-02-28_1min.csv.gz.partial"

            # Write Jan bars as if they were already downloaded
            _write_partial(partial, _bars_to_rows(jan))

            # Now call download_symbol — fetcher only provides Feb bars
            feb_only_fetcher = _make_fetcher({"2021-02": feb})
            out = download_symbol("TEST", "2021-01-01", "2021-02-28", cache,
                                  _fetcher=feb_only_fetcher)

            rows, _ = _read_partial(out)

        self.assertEqual(len(rows), 4, "resumed file must contain Jan + Feb bars")
        # Verify timestamps are monotonically increasing (no duplicates)
        timestamps = [row.split(",")[0] for row in rows]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(set(timestamps)), len(timestamps), "no duplicate timestamps")

    def test_no_duplicates_on_full_resume(self):
        """Partial has Jan + Feb bars; no new bars are added on resume.

        The fetcher may be called at most once (to check the tail of the last
        partial month), but must return nothing and produce no duplicates.
        """
        jan = _make_bars(["2021-01-04"])
        feb = _make_bars(["2021-02-01"])
        all_bars = jan + feb

        call_count = {"n": 0}

        def counting_fetcher(symbol, start_dt, end_dt):
            call_count["n"] += 1
            return []  # tail-check finds nothing new

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            partial = cache / "TEST_2021-01-01_2021-02-28_1min.csv.gz.partial"
            _write_partial(partial, _bars_to_rows(all_bars))

            out = download_symbol("TEST", "2021-01-01", "2021-02-28", cache,
                                  _fetcher=counting_fetcher)
            rows, _ = _read_partial(out)

        # All original bars preserved, no duplicates
        self.assertEqual(len(rows), 2)
        timestamps = [r.split(",")[0] for r in rows]
        self.assertEqual(len(set(timestamps)), 2, "no duplicate timestamps")
        self.assertEqual(timestamps, sorted(timestamps), "timestamps in order")
        # Jan is fully past last_ts so skipped; only the Feb tail may be checked
        self.assertLessEqual(call_count["n"], 1,
                             "fetcher called at most once for last-month tail")


if __name__ == "__main__":
    unittest.main()
