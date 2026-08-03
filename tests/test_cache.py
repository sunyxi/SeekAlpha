"""Tests for cache integrity validation (src/orb/cache.py).

All tests are offline — no network calls, no alpaca-py dependency.
"""

from __future__ import annotations

import gzip
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from orb.cache import ValidationResult, validate_cache

NY = ZoneInfo("America/New_York")

_HEADER = "ts,open,high,low,close,volume"
_START = "2021-01-01"
_END = "2021-01-31"


def _write_gz(path: Path, lines: list[str], header: str = _HEADER) -> None:
    """Write a well-formed gzip CSV to path."""
    with gzip.open(path, "wt") as f:
        f.write(header + "\n")
        for ln in lines:
            f.write(ln + "\n")


def _ts(day: int, hour: int, minute: int, tz: str = "-05:00") -> str:
    return f"2021-01-{day:02d}T{hour:02d}:{minute:02d}:00{tz}"


class TestValidateCacheHappyPath(unittest.TestCase):

    def test_valid_two_row_file_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                f"{_ts(4, 9, 32)},100,101,99,100,1000",
            ])
            r = validate_cache(p, _START, _END)
        self.assertTrue(r.ok)
        self.assertIsNone(r.error)
        self.assertEqual(r.row_count, 2)

    def test_row_count_matches(self):
        rows = [f"{_ts(4, 9, 30 + i)},100,101,99,100,1000" for i in range(1, 11)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, rows)
            r = validate_cache(p, _START, _END)
        self.assertEqual(r.row_count, 10)

    def test_first_and_last_ts_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                f"{_ts(4, 9, 32)},100,101,99,100,1000",
                f"{_ts(4, 9, 33)},100,101,99,100,1000",
            ])
            r = validate_cache(p, _START, _END)
        self.assertEqual(r.first_ts, datetime.fromisoformat(_ts(4, 9, 31)))
        self.assertEqual(r.last_ts,  datetime.fromisoformat(_ts(4, 9, 33)))

    def test_ok_is_true_sets_no_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [f"{_ts(4, 9, 31)},100,101,99,100,1000"])
            r = validate_cache(p, _START, _END)
        self.assertTrue(r.ok)
        self.assertIsNone(r.error)


class TestValidateCacheGzipErrors(unittest.TestCase):

    def test_truncated_gzip_detected(self):
        """The required test: a hand-crafted truncated gzip file fails validation."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            # Write valid gzip with enough data to survive truncation
            _write_gz(p, [
                f"{_ts(4, 9, 30 + i)},100,101,99,100,1000"
                for i in range(1, 50)
            ])
            # Truncate to 1/3 of original size — guaranteed to be mid-stream
            size = p.stat().st_size
            with open(p, "r+b") as f:
                f.truncate(size // 3)

            r = validate_cache(p, _START, _END)

        self.assertFalse(r.ok)
        self.assertIsNotNone(r.error)
        # Error must mention gzip
        self.assertIn("gzip", r.error.lower())

    def test_not_a_gzip_file_detected(self):
        """Plain text written to a .gz path must fail gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            p.write_text("this is not a gzip file\n")
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("gzip", r.error.lower())

    def test_nonexistent_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.csv.gz"
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIsNotNone(r.error)

    def test_zero_byte_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.csv.gz"
            p.write_bytes(b"")
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)


class TestValidateCacheSchemaErrors(unittest.TestCase):

    def test_wrong_header_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [f"{_ts(4, 9, 31)},100,101,99,100,1000"],
                      header="timestamp,o,h,l,c,v")
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("header", r.error)

    def test_bad_timestamp_row_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                "not-a-date,100,101,99,100,1000",
            ])
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("timestamp", r.error)

    def test_no_data_rows_detected(self):
        """File with header only (no data rows) must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [])
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("no data rows", r.error)
        self.assertEqual(r.row_count, 0)


class TestValidateCacheTimestampOrdering(unittest.TestCase):

    def test_duplicate_timestamps_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                f"{_ts(4, 9, 31)},100,101,99,100,1000",  # duplicate
            ])
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("duplicate", r.error)

    def test_out_of_order_timestamps_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 32)},100,101,99,100,1000",
                f"{_ts(4, 9, 31)},100,101,99,100,1000",  # earlier than previous
            ])
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertIn("precedes", r.error)

    def test_monotonic_timestamps_across_days_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                f"{_ts(4, 15, 59)},100,101,99,100,1000",
                f"{_ts(5, 9, 31)},100,101,99,100,1000",  # next day — fine
                f"{_ts(5, 10, 0)},100,101,99,100,1000",
            ])
            r = validate_cache(p, _START, _END)
        self.assertTrue(r.ok)
        self.assertEqual(r.row_count, 4)

    def test_row_count_at_error_point_reported(self):
        """row_count in the result reflects rows seen before the error."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file.csv.gz"
            _write_gz(p, [
                f"{_ts(4, 9, 31)},100,101,99,100,1000",
                f"{_ts(4, 9, 32)},100,101,99,100,1000",
                f"{_ts(4, 9, 32)},100,101,99,100,1000",  # duplicate at row 4
            ])
            r = validate_cache(p, _START, _END)
        self.assertFalse(r.ok)
        self.assertEqual(r.row_count, 2)  # 2 rows accepted before error

    def test_result_never_raises(self):
        """validate_cache must return a result for any input, never raise."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "garbage.csv.gz"
            p.write_bytes(b"\xff\xfe" * 100)  # garbage bytes
            try:
                r = validate_cache(p, _START, _END)
                self.assertFalse(r.ok)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"validate_cache raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
