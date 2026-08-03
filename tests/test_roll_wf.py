"""Tests for scripts/roll_wf.py — rolling walk-forward update.

All tests use temp directories and synthetic data; no Alpaca keys required.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import pathlib
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from orb.cache import write_gzip_csv
from orb.core.orb_core import grid_spec_hash, default_candidate_grid

NY = ZoneInfo("America/New_York")
_HASH = grid_spec_hash(default_candidate_grid())


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _minute_row(dt: datetime) -> str:
    ts = dt.astimezone().isoformat()
    return f"{ts},100.0,101.0,99.0,100.5,10000"


def _make_cache(tmp: Path, symbol: str, start: str, end: str) -> Path:
    """Write a synthetic 1-min bar gzip CSV covering the date range."""
    rows = []
    d = date.fromisoformat(start)
    d_end = date.fromisoformat(end)
    while d <= d_end:
        # 3 bars per day, starting at 09:31 ET
        for offset in range(3):
            dt = datetime(d.year, d.month, d.day, 9, 31 + offset, tzinfo=NY)
            rows.append(_minute_row(dt))
        d += timedelta(days=1)
    path = tmp / f"{symbol}_{start}_{end}_1min.csv.gz"
    write_gzip_csv(path, rows)
    return path


def _make_old_trades(tmp: Path, start: str, end: str) -> Path:
    """Write a minimal old-trades shard directory with meta.json."""
    shard_dir = tmp / "shard0of1"
    shard_dir.mkdir(parents=True)
    meta = {
        "schema_version": 1,
        "grid_spec_hash": _HASH,
        "candidate_shard": 0,
        "candidate_shards": 1,
        "universe": ["SPY"],
        "start": start,
        "end": end,
        "resolution": "1min(iex)->5minute",
        "timezone": "America/New_York",
        "normalization": "adjusted(all)",
        "extended_hours": False,
        "data_source": "alpaca-iex-free",
        "trade_count": 2,
    }
    (shard_dir / "meta.json").write_text(json.dumps(meta))
    trades = [
        {
            "candidate_id": "abc123",
            "symbol": "SPY",
            "signal_time": f"{start}T09:35:00",
            "entry_time":  f"{start}T09:35:00",
            "exit_time":   f"{start}T10:00:00",
            "entry_price": 100.0, "stop_price": 98.0, "target_price": 104.0,
            "exit_price": 104.0, "exit_reason": "target",
            "quantity": 10, "gross_pnl": 40.0,
            "entry_notional": 1000.0, "exit_notional": 1040.0,
        },
        {
            "candidate_id": "abc123",
            "symbol": "SPY",
            "signal_time": f"{start}T10:35:00",
            "entry_time":  f"{start}T10:35:00",
            "exit_time":   f"{start}T11:00:00",
            "entry_price": 100.0, "stop_price": 98.0, "target_price": 104.0,
            "exit_price": 98.0, "exit_reason": "stop",
            "quantity": 10, "gross_pnl": -20.0,
            "entry_notional": 1000.0, "exit_notional": 980.0,
        },
    ]
    (shard_dir / "trades_0000.json").write_text(json.dumps(trades))
    return tmp


# ---------------------------------------------------------------------------
# TestRollWfImport
# ---------------------------------------------------------------------------

class TestRollWfImport(unittest.TestCase):

    def test_importable(self):
        import roll_wf
        self.assertTrue(hasattr(roll_wf, "roll_wf"))

    def test_has_roll_symbol(self):
        import roll_wf
        self.assertTrue(hasattr(roll_wf, "roll_symbol"))


# ---------------------------------------------------------------------------
# TestRollSymbol
# ---------------------------------------------------------------------------

class TestRollSymbol(unittest.TestCase):

    def setUp(self):
        import roll_wf
        self.rw = roll_wf
        self.grid = default_candidate_grid()

    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cache_path = _make_cache(tmp, "SPY", "2021-01-04", "2021-01-06")
            result = self.rw.roll_symbol(
                "SPY", cache_path, "2021-01-04", "2021-01-06", self.grid
            )
            self.assertIsInstance(result, list)

    def test_trades_within_date_range(self):
        with tempfile.TemporaryDirectory() as d:
            cache_path = _make_cache(Path(d), "SPY", "2021-01-04", "2021-01-10")
            result = self.rw.roll_symbol(
                "SPY", cache_path, "2021-01-06", "2021-01-08", self.grid
            )
            for t in result:
                d_entry = t["entry_time"][:10]
                self.assertGreaterEqual(d_entry, "2021-01-06")
                self.assertLessEqual(d_entry, "2021-01-08")

    def test_trade_keys_present(self):
        with tempfile.TemporaryDirectory() as d:
            cache_path = _make_cache(Path(d), "SPY", "2021-01-04", "2021-01-05")
            result = self.rw.roll_symbol(
                "SPY", cache_path, "2021-01-04", "2021-01-05", self.grid
            )
            expected_keys = {
                "candidate_id", "symbol", "entry_time", "exit_time",
                "entry_price", "exit_price", "gross_pnl",
                "entry_notional", "exit_notional",
            }
            for t in result:
                self.assertTrue(expected_keys <= set(t.keys()))


# ---------------------------------------------------------------------------
# TestRollWf
# ---------------------------------------------------------------------------

class TestRollWf(unittest.TestCase):

    def setUp(self):
        import roll_wf
        self.rw = roll_wf

    def _run(self, old_dir, cache_dir, new_start, new_end, out_dir,
             symbols=None):
        self.rw.roll_wf(
            old_trades_dir=old_dir,
            cache_dir=cache_dir,
            new_start=new_start,
            new_end=new_end,
            out_dir=out_dir,
            symbols=symbols or ["SPY"],
        )

    def test_output_directory_created(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = _make_old_trades(tmp / "old", "2021-01-04", "2021-01-31")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            out = tmp / "out"
            self._run(old_dir, tmp / "cache", "2021-02-01", "2021-02-05", out)
            self.assertTrue(out.exists())

    def test_meta_json_written(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = _make_old_trades(tmp / "old", "2021-01-04", "2021-01-31")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            out = tmp / "out"
            self._run(old_dir, tmp / "cache", "2021-02-01", "2021-02-05", out)
            shard = out / "shard0of1"
            meta_path = shard / "meta.json"
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["grid_spec_hash"], _HASH)

    def test_date_range_covers_old_and_new(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = _make_old_trades(tmp / "old", "2021-01-04", "2021-01-31")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            out = tmp / "out"
            self._run(old_dir, tmp / "cache", "2021-02-01", "2021-02-05", out)
            meta = json.loads((out / "shard0of1" / "meta.json").read_text())
            self.assertEqual(meta["start"], "2021-01-04")
            self.assertEqual(meta["end"], "2021-02-05")

    def test_old_trades_included(self):
        """Old trades must appear in merged output."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = _make_old_trades(tmp / "old", "2021-01-04", "2021-01-31")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            out = tmp / "out"
            self._run(old_dir, tmp / "cache", "2021-02-01", "2021-02-05", out)

            all_trades = []
            for p in sorted((out / "shard0of1").glob("trades_*.json")):
                all_trades.extend(json.loads(p.read_text()))
            # Old trades had dates starting 2021-01-04
            old_dates = {t["entry_time"][:10] for t in all_trades
                         if t["entry_time"][:10] < "2021-02-01"}
            self.assertGreater(len(old_dates), 0)

    def test_hash_mismatch_exits(self):
        """If old meta.json has a different grid_spec_hash, abort."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = tmp / "old"
            (old_dir / "shard0of1").mkdir(parents=True)
            bad_meta = {
                "schema_version": 1,
                "grid_spec_hash": "deadbeef00000000",  # wrong hash
                "candidate_shard": 0, "candidate_shards": 1,
                "universe": ["SPY"], "start": "2021-01-04", "end": "2021-01-31",
            }
            (old_dir / "shard0of1" / "meta.json").write_text(json.dumps(bad_meta))
            (old_dir / "shard0of1" / "trades_0000.json").write_text("[]")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            with self.assertRaises(SystemExit):
                self._run(old_dir, tmp / "cache",
                          "2021-02-01", "2021-02-05", tmp / "out")

    def test_no_duplicate_trades(self):
        """(candidate_id, symbol, entry_time) must be unique in merged output."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            old_dir = _make_old_trades(tmp / "old", "2021-01-04", "2021-01-31")
            _make_cache(tmp / "cache", "SPY", "2021-01-04", "2021-02-10")
            out = tmp / "out"
            self._run(old_dir, tmp / "cache", "2021-02-01", "2021-02-05", out)

            all_trades = []
            for p in sorted((out / "shard0of1").glob("trades_*.json")):
                all_trades.extend(json.loads(p.read_text()))
            keys = [(t["candidate_id"], t["symbol"], t["entry_time"])
                    for t in all_trades]
            self.assertEqual(len(keys), len(set(keys)), "duplicate trades found")


# ---------------------------------------------------------------------------
# TestStdlibOnly
# ---------------------------------------------------------------------------

class TestStdlibOnly(unittest.TestCase):

    def test_no_alpaca_top_level(self):
        src = pathlib.Path(__file__).parent.parent / "scripts" / "roll_wf.py"
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    continue
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else ([node.module] if node.module else []))
                for name in names:
                    if name and name.split(".")[0] == "alpaca":
                        # Find the enclosing function — top-level alpaca import is banned
                        self.fail(f"top-level alpaca import in roll_wf.py: {name}")


if __name__ == "__main__":
    unittest.main()
