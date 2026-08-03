#!/usr/bin/env python3
"""Incremental 1-minute bar cache updater.

Extends an existing ``<SYMBOL>_<start>_<end>_1min.csv.gz`` file forward to a
new end date by downloading only the missing months from Alpaca IEX.

The existing historical file is never re-downloaded.  A new file covering the
full extended range is written atomically before the old one is removed.

Alpaca credentials are read from environment variables only — never written to
disk (凭证零落盘 rule).

Usage
-----
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    python3 scripts/update_cache.py \\
        --cache-dir data --end 2026-07-31 \\
        --symbols SPY QQQ

If ``--end`` is omitted, defaults to yesterday (exchange-local date).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from orb.cache import find_cached, read_all_rows, write_gzip_csv

NY = ZoneInfo("America/New_York")

_RTH_START_HOUR, _RTH_START_MIN = 9, 30
_RTH_END_HOUR,   _RTH_END_MIN   = 16, 0

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "AVGO", "NFLX", "ORCL", "CRM",
    "JPM", "BAC", "GS", "V", "MA",
    "XOM", "CVX", "UNH", "JNJ", "LLY",
    "WMT", "COST", "HD", "CAT", "BA",
]


# ---------------------------------------------------------------------------
# Internal helpers

def _month_ranges(start: str, end: str) -> list[tuple[datetime, datetime]]:
    """Break [start, end] into monthly download chunks."""
    from datetime import time as _time
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    rth_start = _time(_RTH_START_HOUR, _RTH_START_MIN)
    rth_end   = _time(_RTH_END_HOUR,   _RTH_END_MIN)
    ranges: list[tuple[datetime, datetime]] = []
    cur = s.replace(day=1)
    while cur <= e:
        nxt = (cur.replace(month=cur.month + 1) if cur.month < 12
               else cur.replace(year=cur.year + 1, month=1))
        chunk_end = min(nxt - timedelta(days=1), e)
        ranges.append((
            datetime.combine(cur, rth_start).replace(tzinfo=NY),
            datetime.combine(chunk_end, rth_end).replace(tzinfo=NY),
        ))
        cur = nxt
    return ranges


def _alpaca_fetch(symbol: str, start_dt: datetime, end_dt: datetime) -> list[str]:
    """Download one monthly chunk from Alpaca IEX.

    Credentials are read from ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
    Returns list of CSV row strings (no header).
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("set ALPACA_API_KEY and ALPACA_SECRET_KEY (free paper keys)")
    client = StockHistoricalDataClient(key, sec)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        adjustment="all",
        feed="iex",
    )
    bars = client.get_stock_bars(req).data.get(symbol, [])
    return [
        f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}"
        for b in bars
    ]


def _fetch_with_retry(
    symbol: str,
    chunk_start: datetime,
    chunk_end: datetime,
    max_retries: int,
    backoff_base: float,
    fetcher: Callable,
) -> list[str]:
    for attempt in range(max_retries):
        try:
            return fetcher(symbol, chunk_start, chunk_end)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = backoff_base ** attempt
            print(
                f"[retry] {symbol} {chunk_start.date()} "
                f"attempt {attempt + 1}/{max_retries} ({exc}); "
                f"retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    return []


# ---------------------------------------------------------------------------
# Public entry point (importable by tests)

def update_symbol(
    symbol: str,
    cache_dir: Path,
    new_end: str,
    *,
    _fetcher: Optional[Callable] = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Optional[Path]:
    """Extend the cached file for ``symbol`` to cover up to ``new_end``.

    Parameters
    ----------
    symbol    : ticker symbol (must match existing cache filename prefix)
    cache_dir : directory containing the existing ``.csv.gz`` files
    new_end   : ISO-8601 date string (inclusive) for the desired coverage end
    _fetcher  : override Alpaca API call (for tests); receives
                (symbol, start_dt, end_dt) and returns list of CSV row strings

    Returns
    -------
    Path to the new extended cache file, or None if already up to date.
    Raises SystemExit if no existing cache file is found.
    """
    cache_dir = Path(cache_dir)
    fetcher = _fetcher if _fetcher is not None else _alpaca_fetch

    existing = find_cached(cache_dir, symbol)
    if existing is None:
        sys.exit(
            f"no existing cache for {symbol} in {cache_dir}; "
            "run local_pump.py first to create the initial cache"
        )

    old_path, old_start, old_end = existing
    if old_end >= new_end:
        print(f"[update] {symbol}: already covers up to {old_end}, skip", flush=True)
        return None

    # Download only the months not yet in the cache
    fetch_start = (date.fromisoformat(old_end) + timedelta(days=1)).isoformat()
    print(f"[update] {symbol}: extending {old_end} -> {new_end}", flush=True)

    new_rows: list[str] = []
    for chunk_start, chunk_end in _month_ranges(fetch_start, new_end):
        rows = _fetch_with_retry(
            symbol, chunk_start, chunk_end, max_retries, backoff_base, fetcher
        )
        new_rows.extend(rows)
        if rows:
            print(f"[update] {symbol}: {chunk_start.date()} "
                  f"got {len(rows)} bars", flush=True)

    old_rows = read_all_rows(old_path)
    combined = old_rows + new_rows  # already sorted: old < new by date range
    new_path = cache_dir / f"{symbol}_{old_start}_{new_end}_1min.csv.gz"

    write_gzip_csv(new_path, combined)
    old_path.unlink()
    print(f"[update] {symbol}: {len(old_rows)} + {len(new_rows)} rows "
          f"-> {new_path.name}", flush=True)
    return new_path


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    yesterday = (datetime.now(NY).date() - timedelta(days=1)).isoformat()

    ap = argparse.ArgumentParser(
        description="Extend cached 1-min bar files forward to a new end date."
    )
    ap.add_argument("--cache-dir", default="data",
                    help="Directory containing existing .csv.gz cache files")
    ap.add_argument("--end", default=yesterday,
                    help="New end date (inclusive, ISO-8601); default: yesterday")
    ap.add_argument("--symbols", nargs="*", default=UNIVERSE,
                    help="Symbols to update (default: full UNIVERSE)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff-base", type=float, default=2.0)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    for sym in args.symbols:
        update_symbol(sym, cache_dir, args.end,
                      max_retries=args.max_retries,
                      backoff_base=args.backoff_base)


if __name__ == "__main__":
    main()
