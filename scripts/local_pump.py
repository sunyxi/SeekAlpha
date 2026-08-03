#!/usr/bin/env python3
"""Local research pump: Alpaca free (IEX) minute bars -> ORB core -> trade logs.

Replaces the QuantConnect data pump entirely. Output directory layout is
identical to the ObjectStore download, so scripts/wf_select.py consumes it
unchanged.

Setup (one-time):
    pip install alpaca-py pandas
    export ALPACA_API_KEY=...      # free paper account, no funding needed
    export ALPACA_SECRET_KEY=...

Usage:
    python3 scripts/local_pump.py --start 2021-01-04 --end 2026-06-30 \\
        --cache-dir data --out-dir runs/orb045
    # optional candidate sharding to spread CPU time across runs/machines:
    #   --shard 0 --shards 4

Then:
    python3 scripts/wf_select.py --trades-dir runs/orb045 \\
        --report-output reports/orb045-wf.json

Notes / limitations (must be stated in any report):
  - IEX feed covers a single exchange's prints; absolute volume is lower than
    consolidated tape. RVOL stays valid because numerator and denominator use
    the same feed.
  - adjustment="all" folds splits and dividends into prices (corporate-action
    mode: adjusted).
  - Bars are consolidated locally and deterministically from 1-minute bars to
    5-minute RTH bars; bar `end` timestamps follow exchange time
    (America/New_York), matching the core's conventions.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from orb.core import Bar, SymbolEngine, default_candidate_grid, grid_spec_hash, SCHEMA_VERSION
from orb.calendar import is_trading_day as _is_trading_day, session_end as _calendar_session_end
from orb.cache import validate_cache
from orb.quality import compute_quality, format_quality_report, write_quality_json

NY = ZoneInfo("America/New_York")
RTH_START_TIME = datetime.min.time().replace(hour=9, minute=30)
RTH_END_TIME   = datetime.min.time().replace(hour=16, minute=0)

from datetime import time as _time
RTH_START = _time(9, 30)
RTH_END   = _time(16, 0)

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "AVGO", "NFLX", "ORCL", "CRM",
    "JPM", "BAC", "GS", "V", "MA",
    "XOM", "CVX", "UNH", "JNJ", "LLY",
    "WMT", "COST", "HD", "CAT", "BA",
]


# --------------------------------------------------------------- downloading

def _month_ranges(start: str, end: str) -> list[tuple[datetime, datetime]]:
    """Break [start, end] into monthly chunks for resumable downloading."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    ranges: list[tuple[datetime, datetime]] = []
    cur = s.replace(day=1)
    while cur <= e:
        nxt = (cur.replace(month=cur.month + 1) if cur.month < 12
               else cur.replace(year=cur.year + 1, month=1))
        chunk_end = min(nxt - timedelta(days=1), e)
        ranges.append((
            datetime.combine(cur, RTH_START).replace(tzinfo=NY),
            datetime.combine(chunk_end, RTH_END).replace(tzinfo=NY),
        ))
        cur = nxt
    return ranges


def _alpaca_fetch(symbol: str, start_dt: datetime, end_dt: datetime) -> list:
    """Isolated Alpaca API call — imported lazily so core remains dep-free."""
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
    return client.get_stock_bars(req).data.get(symbol, [])


def _bars_to_rows(bars: list) -> list[str]:
    """Convert Alpaca bar objects to CSV row strings (no header)."""
    return [
        f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}"
        for b in bars
    ]


def _read_partial(partial: Path) -> tuple[list[str], Optional[datetime]]:
    """Read existing rows and last timestamp from a .partial gzip CSV.

    Returns (rows, last_ts). last_ts is None if the file is empty or unreadable.
    """
    rows: list[str] = []
    last_ts: Optional[datetime] = None
    try:
        with gzip.open(partial, "rt") as f:
            next(f)  # skip header
            for line in f:
                line = line.rstrip("\n")
                if line:
                    rows.append(line)
                    last_ts = datetime.fromisoformat(line.split(",")[0])
    except Exception:
        pass
    return rows, last_ts


def _write_partial(partial: Path, rows: list[str]) -> None:
    """Atomically write rows to a .partial gzip CSV (header included)."""
    tmp = Path(str(partial) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        f.write("ts,open,high,low,close,volume\n")
        for row in rows:
            f.write(row + "\n")
    os.replace(tmp, partial)


def _fetch_chunk_with_retry(
    symbol: str,
    chunk_start: datetime,
    chunk_end: datetime,
    max_retries: int,
    backoff_base: float,
    fetcher: Callable,
) -> list[str]:
    """Fetch one monthly chunk with exponential back-off retry.

    Returns list of CSV row strings (no header). Raises on exhausted retries.
    """
    for attempt in range(max_retries):
        try:
            bars = fetcher(symbol, chunk_start, chunk_end)
            return _bars_to_rows(bars)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = backoff_base ** attempt
            print(
                f"[retry] {symbol} chunk {chunk_start.date()} "
                f"attempt {attempt + 1}/{max_retries} failed ({exc}); "
                f"retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    return []  # unreachable


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    _fetcher: Optional[Callable] = None,
) -> Path:
    """Download 1-min IEX bars to a gzip CSV cache; resume if interrupted.

    Downloads in monthly chunks so a crash can be resumed from the last
    complete chunk. Pass `_fetcher` to override the Alpaca API call (tests).
    """
    out = cache_dir / f"{symbol}_{start}_{end}_1min.csv.gz"
    if out.exists():
        return out

    fetcher = _fetcher if _fetcher is not None else _alpaca_fetch
    partial = Path(str(out) + ".partial")  # avoids pathlib's single-suffix replacement

    rows, last_ts = _read_partial(partial) if partial.exists() else ([], None)
    resume_msg = f" (resuming from {last_ts.date()})" if last_ts else ""
    print(f"[download] {symbol} {start}..{end}{resume_msg}", flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start, chunk_end in _month_ranges(start, end):
        # Skip chunks already fully covered by the partial file
        if last_ts is not None:
            if chunk_end <= last_ts:
                continue
            # Partially covered: advance chunk_start past last seen bar
            if chunk_start <= last_ts:
                chunk_start = last_ts + timedelta(minutes=1)

        new_rows = _fetch_chunk_with_retry(
            symbol, chunk_start, chunk_end, max_retries, backoff_base, fetcher
        )
        rows.extend(new_rows)
        if new_rows:
            last_ts = datetime.fromisoformat(new_rows[-1].split(",")[0])
        _write_partial(partial, rows)  # save progress after each chunk

    os.replace(partial, out)
    print(f"[download] {symbol}: {len(rows)} minute bars cached", flush=True)
    return out


# ------------------------------------------------------------- consolidation

def load_minute_bars(path: Path):
    """Load RTH 1-min bars from a gzip CSV, honouring half-day session ends."""
    rows = []
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            ts, o, h, l, c, v = line.rstrip("\n").split(",")
            t = datetime.fromisoformat(ts).astimezone(NY)
            sess_close = _calendar_session_end(t.date())  # 16:00 or 13:00
            if RTH_START <= t.time() < sess_close:
                rows.append((t, float(o), float(h), float(l),
                             float(c), float(v)))
    rows.sort(key=lambda r: r[0])
    return rows


def consolidate_5min(minute_rows):
    """Deterministic 1m -> 5m consolidation aligned to 9:30 ET.

    Returns list of core Bars (naive NY-local datetimes, `end` = bucket end),
    grouped in chronological order. Buckets with zero minute bars are skipped.
    """
    out = []
    cur_key = None
    o = h = l = c = None
    v = 0.0
    for t, bo, bh, bl, bc, bv in minute_rows:
        session_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
        bucket = int((t - session_open).total_seconds() // 300)
        key = (t.date(), bucket)
        if key != cur_key:
            if cur_key is not None:
                out.append(_mk_bar(cur_key, o, h, l, c, v))
            cur_key, o, h, l, c, v = key, bo, bh, bl, bc, bv
        else:
            h, l, c, v = max(h, bh), min(l, bl), bc, v + bv
    if cur_key is not None:
        out.append(_mk_bar(cur_key, o, h, l, c, v))
    return out


def _mk_bar(key, o, h, l, c, v) -> Bar:
    day, bucket = key
    end = datetime.combine(day, RTH_START) + timedelta(minutes=5 * (bucket + 1))
    return Bar(end=end, open=o, high=h, low=l, close=c, volume=v)


# --------------------------------------------------------------- validation

def _ensure_valid_cache(
    symbol: str,
    path: Path,
    start: str,
    end: str,
    max_retries: int,
    backoff_base: float,
    _fetcher=None,
) -> Path:
    """Validate cached file; delete and re-download once if corrupt."""
    result = validate_cache(path, start, end)
    if result.ok:
        print(f"[validate] {symbol}: {result.row_count:,} rows OK", flush=True)
        return path
    print(
        f"[validate] {symbol}: corrupt cache ({result.error}); "
        "deleting and re-downloading",
        flush=True,
    )
    path.unlink(missing_ok=True)
    new_path = download_symbol(
        symbol, start, end, path.parent,
        max_retries=max_retries, backoff_base=backoff_base, _fetcher=_fetcher,
    )
    result2 = validate_cache(new_path, start, end)
    if not result2.ok:
        sys.exit(
            f"[validate] {symbol}: cache still invalid after re-download: "
            f"{result2.error}"
        )
    return new_path


# --------------------------------------------------------------------- main

def run_symbol(symbol: str, bars, grid) -> list:
    eng = SymbolEngine(symbol, grid)
    cur_date = None
    for b in bars:
        if cur_date is not None and b.end.date() != cur_date:
            eng.on_session_end()
        cur_date = b.end.date()
        eng.on_bar(b)
    eng.on_session_end()
    return [t.to_dict() for t in eng.trades]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--symbols", nargs="*", default=UNIVERSE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Alpaca API retry attempts per chunk (default: 3)")
    ap.add_argument("--backoff-base", type=float, default=2.0,
                    help="Exponential back-off base in seconds (default: 2.0)")
    ap.add_argument("--quality-report", default=None, metavar="PATH",
                    help="Write per-symbol quality stats JSON to PATH")
    ap.add_argument("--strict-quality", action="store_true",
                    help="Exit non-zero if any symbol gap rate > 1%%")
    args = ap.parse_args()

    full_grid = default_candidate_grid()
    ghash = grid_spec_hash(full_grid)
    grid = [c for i, c in enumerate(full_grid)
            if i % args.shards == args.shard]
    print(f"grid_hash={ghash} candidates={len(grid)} "
          f"shard={args.shard}/{args.shards}")

    cache = Path(args.cache_dir)
    trades = []
    quality_results = []
    for sym in args.symbols:
        path = download_symbol(sym, args.start, args.end, cache,
                               max_retries=args.max_retries,
                               backoff_base=args.backoff_base)
        path = _ensure_valid_cache(sym, path, args.start, args.end,
                                   args.max_retries, args.backoff_base)
        rows = load_minute_bars(path)
        q = compute_quality(sym, rows, args.start, args.end,
                            _is_trading_day, _calendar_session_end)
        quality_results.append(q)
        bars = consolidate_5min(rows)
        st = run_symbol(sym, bars, grid)
        trades.extend(st)
        print(f"[run] {sym}: {len(bars)} 5m bars -> {len(st)} trades",
              flush=True)

    print(format_quality_report(quality_results), flush=True)
    if args.quality_report:
        write_quality_json(quality_results, args.start, args.end,
                           Path(args.quality_report))
        print(f"[quality] report written to {args.quality_report}", flush=True)
    if args.strict_quality:
        warned = [q.symbol for q in quality_results if q.has_warning()]
        if warned:
            sys.exit(f"[quality] strict mode: gap rate > 1% for: {', '.join(warned)}")

    out = Path(args.out_dir) / f"shard{args.shard}of{args.shards}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "grid_spec_hash": ghash,
        "candidate_shard": args.shard,
        "candidate_shards": args.shards,
        "universe": args.symbols,
        "start": args.start, "end": args.end,
        "resolution": "1min(iex)->5minute",
        "timezone": "America/New_York",
        "normalization": "adjusted(all)",
        "extended_hours": False,
        "data_source": "alpaca-iex-free",
        "trade_count": len(trades),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    CHUNK = 5000
    for i in range(0, max(len(trades), 1), CHUNK):
        (out / f"trades_{i // CHUNK:04d}.json").write_text(
            json.dumps(trades[i:i + CHUNK]))
    print(f"saved {len(trades)} trades under {out}/")


if __name__ == "__main__":
    main()
