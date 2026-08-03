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
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from orb.core import Bar, SymbolEngine, default_candidate_grid, grid_spec_hash, SCHEMA_VERSION

NY = ZoneInfo("America/New_York")
RTH_START, RTH_END = time(9, 30), time(16, 0)

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "AVGO", "NFLX", "ORCL", "CRM",
    "JPM", "BAC", "GS", "V", "MA",
    "XOM", "CVX", "UNH", "JNJ", "LLY",
    "WMT", "COST", "HD", "CAT", "BA",
]


# --------------------------------------------------------------- downloading

def download_symbol(symbol: str, start: str, end: str, cache_dir: Path) -> Path:
    """Download 1-minute IEX bars to a gzip CSV cache; skip if cached."""
    out = cache_dir / f"{symbol}_{start}_{end}_1min.csv.gz"
    if out.exists():
        return out
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
        start=datetime.fromisoformat(start).replace(tzinfo=NY),
        end=datetime.fromisoformat(end).replace(tzinfo=NY),
        adjustment="all",
        feed="iex",
    )
    print(f"[download] {symbol} {start}..{end}", flush=True)
    bars = client.get_stock_bars(req).data.get(symbol, [])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        f.write("ts,open,high,low,close,volume\n")
        for b in bars:
            f.write(f"{b.timestamp.isoformat()},{b.open},{b.high},"
                    f"{b.low},{b.close},{b.volume}\n")
    os.replace(tmp, out)
    print(f"[download] {symbol}: {len(bars)} minute bars cached", flush=True)
    return out


# ------------------------------------------------------------- consolidation

def load_minute_bars(path: Path):
    """Yield (ny_datetime_start, o, h, l, c, v) for RTH minutes, sorted."""
    rows = []
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            ts, o, h, l, c, v = line.rstrip("\n").split(",")
            t = datetime.fromisoformat(ts).astimezone(NY)
            if RTH_START <= t.time() < RTH_END:
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
    args = ap.parse_args()

    full_grid = default_candidate_grid()
    ghash = grid_spec_hash(full_grid)
    grid = [c for i, c in enumerate(full_grid)
            if i % args.shards == args.shard]
    print(f"grid_hash={ghash} candidates={len(grid)} "
          f"shard={args.shard}/{args.shards}")

    cache = Path(args.cache_dir)
    trades = []
    for sym in args.symbols:
        path = download_symbol(sym, args.start, args.end, cache)
        rows = load_minute_bars(path)
        bars = consolidate_5min(rows)
        st = run_symbol(sym, bars, grid)
        trades.extend(st)
        print(f"[run] {sym}: {len(bars)} 5m bars -> {len(st)} trades",
              flush=True)

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
