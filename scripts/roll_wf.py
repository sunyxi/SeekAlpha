#!/usr/bin/env python3
"""Rolling walk-forward update step.

Extends an existing ORB trade-log directory by running the ORB engine on a new
date range (new_start..new_end) using already-cached bar data, then merges the
new trades with the existing ones into a single output directory that
wf_select.py can consume unchanged.

Typical workflow after downloading a new month with update_cache.py:

    python3 scripts/roll_wf.py \\
        --old-trades-dir runs/orb045 \\
        --cache-dir      data \\
        --new-start      2026-07-01 \\
        --new-end        2026-07-31 \\
        --out-dir        runs/orb045-extended

The output directory has the same shard layout as local_pump.py produces and
can be passed directly to wf_select.py:

    python3 scripts/wf_select.py \\
        --trades-dir runs/orb045-extended \\
        --report-output reports/orb045-extended-wf.json

No network access; no Alpaca credentials needed (reads from local cache).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from orb.cache import find_cached, read_all_rows
from orb.calendar import is_trading_day as _is_trading_day, session_end as _session_end
from orb.core.orb_core import (
    Bar,
    SymbolEngine,
    default_candidate_grid,
    grid_spec_hash,
    SCHEMA_VERSION,
)

NY = ZoneInfo("America/New_York")
from datetime import time as _time
_RTH_START = _time(9, 30)
_CHUNK = 5000


# ---------------------------------------------------------------------------
# Bar loading (self-contained; mirrors local_pump.py for clarity)

def _load_minute_bars_filtered(path: Path, new_start: str, new_end: str):
    """Load RTH 1-min bars from a gzip CSV, restricted to [new_start, new_end]."""
    d_start = date.fromisoformat(new_start)
    d_end   = date.fromisoformat(new_end)
    rows = []
    with gzip.open(path, "rt") as f:
        next(f)  # skip header
        for line in f:
            ts, o, h, l, c, v = line.rstrip("\n").split(",")
            t = datetime.fromisoformat(ts).astimezone(NY)
            if not (d_start <= t.date() <= d_end):
                continue
            sess_close = _session_end(t.date())
            if _RTH_START <= t.time() < sess_close:
                rows.append((t, float(o), float(h), float(l), float(c), float(v)))
    rows.sort(key=lambda r: r[0])
    return rows


def _consolidate_5min(minute_rows) -> list[Bar]:
    """1-min -> 5-min consolidation aligned to 9:30 ET (same logic as local_pump.py)."""
    out: list[Bar] = []
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
    end = datetime.combine(day, _RTH_START) + timedelta(minutes=5 * (bucket + 1))
    return Bar(end=end, open=o, high=h, low=l, close=c, volume=v)


# ---------------------------------------------------------------------------
# Engine execution

def roll_symbol(
    symbol: str,
    cache_path: Path,
    new_start: str,
    new_end: str,
    grid: list,
) -> list[dict]:
    """Run the ORB engine over [new_start, new_end] bars for one symbol.

    Returns a list of trade dicts (same schema as local_pump.py output).
    """
    minute_rows = _load_minute_bars_filtered(cache_path, new_start, new_end)
    bars = _consolidate_5min(minute_rows)

    eng = SymbolEngine(symbol, grid)
    cur_date = None
    for b in bars:
        if cur_date is not None and b.end.date() != cur_date:
            eng.on_session_end()
        cur_date = b.end.date()
        eng.on_bar(b)
    eng.on_session_end()
    return [t.to_dict() for t in eng.trades]


# ---------------------------------------------------------------------------
# Shard loading

def _load_old_trades(trades_dir: Path) -> tuple[list[dict], dict]:
    """Load all trades and meta from an existing shard directory.

    Returns (trades_list, meta_dict).  Validates grid_spec_hash.
    """
    shard_dirs = sorted(glob.glob(str(trades_dir / "shard*")))
    if not shard_dirs:
        shard_dirs = [str(trades_dir)]

    metas, trades = [], []
    seen: set = set()
    for sd in shard_dirs:
        mpath = os.path.join(sd, "meta.json")
        if not os.path.exists(mpath):
            sys.exit(f"missing meta.json in {sd}")
        with open(mpath) as _f:
            meta = json.load(_f)
        metas.append(meta)
        for f in sorted(glob.glob(os.path.join(sd, "trades_*.json"))):
            with open(f) as _f:
                batch = json.load(_f)
            for t in batch:
                k = (t["candidate_id"], t["symbol"], t["entry_time"])
                if k not in seen:
                    seen.add(k)
                    trades.append(t)

    hashes = {m["grid_spec_hash"] for m in metas}
    if len(hashes) != 1:
        sys.exit(f"inconsistent grid_spec_hash across shards: {hashes}")

    # Merge meta fields
    merged_meta = dict(metas[0])
    merged_meta["start"] = min(m["start"] for m in metas)
    merged_meta["end"]   = max(m["end"]   for m in metas)
    return trades, merged_meta


# ---------------------------------------------------------------------------
# Main rolling update

def roll_wf(
    old_trades_dir: Path,
    cache_dir: Path,
    new_start: str,
    new_end: str,
    out_dir: Path,
    symbols: list[str] | None = None,
    shard: int = 0,
    shards: int = 1,
) -> None:
    """Extend old_trades_dir with new_start..new_end data and write to out_dir.

    Parameters
    ----------
    old_trades_dir : existing wf trade-log directory (shard0of1/ layout)
    cache_dir      : directory containing 1-min bar gzip CSV files
    new_start      : first date of the new period (ISO-8601)
    new_end        : last date of the new period (ISO-8601)
    out_dir        : merged output directory (must not exist)
    symbols        : list of symbols to process; defaults to meta universe
    shard / shards : candidate sharding (default: single shard)
    """
    old_trades_dir = Path(old_trades_dir)
    cache_dir      = Path(cache_dir)
    out_dir        = Path(out_dir)

    old_trades, old_meta = _load_old_trades(old_trades_dir)

    full_grid = default_candidate_grid()
    current_hash = grid_spec_hash(full_grid)
    if old_meta["grid_spec_hash"] != current_hash:
        sys.exit(
            f"grid_spec_hash mismatch: old={old_meta['grid_spec_hash']} "
            f"current={current_hash}; run is invalid after grid modification"
        )

    grid = [c for i, c in enumerate(full_grid) if i % shards == shard]
    syms = symbols or old_meta.get("universe", [])

    new_trades: list[dict] = []
    for sym in syms:
        existing = find_cached(cache_dir, sym)
        if existing is None:
            print(f"[roll] {sym}: no cache found in {cache_dir}, skipping", flush=True)
            continue
        cache_path, _, _ = existing
        sym_trades = roll_symbol(sym, cache_path, new_start, new_end, grid)
        new_trades.extend(sym_trades)
        print(f"[roll] {sym}: {len(sym_trades)} new trades", flush=True)

    # Deduplicate: old trades by key, then new trades
    seen: set = set()
    merged: list[dict] = []
    for t in old_trades + new_trades:
        k = (t["candidate_id"], t["symbol"], t["entry_time"])
        if k not in seen:
            seen.add(k)
            merged.append(t)

    # Write output
    shard_out = out_dir / f"shard{shard}of{shards}"
    shard_out.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "schema_version": SCHEMA_VERSION,
        "grid_spec_hash": current_hash,
        "candidate_shard": shard,
        "candidate_shards": shards,
        "universe": syms,
        "start": min(old_meta["start"], new_start),
        "end": max(old_meta["end"], new_end),
        "resolution": "1min(iex)->5minute",
        "timezone": "America/New_York",
        "normalization": "adjusted(all)",
        "extended_hours": False,
        "data_source": "alpaca-iex-free",
        "trade_count": len(merged),
        "rolled_from": str(old_trades_dir),
        "new_period": {"start": new_start, "end": new_end},
    }
    (shard_out / "meta.json").write_text(json.dumps(meta_out, indent=2))

    for i in range(0, max(len(merged), 1), _CHUNK):
        (shard_out / f"trades_{i // _CHUNK:04d}.json").write_text(
            json.dumps(merged[i:i + _CHUNK])
        )
    print(f"[roll] merged {len(old_trades)} old + {len(new_trades)} new "
          f"= {len(merged)} trades -> {shard_out}/", flush=True)


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extend an existing ORB trade-log directory with a new period."
    )
    ap.add_argument("--old-trades-dir", required=True,
                    help="Existing trade-log directory (shard layout from local_pump.py)")
    ap.add_argument("--cache-dir", default="data",
                    help="Directory containing .csv.gz bar cache files")
    ap.add_argument("--new-start", required=True, help="Start of new period (YYYY-MM-DD)")
    ap.add_argument("--new-end",   required=True, help="End of new period (YYYY-MM-DD)")
    ap.add_argument("--out-dir",   required=True,
                    help="Output directory for merged trades (must not exist)")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Symbols to update (default: all symbols from old meta.json)")
    ap.add_argument("--shard",  type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    roll_wf(
        old_trades_dir=Path(args.old_trades_dir),
        cache_dir=Path(args.cache_dir),
        new_start=args.new_start,
        new_end=args.new_end,
        out_dir=Path(args.out_dir),
        symbols=args.symbols,
        shard=args.shard,
        shards=args.shards,
    )


if __name__ == "__main__":
    main()
