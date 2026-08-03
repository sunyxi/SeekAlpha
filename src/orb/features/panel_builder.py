"""Daily OHLCV panel builder from 1-minute gzip CSV cache.

Reads ``<cache_dir>/<SYMBOL>_*_1min.csv.gz`` files produced by
``scripts/local_pump.py``, filters to Regular Trading Hours (09:30–15:59 ET),
aggregates to daily OHLCV, aligns all symbols to the NYSE trading calendar,
and forward-fills missing days.

Public API
----------
build_panel(cache_dir, symbols=None, start=None, end=None) -> DailyPanel

DailyPanel.to_alpha101_kwargs() -> dict
    Returns {open_, high, low, close, volume} of shape (T, N) for Alpha101.

Missing-day forward-fill behaviour (documented):
    open = high = low = close = previous day's close; volume = 0.
    Days before the first bar for a symbol are left as NaN.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from orb.calendar import is_trading_day

_NY = ZoneInfo("America/New_York")
_RTH_START = time(9, 30)
_RTH_END   = time(16, 0)   # exclusive: bars starting at 16:00 are excluded

# Panel column indices
OPEN, HIGH, LOW, CLOSE, VOLUME = 0, 1, 2, 3, 4
COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Public data container

@dataclass(frozen=True)
class DailyPanel:
    """Aligned daily OHLCV panel for N symbols over T trading days.

    Attributes
    ----------
    symbols : tuple[str, ...], length N
        Symbol names in alphabetical order.
    dates : tuple[date, ...], length T
        NYSE trading days in ascending order.
    panel : ndarray, shape (N, T, 5)
        Axis-2 columns: OPEN=0, HIGH=1, LOW=2, CLOSE=3, VOLUME=4.
        NaN for days before the first observed bar for a symbol.
        Forward-filled (O=H=L=C=prev_close, vol=0) for all subsequent
        missing days.
    """

    symbols: tuple[str, ...]
    dates: tuple[date, ...]
    panel: np.ndarray  # shape (N, T, 5)

    def to_alpha101_kwargs(self) -> dict[str, np.ndarray]:
        """Return arrays of shape (T, N) suitable for ``Alpha101(**kwargs)``."""
        return {
            "open_":  self.panel[:, :, OPEN].T,
            "high":   self.panel[:, :, HIGH].T,
            "low":    self.panel[:, :, LOW].T,
            "close":  self.panel[:, :, CLOSE].T,
            "volume": self.panel[:, :, VOLUME].T,
        }


# ---------------------------------------------------------------------------
# Internal helpers

def _rth_et_time(ts_str: str) -> tuple[date, time] | None:
    """Return (ET date, ET time) if the bar falls within RTH, else None.

    Handles both EST (UTC-5, winter) and EDT (UTC-4, summer) automatically
    via ZoneInfo("America/New_York").
    """
    dt_utc = datetime.fromisoformat(ts_str)
    dt_et = dt_utc.astimezone(_NY)
    t = dt_et.time().replace(tzinfo=None)
    if t < _RTH_START or t >= _RTH_END:
        return None
    return dt_et.date(), t


def _aggregate_symbol(path: Path) -> dict[date, np.ndarray]:
    """Parse one gzip CSV; return {date: ndarray([O, H, L, C, V])} for RTH only.

    Aggregation rules per calendar day:
      open   = first RTH bar's open
      high   = max of all RTH bar highs
      low    = min of all RTH bar lows
      close  = last RTH bar's close
      volume = sum of all RTH bar volumes
    """
    daily: dict[date, list[float]] = {}
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header: ts,open,high,low,close,volume
        for row in reader:
            if len(row) < 6:
                continue
            result = _rth_et_time(row[0])
            if result is None:
                continue
            d, _ = result
            o, h, lo, c, v = (float(row[k]) for k in range(1, 6))
            if d not in daily:
                daily[d] = [o, h, lo, c, v]
            else:
                acc = daily[d]
                if h  > acc[HIGH]:   acc[HIGH]   = h
                if lo < acc[LOW]:    acc[LOW]    = lo
                acc[CLOSE]  = c      # keep overwriting — last bar wins
                acc[VOLUME] += v
    return {d: np.array(vals, dtype=float) for d, vals in daily.items()}


def _trading_days_between(start: date, end: date) -> list[date]:
    """Return every NYSE trading day in the closed interval [start, end]."""
    out: list[date] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Public API

def build_panel(
    cache_dir: Path | str,
    symbols: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> DailyPanel:
    """Build a daily OHLCV panel from the 1-minute gzip CSV cache.

    Parameters
    ----------
    cache_dir :
        Directory containing ``<SYMBOL>_*_1min.csv.gz`` files.
    symbols :
        Subset of symbols to load. Defaults to all files in cache_dir.
    start, end :
        Inclusive date range. Defaults to the extremes found in the data.

    Returns
    -------
    DailyPanel
        Symbols sorted alphabetically; dates sorted ascending.
        Missing days are forward-filled (O=H=L=C=prev_close, vol=0).
        Days before the first bar for a symbol are NaN.
    """
    cache_dir = Path(cache_dir)

    # Discover one file per symbol (last alphabetically wins ties)
    files: dict[str, Path] = {}
    for p in sorted(cache_dir.glob("*_1min.csv.gz")):
        sym = p.name.split("_")[0]
        if symbols is None or sym in symbols:
            files[sym] = p  # sorted() ensures later entries overwrite earlier

    if not files:
        raise FileNotFoundError(f"no *_1min.csv.gz files found in {cache_dir}")

    sorted_syms = sorted(files.keys())

    # Aggregate per-symbol daily bars
    sym_data: dict[str, dict[date, np.ndarray]] = {}
    all_dates: set[date] = set()
    for sym in sorted_syms:
        agg = _aggregate_symbol(files[sym])
        sym_data[sym] = agg
        all_dates.update(agg.keys())

    if not all_dates:
        raise ValueError("no RTH bars found in any cache file")

    # Determine master trading-day list
    data_min = min(all_dates)
    data_max = max(all_dates)
    eff_start = start if start is not None else data_min
    eff_end   = end   if end   is not None else data_max
    master_dates = _trading_days_between(eff_start, eff_end)

    if not master_dates:
        raise ValueError(f"no trading days in [{eff_start}, {eff_end}]")

    T = len(master_dates)
    N = len(sorted_syms)
    panel = np.full((N, T, 5), np.nan, dtype=float)

    # Fill panel with forward-fill for missing days
    for i, sym in enumerate(sorted_syms):
        agg = sym_data[sym]
        prev_close = np.nan
        for j, d in enumerate(master_dates):
            if d in agg:
                panel[i, j] = agg[d]
                prev_close = agg[d][CLOSE]
            elif np.isfinite(prev_close):
                # forward-fill: flat day, zero volume
                panel[i, j] = [prev_close, prev_close, prev_close, prev_close, 0.0]
            # else: leave NaN — no prior data for this symbol yet

    return DailyPanel(
        symbols=tuple(sorted_syms),
        dates=tuple(master_dates),
        panel=panel,
    )
