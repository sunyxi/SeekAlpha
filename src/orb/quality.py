"""Per-symbol data quality statistics for 1-minute bar cache files.

Public API
----------
compute_quality(symbol, minute_rows, start, end, is_trading_day, session_end_fn)
    -> SymbolQuality

format_quality_report(results) -> str
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

_RTH_START = time(9, 30)
_EARLY_CLOSE = time(13, 0)


def _expected_bars(sess_end: time) -> int:
    """Minutes from RTH open (9:30) to session close (exclusive)."""
    return sess_end.hour * 60 + sess_end.minute - 9 * 60 - 30


@dataclass
class SymbolQuality:
    """Quality metrics for one symbol over the simulation date range."""

    symbol: str
    total_sessions: int       # expected trading days in [start, end]
    full_bar_sessions: int    # sessions where bar_count >= expected
    partial_sessions: int     # sessions where 0 < bar_count < expected
    zero_bar_sessions: int    # sessions with no bars at all (gaps)
    stale_bar_count: int      # bars where volume == 0
    gap_rate_pct: float       # zero_bar_sessions / total_sessions × 100
    max_consecutive_gaps: int # longest run of consecutive zero-bar sessions

    def has_warning(self, threshold_pct: float = 1.0) -> bool:
        return self.gap_rate_pct > threshold_pct

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_warning"] = self.has_warning()
        return d


def compute_quality(
    symbol: str,
    minute_rows: list,
    start: str,
    end: str,
    is_trading_day: Callable[[date], bool],
    session_end_fn: Callable[[date], time],
) -> SymbolQuality:
    """Compute data quality metrics for one symbol.

    Parameters
    ----------
    symbol:
        Ticker symbol (for labelling only).
    minute_rows:
        Output of ``load_minute_bars`` — list of
        ``(tz-aware datetime, open, high, low, close, volume)`` tuples.
    start, end:
        ISO-8601 date strings defining the expected data range (inclusive).
    is_trading_day:
        Callable returning True for NYSE trading days.  ``ValueError`` is
        silently swallowed for out-of-range dates (calendar edge cases).
    session_end_fn:
        Callable returning the session-end time for a trading day.
    """
    bars_per_day: dict[date, int] = defaultdict(int)
    stale_bar_count = 0

    for row in minute_rows:
        t: datetime = row[0]
        v: float = row[5]
        bars_per_day[t.date()] += 1
        if v == 0.0:
            stale_bar_count += 1

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)

    total_sessions = 0
    full_bar_sessions = 0
    partial_sessions = 0
    zero_bar_sessions = 0
    max_consecutive_gaps = 0
    current_run = 0

    cur = s
    while cur <= e:
        try:
            trading = is_trading_day(cur)
        except ValueError:
            cur += timedelta(days=1)
            continue

        if trading:
            total_sessions += 1
            expected = _expected_bars(session_end_fn(cur))
            actual = bars_per_day.get(cur, 0)

            if actual == 0:
                zero_bar_sessions += 1
                current_run += 1
                max_consecutive_gaps = max(max_consecutive_gaps, current_run)
            else:
                current_run = 0
                if actual >= expected:
                    full_bar_sessions += 1
                else:
                    partial_sessions += 1

        cur += timedelta(days=1)

    gap_rate_pct = (
        round(zero_bar_sessions / total_sessions * 100.0, 4)
        if total_sessions > 0
        else 0.0
    )

    return SymbolQuality(
        symbol=symbol,
        total_sessions=total_sessions,
        full_bar_sessions=full_bar_sessions,
        partial_sessions=partial_sessions,
        zero_bar_sessions=zero_bar_sessions,
        stale_bar_count=stale_bar_count,
        gap_rate_pct=gap_rate_pct,
        max_consecutive_gaps=max_consecutive_gaps,
    )


def format_quality_report(results: list[SymbolQuality]) -> str:
    """Return a human-readable table of quality metrics."""
    if not results:
        return "=== Data Quality Report: no symbols ==="

    header = (
        f"{'Symbol':<8} {'Sessions':>8} {'Full':>6} "
        f"{'Partial':>8} {'Missing':>8} {'Gap%':>7} "
        f"{'MaxRun':>7} {'Stale':>7}"
    )
    sep = "-" * len(header)
    lines = ["=== Data Quality Report ===", header, sep]

    for q in results:
        warn = "  *** WARNING" if q.has_warning() else ""
        lines.append(
            f"{q.symbol:<8} {q.total_sessions:>8} {q.full_bar_sessions:>6} "
            f"{q.partial_sessions:>8} {q.zero_bar_sessions:>8} "
            f"{q.gap_rate_pct:>7.2f} {q.max_consecutive_gaps:>7} "
            f"{q.stale_bar_count:>7}{warn}"
        )

    warnings = [q.symbol for q in results if q.has_warning()]
    if warnings:
        lines.append(sep)
        lines.append(f"WARNING: gap rate > 1% for: {', '.join(warnings)}")

    return "\n".join(lines)


def write_quality_json(
    results: list[SymbolQuality],
    start: str,
    end: str,
    out_path: Path,
) -> None:
    """Write quality report JSON atomically (temp-file swap)."""
    import os

    data = {
        "generated_at": datetime.now().isoformat(),
        "start": start,
        "end": end,
        "symbols": [q.to_dict() for q in results],
        "summary": {
            "any_warnings": any(q.has_warning() for q in results),
            "warned_symbols": [q.symbol for q in results if q.has_warning()],
        },
    }
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, out_path)
