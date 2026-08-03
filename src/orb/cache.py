"""Cache management for 1-minute bar gzip CSV files.

Public API
----------
validate_cache(path, start, end)   -> ValidationResult
find_cached(cache_dir, symbol)     -> Optional[tuple[Path, str, str]]
read_all_rows(path)                -> list[str]
write_gzip_csv(path, rows)         -> None
"""

from __future__ import annotations

import gzip
import os
import re
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_EXPECTED_HEADER = "ts,open,high,low,close,volume"

# Filename pattern: <SYMBOL>_<YYYY-MM-DD>_<YYYY-MM-DD>_1min.csv.gz
_CACHE_PAT = re.compile(
    r"^(.+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_1min\.csv\.gz$"
)


def find_cached(cache_dir: Path, symbol: str) -> Optional[tuple[Path, str, str]]:
    """Find the most recent cached file for `symbol` in `cache_dir`.

    Returns (path, start_str, end_str) where start_str and end_str are
    ISO-8601 date strings parsed from the filename, or None if no matching
    file exists.  When multiple files match (e.g. from a partial extend),
    the one with the latest end date is returned.
    """
    cache_dir = Path(cache_dir)
    best: Optional[tuple[Path, str, str]] = None
    for p in cache_dir.iterdir() if cache_dir.exists() else []:
        m = _CACHE_PAT.match(p.name)
        if m and m.group(1) == symbol:
            start, end = m.group(2), m.group(3)
            if best is None or end > best[2]:
                best = (p, start, end)
    return best


def read_all_rows(path: Path) -> list[str]:
    """Return all data rows from a gzip CSV (header excluded, no trailing newline)."""
    rows: list[str] = []
    with gzip.open(path, "rt") as f:
        next(f)  # skip header
        for line in f:
            line = line.rstrip("\n")
            if line:
                rows.append(line)
    return rows


def write_gzip_csv(path: Path, rows: list[str]) -> None:
    """Write header + rows to a gzip CSV atomically via a .tmp file."""
    tmp = Path(str(path) + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tmp, "wt") as f:
        f.write(_EXPECTED_HEADER + "\n")
        for row in rows:
            f.write(row + "\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a cache integrity check."""

    path: Path
    ok: bool
    error: Optional[str]
    row_count: int
    first_ts: Optional[datetime]
    last_ts: Optional[datetime]


def validate_cache(path: Path, start: str, end: str) -> ValidationResult:
    """Validate a cached 1-min bar gzip CSV for integrity.

    Checks performed (in order):
    1. File is a readable gzip archive — not truncated or corrupted.
    2. First line matches the expected header schema.
    3. Every row has a parseable ISO-8601 timestamp.
    4. Timestamps are strictly increasing (no duplicates, no reordering).
    5. At least one data row is present.

    ``start`` and ``end`` (ISO-8601 date strings, inclusive) are available for
    future coverage-gap checks against the trading calendar; they are currently
    included in the ValidationResult for diagnostic purposes only.

    Never raises — all errors are captured in ValidationResult.error.
    """
    row_count = 0
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    prev_ts: Optional[datetime] = None

    try:
        with gzip.open(path, "rt") as f:
            header = f.readline().rstrip("\r\n")
            if header != _EXPECTED_HEADER:
                return ValidationResult(
                    path=path,
                    ok=False,
                    error=f"unexpected header: {header!r}",
                    row_count=0,
                    first_ts=None,
                    last_ts=None,
                )

            for lineno, line in enumerate(f, start=2):
                line = line.rstrip("\r\n")
                if not line:
                    continue

                try:
                    ts = datetime.fromisoformat(line.split(",")[0])
                except (ValueError, IndexError) as exc:
                    return ValidationResult(
                        path=path,
                        ok=False,
                        error=f"line {lineno}: bad timestamp: {exc}",
                        row_count=row_count,
                        first_ts=first_ts,
                        last_ts=last_ts,
                    )

                if prev_ts is not None:
                    if ts == prev_ts:
                        return ValidationResult(
                            path=path,
                            ok=False,
                            error=f"line {lineno}: duplicate timestamp {ts.isoformat()}",
                            row_count=row_count,
                            first_ts=first_ts,
                            last_ts=last_ts,
                        )
                    if ts < prev_ts:
                        return ValidationResult(
                            path=path,
                            ok=False,
                            error=(
                                f"line {lineno}: timestamp {ts.isoformat()} "
                                f"precedes previous {prev_ts.isoformat()} (not sorted)"
                            ),
                            row_count=row_count,
                            first_ts=first_ts,
                            last_ts=last_ts,
                        )

                row_count += 1
                if first_ts is None:
                    first_ts = ts
                last_ts = prev_ts = ts

    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as exc:
        return ValidationResult(
            path=path,
            ok=False,
            error=f"gzip error: {type(exc).__name__}: {exc}",
            row_count=row_count,
            first_ts=first_ts,
            last_ts=last_ts,
        )

    if row_count == 0:
        return ValidationResult(
            path=path,
            ok=False,
            error="file contains no data rows",
            row_count=0,
            first_ts=None,
            last_ts=None,
        )

    return ValidationResult(
        path=path,
        ok=True,
        error=None,
        row_count=row_count,
        first_ts=first_ts,
        last_ts=last_ts,
    )
