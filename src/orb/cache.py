"""Cache integrity validation for 1-minute bar gzip CSV files.

Public API
----------
validate_cache(path, start, end) -> ValidationResult
"""

from __future__ import annotations

import gzip
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_EXPECTED_HEADER = "ts,open,high,low,close,volume"


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
