#!/usr/bin/env python3
"""Export the existing SeekAlpha cache as a Qlib-compatible daily CSV."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orb.features.panel_builder import build_panel
from orb.qlib_adapter import require_qlib, to_qlib_frame


def _atomic_create_csv(frame, output: Path) -> None:
    if output.exists():
        sys.exit(f"output path exists; create-only policy refuses overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=output.parent, suffix=".tmp.csv")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            frame.reset_index().to_csv(handle, index=False)
        os.link(tmp_name, output)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verify-qlib",
        action="store_true",
        help="also import the optional pyqlib runtime and print its version",
    )
    args = parser.parse_args()

    qlib_runtime = require_qlib() if args.verify_qlib else None
    panel = build_panel(
        Path(args.cache_dir),
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    frame = to_qlib_frame(panel)
    _atomic_create_csv(frame, Path(args.output))
    print(
        f"[qlib_poc] wrote {len(frame)} rows for {len(panel.symbols)} symbols "
        f"to {args.output}"
    )
    if qlib_runtime is not None:
        print(f"[qlib_poc] pyqlib runtime: {qlib_runtime.__version__}")


if __name__ == "__main__":
    main()
