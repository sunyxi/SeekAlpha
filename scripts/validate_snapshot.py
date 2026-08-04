#!/usr/bin/env python3
"""Resolve and verify an immutable research snapshot manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orb.immutable_cache import ImmutableResearchCache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    snapshot = ImmutableResearchCache(args.cache_root).resolve_snapshot(args.snapshot_id)
    print(f"passed: {snapshot['snapshot_id']} files={len(snapshot['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
