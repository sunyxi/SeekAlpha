#!/usr/bin/env python3
"""Validate the selected data-source and point-in-time universe policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orb.data_source_policy import DataSourcePolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("src/orb/data_source_policy.json"))
    args = parser.parse_args()
    policy = DataSourcePolicy.load(args.policy)
    print(f"passed: {policy.policy_id} primary={policy.primary.provider} fallback={policy.fallback.provider}")
    print(f"blockers: {len(policy.blockers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
