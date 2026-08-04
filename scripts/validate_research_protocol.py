#!/usr/bin/env python3
"""Validate and print the frozen SeekAlpha research protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support direct execution from a checkout without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orb.research_protocol import ResearchProtocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("src/orb/research_protocol.json"),
        help="path to the machine-readable protocol",
    )
    parser.add_argument("--json", action="store_true", help="print canonical protocol metadata")
    args = parser.parse_args()
    protocol = ResearchProtocol.load(args.protocol)
    if args.json:
        print(json.dumps({
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
            "retention_available_after": protocol.retention_available_after.isoformat(),
        }, sort_keys=True))
    else:
        print(f"passed: {protocol.protocol_id} ({protocol.protocol_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
