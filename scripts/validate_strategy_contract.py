#!/usr/bin/env python3
"""Validate a strategy specification and its decision report together."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orb.strategy_contract import DecisionReport, StrategySpec, create_only_write_report, render_decision_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    spec = StrategySpec.load(args.spec)
    report = DecisionReport.validate(json.loads(args.report.read_text(encoding="utf-8")), spec)
    if args.summary_output:
        output = render_decision_summary(spec, report)
        path = args.summary_output
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(output, encoding="utf-8")
        try:
            import os
            os.link(temporary, path)
        except FileExistsError:
            raise SystemExit(f"summary already exists: {path}")
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    print(f"passed: {report.data['report_id']} ({report.decision_state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
