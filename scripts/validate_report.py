#!/usr/bin/env python3
"""Walk-forward report schema validator.

Verifies schema_version, required fields, gate-metric completeness,
grid_spec_hash match against the frozen 192-candidate grid, and valid
decision value.

Usage
-----
    python3 scripts/validate_report.py --report <path>

Exit codes: 0 = valid, 1 = one or more validation errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orb.core import default_candidate_grid, grid_spec_hash as _compute_grid_hash

_SCHEMA_VERSION = 1
_VALID_DECISIONS = frozenset({"Candidate", "No-Go"})

_REQUIRED_TOP = frozenset({
    "schema_version", "generated_at", "input_meta",
    "fold_definition", "selection_gates", "decision_gates",
    "folds", "outer_test_metrics_by_cost",
    "symbol_attribution_baseline", "decision", "decision_reasons",
})

_REQUIRED_FOLD_DEF = frozenset({"train_days", "test_days", "step_days"})

_REQUIRED_SEL_GATES = frozenset({
    "min_trades", "min_train_sharpe", "min_train_pf", "validation",
})

_REQUIRED_DEC_GATES = frozenset({
    "min_outer_trades", "min_outer_sharpe", "min_outer_pf",
})

_REQUIRED_COST_SCENARIOS = frozenset({"zero", "baseline", "double"})

_REQUIRED_COST_FIELDS = frozenset({
    "trades", "win_rate", "profit_factor", "sharpe",
    "mean_net_bps", "total_net_pnl", "max_drawdown_frac",
})

# Cached at module load so the test suite can override cheaply via monkeypatch.
_FROZEN_GRID_HASH: str = _compute_grid_hash(default_candidate_grid())


def validate(data: dict) -> list[str]:
    """Return a list of error strings for *data* (empty list = valid).

    Accepts a pre-parsed dict so callers can validate synthetic reports
    without writing to disk.
    """
    errors: list[str] = []

    # 1. Schema version
    sv = data.get("schema_version")
    if sv != _SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {_SCHEMA_VERSION}, got {sv!r}"
        )

    # 2. Required top-level fields
    for f in sorted(_REQUIRED_TOP - set(data)):
        errors.append(f"missing required field: {f!r}")

    # 3. fold_definition sub-fields
    for f in sorted(_REQUIRED_FOLD_DEF - set(data.get("fold_definition") or {})):
        errors.append(f"fold_definition missing field: {f!r}")

    # 4. selection_gates sub-fields
    for f in sorted(_REQUIRED_SEL_GATES - set(data.get("selection_gates") or {})):
        errors.append(f"selection_gates missing field: {f!r}")

    # 5. decision_gates sub-fields
    for f in sorted(_REQUIRED_DEC_GATES - set(data.get("decision_gates") or {})):
        errors.append(f"decision_gates missing field: {f!r}")

    # 6. Cost scenarios and their metrics
    cost = data.get("outer_test_metrics_by_cost") or {}
    for s in sorted(_REQUIRED_COST_SCENARIOS - set(cost)):
        errors.append(f"outer_test_metrics_by_cost missing scenario: {s!r}")
    for s in sorted(_REQUIRED_COST_SCENARIOS & set(cost)):
        for f in sorted(_REQUIRED_COST_FIELDS - set(cost[s] or {})):
            errors.append(
                f"outer_test_metrics_by_cost[{s!r}] missing field: {f!r}"
            )

    # 7. decision value
    decision = data.get("decision")
    if decision not in _VALID_DECISIONS:
        errors.append(
            f"decision: expected one of {sorted(_VALID_DECISIONS)}, got {decision!r}"
        )

    # 8. decision_reasons is a list
    dr = data.get("decision_reasons")
    if not isinstance(dr, list):
        errors.append(
            f"decision_reasons: expected list, got {type(dr).__name__!r}"
        )

    # 9. symbol_attribution_baseline is a non-empty dict
    sa = data.get("symbol_attribution_baseline")
    if not isinstance(sa, dict) or not sa:
        errors.append("symbol_attribution_baseline: expected non-empty dict")

    # 10. grid_spec_hash matches the frozen grid
    meta = data.get("input_meta")
    if not isinstance(meta, list) or not meta:
        errors.append("input_meta: expected non-empty list")
    else:
        report_hash = meta[0].get("grid_spec_hash")
        if report_hash != _FROZEN_GRID_HASH:
            errors.append(
                f"grid_spec_hash mismatch: report has {report_hash!r}, "
                f"frozen grid is {_FROZEN_GRID_HASH!r}"
            )

    return errors


def validate_file(path: Path) -> list[str]:
    """Load JSON from *path* and validate. Returns error strings."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"JSON parse error: {exc}"]
    except OSError as exc:
        return [f"cannot read file: {exc}"]
    return validate(data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate a walk-forward report JSON against the frozen schema."
    )
    ap.add_argument("--report", required=True, metavar="PATH",
                    help="Path to the walk-forward report JSON")
    args = ap.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"ERROR: {path}: not found", file=sys.stderr)
        sys.exit(1)

    errors = validate_file(path)
    if errors:
        print(f"INVALID  {path}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"OK  {path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
