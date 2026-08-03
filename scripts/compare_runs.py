#!/usr/bin/env python3
"""Paired statistical comparison of two wf_select.py report JSONs.

Compares a baseline (no meta-labeling) run against an ML (meta-label) run
on the same outer-test fold boundaries, producing a paired t-test report.

Usage
-----
    python3 scripts/compare_runs.py \\
        --baseline reports/orb045-wf.json \\
        --ml       reports/orb045-ml.json \\
        --output   reports/orb045-comparison.json

    # Allow mismatched input metadata (recorded in output; use with caution):
    python3 scripts/compare_runs.py --allow-input-mismatch ...

Output is create-only (refuses to overwrite existing file, atomic via os.link).
Stdlib-only — no numpy, scipy, or other optional dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime

# Fields that must match between the two reports' input_meta for a valid comparison.
_INPUT_META_REQUIRED_MATCH: list[str] = [
    "grid_spec_hash",
    "universe",
    "start",
    "end",
    "resolution",
    "normalization",
    "data_source",
    "trade_count",
]

# Metrics extracted per fold for the paired comparison.
_COMPARED_METRICS: list[str] = [
    "sharpe",
    "profit_factor",
    "mean_net_bps",
    "win_rate",
    "total_net_pnl",
]

# Significance threshold (two-sided, normal approximation).
_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Core functions (importable by tests)

def _extract_input_meta(report: dict) -> dict:
    """Return a flat dict of input_meta fields from the first shard."""
    meta = report.get("input_meta")
    if isinstance(meta, list):
        return meta[0] if meta else {}
    if isinstance(meta, dict):
        return meta
    return {}


def check_input_meta(
    baseline_report: dict,
    ml_report: dict,
) -> dict[str, dict]:
    """Compare input_meta fields between two reports.

    Returns a dict mapping field name → {"baseline": v1, "ml": v2}
    for every field in _INPUT_META_REQUIRED_MATCH that differs between
    the two reports.  An empty dict means all checked fields match.
    """
    base_meta = _extract_input_meta(baseline_report)
    ml_meta   = _extract_input_meta(ml_report)
    mismatches: dict[str, dict] = {}
    for field in _INPUT_META_REQUIRED_MATCH:
        bv = base_meta.get(field)
        mv = ml_meta.get(field)
        if bv != mv:
            mismatches[field] = {"baseline": bv, "ml": mv}
    return mismatches


def fold_metrics(folds: list[dict]) -> dict[int, dict]:
    """Return {fold_index: test_metrics} for folds where a candidate was selected."""
    return {
        f["fold"]: f["test_metrics"]
        for f in folds
        if f.get("selected") is not None and "test_metrics" in f
    }


def paired_ttest(diffs: list[float]) -> dict:
    """Two-sided paired t-test via normal approximation.

    Parameters
    ----------
    diffs : list of float
        Per-fold differences (ML - baseline) for a single metric.
        Must have length >= 2.

    Returns
    -------
    dict with keys: n, mean_diff, std_diff, t_stat, p_value, significant
    """
    n = len(diffs)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    mean_d = statistics.mean(diffs)
    std_d  = statistics.stdev(diffs)      # sample std, ddof=1

    if std_d == 0.0:
        t_stat  = float("nan")
        p_value = float("nan")
    else:
        se     = std_d / math.sqrt(n)
        t_stat = mean_d / se
        # Two-sided p-value using normal (Gaussian) approximation.
        # Consistent with the approach used in ic_eval.py.
        p_value = math.erfc(abs(t_stat) / math.sqrt(2))

    return {
        "n":          n,
        "mean_diff":  mean_d,
        "std_diff":   std_d,
        "t_stat":     t_stat,
        "p_value":    p_value,
        "significant": (math.isfinite(p_value) and p_value < _ALPHA),
    }


def compare(
    baseline_report: dict,
    ml_report: dict,
    input_mismatch: dict | None = None,
) -> dict:
    """Compute per-metric paired comparison between two wf_select reports.

    Folds are matched by fold index.  Only folds where both reports selected
    a candidate (test_metrics present) are included.

    Parameters
    ----------
    input_mismatch : dict or None
        If not None, the result of check_input_meta(); recorded in the output
        under "input_mismatch".  None means the caller has verified inputs match.

    Returns a comparison dict suitable for JSON serialisation.
    """
    base_m = fold_metrics(baseline_report.get("folds", []))
    ml_m   = fold_metrics(ml_report.get("folds", []))

    common = sorted(base_m.keys() & ml_m.keys())
    result: dict = {
        "schema_version": 1,
        "common_folds":   len(common),
        "metrics_compared": _COMPARED_METRICS,
    }

    if len(common) < 2:
        result["paired_comparison"] = None
        result["verdict"] = "insufficient_data"
        return result

    paired: dict = {}
    for metric in _COMPARED_METRICS:
        diffs = [
            ml_m[k].get(metric, float("nan")) - base_m[k].get(metric, float("nan"))
            for k in common
            if math.isfinite(ml_m[k].get(metric, float("nan")))
            and math.isfinite(base_m[k].get(metric, float("nan")))
        ]
        if len(diffs) < 2:
            paired[metric] = {"n": len(diffs), "verdict": "insufficient_data"}
            continue

        stats = paired_ttest(diffs)
        paired[metric] = {
            **stats,
            "baseline_mean": statistics.mean(base_m[k].get(metric, 0.0) for k in common),
            "ml_mean":       statistics.mean(ml_m[k].get(metric, 0.0)   for k in common),
        }

    result["paired_comparison"] = paired

    if input_mismatch is not None:
        result["input_mismatch"] = input_mismatch

    # Overall verdict based on mean_net_bps (primary) + sharpe
    sharpe_better = (
        paired.get("sharpe", {}).get("significant") is True
        and paired.get("sharpe", {}).get("mean_diff", 0.0) > 0
    )
    bps_better = (
        paired.get("mean_net_bps", {}).get("significant") is True
        and paired.get("mean_net_bps", {}).get("mean_diff", 0.0) > 0
    )
    sharpe_worse = (
        paired.get("sharpe", {}).get("significant") is True
        and paired.get("sharpe", {}).get("mean_diff", 0.0) < 0
    )

    if sharpe_better or bps_better:
        result["verdict"] = "ml_significantly_better"
    elif sharpe_worse:
        result["verdict"] = "ml_significantly_worse"
    else:
        result["verdict"] = "no_significant_improvement"

    return result


def write_report(
    baseline_report: dict,
    ml_report: dict,
    output_path: str,
    *,
    allow_input_mismatch: bool = False,
) -> None:
    """Compare and write report JSON to output_path (create-only, atomic).

    Raises SystemExit with a non-zero code if input_meta fields differ between
    the two reports, unless allow_input_mismatch=True is passed.  When allowed,
    the mismatch is recorded in the output under "input_mismatch".
    """
    if os.path.exists(output_path):
        sys.exit(f"output path exists; create-only policy refuses overwrite: {output_path}")

    mismatches = check_input_meta(baseline_report, ml_report)
    if mismatches and not allow_input_mismatch:
        diff_fields = ", ".join(mismatches.keys())
        sys.exit(
            f"input_meta mismatch between reports — fields differ: {diff_fields}\n"
            f"Details: {mismatches}\n"
            "Pass --allow-input-mismatch to proceed (mismatch will be recorded in output)."
        )

    cmp = compare(baseline_report, ml_report, input_mismatch=mismatches or None)
    cmp["generated_at"] = datetime.now().astimezone().isoformat()
    cmp["baseline_report"] = baseline_report.get("_source_path", "")
    cmp["ml_report"]       = ml_report.get("_source_path", "")

    tmp = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(cmp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, output_path)
    os.unlink(tmp)


# ---------------------------------------------------------------------------
# CLI

def _load(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"invalid JSON in {path}: {e}")
    data["_source_path"] = path
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired ML vs baseline comparison of two wf_select reports."
    )
    ap.add_argument("--baseline", required=True,
                    help="Path to baseline (no meta-label) wf_select report JSON")
    ap.add_argument("--ml",       required=True,
                    help="Path to ML (meta-label) wf_select report JSON")
    ap.add_argument("--output",   required=True,
                    help="Output path for the comparison report JSON (create-only)")
    ap.add_argument("--allow-input-mismatch", action="store_true", default=False,
                    help="Allow mismatched input_meta fields (recorded in output). "
                         "Use only when the mismatch is understood and accepted.")
    args = ap.parse_args()

    baseline_report = _load(args.baseline)
    ml_report       = _load(args.ml)

    write_report(baseline_report, ml_report, args.output,
                 allow_input_mismatch=args.allow_input_mismatch)
    print(f"verdict: {json.load(open(args.output))['verdict']}")
    print(f"report  -> {args.output}")


if __name__ == "__main__":
    main()
