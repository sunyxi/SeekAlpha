#!/usr/bin/env python3
"""Drift monitoring report for ORB walk-forward performance.

Detects whether the strategy's most-recent fold performance has shifted
significantly from the historical reference by computing per-metric z-scores
across the fold-level test_metrics recorded in a wf_select.py report JSON.

Algorithm
---------
1. Extract all folds where a candidate was selected (test_metrics present).
2. Split chronologically: reference = all but the last ``recent_folds`` selected
   folds; recent = the last ``recent_folds`` selected folds.
3. For each tracked metric: compute z = (recent_mean - ref_mean) / ref_std.
4. Per-metric verdict: |z| < 1 → stable, 1 ≤ |z| < 2 → moderate_drift,
   |z| ≥ 2 → significant_drift.  Negative z for performance metrics (sharpe,
   bps, win_rate, pf) means degradation.
5. Overall verdict: worst single-metric verdict that represents a degradation
   (negative z); improvements are not flagged as drift.

Output is create-only, atomic (os.link swap), consistent with wf_select.py.
Stdlib-only — no numpy, scipy, or other optional dependencies.

Usage
-----
    python3 scripts/drift_monitor.py \\
        --report reports/orb045-extended-wf.json \\
        --output reports/orb045-drift.json \\
        [--recent-folds 3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime

# Metrics to track (all are "higher is better" for performance assessment)
_TRACKED_METRICS: list[str] = [
    "sharpe",
    "profit_factor",
    "mean_net_bps",
    "win_rate",
]

# z-score thresholds for verdict classification
_MODERATE_Z:     float = 1.0
_SIGNIFICANT_Z:  float = 2.0

# Minimum reference folds needed for a valid z-score
_MIN_REF_FOLDS: int = 3


# ---------------------------------------------------------------------------
# Core functions (importable by tests)

def split_folds(
    folds: list[dict],
    recent_folds: int,
) -> tuple[list[dict], list[dict]]:
    """Split selected folds into (reference, recent).

    Only folds where a candidate was selected (test_metrics present) are
    included.  The most recent ``recent_folds`` selected folds become the
    "recent" window; all earlier selected folds become the "reference".

    Returns (reference, recent).  reference is empty when there are fewer
    than (recent_folds + _MIN_REF_FOLDS) selected folds in total.
    """
    selected = [
        f for f in folds
        if f.get("selected") is not None and "test_metrics" in f
    ]
    # Preserve chronological order (folds are written in order by wf_select.py)
    selected.sort(key=lambda f: f["fold"])

    if len(selected) <= recent_folds or len(selected) - recent_folds < _MIN_REF_FOLDS:
        return [], selected  # caller treats empty reference as insufficient

    reference = selected[:-recent_folds]
    recent    = selected[-recent_folds:]
    return reference, recent


def metric_zscore(ref_values: list[float], rec_values: list[float]) -> float:
    """Two-sample z-score: (mean_rec - mean_ref) / std_ref.

    Returns float("nan") if std_ref == 0 or len(ref_values) < 2.
    """
    if len(ref_values) < 2 or not rec_values:
        return float("nan")
    try:
        std_ref = statistics.stdev(ref_values)
    except statistics.StatisticsError:
        return float("nan")
    if std_ref == 0.0:
        return float("nan")
    mean_ref = statistics.mean(ref_values)
    mean_rec = statistics.mean(rec_values)
    return (mean_rec - mean_ref) / std_ref


def _metric_verdict(z: float) -> str:
    """Per-metric verdict based on absolute z magnitude (degradation flagged)."""
    if not math.isfinite(z):
        return "insufficient_data"
    if z >= -_MODERATE_Z:          # stable or improving
        return "stable"
    if z >= -_SIGNIFICANT_Z:       # moderate degradation
        return "moderate_drift"
    return "significant_drift"     # significant degradation


def _overall_verdict(metric_results: dict) -> str:
    """Worst verdict across all metrics (ignoring improvements)."""
    order = ("significant_drift", "moderate_drift", "stable", "insufficient_data")
    worst = "stable"
    for name, m in metric_results.items():
        v = m.get("verdict", "stable")
        if order.index(v) < order.index(worst):
            worst = v
    return worst


def monitor(report: dict, recent_folds: int = 3) -> dict:
    """Compute drift signals from a wf_select.py report JSON.

    Parameters
    ----------
    report       : parsed wf_select.py output dict
    recent_folds : number of trailing selected folds treated as "recent"

    Returns a dict suitable for JSON serialisation.
    """
    folds = report.get("folds", [])
    reference, recent = split_folds(folds, recent_folds)

    result: dict = {
        "schema_version":    1,
        "recent_folds":      len(recent),
        "reference_folds":   len(reference),
    }

    if len(reference) < _MIN_REF_FOLDS:
        result["verdict"] = "insufficient_data"
        result["metrics"] = {}
        return result

    metrics_out: dict = {}
    for metric in _TRACKED_METRICS:
        ref_vals = [f["test_metrics"].get(metric, float("nan"))
                    for f in reference
                    if math.isfinite(f["test_metrics"].get(metric, float("nan")))]
        rec_vals = [f["test_metrics"].get(metric, float("nan"))
                    for f in recent
                    if math.isfinite(f["test_metrics"].get(metric, float("nan")))]

        z = metric_zscore(ref_vals, rec_vals)
        metrics_out[metric] = {
            "ref_mean":   statistics.mean(ref_vals) if ref_vals else float("nan"),
            "rec_mean":   statistics.mean(rec_vals) if rec_vals else float("nan"),
            "ref_std":    (statistics.stdev(ref_vals)
                           if len(ref_vals) >= 2 else float("nan")),
            "z_score":    z,
            "verdict":    _metric_verdict(z),
        }

    result["metrics"] = metrics_out
    result["verdict"] = _overall_verdict(metrics_out)
    return result


def write_report(
    wf_report: dict,
    output_path: str,
    *,
    recent_folds: int = 3,
) -> None:
    """Run monitor() and write result to output_path (create-only, atomic)."""
    if os.path.exists(output_path):
        sys.exit(f"output path exists; create-only policy refuses overwrite: {output_path}")

    result = monitor(wf_report, recent_folds=recent_folds)
    result["generated_at"]  = datetime.now().astimezone().isoformat()
    result["source_report"] = wf_report.get("_source_path", "")
    result["recent_folds_requested"] = recent_folds

    tmp = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
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
        description="Detect performance drift in a wf_select.py walk-forward report."
    )
    ap.add_argument("--report",  required=True,
                    help="wf_select.py report JSON to analyse")
    ap.add_argument("--output",  required=True,
                    help="Output path for the drift report (create-only)")
    ap.add_argument("--recent-folds", type=int, default=3,
                    help="Number of trailing selected folds treated as 'recent' (default: 3)")
    args = ap.parse_args()

    wf_report = _load(args.report)
    write_report(wf_report, args.output, recent_folds=args.recent_folds)

    with open(args.output) as f:
        out = json.load(f)
    verdict = out["verdict"]
    print(f"verdict: {verdict}")
    for name, m in out.get("metrics", {}).items():
        z = m.get("z_score", float("nan"))
        z_str = f"{z:+.2f}" if math.isfinite(z) else "n/a"
        print(f"  {name:20s}  z={z_str:>7s}  {m['verdict']}")
    print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
