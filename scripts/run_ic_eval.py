#!/usr/bin/env python3
"""Compute Rank IC for all Alpha-101 factors and publish a create-only report.

Usage:
    python3 scripts/run_ic_eval.py \
        --cache-dir data \
        --start 2021-01-04 \
        --end 2026-06-30 \
        --report-output reports/factor-ic-20260803.json \
        --factor-list src/orb/features/factor_list.json

The script:
  1. Builds the daily OHLCV panel from 1-min cache.
  2. Runs Alpha101 on the full panel.
  3. Computes Spearman rank IC for each factor at horizons 1, 5, 20 days.
  4. Summarises IC series (mean, std, IR, t-stat, p-value).
  5. Applies Benjamini-Hochberg FDR correction (q=0.05) across all
     factor × horizon hypotheses.
  6. Writes the factor IC report JSON (create-only; refuses to overwrite).
  7. Writes factor_list.json (also create-only) with the surviving factors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure src/ is on sys.path for editable-install-free runs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from orb.features.alpha101 import Alpha101
from orb.features.ic_eval import (
    fdr_correct,
    fwd_return,
    ic_summary,
    ic_summary_hac,
    ic_summary_nonoverlap,
    rank_ic_series,
)
from orb.features.panel_builder import CLOSE, build_panel

_SCHEMA_VERSION = 1
_HORIZONS = [1, 5, 20]
_FDR_Q = 0.05

# Factor development period boundary (frozen — must match factor_list.json dev_cutoff).
# IC evaluation is prohibited on data beyond this date to prevent look-ahead into
# any future hold-out or live period.
_DEV_CUTOFF = date(2026, 6, 30)


def _atomic_create_only(path: Path, data: dict) -> None:
    """Write JSON atomically; exit 1 if the target already exists."""
    if path.exists():
        sys.exit(f"report path exists; create-only policy refuses overwrite: {path}")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp.json")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.link(tmp_path, path)   # atomic create-only publish
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir",       default="data",
                    help="directory of *_1min.csv.gz files")
    ap.add_argument("--start",           default="2021-01-04")
    ap.add_argument("--end",             default="2026-06-30")
    ap.add_argument("--report-output",   required=True,
                    help="path for factor IC report JSON (create-only)")
    ap.add_argument("--factor-list",     default="src/orb/features/factor_list.json",
                    help="path for surviving factor list JSON (create-only)")
    ap.add_argument("--hac", action="store_true", default=False,
                    help="Use Newey-West HAC standard errors for FDR correction. "
                         "Decision is based on HAC p-values; naive and non-overlapping "
                         "t-stats are also reported. Does not write factor_list.json.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    report_path = Path(args.report_output)
    factor_list_path = Path(args.factor_list)
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    if end > _DEV_CUTOFF:
        sys.exit(
            f"end={end} exceeds the factor development cutoff {_DEV_CUTOFF}. "
            "Re-evaluation with data beyond the declared boundary is prohibited."
        )

    print(f"[ic_eval] building daily panel from {cache_dir} ({start} → {end})")
    panel = build_panel(cache_dir, start=start, end=end)
    T, N = len(panel.dates), len(panel.symbols)
    print(f"[ic_eval] panel: T={T} days, N={N} symbols")

    kwargs = panel.to_alpha101_kwargs()
    close = panel.panel[:, :, CLOSE].T  # (T, N)

    alpha = Alpha101(**kwargs)
    all_names = Alpha101.all_alpha_names()
    print(f"[ic_eval] evaluating {len(all_names)} factors × {len(_HORIZONS)} horizons")

    # Precompute forward returns for each horizon
    fwd = {h: fwd_return(close, h) for h in _HORIZONS}

    # Collect per-(factor, horizon) statistics including HAC when --hac is set
    hypothesis_keys: list[tuple[str, int]] = []
    hypothesis_p:   list[float] = []
    raw_summaries:  dict[str, dict] = {}

    for name in all_names:
        factor = getattr(alpha, name)()
        raw_summaries[name] = {}
        for h in _HORIZONS:
            ic = rank_ic_series(factor, fwd[h])
            s_ols = ic_summary(ic)
            entry: dict = {**s_ols}
            if args.hac:
                lag = h - 1
                s_hac = ic_summary_hac(ic, lag=lag)
                s_no  = ic_summary_nonoverlap(ic, h=h)
                entry.update({
                    "hac_lag":          lag,
                    "hac_t_stat":       s_hac["hac_t_stat"],
                    "hac_p_value":      s_hac["hac_p_value"],
                    "nonoverlap_n":     s_no["n_obs"],
                    "nonoverlap_t":     s_no["t_stat"],
                    "nonoverlap_p":     s_no["p_value"],
                })
            raw_summaries[name][h] = entry
            # FDR p-value: HAC when --hac, else naive OLS
            if args.hac:
                p = entry["hac_p_value"]
            else:
                p = entry["p_value"]
            hypothesis_keys.append((name, h))
            hypothesis_p.append(p if (p == p) else 1.0)

    # BH correction across all (factor, horizon) hypotheses
    p_arr = np.array(hypothesis_p)
    rejected, bh_p = fdr_correct(p_arr, q=_FDR_Q)

    def _r(v: float, places: int = 6) -> float | None:
        return round(float(v), places) if v == v else None

    # Build per-factor report entries
    factors_report: dict = {}
    for idx, (name, h) in enumerate(hypothesis_keys):
        if name not in factors_report:
            factors_report[name] = {}
        e = raw_summaries[name][h]
        entry: dict = {
            "mean_ic":    _r(e["mean_ic"]),
            "std_ic":     _r(e["std_ic"]),
            "ic_ir":      _r(e["ic_ir"]),
            "naive_t":    _r(e["t_stat"]),
            "naive_p":    _r(e["p_value"], 8),
            "n_obs":      e["n_obs"],
        }
        if args.hac:
            entry.update({
                "hac_lag":        e["hac_lag"],
                "hac_t":          _r(e["hac_t_stat"]),
                "hac_p":          _r(e["hac_p_value"], 8),
                "nonoverlap_n":   e["nonoverlap_n"],
                "nonoverlap_t":   _r(e["nonoverlap_t"]),
                "nonoverlap_p":   _r(e["nonoverlap_p"], 8),
                "bh_p":           round(float(bh_p[idx]), 8),
                "passes_fdr":     bool(rejected[idx]),
                "fdr_basis":      "hac",
            })
        else:
            entry.update({
                "bh_p":       round(float(bh_p[idx]), 8),
                "passes_fdr": bool(rejected[idx]),
                "fdr_basis":  "naive",
            })
        factors_report[name][f"{h}d"] = entry

    # Survivors: pass FDR at any horizon
    surviving = sorted(
        name for name in all_names
        if any(factors_report[name][f"{h}d"]["passes_fdr"] for h in _HORIZONS)
    )
    mode = "HAC" if args.hac else "naive"
    print(f"[ic_eval] surviving factors ({mode} FDR q={_FDR_Q}): {len(surviving)} / {len(all_names)}")
    for name in surviving:
        bests = {h: factors_report[name][f"{h}d"] for h in _HORIZONS}
        print(f"  {name}: " + "  ".join(
            f"{h}d IC={d['mean_ic']:+.4f} IR={d['ic_ir']:+.4f}"
            for h, d in bests.items() if d["passes_fdr"]
        ))

    report = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "date_range": {"start": str(start), "end": str(end)},
        "n_symbols": N,
        "n_days": T,
        "horizons_days": _HORIZONS,
        "fdr_q": _FDR_Q,
        "fdr_basis": "hac" if args.hac else "naive",
        "n_hypotheses": len(hypothesis_keys),
        "n_surviving": len(surviving),
        "surviving_factors": surviving,
        "factors": factors_report,
    }

    # Publish report (create-only)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create_only(report_path, report)
    print(f"[ic_eval] report written → {report_path}")

    if not args.hac:
        # Publish factor list (create-only) — only for the original naive run
        factor_list_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_create_only(
            factor_list_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dev_cutoff": str(_DEV_CUTOFF),
                "fdr_q": _FDR_Q,
                "horizons_days": _HORIZONS,
                "factors": surviving,
            },
        )
        print(f"[ic_eval] factor list written → {factor_list_path}")
    else:
        print("[ic_eval] --hac mode: factor_list.json not updated (frozen)")


if __name__ == "__main__":
    main()
