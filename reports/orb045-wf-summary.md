# ORB-045 Walk-Forward Report Summary

**Report file:** `reports/orb045-wf.json` (create-only; not committed to git)
**Generated:** 2026-08-03
**Grid hash:** `68f21a729b407c63`
**Decision:** **No-Go**

## Walk-forward configuration

| Parameter | Value |
| --- | --- |
| Folds | 16 (252-day train / 63-day test / 63-day step) |
| Folds with candidate selection | 5 (folds 4, 5, 6, 7, 8) |
| Inner validation | Last 20 % of train window |

## Outer test metrics

| Cost scenario | Trades | Sharpe | Profit Factor | Mean net bps | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero (0 bps) | 566 | +0.70 | 1.20 | +4.73 | 0.96 % |
| Baseline (2.5 bps/side) | 566 | −0.03 | 0.99 | −0.27 | 2.24 % |
| Double (5.0 bps/side) | 566 | −0.79 | 0.82 | −5.28 | 3.77 % |

## Gate failures (baseline cost)

| Gate | Threshold | Actual | Result |
| --- | --- | --- | --- |
| Outer Sharpe | > 0.5 | −0.03 | FAIL |
| Outer PF | ≥ 1.10 | 0.99 | FAIL |
| Outer mean net bps | > 0 | −0.27 | FAIL |

## Interpretation

The gross ORB signal has measurable edge at zero cost (Sharpe +0.70, PF 1.20). At
realistic IEX execution costs (2.5 bps/side) the edge is fully erased. This is the
expected baseline result for a pure price-breakout strategy without any selectivity
overlay.

Five of sixteen walk-forward folds produced a passing candidate (folds 4–8), with test
windows covering 2023-03-06 to 2024-06-26. The remaining 11 folds selected no candidate:
folds 1, 2, 3, 10, and 12 rejected all 192 candidates because no candidate achieved
train Sharpe ≥ 0 in their respective training windows.

The 566 outer test trades are concentrated in the ~15-month window from 2023-03 to 2024-06.
`wf_select.py` only accumulates outer trades for folds where a candidate is selected; there
is no fallback or bleed-through mechanism. The seven folds with test dates from 2024-06-27
onward produced no selected candidate, with several showing 192/192 Sharpe rejections.
This indicates the strategy's gross edge was regime-specific and had dissipated in the
subsequent period.

**Research implications for M3+:**

1. Factor overlays (Alpha-101, RVOL regime filters) may concentrate trades on
   higher-quality signals and raise effective gross bps sufficiently to survive costs.
2. Tighter execution modelling (limit orders, venue selection) could reduce the
   effective cost per trade.
3. The frozen grid and gates must not be modified in response to this result.
