# ORB-045 ML Overlay Summary (M4)

**Report files:** `reports/orb045-ml.json`, `reports/orb045-ml-bayes.json`,
`reports/orb045-comparison.json` (all create-only; not committed to git)
**Generated:** 2026-08-03
**Decision:** **No-Go** (unchanged from baseline)

## Three-run comparison (baseline cost: 2.5 bps/side)

| Run | Trades | Win rate | Mean net bps | Profit factor | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (no ML) | 566 | 0.449 | −0.27 | 0.991 | −0.03 |
| Meta-label filter | 246 | 0.480 | −1.04 | 0.967 | −0.10 |
| Meta-label + Bayesian search | 315 | 0.451 | −1.84 | 0.938 | −0.21 |

All three runs fail the final decision gates (Sharpe > 0.5, PF ≥ 1.10, mean net bps > 0).

## Per-fold comparison (folds 4–8; baseline cost)

| Fold | Test window | Baseline trades | Baseline bps | ML trades | ML bps | ML+Bayes trades | ML+Bayes bps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2023-03-06 → 2023-06-07 | 117 | +5.28 | 53 | +4.89 | 68 | −7.60 |
| 5 | 2023-06-08 → 2023-09-08 | 124 | +1.62 | 59 | +1.09 | 50 | +1.21 |
| 6 | 2023-09-11 → 2023-12-12 | 101 | +1.27 | 37 | −4.28 | 40 | +1.95 |
| 7 | 2023-12-13 → 2024-03-25 | 103 | −2.80 | 41 | −1.60 | 53 | +8.76 |
| 8 | 2024-03-26 → 2024-06-26 | 121 | −6.73 | 56 | −6.37 | 104 | −6.40 |

## Diagnostics

### (a) Model predictive power near zero

Meta-label CV log-loss values across folds:

| Fold | Meta-label cv_log_loss | Meta-label+Bayes cv_log_loss |
| ---: | ---: | ---: |
| 4 | 0.6997 | 0.6843 |
| 5 | 0.6813 | 0.6677 |
| 6 | 0.6825 | 0.6764 |
| 7 | 0.6814 | 0.6750 |
| 8 | 0.6944 | 0.6933 |

The binary no-information baseline is ln(2) = 0.6931. All meta-label CV log-losses fall in
the range 0.681–0.700, straddling this baseline. The model achieves essentially no
predictive power on the inner-validation set across all folds.

### (b) Win-rate increase does not translate to return improvement

Filtering raises win rate by ~3 percentage points (0.449 → 0.480) but mean net bps
deteriorates from −0.27 to −1.04. This is consistent with the known failure mode of binary
win/loss meta-labels applied to ORB-style trades: the label ignores return magnitude, so the
filter can correctly identify "more wins" while discarding the largest-gain trades, which have
disproportionate impact on mean bps.

### (c) Bayesian search does not improve on default parameters

The meta-label + Bayesian search run is uniformly worse than the fixed-parameter meta-label
run (mean net bps −1.84 vs −1.04; Sharpe −0.21 vs −0.10). Optuna's objective minimises inner-CV
log-loss, which is misaligned with the economic objective. When the model has near-zero
predictive power, tuning log-loss further does not improve trade selection.

## Paired statistical test

Source: `reports/orb045-comparison.json`

Comparison: meta-label (orb045-ml.json) vs baseline (orb045-wf.json)

**Input mismatch note:** The two reports were produced from different data-pump runs with
`input_meta.trade_count = 959233` (baseline) vs `959221` (ML). The 12-trade discrepancy
(0.0013 % of inputs) is attributed to Alpaca IEX backfill between runs; see
`docs/limitations.md`. The difference is too small to change the paired-test direction, but
the paired-comparison precondition (identical inputs) was technically violated.

| Metric | n folds | Mean diff (ML − base) | p-value | Significant? |
| --- | ---: | ---: | ---: | --- |
| Sharpe | 5 | +0.120 | 0.80 | No |
| Profit factor | 5 | −0.029 | 0.51 | No |
| Mean net bps | 5 | −0.981 | 0.41 | No |
| Win rate | 5 | +0.025 | 0.14 | No |
| Total net PnL | 5 | −21.6 | 0.89 | No |

**Statistical power note:** With n = 5 common folds and std_diff ≈ 1.05 for Sharpe,
the test has very low power to detect effects smaller than ~2 Sharpe units. The result
"no significant improvement" should be read as "insufficient power to detect an effect,"
not as "ML is proven ineffective." A longer live-forward period with more folds would be
needed for a conclusive test.

## Conclusion

**M4 = No-Go.** All three runs fail the pre-declared gates at baseline cost. The ML overlay
does not improve on the No-Go baseline. The paired comparison confirms no statistically
significant improvement, though the test is underpowered (n = 5 folds).
