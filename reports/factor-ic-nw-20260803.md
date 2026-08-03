# Factor IC Evaluation — HAC-Corrected (Newey-West) — 2026-08-03

Supersedes [factor-ic-20260803.md](factor-ic-20260803.md) for decision purposes.

## Run parameters

| Parameter | Value |
|-----------|-------|
| Panel date range | 2021-01-04 → 2026-06-30 |
| Symbols | 31 (full ORB universe) |
| Trading days | 1,378 |
| Factors evaluated | 42 (Alpha-101) |
| Horizons | 1d, 5d, 20d |
| FDR method | Benjamini-Hochberg q = 0.05 (HAC p-values) |
| Total hypotheses | 126 (42 × 3) |
| SE estimator | Newey-West Bartlett kernel, lag = H − 1 per horizon |
| Secondary check | Non-overlapping subsample (every-H observation) |

## Command

```bash
python3 scripts/run_ic_eval.py \
    --cache-dir data \
    --start 2021-01-04 \
    --end 2026-06-30 \
    --report-output reports/factor-ic-nw-20260803.json \
    --factor-list src/orb/features/factor_list.json \
    --hac
```

## Result: 1 / 42 factors survive FDR (HAC basis)

| Factor | Horizon | Naive t | HAC t (lag=H−1) | Non-overlap t | HAC BH-p | Passes FDR? |
|--------|---------|--------:|----------------:|--------------:|---------:|:-----------:|
| alpha042 | 20d | +8.37 | +3.59 | +1.22 | 0.0017 | **Yes** |

All other factors: HAC t-stats fall below the BH threshold at q = 0.05.

## Three-statistic comparison for previously surviving factors

The original naive analysis identified 6 survivors, all at 20d horizon. Under HAC:

| Factor | Horizon | Naive t | Naive BH-p | HAC t | HAC BH-p | Passes HAC FDR? |
|--------|---------|--------:|-----------:|------:|---------:|:---------------:|
| alpha003 | 20d | +5.71 | 0.0012 | +2.35 | 0.0936 | No |
| alpha024 | 20d | +3.29 | 0.0142 | +1.23 | 0.2177 | No |
| alpha032 | 5d  | +3.44 | 0.0119 | +2.04 | 0.0414 | No |
| alpha032 | 20d | +5.63 | 0.0012 | +2.07 | 0.0389 | No |
| alpha035 | 20d | +3.18 | 0.0142 | +1.72 | 0.0850 | No |
| alpha042 | 5d  | +4.70 | 0.0024 | +3.30 | 0.0099 | No |
| alpha042 | **20d** | **+8.37** | **< 0.001** | **+3.59** | **0.0017** | **Yes** |
| alpha050 | 20d | +4.17 | 0.0060 | +1.58 | 0.1133 | No |

Notes:
- 1d horizon: HAC t = naive t exactly (lag = 0, degenerate case).
- 5d and 20d: HAC t is substantially smaller, consistent with autocorrelated IC from overlapping returns.
- Non-overlapping t-stats (independent check) are weaker still, confirming the autocorrelation diagnosis.

## Why naive t-stats were inflated

At a 20-day forward-return horizon, each consecutive IC observation shares 19 days of forward
returns with its predecessor. This creates positive autocorrelation at lags 1–19 in the IC series.
The OLS (naive) t-statistic assumes i.i.d. observations; its effective sample size is n = 1 378 even
though the actual number of independent observations is closer to 1 378 / 20 ≈ 69.

The Newey-West estimator with lag = H − 1 = 19 accounts for this by inflating the standard error
using the sample autocovariance function with a Bartlett downweighting kernel. At lag=0 (1d horizon),
the HAC SE reduces exactly to the OLS SE, verifying the implementation.

## Implication for ADR-004 and M4

ADR-004 states "six FDR-surviving Alpha-101 factors as features." Under the HAC-corrected analysis,
only 1 factor (alpha042, 20d) survives FDR. The meta-labeling model in M4 was trained with 6
features; with the corrected evidence base there is effectively 1 signal-bearing feature.
This does not change the M4 No-Go decision (the model showed no predictive power regardless),
but the original ADR-004 premise overstated the feature quality. See the note added to
[ADR-004](../docs/adr/ADR-004-ml-classifier.md).

## factor_list.json status

`src/orb/features/factor_list.json` is frozen and unchanged. The 6 factors listed there were
selected under the naive analysis and remain committed as the formal M3/M4 feature pool.
This report provides corrected evidence about their statistical reliability but does not
retroactively change the M3 output or the M4 training feature set.
