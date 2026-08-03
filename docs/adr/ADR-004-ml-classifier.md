# ADR-004 — LightGBM as the meta-labeling classifier

**Status:** Accepted  
**Date:** 2026-08-03  
**Deciders:** project maintainers  
**Related:** [ADR-003](ADR-003-numpy-feature-layer.md), [ADR-005](ADR-005-optuna-bayes-search.md)

---

## Context

M4-2 (meta-labeling filter) requires a binary classifier that, given the six
FDR-surviving Alpha-101 factors as features and the primary ORB signal as a
side filter, outputs a probability of a winning trade.  Key requirements:

- **Tabular inputs**: features are a small, dense vector (≤ 6 floats) per bar.
- **Class imbalance**: ORB wins are a minority of signals; the classifier must
  handle imbalanced labels without manual resampling.
- **Calibration**: the output probability is used as a confidence gate; it must
  be monotone in expected win-rate, not necessarily perfectly calibrated.
- **Purged time-series CV**: the inner loop runs purged k-fold cross-validation
  with an embargo gap equal to the label horizon (20 days); the library must
  support `sample_weight` to zero-out purged observations.
- **Speed**: the feature matrix per fold is ≤ 1 500 rows × 6 columns;
  fit time is not a bottleneck, but must finish in seconds for 40+ Optuna trials.
- **No compilation step**: CI must install without a CUDA/GPU build chain.

## Decision

**Use LightGBM ≥ 4.0 as the classifier.**

```toml
[project.optional-dependencies]
ml = ["lightgbm>=4.0", "optuna>=3.6"]
```

LightGBM is installed in the `ml` optional group only; the core, features, and
walk-forward layers remain unaffected.

## Motivation

| Criterion | LightGBM | scikit-learn RF | scikit-learn LR |
|-----------|----------|-----------------|-----------------|
| Class imbalance | `scale_pos_weight` — native | `class_weight='balanced'` | `class_weight='balanced'` |
| Non-linear factor interactions | Gradient boosting — yes | Random forest — yes | Logistic — no |
| Sample weight for purging | Native (`weight` param) | Native | Native |
| Training speed (1 500 rows) | < 0.5 s per trial | < 0.5 s per trial | < 0.1 s per trial |
| Optuna integration | `lightgbm.cv` + `optuna-integration` | custom loop | custom loop |
| No GPU/CUDA build needed | CPU-only default | n/a | n/a |
| Probability output | `predict_proba` | `predict_proba` | `predict_proba` |

LightGBM's histogram-based GBDT consistently outperforms Random Forest on small,
dense tabular data in financial research (Buehler et al. 2019; de Prado 2020
§6.4). Logistic regression cannot capture the non-linear interactions between
momentum and volume factors that are visible in the IC analysis.

A scikit-learn `RandomForestClassifier` is retained as the comparison **baseline**
in the paired comparison report (M4-4), using the same purged CV. The baseline
never selects hyperparameters; its default parameters are fixed before any
outer-test data is seen (Rule 2 in AGENT.md).

## Constraints

- **No outer-test data during training**: enforced by the purged CV framework and
  the walk-forward fold structure (Rule 1 in AGENT.md).
- **Search space declared before first outer-test fold**: the LightGBM search
  space (ranges for `num_leaves`, `learning_rate`, `min_child_samples`,
  `scale_pos_weight`) is committed in a JSON spec file before M4's outer-test run.
- **Fixed random seed**: `seed=42` in all LightGBM and Optuna calls (Rule 6).
- **scikit-learn not a required dep**: the baseline RF is instantiated via the
  `sklearn` import that is only needed for the paired comparison script.

## Alternatives considered

| Alternative | Why rejected |
|-------------|-------------|
| scikit-learn RandomForest (primary) | Slightly worse on tabular data than GBDT for N < 2 000 rows per fold; harder to tune via Optuna without the `lightgbm.cv` shortcut |
| XGBoost | Heavier install (Cython, optional CUDA); marginally slower on CPU for small datasets |
| CatBoost | Large install; no advantage on numeric-only features |
| Neural network (PyTorch / sklearn MLP) | Overkill for 6 features; slower than GBDT; requires careful normalisation |
| Isotonic regression post-hoc calibration | Out of scope for M4; can be added as a post-processing step later |
