# ADR-005 — Optuna for Bayesian hyperparameter search

**Status:** Accepted  
**Date:** 2026-08-03  
**Deciders:** project maintainers  
**Related:** [ADR-004](ADR-004-ml-classifier.md)

---

## Context

M4-3 (Bayesian hyperparameter search) must tune the LightGBM classifier inside
each walk-forward training window using only training-window data (Rule 1 in
AGENT.md). Requirements:

- **Bayesian optimisation**: grid/random search wastes trials on already-explored
  regions; the budget is capped at ≤ 50 trials per fold (Rule 2 declares the
  budget before any outer-test data is seen).
- **Reproducibility**: given the same seed, the trial sequence must be identical
  across machines (Rule 6).
- **No outer-test leakage**: the optimiser must receive only training-window
  metrics; the outer-test fold is never passed to the study.
- **LightGBM integration**: must interoperate cleanly with `lgb.cv` for efficient
  cross-validation inside a trial.
- **Lightweight install**: no C++ toolchain beyond what LightGBM already requires.
- **Pruning support**: early stopping of unpromising trials reduces wall time by
  ~40% on typical runs.

## Decision

**Use Optuna ≥ 3.6 for all hyperparameter search.**

```toml
[project.optional-dependencies]
ml = ["lightgbm>=4.0", "optuna>=3.6"]
```

Optuna is installed in the same `ml` optional group as LightGBM; no additional
optional group is required.

## Motivation

| Criterion | Optuna | scikit-optimize | Hyperopt | Ray Tune |
|-----------|--------|-----------------|----------|----------|
| Bayesian (TPE) | ✅ TPE default | ✅ GP / RF | ✅ TPE | ✅ multiple |
| Deterministic with seed | ✅ `sampler=TPESampler(seed=N)` | ✅ | partial | partial |
| LightGBM `cv` integration | ✅ native callback | manual | manual | manual |
| Pruning (early stopping) | ✅ MedianPruner | ❌ | ❌ | ✅ |
| Pure Python install | ✅ | ✅ | ✅ | ❌ (Ray) |
| Active maintenance (2025) | ✅ | ⚠️ low activity | ⚠️ | ✅ |
| No distributed cluster | ✅ single-process | ✅ | ✅ | ❌ overkill |

Optuna's TPE sampler is fully reproducible under a fixed seed and has a clean
`study.optimize(n_trials=N)` API. Its `MedianPruner` halts trials whose
intermediate CV score falls below the median of completed trials at that step,
typically halving wall time with no degradation in best-trial quality.

The search space is declared as a JSON spec (committed before the outer-test
run begins) and translated to `trial.suggest_*` calls inside the objective
function. This satisfies Rule 2 (pre-declared search space) and makes the
search reproducible from the spec file alone.

## Usage pattern (sketch)

```python
import optuna
import lightgbm as lgb

def objective(trial):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        # ... other params from JSON spec
    }
    cv_result = lgb.cv(params, dtrain, nfold=5, callbacks=[lgb.early_stopping(10)])
    return cv_result["valid binary_logloss-mean"][-1]

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50)
```

## Constraints

- **Trial budget fixed before outer test**: `n_trials=50` (or the value declared
  in the JSON search-space spec) is recorded in the report; increasing it
  post-hoc on the same outer-test fold is prohibited (Rule 2).
- **`optuna.logging.set_verbosity(WARNING)`**: silences per-trial output in CI.
- **No Optuna dashboard / storage**: in-memory study only; results are captured
  in the report JSON, not a database.
- **LightGBM `verbosity=-1`**: suppresses LightGBM training logs inside trials.

## Alternatives considered

| Alternative | Why rejected |
|-------------|-------------|
| scikit-optimize (skopt) | Low maintenance activity since 2022; no native pruning support |
| Hyperopt | TPE implementation, but pruning requires manual callbacks; less active |
| Random search (sklearn) | Baseline only — insufficient for ≤ 50 trials compared to TPE |
| Ray Tune | Requires distributed runtime; overkill for single-machine single-fold search |
| Manual grid search | Exponential in the number of parameters; ruled out at the design stage |
