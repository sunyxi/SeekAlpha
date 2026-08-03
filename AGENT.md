# AGENT.md — ML Discipline Rules and Frozen Invariants

> This file is a **permanent, machine-readable contract** for all modelling and
> ML work in this repository. Every agent, contributor, and automated system
> must read and comply with these rules before making any change that touches
> simulation, feature computation, model training, hyperparameter selection,
> or reporting.
>
> These rules cannot be relaxed by citing "research needs" or "one more trial".
> A correct No-Go is a valid and complete research outcome.

---

## 1. ML Discipline Rules (Eight Rules — Verbatim)

**Rule 1 — Training-window confinement**
All learning (hyperparameter search, model training, feature selection, threshold
selection) happens only inside the current fold's training window. Everything is
frozen before the fold's outer-test period begins.

**Rule 2 — Pre-declared search space**
The search space, optimization budget (number of trials), model family, and
feature pool are declared and their SHA-256 hash is recorded in the report
before any outer-test data is seen. Post-hoc expansion of the search space is
prohibited.

**Rule 3 — Outer test and retention period are read-once**
The outer test and the final retention period never participate in any selection,
threshold calibration, or model fitting. The retention period is consumed exactly
once. It may not be re-visited after the first read.

**Rule 4 — Purged time-series cross-validation**
All time-series cross-validation inside a training window must apply purging
(remove samples whose labels overlap with the validation period) and embargo
(gap between train and val equal to the label horizon). Label-leakage via
overlapping returns is not permitted.

**Rule 5 — Paired comparison mandatory**
Every ML optimization result must be reported alongside the corresponding
no-ML baseline on the same outer-test period and the same cost scenarios
(zero / baseline / double). Reporting only the ML variant is prohibited.

**Rule 6 — Fixed random seed**
All stochastic components (model initialisation, optimizer sampling, data
shuffling) use a fixed random seed declared before the run. The seed is recorded
in the report to allow exact reproduction.

**Rule 7 — Pre-declared gate is final**
Results that do not meet the pre-declared decision gates (see §3) remain No-Go.
"Run another optimisation round on the same outer test" is prohibited.

**Rule 8 — Rolling window is forward-only**
In the continuous research loop (M5), new data is appended and the walk-forward
rolls forward. Previously consumed outer-test periods and retention periods are
never re-evaluated with new models or new parameters.

---

## 2. Frozen Items

The following items are frozen. They may not be modified after the first
simulation results have been produced. Any change requires explicit written
justification, a new frozen value, and invalidation of all prior results.

| Item | Frozen value / location |
|------|------------------------|
| Candidate grid | `default_candidate_grid()` in `src/orb/core/orb_core.py` |
| Grid spec hash | SHA-256 prefix of the serialised grid; verified in every `meta.json` |
| Fold definition | train 252 d / test 63 d / step 63 d (`wf_select.py: TRAIN_DAYS, TEST_DAYS, STEP_DAYS`) |
| Fold selection gates | `SEL_MIN_TRADES=30`, `SEL_MIN_SHARPE=0.0`, `SEL_MIN_PF=1.0`, val net PnL > 0 |
| Final decision gates | `GATE_MIN_TRADES=100`, `GATE_MIN_SHARPE=0.5`, `GATE_MIN_PF=1.10`, mean net bps > 0 |
| RVOL definition | median of last 20 trading days' cumulative volume at the same intraday minute mark; minimum 10 days of history required |
| Cost scenarios | zero (0 bps), baseline (2.5 bps/side), double (5.0 bps/side) |
| Research time boundary | Simulation date range declared at run time; outer test must not extend beyond it |
| ML search space (M4) | To be declared in a JSON spec file before M4 outer-test run; hash recorded in report |

---

## 3. Dependency Policy

Dependencies are added by milestone. Each new third-party dependency requires an
Architecture Decision Record (ADR) in `docs/adr/` before the first line of code
that imports it is merged.

| Layer / Milestone | Allowed dependencies |
|-------------------|---------------------|
| Core engine (`src/orb/core/`) | **stdlib only** — no third-party imports, ever |
| Walk-forward evaluator (`scripts/wf_select.py`) | **stdlib only** |
| Data pump (`scripts/local_pump.py`) | `alpaca-py` (declared in ADR-002) |
| Feature layer, M3+ (`src/orb/features/`) | `numpy` (requires ADR-003 before merge) |
| ML layer, M4+ (`src/orb/ml/`) | `scikit-learn` and/or `lightgbm` (requires ADR-004); `optuna` (requires ADR-005) |
| Test suite | `pytest` only |

Installing optional groups:

```bash
pip install -e ".[data]"   # adds alpaca-py
pip install -e ".[test]"   # adds pytest
pip install -e ".[ml]"     # adds scikit-learn/lightgbm/optuna (M4)
```

---

## 4. Reporting Discipline

- Reports are **create-only**: the evaluator writes via an atomic hard-link swap
  and refuses to overwrite an existing report file.
- Every report must record: `grid_spec_hash`, fold definition, all gate values,
  cost scenarios, and (for ML reports) the search-space hash, trial budget,
  random seed, and library versions.
- A `Candidate` decision is a licence to continue research, not a trading
  authorisation. This system has no order-routing capability and must never
  acquire one.
