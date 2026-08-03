# Roadmap

> **Rule**: Roadmap is a commitment. Scope changes require updating this file
> (and the associated GitHub Issue) before any code changes.
> Milestones must be completed in order: infrastructure first, ML last.

---

## M0 — Repository Skeleton

**Goal**: Establish the project structure that all future work builds on.

Migrate the reference implementation as-is into a clean `src/` layout.
Set up CI (GitHub Actions) that runs the test suite on every push/PR.
Write `AGENT.md` (ML discipline rules, frozen invariants, dependency policy).
Configure `.gitignore` to exclude data caches, run outputs, credentials, and
compiled artefacts. Verify all six existing fixture tests pass in CI.

**Outputs**: passing CI badge, `AGENT.md`, `src/` layout, test-suite green.

**Definition of Done**: `python -m pytest` (or `unittest discover`) green in CI;
`AGENT.md` committed; no credentials or large binary files tracked by git.

---

## M1 — Data Layer Hardening

**Goal**: Make the data pipeline robust enough for multi-year, multi-symbol
production research runs without manual intervention.

Add resume / partial-download so an interrupted download restarts at the right
offset rather than re-fetching from the beginning. Add configurable retry with
exponential back-off for transient API failures. Implement a local NYSE trading
calendar (holiday list + half-day list) so the pump correctly handles early
closes, avoids requesting data on market holidays, and marks sessions properly.
Handle DST transitions deterministically. On each run, emit a data-quality
summary: per-symbol gap rate, stale-bar count, sessions with fewer than the
expected bar count, and any days where the 1-min cache is present but contains
zero RTH bars.

**Outputs**: hardened `local_pump.py`, trading calendar module,
data-quality report emitted to stdout / log file.

**Definition of Done**: a simulated interrupted download resumes correctly
(unit test); DST edge cases pass (unit tests); full 31-symbol download produces
a quality report with no unexpected gaps on known trading days.
See ISSUE-001 for detailed acceptance criteria.

---

## M2 — ORB Baseline Evidence

**Goal**: Produce the first real-data Candidate / No-Go decision for the 192-
candidate grid over the full 2021-01-04 to 2026-06-30 date range.

Run `local_pump.py` over all 31 symbols (optionally sharded). Run `wf_select.py`
and publish the report under `reports/orb045-wf.json`. Record the exact command
invocations, data provenance, and environment in a companion `reports/orb045-run.md`.

**Outputs**: `reports/orb045-wf.json` (create-only), run provenance document.

**Definition of Done**: report file exists, `decision` field is either
`"Candidate"` or `"No-Go"` with all gate fields populated, `grid_spec_hash`
matches the frozen grid, provenance document committed alongside the report.
No subsequent code changes may re-run against the same outer test.

---

## M3 — Alpha-101 Factor Layer

**Goal**: Build a principled daily-frequency factor library and IC evaluation
harness to identify which cross-sectional signals carry genuine predictive power
over ORB entry returns — without contaminating the ORB walk-forward results.

Implement the Alpha-101 operator library (numpy-only; ADR-003 required) covering
at minimum 40 factors. Build a daily factor panel from the existing 1-min cache
(OHLCV aggregation to daily bars). Implement a rank-IC evaluation pipeline:
for each factor, compute rank IC against multiple forward-return horizons (1d,
5d, 20d); apply Benjamini–Hochberg FDR correction at q ≤ 0.05; publish a factor
report listing which factors survive. The surviving factor list is frozen after
this milestone and becomes the feature pool for M4's meta-labeling model.

**Outputs**: `src/features/alpha101.py`, factor panel builder, IC evaluator,
FDR module, `reports/factor-ic-<date>.json` (create-only), frozen factor list.

**Definition of Done**: ≥ 40 factors implemented with unit tests; IC pipeline
produces reproducible results (fixed seed); factor report published with FDR
results; surviving-factor list committed and frozen.
See ISSUE-002 for detailed acceptance criteria.

---

## M4 — ML Optimization Layer

Status: DoD achieved — decision No-Go

Reports: `reports/orb045-ml.json`, `reports/orb045-ml-bayes.json`,
`reports/orb045-comparison.json`, `reports/orb045-ml-summary.md`

**Goal**: Determine whether a train-window-only ML overlay meaningfully improves
the ORB strategy's risk-adjusted performance on the outer test, and whether
Bayesian hyperparameter search outperforms the fixed grid — using the strict
ML discipline rules in `AGENT.md`.

**(a) Meta-labeling filter**: Train a gradient-boosted classifier inside each
WF fold's training window to predict P(trade wins). Features are drawn from ORB
trade attributes plus the M3 surviving factors aligned to entry date. Use purged
time-series CV + embargo within the training window. At test time, only trades
with P(win) above a pre-declared threshold enter the filtered outer-test set.

**(b) Bayesian hyperparameter search**: Replace the fixed 192-candidate grid scan
in the WF inner loop with a TPE/GP optimizer (optuna; ADR-005 required). The
search space and trial budget are pre-declared and hashed into the report before
any outer-test data is seen.

Both sub-tasks produce a paired comparison report: filtered vs. unfiltered (or
Bayesian vs. grid) on the identical outer-test window and cost scenarios.
Random seeds are fixed and recorded.

**Outputs**: `src/ml/meta_label.py`, `src/ml/bayes_search.py`, paired report
JSONs (create-only).

**Definition of Done**: paired comparison published; ML components have unit
tests; search space + seed committed before the outer-test run; `AGENT.md`
disciplines provably followed (no outer-test data seen during training).
ADR-004 and ADR-005 committed before code merged.

**Evidence summary**: All three runs (baseline / meta-label / meta-label+Bayes)
failed the final gates at baseline cost. Meta-label CV log-loss values (0.681–0.700)
were at or above the no-information baseline ln(2) = 0.693, indicating no
predictive power on inner validation. Paired comparison (n = 5 folds) found no
statistically significant improvement; the test is underpowered. See
`reports/orb045-ml-summary.md` for full details.

---

## M5 — Continuous Research Loop

Status: incomplete — drift report produced but DoD not met (see below)

Report: `reports/orb045-drift.json`

**Goal**: Establish a repeatable monthly update cycle so new market data can
be incorporated into a rolling walk-forward assessment without compromising
the disciplines from prior milestones.

Implement an incremental download step (append new months to the cache, not
re-download). Define the rolling-window schedule: when new month N arrives,
extend the WF by one step, produce a new outer-test fold, compare it to the
prior fold's baseline, and publish a drift-monitoring report. Drift metrics
include: candidate selection stability (same `candidate_id` chosen?), outer
Sharpe trend, and cost-scenario sensitivity. The loop is research-only —
no order routing or live connection of any kind.

**Outputs**: `scripts/rolling_update.py`, drift-monitoring report schema,
cron / scheduling documentation.

**Definition of Done**: end-to-end rolling update runs from a new month's
cache append through to a published drift report, with no manual steps;
a test using synthetic new-month data verifies the pipeline; documentation
explains the cadence and operator checklist.

**Current state**: `reports/orb045-drift.json` was produced by `drift_monitor.py`
against the baseline WF report. However, `reference_folds = 0` and
`verdict = "insufficient_data"`: only 5 folds contain `test_metrics`, all of which
fall within the "recent" window (last 3 selected folds). There are fewer than 3
folds remaining for the reference window (drift_monitor.py requires at least
`_MIN_REF_FOLDS = 3` reference folds). No drift metrics can be computed.

The pipeline code (incremental download, rolling WF step, drift report) is
implemented and tested, but the DoD requires a published drift report with
usable metrics. That requires more selected folds — either a longer live-forward
period or a regime in which more folds pass selection gates. DoD not yet met.
