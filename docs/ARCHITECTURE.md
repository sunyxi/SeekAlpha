# Architecture

> **Rule**: Architecture is a commitment. Any deviation from this document requires
> updating this file first, with a record of the reason, before changing code.

---

## 1. Layered Module Map

```text
┌─────────────────────────────────────────────────────────────────────┐
│  CLI Layer                                                          │
│  orb <subcommand>  (scripts/local_pump.py · scripts/wf_select.py)  │
│  Future: unified `orb` entry point                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────────────────┐
│  Report Layer                                                       │
│  wf_select.py: create-only JSON report, Gate status, provenance    │
│  research_protocol.py: frozen windows, budgets, gates, retention log│
│  Future: IC report (factor panel → rank-IC table, FDR result)      │
└────────────────────────┬────────────────────────────────────────────┘
                         │ reads trade logs / IC results
┌────────────────────────▼────────────────────────────────────────────┐
│  Validation Layer                                                   │
│  Nested walk-forward evaluator (252/63/63, outer test aggregation)  │
│  Future: Bayesian hyperparam search (TPE/GP, train-window only)     │
│  Future: IC evaluator + FDR correction (Benjamini–Hochberg)         │
└────────────────────────┬────────────────────────────────────────────┘
                         │ consumes
┌────────────────────────▼────────────────────────────────────────────┐
│  Strategy Layer                                                     │
│  ORB signal core: SymbolEngine, CandidateParams, Trade              │
│  *** ZERO third-party dependencies — stdlib only ***                │
│  Future: meta-labeling filter (train-window retrained each fold)    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ consumes
┌────────────────────────▼────────────────────────────────────────────┐
│  Feature Layer                                                      │
│  ORB features: point-in-time ATR (Wilder-14), RVOL (median-20d),   │
│                VWAP, opening-range windows                          │
│  Optional: Qlib-compatible panel adapter (isolated research only)   │
│  Future: Alpha-101 operator library (numpy only)                    │
│  Future: daily factor panel (cross-sectional rank, IC evaluation)   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ reads
┌────────────────────────▼────────────────────────────────────────────┐
│  Data Layer                                                         │
│  Alpaca IEX pump: download → gzip CSV cache (idempotent)           │
│  1-min → 5-min RTH consolidation (NY-aligned, deterministic)        │
│  Future: resume / retry / partial-download for long date ranges     │
│  Future: trading calendar (NYSE half-days, DST, holiday list)       │
│  Future: data quality statistics (gap rate, stale-bar detection)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Dependency directions

- Each layer depends **only** on layers below it. No upward references.
- The Strategy Layer (core) and Validation Layer must remain pure stdlib with
  zero third-party imports. This is enforced by the test suite and CI.
- The Feature Layer may use `numpy` from M3 onward (ADR required).
- The Data Layer may use `alpaca-py`. All third-party imports are isolated here.
- ML dependencies (`scikit-learn`, `lightgbm`, `optuna`) are permitted only in
  the ML sub-module of the Validation Layer, guarded by a separate ADR (M4).
- Qlib dependencies are permitted only under `src/orb/qlib_adapter/`. The
  adapter consumes the existing daily panel and cannot be referenced by the
  core engine or `wf_select.py`.

---

## 2. Data Flow

### ORB simulation pipeline (intraday)

```text
Alpaca IEX API
   │  (alpaca-py, free paper keys)
   ▼
1-minute RTH bars → gzip CSV cache
   data/<SYMBOL>_<start>_<end>_1min.csv.gz
   │  (cached; re-download skipped if file exists)
   ▼
consolidate_5min()                       [local_pump.py]
   │  aligned to 9:30 ET, naive NY datetime, bar.end = bucket close
   ▼
Bar objects  →  SymbolEngine.on_bar()    [core/orb_core.py]
   │  point-in-time: ATR, RVOL, VWAP, OR windows
   │  per-candidate lifecycle: pending → position → exit
   │  no costs applied
   ▼
Trade objects  {gross_pnl, entry_notional, exit_notional, ...}
   ▼
trades_NNNN.json + meta.json             [runs/<name>/shard<N>of<M>/]
   │  (up to 5 000 trades/file; atomic write via .tmp)
   ▼
wf_select.py                             [validation layer]
   │  validates grid_spec_hash, shard completeness, no duplicates
   │  nested WF: protocol-controlled train/validation/test windows
   │  fold gate → best candidate per fold → outer test aggregation
   │  cost scenarios: zero / baseline (2.5 bps) / double (5 bps)
   ▼
report JSON  →  Candidate | No-Go        [reports/<name>-wf.json]
   (atomic create-only: link + unlink; refuses to overwrite)
```

### Factor research pipeline (daily) — M3 and later

```text
Alpaca IEX daily bars (or derived from 1-min cache)
   │
   ▼
Daily factor panel  →  Alpha-101 operator library   [feature layer]
   │  cross-sectional rank normalization
   │  forward returns (1d, 5d, 20d)
   ▼
Rank IC evaluation  →  IC time series per factor    [validation layer]
   │  FDR correction (Benjamini–Hochberg, q ≤ 0.05)
   │  only factors surviving FDR enter the feature list
   ▼
Factor feature list  →  (frozen)  →  meta-labeling feature pool [M4]
```

### ML training cycle (train-window only) — M4 and later

```text
Each WF fold's training window
   ├── ORB features from trade records
   └── surviving daily factors aligned to entry date
              │
              ▼
   Meta-labeling: gradient-boosted model
      train on inner-train subset, evaluate on inner-val
      purged time-series CV + embargo (no leakage)
              │
              ▼
   Filter: predicted P(win) threshold
   Applied to outer test → filtered trade set
              │
              ▼
   WF report with pre/post-filter comparison
   (same outer test window, same cost scenarios)
```

---

## 3. Critical Invariants

| # | Invariant | Where enforced |
| --- | --------- | -------------- |
| 1 | **No future data**: ATR, RVOL, VWAP are point-in-time (computed from bars already seen before the signal bar closes) | `orb_core.py`, test suite |
| 2 | **Next-bar entry**: signal at bar-t close → entry at bar-(t+1) open, same session only; stale signals die silently | `_step_candidate`, tests |
| 3 | **Stop-first**: when a bar's range spans both stop and target, stop wins | `_step_candidate`, tests |
| 4 | **Long-only, no overnight**: `_finalize_session()` flattens all open positions at session end | `_finalize_session`, tests |
| 5 | **Costs never in core**: `Trade` carries gross PnL and notionals; cost scenarios applied offline | `wf_select.py` only |
| 6 | **Frozen grid**: `default_candidate_grid()` and its SHA-256 `grid_spec_hash` must never change after results exist | `meta.json`, CI hash check |
| 7 | **Frozen gates**: `SEL_*` and `GATE_*` thresholds in `wf_select.py` declared before results are seen | comments + code review |
| 8 | **Create-only reports**: reports refuse to overwrite; atomic publish via hard-link swap | `wf_select.py:os.link` |
| 9 | **Retention period once-only**: outer test data and final hold-out period are consumed exactly once | process discipline |
| 10 | **Zero credentials on disk**: Alpaca keys read from env vars only, never written to any file | `local_pump.py` |
| 11 | **Determinism**: same input → same output; no random state outside explicitly seeded ML (M4) | stdlib, seeded RNG in M4 |
| 12 | **Research protocol**: development, outer-test, and retention windows are disjoint; budgets, gates, costs, purge, embargo, and seeds are machine-readable and hashed | `research_protocol.json`, `research_protocol.py` |
| 13 | **Retention read-once**: a retention experiment ID is atomically reserved before its loader runs; failed loads remain consumed | `RetentionLedger` |

---

## 4. ML Component Positioning

ML is confined to the **Validation Layer**, inside each fold's training window.
It is never permitted to cross into the Strategy or Feature layers in a way that
could observe outer test results.

```text
Validation Layer
 ├── wf_select.py  (walk-forward evaluator, pure stdlib)
 └── ml/  [M4]
      ├── meta_label.py   (gradient-boosted P(win) model)
      │     deps: scikit-learn or lightgbm (ADR required)
      └── bayes_search.py (TPE/GP optimizer for WF inner loop)
            deps: optuna (ADR required)
```

The ML sub-module:

- Is retrained fresh for every WF fold on that fold's training window.
- Uses purged time-series CV + embargo within the training window.
- Has its search space, trial budget, model family, and feature pool
  declared and hashed into the report before any outer-test is seen.
- Always produces a **paired comparison**: filtered vs. unfiltered results
  on the same outer-test period and cost scenarios.
- Uses a fixed random seed, recorded in the report for reproducibility.

---

## 5. Architecture Decision Records (ADR index)

ADRs live in `docs/adr/`. Each third-party dependency introduction requires one.

| ID | Status | Title |
| -- | ------ | ----- |
| [ADR-001](adr/ADR-001-stdlib-only-core.md) | Accepted | Use stdlib-only core to guarantee portability and determinism |
| [ADR-002](adr/ADR-002-iex-feed-rvol-consistency.md) | Accepted | IEX feed for RVOL (self-consistent numerator/denominator) |
| ADR-003 | Pending (M3) | Introduce numpy in feature layer for Alpha-101 operators |
| ADR-004 | Pending (M4) | Introduce scikit-learn / lightgbm for meta-labeling |
| ADR-005 | Pending (M4) | Introduce optuna for Bayesian hyperparameter search |
| [ADR-006](adr/ADR-006-optional-qlib-adapter.md) | Accepted | Add an isolated optional Microsoft Qlib research adapter |
| [ADR-007](adr/ADR-007-research-protocol-retention-ledger.md) | Accepted | Freeze research protocol and enforce read-once retention access |
