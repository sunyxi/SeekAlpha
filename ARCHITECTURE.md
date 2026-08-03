# Architecture

## System overview

An Opening Range Breakout (ORB) research and backtesting tool that scans a
frozen 192-candidate parameter grid across 31 symbols over multi-year 1-minute
bar data, evaluates candidates through a nested walk-forward framework, and
applies an ML meta-labeling filter to refine the final signal.

```
Alpaca IEX 1-min bars (gzip CSV cache in data/)
        │
        ▼  panel_builder.py  [src/orb/features/]
Daily OHLCV panel — shape (N, T, 5), NYSE-calendar aligned
        │
        ├──▶  alpha101.py  [src/orb/features/]
        │     42 Alpha-101 cross-sectional factors — shape (T, N)
        │         │
        │         ▼  ic_eval.py  [src/orb/features/]
        │     Rank IC → BH FDR → factor_list.json (6 survivors, frozen)
        │
        ├──▶  local_pump.py  [scripts/]
        │     5-min RTH bars → SymbolEngine → trade logs
        │         │
        │         ▼  wf_select.py  [scripts/]
        │     Nested walk-forward → WF report JSON (create-only)
        │
        └──▶  ml/  [src/orb/ml/]  ← M4
              Meta-labeling filter (LightGBM + Optuna)
              Paired comparison report (ML vs ORB baseline)
```

## Layer policy

| Layer | Location | Allowed dependencies |
|-------|----------|---------------------|
| Core engine | `src/orb/core/` | stdlib only (ADR-001) |
| Walk-forward evaluator | `scripts/wf_select.py` | stdlib only (ADR-001) |
| Data pump | `scripts/local_pump.py` | `alpaca-py` (ADR-002) |
| Feature layer | `src/orb/features/` | `numpy` (ADR-003) |
| ML layer | `src/orb/ml/` | `lightgbm`, `optuna` (ADR-004, ADR-005) |
| Test suite | `tests/` | `pytest`, `numpy` |

## Key frozen invariants

See `AGENT.md §2` for the complete list. The most critical items:

| Item | Value |
|------|-------|
| Candidate grid hash | `68f21a729b407c63` |
| Factor development cutoff | `2026-06-30` |
| Factor list SHA-256 | `41788b5b97629965a52799936f691e1bde391801d6df995712cd301df59d80ec` |
| Fold definition | train 252 d / test 63 d / step 63 d |
| Cost scenarios | zero (0 bps) / baseline (2.5 bps/side) / double (5 bps/side) |

## Data flow detail

### Raw data

`data/<SYMBOL>_<start>_<end>_1min.csv.gz` — UTC-timestamped 1-minute bars,
downloaded once via `scripts/local_pump.py` from Alpaca IEX (adjusted for
corporate actions). Columns: `ts, open, high, low, close, volume`.

### Feature panel

`panel_builder.build_panel()` reads the gzip CSVs, filters to Regular Trading
Hours (09:30–15:59 ET), aggregates to daily OHLCV, aligns to the NYSE trading
calendar (2021–2030 supported), and forward-fills missing days. Output is a
`DailyPanel` with shape `(N, T, 5)`.

`Alpha101(**panel.to_alpha101_kwargs())` produces 42 cross-sectional factors of
shape `(T, N)`. All operators are point-in-time (no look-ahead); verified by
`tests/test_alpha101.py::TestNoFutureData`.

### Walk-forward ORB evaluation

`local_pump.py` consolidates 1-min bars to 5-min RTH bars and feeds them to
`SymbolEngine` (one pass per symbol, streaming). Output: per-shard trade JSON
files with gross PnL and notionals (no costs applied in core).

`wf_select.py` loads all shards, validates `grid_spec_hash`, runs nested
walk-forward (252-day train / 63-day test / 63-day step), and publishes an
atomic create-only report JSON.

### ML meta-labeling (M4)

`src/orb/ml/` (to be implemented in M4-2 through M4-4):

- **Meta-labeling filter**: LightGBM binary classifier trained on the 6
  FDR-surviving factors. Uses purged time-series CV with a 20-day embargo
  (matching the 20-day IC horizon) inside each walk-forward training window.
- **Bayesian search**: Optuna TPE sampler, ≤ 50 trials, fixed seed=42. Search
  space committed as a JSON spec before the outer-test run.
- **Paired comparison**: ML-filtered results compared to the raw ORB baseline
  on the same outer-test periods and same three cost scenarios.

## Report discipline

All reports are **create-only** (refuse to overwrite) and published atomically
via `os.link()` hard-link swap. The only committed report files are `.md`
provenance documents; large `.json` reports are gitignored. See `.gitignore`.

## Architecture Decision Records

| ADR | Decision |
|-----|---------|
| [ADR-001](docs/adr/ADR-001-stdlib-only-core.md) | Core engine and WF evaluator: stdlib only |
| [ADR-002](docs/adr/ADR-002-iex-feed-rvol-consistency.md) | Alpaca IEX feed for data download |
| [ADR-003](docs/adr/ADR-003-numpy-feature-layer.md) | numpy in `src/orb/features/` only |
| [ADR-004](docs/adr/ADR-004-ml-classifier.md) | LightGBM for meta-labeling classifier |
| [ADR-005](docs/adr/ADR-005-optuna-bayes-search.md) | Optuna for Bayesian hyperparameter search |
