# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An Opening Range Breakout (ORB) research and backtesting tool. It scans a frozen grid of 192 candidate parameter sets across 31 symbols over multi-year 1-minute bar data, then evaluates them through a nested walk-forward framework to produce a binary Candidate / No-Go decision. No QuantConnect dependency — pure Python, offline.

## Commands

```bash
# Run tests (no network, no account required)
python3 -m unittest discover -s tests -v

# Run a single test
python3 -m unittest tests.test_orb_core.TestOrbCore.test_breakout_lifecycle_next_bar_entry_and_target

# Download data and simulate all 192 candidates (requires Alpaca keys)
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
python3 scripts/local_pump.py --start 2021-01-04 --end 2026-06-30 \
  --cache-dir data --out-dir runs/orb045

# Optional: shard candidates across runs to parallelise CPU work
python3 scripts/local_pump.py ... --out-dir runs/orb045 --shard 0 --shards 4
python3 scripts/local_pump.py ... --out-dir runs/orb045 --shard 1 --shards 4
# (then shards 2, 3)

# Nested walk-forward selection and report (aggregates all shards automatically)
python3 scripts/wf_select.py --trades-dir runs/orb045 \
  --report-output reports/orb045-wf.json
```

Python 3.12. The only external dependency is `alpaca-py` (only needed by `local_pump.py`). The core and `wf_select.py` are stdlib-only.

## Architecture

### Data flow

```
Alpaca IEX 1-min bars (cached as gzip CSVs in data/)
    │
    ▼  consolidate_5min()  [local_pump.py]
5-minute RTH Bar objects (naive NY-local datetimes, bar.end = bucket end)
    │
    ▼  SymbolEngine.on_bar()  [core/orb_core.py]
Trade objects (gross PnL + notionals, NO costs applied)
    │
    ▼  trades_NNNN.json + meta.json  [runs/<name>/shard<N>of<M>/]
    │
    ▼  wf_select.py  (loads shards, validates grid_spec_hash)
Nested walk-forward report JSON  →  Candidate | No-Go
```

### `core/orb_core.py` — the engine (zero external deps)

- **`Bar`** — immutable 5-minute bar; `end` is the bar's close time in exchange-local naive datetime.
- **`CandidateParams`** — frozen dataclass of 10 parameters; `candidate_id` is a SHA-256 hash of the params, ensuring stable identity across runs.
- **`default_candidate_grid()`** — produces the frozen 192-candidate `itertools.product` grid. **Do not modify after any results exist.**
- **`SymbolEngine`** — runs feature computation (ATR, RVOL, VWAP) and all candidate lifecycles for one symbol in a single bar-by-bar pass. Maintains session state in `_SymbolDay`; call `on_session_end()` at day boundaries.

Key invariants baked into the engine:
- All features (VWAP, RVOL, ATR) are **point-in-time** — computed from bars already seen.
- **Next-bar entry**: signal generated at close of bar `t` → entry at open of bar `t+1` (same session only; stale pending signals die silently if the session ends).
- **Stop-first**: when a bar's range spans both stop and target, the stop wins.
- **Long-only, no overnight**: `_finalize_session()` flattens all open positions at session close.
- **Costs are never applied in the core.** `Trade` carries `gross_pnl`, `entry_notional`, and `exit_notional` so cost scenarios are applied offline.

ATR uses Wilder smoothing (period 14); RVOL uses the median of the last 20 trading days' cumulative volume at the same intraday minute mark (minimum 10 days of history required).

### `scripts/local_pump.py` — data pump

Downloads Alpaca IEX 1-min bars to `data/<SYMBOL>_<start>_<end>_1min.csv.gz` (idempotent). Consolidates to 5-min RTH bars, feeds them to `SymbolEngine`, and writes per-shard output:

```
runs/<name>/shard<N>of<M>/
    meta.json          # grid_spec_hash, shard indices, universe, date range
    trades_0000.json   # up to 5000 trades per file
    trades_0001.json
    ...
```

Sharding splits the **candidate list** only (modulo assignment), not dates or symbols. The `grid_spec_hash` in every `meta.json` must be identical for the aggregator to accept the shards.

### `scripts/wf_select.py` — walk-forward evaluator

Loads all shards (or a single un-sharded directory), validates shard completeness and hash consistency, then runs nested walk-forward:

- **Folds**: 252-day train / 63-day test, stepping 63 days.
- **Inner split**: last 20% of train window is validation.
- **Fold selection gate** (training window, baseline cost 2.5 bps/side): min 30 trades, train Sharpe ≥ 0, train PF ≥ 1.0, validation net PnL > 0. Folds with no passing candidate select nothing.
- **Final gate** (aggregated outer test, baseline cost): ≥ 100 trades, Sharpe > 0.5, PF ≥ 1.10, mean net bps > 0.
- Cost scenarios evaluated: `zero` (0 bps), `baseline` (2.5 bps/side), `double` (5 bps/side).
- Report is written atomically (hard-link swap) and is **create-only** — it will refuse to overwrite an existing file.

## Critical constraints

- **Never modify `default_candidate_grid()`** after any simulation results have been produced. The `grid_spec_hash` links results to the exact grid; any change silently invalidates all prior runs.
- **Never modify frozen decision gates** in `wf_select.py` after looking at results. All thresholds (`SEL_*`, `GATE_*`) are pre-declared to prevent overfitting.
- The test suite in `tests/test_orb_core.py` uses deterministic synthetic bars and covers: next-bar entry, no-history guard, stop-first ambiguity, session-close flatten, signal cutoff, and candidate-id stability. Run it after any change to `orb_core.py`.
