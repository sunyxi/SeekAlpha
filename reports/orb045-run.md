# ORB-045 Simulation Run Provenance

## Run summary

| Field | Value |
| --- | --- |
| Run date | 2026-08-03 |
| Python version | 3.12.11 |
| Data source | Alpaca IEX 1-min bars, `adjustment="all"` |
| Date range | 2021-01-04 → 2026-06-30 |
| Universe | 31 symbols (see below) |
| Candidates | 192 (`grid_spec_hash = 68f21a729b407c63`) |
| Total trades (input) | 959,233 |
| Trade date range | 2021-01-25 → 2026-06-29 |
| Output directory | `runs/orb045/shard0of1/` |
| WF report | `reports/orb045-wf.json` |
| Decision | **No-Go** |

## Commands executed

```bash
# Step 1 — Data download and simulation (single shard, all 31 symbols)
python3 scripts/local_pump.py \
    --start 2021-01-04 \
    --end   2026-06-30 \
    --cache-dir data \
    --out-dir   runs/orb045

# Step 2 — Nested walk-forward evaluation
python3 scripts/wf_select.py \
    --trades-dir      runs/orb045 \
    --report-output   reports/orb045-wf.json
```

## Universe (31 symbols)

```text
SPY  QQQ  IWM  DIA
AAPL MSFT NVDA AMZN META GOOGL
TSLA AMD  AVGO NFLX ORCL CRM
JPM  BAC  GS   V    MA
XOM  CVX  UNH  JNJ  LLY
WMT  COST HD   CAT  BA
```

## Walk-forward configuration

| Parameter | Value |
| --- | --- |
| Train window | 252 trading days |
| Test window | 63 trading days |
| Step | 63 trading days |
| Inner validation | Last 20 % of train window |
| Fold selection gate | min 30 trades, train Sharpe ≥ 0, train PF ≥ 1.0, val net PnL > 0 |
| Final gate | ≥ 100 outer trades, Sharpe > 0.5, PF ≥ 1.10, mean net bps > 0 |
| Baseline cost | 2.5 bps/side |

## Outer test results (16 folds, 5 with candidate selection — folds 4–8)

| Cost scenario | Trades | Sharpe | Profit Factor | Mean net bps |
| --- | ---: | ---: | ---: | ---: |
| Zero (0 bps) | 566 | 0.70 | 1.20 | +4.73 |
| Baseline (2.5 bps/side) | 566 | −0.03 | 0.99 | −0.27 |
| Double (5 bps/side) | 566 | −0.79 | 0.82 | −5.28 |

**Decision reasons:**

- Outer Sharpe −0.03 ≤ 0.5 (gate: > 0.5)
- Outer PF 0.99 < 1.1 (gate: ≥ 1.10)
- Outer mean net bps ≤ 0 (gate: > 0)

## Per-fold results (selected folds only; baseline 2.5 bps/side)

| Fold | Train window | Test window | Selected candidate | Outer trades | Mean net bps | Sharpe |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 4 | 2022-02-09 → 2023-03-03 | 2023-03-06 → 2023-06-07 | c692bf06525d6 | 117 | +5.28 | +1.44 |
| 5 | 2022-05-12 → 2023-06-07 | 2023-06-08 → 2023-09-08 | cc17e39651c0d | 124 | +1.62 | +0.35 |
| 6 | 2022-08-18 → 2023-09-08 | 2023-09-11 → 2023-12-12 | c6b0405184468 | 101 | +1.27 | +0.31 |
| 7 | 2022-11-23 → 2023-12-12 | 2023-12-13 → 2024-03-25 | c333a51ead3dd | 103 | −2.80 | −0.83 |
| 8 | 2023-03-06 → 2024-03-25 | 2024-03-26 → 2024-06-26 | c78a02f455318 | 121 | −6.73 | −3.13 |

Folds 0–3 and 9–15: no candidate passed the fold selection gate (selected = null, no outer trades).

## Regime observation

The 5 folds that produced a selection (folds 4–8) cover a continuous test window from
2023-03-06 to 2024-06-26 — approximately 15 months. The 566 outer trades are therefore
concentrated in this sub-period, not spread across the full 2021–2026 date range.

The subsequent 7 folds (9–15, test dates 2024-06-27 → 2026-05-12) produced no selected
candidate. Among these, folds 10 and 12 show `rejection_reasons: {"train_sharpe": 192}`,
meaning all 192 candidates failed the train Sharpe ≥ 0 gate within those training windows.
Folds 1, 2, 3 (2022 test windows) also show the same 192/192 Sharpe rejection pattern.

This pattern indicates that strategy profitability, even at gross-zero cost, was regime-specific
to the 2022-02 → 2024-06 training period. No additional research conclusions are drawn here.

## Key observations

- At zero cost the strategy shows positive edge (Sharpe 0.70, PF 1.20) — the gross signal exists.
- 2.5 bps/side execution costs fully erase the edge (PF drops from 1.20 to 0.99).
- The 566 outer trades come from 5 selected folds (4–8) covering 2023-03 to 2024-06.
  `wf_select.py` only accumulates outer trades for folds where a candidate is selected;
  there is no fallback or bleed-through path.
- **This is the correct baseline result**: strategy is not profitable at realistic costs with
  the current parameter grid. M3+ research should focus on cost reduction (tighter execution)
  and/or factor overlays that improve candidate selectivity.

## Reproducibility notes

- Cache files in `data/` are idempotent re-downloads from Alpaca IEX (free paper keys).
- `grid_spec_hash = 68f21a729b407c63` is the SHA-256 prefix of the frozen 192-candidate grid
  produced by `default_candidate_grid()` in `src/orb/core/orb_core.py`. **Do not modify the
  grid after this run.**
- The WF report JSON is create-only (atomic hard-link swap); re-running `wf_select.py` with
  the same `--report-output` path will fail to prevent accidental overwrite.
- Note: a second run of `local_pump.py` with the same parameters produced `trade_count = 959221`
  (12 fewer trades). This is believed to be caused by Alpaca IEX historical data backfill between
  runs; see `docs/limitations.md` for diagnosis. The WF report is bound to the first run
  (959,233 trades).
