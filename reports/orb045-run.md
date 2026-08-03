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
| Total trades | 959,233 |
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

## Outer test results (16 folds, 0 folds with candidate selection)

| Cost scenario | Trades | Sharpe | Profit Factor | Mean net bps |
| --- | ---: | ---: | ---: | ---: |
| Zero (0 bps) | 566 | 0.70 | 1.20 | +4.73 |
| Baseline (2.5 bps/side) | 566 | −0.03 | 0.99 | −0.27 |
| Double (5 bps/side) | 566 | −0.79 | 0.82 | −5.28 |

**Decision reasons:**
- Outer Sharpe −0.03 ≤ 0.5 (gate: > 0.5)
- Outer PF 0.99 < 1.1 (gate: ≥ 1.10)
- Outer mean net bps ≤ 0 (gate: > 0)

## Key observations

- At zero cost the strategy shows positive edge (Sharpe 0.70, PF 1.20) — the gross signal exists.
- 2.5 bps/side execution costs fully erase the edge (PF drops from 1.20 to 0.99).
- No fold reached a passing candidate; the 566 outer test trades arise from a fallback or
  bleed-through path — see `wf_select.py` logic for details.
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
