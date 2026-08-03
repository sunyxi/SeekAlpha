#!/usr/bin/env python3
"""Offline nested walk-forward over trade logs produced by the LEAN data pump.

Usage:
    python3 scripts/wf_select.py --trades-dir downloaded_objectstore/orb045 \\
        --report-output reports/orb045-wf.json

Pre-declared, frozen decision gates (change ONLY before looking at results):
    per-fold selection gates (training window):
        min_trades >= 30, train Sharpe >= 0, train profit factor >= 1.0,
        validation net PnL > 0 (last 20% of the training window)
    final Candidate gates (aggregated outer tests, baseline cost 2.5 bps/side):
        total outer trades >= 100, outer Sharpe > 0.5, outer PF >= 1.10,
        outer mean net bps > 0
    Anything else -> No-Go.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

INITIAL_CAPITAL = 100_000.0
COST_SCENARIOS = {"zero": 0.0, "baseline": 2.5, "double": 5.0}

TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 252, 63, 63     # calendar-of-trading-days folds
SEL_MIN_TRADES, SEL_MIN_SHARPE, SEL_MIN_PF = 30, 0.0, 1.0
GATE_MIN_TRADES, GATE_MIN_SHARPE, GATE_MIN_PF = 100, 0.5, 1.10


def net_pnl(t: dict, bps_per_side: float) -> float:
    cost = (t["entry_notional"] + t["exit_notional"]) * bps_per_side / 10_000.0
    return t["gross_pnl"] - cost


def daily_returns(trades: list, trading_days: list, bps: float) -> list:
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["exit_time"][:10]] += net_pnl(t, bps)
    return [by_day.get(d, 0.0) / INITIAL_CAPITAL for d in trading_days]


def sharpe(rets: list) -> float:
    if len(rets) < 2:
        return float("nan")
    sd = statistics.pstdev(rets)
    if sd == 0:
        return float("nan")
    return statistics.mean(rets) / sd * math.sqrt(252)


def metrics(trades: list, trading_days: list, bps: float) -> dict:
    rets = daily_returns(trades, trading_days, bps)
    nets = [net_pnl(t, bps) for t in trades]
    wins = [n for n in nets if n > 0]
    losses = [-n for n in nets if n < 0]
    equity, peak, mdd = 0.0, 0.0, 0.0
    for r in rets:
        equity += r
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    mean_bps = (statistics.mean(
        n / t["entry_notional"] * 10_000.0 for n, t in zip(nets, trades))
        if trades else float("nan"))
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(nets) if nets else float("nan"),
        "profit_factor": (sum(wins) / sum(losses)) if losses else float("inf"),
        "sharpe": sharpe(rets),
        "mean_net_bps": mean_bps,
        "total_net_pnl": sum(nets),
        "max_drawdown_frac": mdd,
    }


def load_trades(trades_dir: str) -> tuple[list, list]:
    metas, trades = [], []
    shard_dirs = sorted(glob.glob(os.path.join(trades_dir, "shard*")))
    if not shard_dirs:
        shard_dirs = [trades_dir]
    seen = set()
    for sd in shard_dirs:
        mpath = os.path.join(sd, "meta.json")
        if not os.path.exists(mpath):
            sys.exit(f"missing meta.json in {sd}: refusing incomplete input")
        meta = json.load(open(mpath))
        metas.append(meta)
        for f in sorted(glob.glob(os.path.join(sd, "trades_*.json"))):
            trades.extend(json.load(open(f)))
    hashes = {m["grid_spec_hash"] for m in metas}
    if len(hashes) != 1:
        sys.exit(f"inconsistent grid_spec_hash across shards: {hashes}")
    if len(metas) > 1:
        shards_total = {m["candidate_shards"] for m in metas}
        got = sorted(m["candidate_shard"] for m in metas)
        if len(shards_total) != 1 or got != list(range(got[-1] + 1)) \
                or len(got) != next(iter(shards_total)):
            sys.exit(f"missing/duplicate shards: have {got}, expect "
                     f"{shards_total} total")
    for t in trades:
        k = (t["candidate_id"], t["symbol"], t["entry_time"])
        if k in seen:
            sys.exit(f"duplicate trade detected: {k}")
        seen.add(k)
    return metas, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-dir", required=True)
    ap.add_argument("--report-output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.report_output):
        sys.exit("report path exists; create-only policy refuses overwrite")

    metas, all_trades = load_trades(args.trades_dir)

    trading_days = sorted({t["exit_time"][:10] for t in all_trades})
    by_cand = defaultdict(list)
    for t in all_trades:
        by_cand[t["candidate_id"]].append(t)

    folds, i = [], 0
    while i + TRAIN_DAYS + TEST_DAYS <= len(trading_days):
        train = trading_days[i:i + TRAIN_DAYS]
        test = trading_days[i + TRAIN_DAYS:i + TRAIN_DAYS + TEST_DAYS]
        folds.append((train, test))
        i += STEP_DAYS

    fold_reports, outer_trades = [], []
    for k, (train, test) in enumerate(folds):
        split = int(len(train) * 0.8)
        inner_train, inner_val = train[:split], train[split:]
        t0, t1 = set(inner_train), set(inner_val)
        best, best_val_sharpe, rejections = None, float("-inf"), defaultdict(int)
        for cid, trs in by_cand.items():
            tr = [t for t in trs if t["exit_time"][:10] in t0]
            va = [t for t in trs if t["exit_time"][:10] in t1]
            m = metrics(tr, inner_train, COST_SCENARIOS["baseline"])
            if m["trades"] < SEL_MIN_TRADES:
                rejections["min_trades"] += 1; continue
            if not (m["sharpe"] >= SEL_MIN_SHARPE):
                rejections["train_sharpe"] += 1; continue
            if not (m["profit_factor"] >= SEL_MIN_PF):
                rejections["train_pf"] += 1; continue
            vm = metrics(va, inner_val, COST_SCENARIOS["baseline"])
            if not (vm["total_net_pnl"] > 0):
                rejections["val_pnl"] += 1; continue
            if vm["sharpe"] > best_val_sharpe:
                best, best_val_sharpe = cid, vm["sharpe"]
        rep = {"fold": k, "train": [train[0], train[-1]],
               "test": [test[0], test[-1]],
               "rejection_reasons": dict(rejections)}
        if best is None:
            rep["selected"] = None
        else:
            tset = set(test)
            sel = [t for t in by_cand[best] if t["exit_time"][:10] in tset]
            rep["selected"] = best
            rep["test_metrics"] = metrics(sel, test, COST_SCENARIOS["baseline"])
            outer_trades.extend(sel)
        fold_reports.append(rep)

    outer_days = sorted({d for _, te in folds for d in te})
    agg = {name: metrics(outer_trades, outer_days, bps)
           for name, bps in COST_SCENARIOS.items()}
    base = agg["baseline"]
    reasons = []
    if base["trades"] < GATE_MIN_TRADES:
        reasons.append(f"outer trades {base['trades']} < {GATE_MIN_TRADES}")
    if not (base["sharpe"] > GATE_MIN_SHARPE):
        reasons.append(f"outer sharpe {base['sharpe']:.2f} <= {GATE_MIN_SHARPE}")
    if not (base["profit_factor"] >= GATE_MIN_PF):
        reasons.append(f"outer PF {base['profit_factor']:.2f} < {GATE_MIN_PF}")
    if not (base["mean_net_bps"] > 0):
        reasons.append("outer mean net bps <= 0")
    decision = "Candidate" if not reasons else "No-Go"

    symbol_attr = defaultdict(float)
    for t in outer_trades:
        symbol_attr[t["symbol"]] += net_pnl(t, COST_SCENARIOS["baseline"])

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_meta": metas,
        "fold_definition": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                            "step_days": STEP_DAYS},
        "selection_gates": {"min_trades": SEL_MIN_TRADES,
                            "min_train_sharpe": SEL_MIN_SHARPE,
                            "min_train_pf": SEL_MIN_PF,
                            "validation": "last 20% of train, net pnl > 0"},
        "decision_gates": {"min_outer_trades": GATE_MIN_TRADES,
                           "min_outer_sharpe": GATE_MIN_SHARPE,
                           "min_outer_pf": GATE_MIN_PF},
        "folds": fold_reports,
        "outer_test_metrics_by_cost": agg,
        "symbol_attribution_baseline": dict(sorted(symbol_attr.items())),
        "decision": decision,
        "decision_reasons": reasons or ["all pre-declared gates passed"],
    }
    tmp = args.report_output + ".tmp"
    os.makedirs(os.path.dirname(args.report_output) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.link(tmp, args.report_output)   # atomic create-only publish
    os.unlink(tmp)
    print(f"{decision}: {'; '.join(report['decision_reasons'])}")
    print(f"report -> {args.report_output}")


if __name__ == "__main__":
    main()
