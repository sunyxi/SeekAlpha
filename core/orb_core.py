"""Broker-independent ORB research core.

Pure Python, no QuantConnect imports. The same engine is driven by:
  - the LEAN adapter (lean/main.py) with real consolidated 5-minute bars, and
  - local fixture tests (tests/) with synthetic deterministic bars.

Design invariants:
  - No feature reads future bars (VWAP/RVOL/ATR are point-in-time).
  - Signal on bar t -> entry at open of bar t+1 (same session only).
  - Stop-first conservative intrabar ambiguity resolution.
  - Long-only, no overnight; session end flattens everything.
  - Costs are NOT applied here; trades carry gross PnL and notionals so
    cost scenarios (zero/baseline/double) are applied offline.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

SCHEMA_VERSION = 1
ATR_PERIOD = 14
RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_HISTORY_DAYS = 10


@dataclass(frozen=True)
class Bar:
    """A 5-minute trade bar. `end` is the bar END time in exchange time."""
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class CandidateParams:
    opening_range_minutes: int          # 15 or 30
    breakout_atr_buffer: float          # e.g. 0.05 -> close > OR_high + 0.05*ATR
    stop_atr_fraction: float            # e.g. 0.6
    stop_range_fraction: float          # e.g. 0.5
    target_r: float                     # e.g. 1.5
    max_hold_minutes: int               # e.g. 120
    signal_cutoff_minutes: int          # after session open, e.g. 90
    min_relative_volume: float          # e.g. 1.5
    min_breakout_close_location: float  # e.g. 0.7
    require_rising_vwap: bool

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return "c" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def default_candidate_grid() -> List[CandidateParams]:
    """Frozen 192-candidate grid. Do not modify after results exist."""
    grid = itertools.product(
        (15, 30),                # opening_range_minutes
        (0.05, 0.10),            # breakout_atr_buffer
        (0.6,),                  # stop_atr_fraction
        (0.5,),                  # stop_range_fraction
        (1.0, 1.5, 2.0),         # target_r
        (60, 120),               # max_hold_minutes
        (60, 90),                # signal_cutoff_minutes
        (1.2, 1.5),              # min_relative_volume
        (0.6, 0.7),              # min_breakout_close_location
        (True,),                 # require_rising_vwap
    )
    return [CandidateParams(*g) for g in grid]


def grid_spec_hash(grid: List[CandidateParams]) -> str:
    payload = json.dumps([asdict(c) for c in grid], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Trade:
    candidate_id: str
    symbol: str
    signal_time: str
    entry_time: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: str
    exit_price: float
    exit_reason: str            # stop | target | max_hold | session_close
    quantity: int
    gross_pnl: float
    entry_notional: float
    exit_notional: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Position:
    entry_time: datetime
    entry_price: float
    stop: float
    target: float
    quantity: int
    signal_time: datetime


@dataclass
class _Pending:
    signal_time: datetime
    session_date: object  # date


@dataclass
class _SymbolDay:
    session_open: datetime
    or_high: float = float("-inf")
    or_low: float = float("inf")
    or_complete: bool = False
    cum_pv: float = 0.0
    cum_vol: float = 0.0
    prev_vwap: Optional[float] = None
    vwap: Optional[float] = None
    day_open: Optional[float] = None
    day_high: float = float("-inf")
    day_low: float = float("inf")
    day_close: Optional[float] = None
    last_bar_end: Optional[datetime] = None
    or_windows: Dict[int, tuple] = field(default_factory=dict)


class SymbolEngine:
    """Runs features once per bar and all candidates' lifecycles for one symbol."""

    def __init__(self, symbol: str, candidates: List[CandidateParams],
                 notional_per_trade: float = 10_000.0):
        self.symbol = symbol
        self.candidates = candidates
        self.notional = notional_per_trade
        self.trades: List[Trade] = []
        # feature state
        self._day: Optional[_SymbolDay] = None
        self._daily_tr: List[float] = []
        self._atr: Optional[float] = None
        self._prev_daily_close: Optional[float] = None
        # minutes_after_open -> deque of past-day cumulative volumes at that mark
        self._rvol_hist: Dict[int, Deque[float]] = {}
        self._today_cumvol_marks: Dict[int, float] = {}
        # per-candidate lifecycle
        self._positions: Dict[str, _Position] = {}
        self._pending: Dict[str, _Pending] = {}

    # ------------------------------------------------------------------ bars

    def on_bar(self, bar: Bar) -> None:
        d = self._day
        if d is None or bar.end.date() != d.session_open.date():
            if d is not None:
                self._finalize_session(d)
            d = _SymbolDay(session_open=bar.end - timedelta(minutes=5))
            self._day = d

        minutes = int((bar.end - d.session_open).total_seconds() // 60)

        # ---- point-in-time features (this bar's OHLCV is "now known") ----
        d.day_open = bar.open if d.day_open is None else d.day_open
        d.day_high = max(d.day_high, bar.high)
        d.day_low = min(d.day_low, bar.low)
        d.day_close = bar.close

        typical = (bar.high + bar.low + bar.close) / 3.0
        d.cum_pv += typical * bar.volume
        d.cum_vol += bar.volume
        d.prev_vwap = d.vwap
        d.vwap = d.cum_pv / d.cum_vol if d.cum_vol > 0 else None
        self._today_cumvol_marks[minutes] = d.cum_vol

        rvol = self._relative_volume(minutes, d.cum_vol)

        d.last_bar_end = bar.end
        for w in {c.opening_range_minutes for c in self.candidates}:
            hi, lo, done = d.or_windows.get(w, (float("-inf"), float("inf"), False))
            if not done:
                if minutes <= w:
                    hi, lo = max(hi, bar.high), min(lo, bar.low)
                if minutes >= w:
                    done = True
                d.or_windows[w] = (hi, lo, done)

        # ---- candidate lifecycles ----
        for c in self.candidates:
            self._step_candidate(c, bar, d, minutes, rvol)

    def _step_candidate(self, c: CandidateParams, bar: Bar, d: _SymbolDay,
                        minutes: int, rvol: Optional[float]) -> None:
        cid = c.candidate_id

        # 1) fill pending entry at this bar's open (same session only)
        pend = self._pending.pop(cid, None)
        if pend is not None:
            if pend.session_date == bar.end.date():
                or_hi, or_lo, done = d.or_windows.get(c.opening_range_minutes, (None, None, False))
                stop_dist = min(
                    c.stop_atr_fraction * self._atr,          # type: ignore[arg-type]
                    c.stop_range_fraction * (or_hi - or_lo),
                )
                if stop_dist > 0 and bar.open > 0:
                    qty = int(self.notional // bar.open)
                    if qty > 0:
                        self._positions[cid] = _Position(
                            entry_time=bar.end - timedelta(minutes=5),
                            entry_price=bar.open,
                            stop=bar.open - stop_dist,
                            target=bar.open + c.target_r * stop_dist,
                            quantity=qty,
                            signal_time=pend.signal_time,
                        )
            # signal on last bar of a session dies silently (next-bar rule)

        # 2) manage open position on this bar (stop-first, then target, time, close)
        pos = self._positions.get(cid)
        if pos is not None and bar.end > pos.entry_time:
            exit_price = exit_reason = None
            if bar.open <= pos.stop:
                exit_price, exit_reason = bar.open, "stop"
            elif bar.low <= pos.stop:
                exit_price, exit_reason = pos.stop, "stop"
            elif bar.open >= pos.target:
                exit_price, exit_reason = bar.open, "target"
            elif bar.high >= pos.target:
                exit_price, exit_reason = pos.target, "target"
            elif (bar.end - pos.entry_time) >= timedelta(minutes=c.max_hold_minutes):
                exit_price, exit_reason = bar.close, "max_hold"
            if exit_price is not None:
                self._close(cid, pos, bar.end, exit_price, exit_reason)
                pos = None

        # 3) evaluate a new signal on this bar's close
        if pos is not None or cid in self._pending:
            return
        if self._atr is None or rvol is None or d.vwap is None:
            return
        or_hi, or_lo, done = d.or_windows.get(c.opening_range_minutes, (None, None, False))
        if not done or minutes <= c.opening_range_minutes:
            return
        if minutes > c.signal_cutoff_minutes:
            return
        if or_hi <= or_lo:
            return
        if bar.close <= or_hi + c.breakout_atr_buffer * self._atr:
            return
        if rvol < c.min_relative_volume:
            return
        rng = bar.high - bar.low
        loc = 1.0 if rng <= 0 else (bar.close - bar.low) / rng
        if loc < c.min_breakout_close_location:
            return
        if bar.close < d.vwap:
            return
        if c.require_rising_vwap and (d.prev_vwap is None or d.vwap <= d.prev_vwap):
            return
        self._pending[cid] = _Pending(signal_time=bar.end,
                                      session_date=bar.end.date())

    # ------------------------------------------------------------- lifecycle

    def _close(self, cid: str, pos: _Position, when: datetime,
               price: float, reason: str) -> None:
        gross = (price - pos.entry_price) * pos.quantity
        self.trades.append(Trade(
            candidate_id=cid, symbol=self.symbol,
            signal_time=pos.signal_time.isoformat(),
            entry_time=pos.entry_time.isoformat(),
            entry_price=pos.entry_price, stop_price=pos.stop,
            target_price=pos.target, exit_time=when.isoformat(),
            exit_price=price, exit_reason=reason, quantity=pos.quantity,
            gross_pnl=gross,
            entry_notional=pos.entry_price * pos.quantity,
            exit_notional=price * pos.quantity,
        ))
        self._positions.pop(cid, None)

    def on_session_end(self, last_bar_close_time: Optional[datetime] = None) -> None:
        """Adapter calls this at end of each trading day (handles half days)."""
        d = self._day
        if d is None:
            return
        self._finalize_session(d)
        self._day = None

    def _finalize_session(self, d: _SymbolDay) -> None:
        when = d.last_bar_end if d.last_bar_end is not None else d.session_open
        for cid in list(self._positions):
            pos = self._positions[cid]
            price = d.day_close if d.day_close is not None else pos.entry_price
            self._close(cid, pos, when, price, "session_close")
        self._pending.clear()
        # daily ATR update (Wilder)
        if d.day_close is not None and d.day_high > d.day_low:
            pc = self._prev_daily_close
            tr = d.day_high - d.day_low
            if pc is not None:
                tr = max(tr, abs(d.day_high - pc), abs(d.day_low - pc))
            self._daily_tr.append(tr)
            if self._atr is None and len(self._daily_tr) >= ATR_PERIOD:
                self._atr = sum(self._daily_tr[-ATR_PERIOD:]) / ATR_PERIOD
            elif self._atr is not None:
                self._atr = (self._atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD
            self._prev_daily_close = d.day_close
        # roll RVOL history
        for m, cum in self._today_cumvol_marks.items():
            self._rvol_hist.setdefault(m, deque(maxlen=RVOL_LOOKBACK_DAYS)).append(cum)
        self._today_cumvol_marks = {}

    # -------------------------------------------------------------- features

    def _relative_volume(self, minutes: int, cum_vol: float) -> Optional[float]:
        hist = self._rvol_hist.get(minutes)
        if hist is None or len(hist) < RVOL_MIN_HISTORY_DAYS:
            return None
        med = statistics.median(hist)
        if med <= 0:
            return None
        return cum_vol / med
