"""Deterministic fixture tests for the broker-independent ORB core.

Runs with no QuantConnect SDK, no network, no account.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

from orb_core import Bar, CandidateParams, SymbolEngine  # noqa: E402

NY_OPEN = datetime(2024, 6, 3, 9, 30)


def one_candidate() -> CandidateParams:
    return CandidateParams(
        opening_range_minutes=15, breakout_atr_buffer=0.05,
        stop_atr_fraction=0.6, stop_range_fraction=0.5, target_r=1.5,
        max_hold_minutes=120, signal_cutoff_minutes=90,
        min_relative_volume=1.2, min_breakout_close_location=0.6,
        require_rising_vwap=True)


def flat_day(day_open: datetime, price: float, vol: float = 100_000,
             bars: int = 78):
    """A boring session: 78 five-minute bars, tiny drift, constant volume."""
    out = []
    for i in range(bars):
        end = day_open + timedelta(minutes=5 * (i + 1))
        out.append(Bar(end=end, open=price, high=price + 0.5,
                       low=price - 0.5, close=price, volume=vol))
    return out


def warm_up(eng: SymbolEngine, days: int = 25, price: float = 100.0):
    """Feed enough flat sessions to establish ATR(14) and RVOL history."""
    for d in range(days):
        day = NY_OPEN - timedelta(days=days - d)
        if day.weekday() >= 5:
            continue
        for b in flat_day(day, price):
            eng.on_bar(b)
        eng.on_session_end()


class TestOrbCore(unittest.TestCase):

    def _breakout_day(self, eng: SymbolEngine):
        """Scripted session: OR forms in first 3 bars, breakout on bar 4
        with 3x volume, next-bar entry on bar 5, target hit on bar 6."""
        d = NY_OPEN
        bars = [
            Bar(d + timedelta(minutes=5), 100.0, 100.8, 99.6, 100.2, 100_000),
            Bar(d + timedelta(minutes=10), 100.2, 100.9, 99.8, 100.5, 100_000),
            Bar(d + timedelta(minutes=15), 100.5, 101.0, 100.0, 100.7, 100_000),
            # breakout: close 102.4 > OR high 101.0 + 0.05*ATR, strong close,
            # heavy volume, above rising VWAP
            Bar(d + timedelta(minutes=20), 100.8, 102.5, 100.8, 102.4, 900_000),
            # entry bar: enters at open 102.5
            Bar(d + timedelta(minutes=25), 102.5, 102.9, 102.3, 102.8, 300_000),
            # target bar
            Bar(d + timedelta(minutes=30), 102.8, 104.5, 102.7, 104.2, 300_000),
        ]
        for b in bars:
            eng.on_bar(b)
        eng.on_session_end()

    def test_breakout_lifecycle_next_bar_entry_and_target(self):
        eng = SymbolEngine("TEST", [one_candidate()])
        warm_up(eng)
        self._breakout_day(eng)
        self.assertEqual(len(eng.trades), 1)
        t = eng.trades[0]
        self.assertEqual(t.entry_price, 102.5)          # next-bar open
        self.assertEqual(t.exit_reason, "target")
        # stop distance = min(0.6*ATR(1.0)=0.6, 0.5*OR width(1.4)=0.7) = 0.6
        self.assertAlmostEqual(t.entry_price - t.stop_price, 0.6, places=6)
        self.assertAlmostEqual(t.target_price - t.entry_price, 0.9, places=6)
        self.assertEqual(t.quantity, int(10_000 // 102.5))
        self.assertGreater(t.gross_pnl, 0)

    def test_no_signal_without_rvol_history(self):
        eng = SymbolEngine("TEST", [one_candidate()])
        # no warm-up: ATR and RVOL are unavailable -> zero trades
        self._breakout_day(eng)
        self.assertEqual(len(eng.trades), 0)

    def test_stop_first_ambiguity(self):
        eng = SymbolEngine("TEST", [one_candidate()])
        warm_up(eng)
        d = NY_OPEN
        bars = [
            Bar(d + timedelta(minutes=5), 100.0, 100.8, 99.6, 100.2, 100_000),
            Bar(d + timedelta(minutes=10), 100.2, 100.9, 99.8, 100.5, 100_000),
            Bar(d + timedelta(minutes=15), 100.5, 101.0, 100.0, 100.7, 100_000),
            Bar(d + timedelta(minutes=20), 100.8, 102.5, 100.8, 102.4, 900_000),
            # entry at 102.5; this bar spans BOTH stop (101.9) and target (103.4)
            Bar(d + timedelta(minutes=25), 102.5, 104.0, 101.5, 103.0, 300_000),
        ]
        for b in bars:
            eng.on_bar(b)
        eng.on_session_end()
        self.assertEqual(len(eng.trades), 1)
        self.assertEqual(eng.trades[0].exit_reason, "stop")

    def test_session_close_flatten_no_overnight(self):
        eng = SymbolEngine("TEST", [one_candidate()])
        warm_up(eng)
        d = NY_OPEN
        bars = [
            Bar(d + timedelta(minutes=5), 100.0, 100.8, 99.6, 100.2, 100_000),
            Bar(d + timedelta(minutes=10), 100.2, 100.9, 99.8, 100.5, 100_000),
            Bar(d + timedelta(minutes=15), 100.5, 101.0, 100.0, 100.7, 100_000),
            Bar(d + timedelta(minutes=20), 100.8, 102.5, 100.8, 102.4, 900_000),
            # entry, then price drifts sideways inside stop/target until close
            Bar(d + timedelta(minutes=25), 102.5, 102.6, 102.4, 102.5, 300_000),
        ]
        for b in bars:
            eng.on_bar(b)
        eng.on_session_end()
        self.assertEqual(len(eng.trades), 1)
        # max_hold is 120m but only one post-entry bar exists -> session close
        self.assertEqual(eng.trades[0].exit_reason, "session_close")

    def test_signal_cutoff(self):
        p = one_candidate()
        cid = p.candidate_id
        eng = SymbolEngine("TEST", [p])
        warm_up(eng)
        d = NY_OPEN
        # quiet until minute 95, breakout at minute 100 (> 90 cutoff)
        bars = []
        for i in range(19):  # bars ending at +5 .. +95
            end = d + timedelta(minutes=5 * (i + 1))
            bars.append(Bar(end, 100.0, 100.5, 99.5, 100.0, 100_000))
        bars.append(Bar(d + timedelta(minutes=100),
                        100.0, 103.0, 100.0, 102.9, 900_000))
        bars.append(Bar(d + timedelta(minutes=105),
                        102.9, 103.2, 102.8, 103.0, 300_000))
        for b in bars:
            eng.on_bar(b)
        eng.on_session_end()
        self.assertEqual(len(eng.trades), 0, "signal after cutoff must not trade")
        self.assertEqual(cid, p.candidate_id)  # id stability

    def test_candidate_id_deterministic(self):
        self.assertEqual(one_candidate().candidate_id,
                         one_candidate().candidate_id)


if __name__ == "__main__":
    unittest.main()
