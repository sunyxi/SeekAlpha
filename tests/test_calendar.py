"""Tests for the NYSE trading calendar (src/orb/calendar.py).

No network calls. Validates holidays, half-days, weekends, DST dates,
and the load_minute_bars half-day bar-trimming integration.
"""

from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from orb.calendar import (
    EARLY_CLOSE,
    MARKET_CLOSE,
    _compute_holidays,
    _compute_half_days,
    is_trading_day,
    session_end,
)

# Import load_minute_bars from local_pump for integration tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from local_pump import load_minute_bars

NY = ZoneInfo("America/New_York")


class TestIsTraidingDay(unittest.TestCase):

    # ---------------------------------------------------------------- weekends
    def test_saturday_is_not_trading_day(self):
        self.assertFalse(is_trading_day(date(2024, 3, 2)))   # Saturday

    def test_sunday_is_not_trading_day(self):
        self.assertFalse(is_trading_day(date(2024, 3, 3)))   # Sunday

    # --------------------------------------------------------------- holidays
    def test_new_years_day_2021(self):
        self.assertFalse(is_trading_day(date(2021, 1, 1)))

    def test_mlk_day_2024(self):
        self.assertFalse(is_trading_day(date(2024, 1, 15)))  # 3rd Mon Jan 2024

    def test_good_friday_2024(self):
        self.assertFalse(is_trading_day(date(2024, 3, 29)))

    def test_memorial_day_2024(self):
        self.assertFalse(is_trading_day(date(2024, 5, 27)))  # last Mon May 2024

    def test_juneteenth_2024(self):
        self.assertFalse(is_trading_day(date(2024, 6, 19)))

    def test_juneteenth_not_observed_in_2021(self):
        # Juneteenth added from 2022; Jun 19 2021 is a normal day (Saturday)
        # But Jun 18 2021 (Friday) should be a trading day
        self.assertTrue(is_trading_day(date(2021, 6, 18)))

    def test_independence_day_2023(self):
        self.assertFalse(is_trading_day(date(2023, 7, 4)))   # Tue

    def test_independence_day_observed_2021(self):
        # Jul 4 2021 is Sunday → observed Mon Jul 5 (closed); Jul 4 itself is a Sunday (closed)
        self.assertFalse(is_trading_day(date(2021, 7, 5)))  # observed closure
        self.assertFalse(is_trading_day(date(2021, 7, 4)))  # Sunday — not a trading day

    def test_labor_day_2024(self):
        self.assertFalse(is_trading_day(date(2024, 9, 2)))   # 1st Mon Sep 2024

    def test_thanksgiving_2024(self):
        self.assertFalse(is_trading_day(date(2024, 11, 28)))  # 4th Thu Nov 2024

    def test_christmas_2024(self):
        self.assertFalse(is_trading_day(date(2024, 12, 25)))  # Wednesday

    def test_christmas_observed_2021(self):
        # Dec 25 2021 is Saturday → observed Dec 24 (Friday)
        self.assertFalse(is_trading_day(date(2021, 12, 24)))

    def test_carter_mourning_day_2025(self):
        self.assertFalse(is_trading_day(date(2025, 1, 9)))

    # ---------------------------------------------------------- normal days
    def test_normal_monday(self):
        self.assertTrue(is_trading_day(date(2024, 3, 4)))    # Mon

    def test_normal_wednesday(self):
        self.assertTrue(is_trading_day(date(2024, 6, 5)))    # Wed

    # --------------------------------------------------- DST transition dates
    def test_day_after_spring_forward_2024(self):
        # DST sprang forward 2024-03-10 (Sun). Mon 2024-03-11 is a normal trading day.
        self.assertTrue(is_trading_day(date(2024, 3, 11)))

    def test_day_after_fall_back_2024(self):
        # DST fell back 2024-11-03 (Sun). Mon 2024-11-04 is a normal trading day.
        self.assertTrue(is_trading_day(date(2024, 11, 4)))

    def test_dst_spring_transition_day_is_sunday(self):
        self.assertFalse(is_trading_day(date(2024, 3, 10)))  # Sunday — not trading

    # ----------------------------------------------------- out of range
    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            is_trading_day(date(2031, 1, 1))

    def test_too_early_raises(self):
        with self.assertRaises(ValueError):
            is_trading_day(date(2020, 12, 31))


class TestSessionEnd(unittest.TestCase):

    # --------------------------------------------------------------- half-days
    def test_black_friday_2024(self):
        self.assertEqual(session_end(date(2024, 11, 29)), EARLY_CLOSE)

    def test_july_3_half_day_2024(self):
        # Jul 4 2024 is Thursday → Jul 3 (Wed) is early close
        self.assertEqual(session_end(date(2024, 7, 3)), EARLY_CLOSE)

    def test_christmas_eve_2024(self):
        # Dec 25 2024 is Wednesday → Dec 24 (Tue) is early close
        self.assertEqual(session_end(date(2024, 12, 24)), EARLY_CLOSE)

    def test_black_friday_2023(self):
        self.assertEqual(session_end(date(2023, 11, 24)), EARLY_CLOSE)

    def test_july_3_half_day_2023(self):
        # Jul 4 2023 is Tuesday → Jul 3 (Mon) is early close
        self.assertEqual(session_end(date(2023, 7, 3)), EARLY_CLOSE)

    # ----------------------------------------- NO half-day (special cases)
    def test_no_july3_half_day_2021(self):
        # Jul 4 2021 is Sunday → observed Jul 5; Jul 3 (Sat) is not a trading day
        # Jul 2 (Fri) should be a normal full-day session
        self.assertEqual(session_end(date(2021, 7, 2)), MARKET_CLOSE)

    def test_no_july3_half_day_2022(self):
        # Jul 4 2022 is Monday (weekday) but Jul 3 is Sunday (not a trading day)
        # Full day before: Jul 1 (Fri)
        self.assertEqual(session_end(date(2022, 7, 1)), MARKET_CLOSE)

    def test_no_christmas_eve_half_day_2021(self):
        # Dec 25 2021 is Saturday → observed Dec 24 (full holiday, not half-day)
        # Dec 23 should be normal
        self.assertEqual(session_end(date(2021, 12, 23)), MARKET_CLOSE)

    # ---------------------------------------------------------- normal days
    def test_normal_session_end(self):
        self.assertEqual(session_end(date(2024, 3, 11)), MARKET_CLOSE)

    def test_normal_session_end_wednesday(self):
        self.assertEqual(session_end(date(2024, 6, 5)), MARKET_CLOSE)


class TestHolidayComputation(unittest.TestCase):
    """Spot-check _compute_holidays for specific years."""

    def test_2024_holiday_count(self):
        h = _compute_holidays(2024)
        # Expected: NY, MLK, Pres, GF, Mem, Juneteenth, IndDay, Labor, Thanksgiving, Christmas
        self.assertGreaterEqual(len(h), 10)

    def test_good_friday_2024(self):
        h = _compute_holidays(2024)
        self.assertIn(date(2024, 3, 29), h)

    def test_good_friday_2021(self):
        h = _compute_holidays(2021)
        self.assertIn(date(2021, 4, 2), h)

    def test_juneteenth_absent_in_2021(self):
        h = _compute_holidays(2021)
        self.assertNotIn(date(2021, 6, 19), h)

    def test_juneteenth_present_in_2022(self):
        h = _compute_holidays(2022)
        self.assertIn(date(2022, 6, 20), h)  # Jun 19 is Sunday → observed Mon Jun 20

    def test_independence_day_sat_2026(self):
        # Jul 4 2026 is Saturday → observed Fri Jul 3
        h = _compute_holidays(2026)
        self.assertIn(date(2026, 7, 3), h)
        self.assertNotIn(date(2026, 7, 4), h)   # Saturday already excluded by weekday check

    def test_no_july3_halfday_when_it_is_a_holiday(self):
        # 2026: Jul 3 is the observed Independence Day (full closure), not a half-day
        h = _compute_holidays(2026)
        hd = _compute_half_days(2026, h)
        self.assertNotIn(date(2026, 7, 3), hd)


class TestLoadMinuteBarsHalfDay(unittest.TestCase):
    """Integration: load_minute_bars trims bars past 13:00 on half-days."""

    def _write_csv(self, path: Path, rows: list[str]) -> None:
        with gzip.open(path, "wt") as f:
            f.write("ts,open,high,low,close,volume\n")
            for r in rows:
                f.write(r + "\n")

    def test_normal_day_keeps_all_rth_bars(self):
        # 2024-06-05 (Wed) is a normal trading day
        day = "2024-06-05"
        rows = [
            f"{day}T09:31:00-04:00,100,101,99,100,1000",
            f"{day}T12:00:00-04:00,100,101,99,100,1000",
            f"{day}T15:59:00-04:00,100,101,99,100,1000",
            f"{day}T16:01:00-04:00,100,101,99,100,1000",  # after close → excluded
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bars.csv.gz"
            self._write_csv(p, rows)
            loaded = load_minute_bars(p)
        # 3 RTH bars (09:31, 12:00, 15:59); 16:01 excluded
        self.assertEqual(len(loaded), 3)

    def test_half_day_trims_bars_after_1300(self):
        # 2024-11-29 (Black Friday) is a half-day: session ends at 13:00
        day = "2024-11-29"
        rows = [
            f"{day}T09:31:00-05:00,100,101,99,100,1000",   # 09:31 → kept
            f"{day}T12:59:00-05:00,100,101,99,100,1000",   # 12:59 → kept
            f"{day}T13:00:00-05:00,100,101,99,100,1000",   # 13:00 → excluded (>= sess_close)
            f"{day}T14:00:00-05:00,100,101,99,100,1000",   # 14:00 → excluded
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bars.csv.gz"
            self._write_csv(p, rows)
            loaded = load_minute_bars(p)
        self.assertEqual(len(loaded), 2)
        # Last bar is at 12:59
        last_bar_time = loaded[-1][0].time()
        self.assertEqual(last_bar_time, time(12, 59))

    def test_july3_halfday_2024_trims_afternoon_bars(self):
        # 2024-07-03 (Wed) is a half-day
        day = "2024-07-03"
        rows = [
            f"{day}T09:31:00-04:00,100,101,99,100,1000",
            f"{day}T12:59:00-04:00,100,101,99,100,1000",
            f"{day}T13:01:00-04:00,100,101,99,100,1000",  # excluded
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bars.csv.gz"
            self._write_csv(p, rows)
            loaded = load_minute_bars(p)
        self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
