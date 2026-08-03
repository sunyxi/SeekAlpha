"""NYSE trading calendar for 2021-2030.

Pure stdlib, no external dependencies. Covers full-day closures (holidays)
and half-day sessions (early close at 13:00 ET) for the date range used in
all simulation runs.

Public API
----------
is_trading_day(d: date) -> bool
    True if NYSE is open on date d.

session_end(d: date) -> time
    16:00 for normal sessions, 13:00 for half-days.
    Behaviour is undefined for non-trading days; call is_trading_day() first.
"""

from __future__ import annotations

from datetime import date, time, timedelta

MARKET_CLOSE: time = time(16, 0)
EARLY_CLOSE:  time = time(13, 0)

_FIRST_YEAR = 2021
_LAST_YEAR  = 2030


# ---------------------------------------------------------------------------
# Internal helpers

def _nth_weekday(year: int, month: int, n: int, weekday: int) -> date:
    """n-th occurrence (1-indexed) of weekday in month. weekday: 0=Mon … 6=Sun."""
    first_of_month = date(year, month, 1)
    delta_to_first = (weekday - first_of_month.weekday()) % 7
    return first_of_month + timedelta(days=delta_to_first + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last occurrence of weekday in month."""
    last_of_month = (
        date(year + 1, 1, 1) - timedelta(days=1)
        if month == 12
        else date(year, month + 1, 1) - timedelta(days=1)
    )
    return last_of_month - timedelta(days=(last_of_month.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """NYSE observance: Saturday → Friday, Sunday → Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """Compute Easter Sunday via the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


# One-off market closures (national days of mourning, extraordinary events).
# Only entries within the supported year range matter for runtime use.
_SPECIAL_CLOSURES: frozenset[date] = frozenset({
    date(2025, 1, 9),   # National day of mourning: President Jimmy Carter
})


# ---------------------------------------------------------------------------
# Holiday and half-day computation

def _compute_holidays(year: int) -> frozenset[date]:
    h: set[date] = set()
    h.add(_observed(date(year, 1, 1)))           # New Year's Day
    h.add(_nth_weekday(year, 1, 3, 0))           # MLK Day (3rd Mon Jan)
    h.add(_nth_weekday(year, 2, 3, 0))           # Presidents' Day (3rd Mon Feb)
    h.add(_easter(year) - timedelta(days=2))      # Good Friday
    h.add(_last_weekday(year, 5, 0))             # Memorial Day (last Mon May)
    if year >= 2022:
        h.add(_observed(date(year, 6, 19)))      # Juneteenth
    h.add(_observed(date(year, 7, 4)))           # Independence Day
    h.add(_nth_weekday(year, 9, 1, 0))           # Labor Day (1st Mon Sep)
    h.add(_nth_weekday(year, 11, 4, 3))          # Thanksgiving (4th Thu Nov)
    h.add(_observed(date(year, 12, 25)))         # Christmas Day
    for d in _SPECIAL_CLOSURES:
        if d.year == year:
            h.add(d)
    return frozenset(h)


def _compute_half_days(year: int, holidays: frozenset[date]) -> frozenset[date]:
    h: set[date] = set()

    # July 3 early close: only when July 4 itself is a weekday (Mon–Fri).
    # If July 4 is Saturday, July 3 is the observed closure (full holiday).
    # If July 4 is Sunday, July 5 is the observed closure; July 3 is normal.
    jul4 = date(year, 7, 4)
    if jul4.weekday() < 5:
        jul3 = date(year, 7, 3)
        if jul3.weekday() < 5 and jul3 not in holidays:
            h.add(jul3)

    # Black Friday (day after Thanksgiving): always an early close.
    thanksgiving = _nth_weekday(year, 11, 4, 3)
    black_friday = thanksgiving + timedelta(days=1)
    if black_friday not in holidays:
        h.add(black_friday)

    # Christmas Eve (Dec 24): early close when Dec 25 is a weekday holiday
    # and Dec 24 is also a weekday that is not itself a holiday.
    dec25 = date(year, 12, 25)
    dec24 = date(year, 12, 24)
    if dec25.weekday() < 5 and dec24.weekday() < 5 and dec24 not in holidays:
        h.add(dec24)

    return frozenset(h)


# Pre-compute calendar tables at module import for the supported range.
_HOLIDAYS:  dict[int, frozenset[date]] = {}
_HALF_DAYS: dict[int, frozenset[date]] = {}

for _y in range(_FIRST_YEAR, _LAST_YEAR + 1):
    _HOLIDAYS[_y]  = _compute_holidays(_y)
    _HALF_DAYS[_y] = _compute_half_days(_y, _HOLIDAYS[_y])


# ---------------------------------------------------------------------------
# Public API

def is_trading_day(d: date) -> bool:
    """Return True if NYSE is open on date d."""
    if d.weekday() >= 5:
        return False
    year = d.year
    if year not in _HOLIDAYS:
        raise ValueError(
            f"calendar covers {_FIRST_YEAR}–{_LAST_YEAR}; got {year}"
        )
    return d not in _HOLIDAYS[year]


def session_end(d: date) -> time:
    """Return NYSE session-end time for date d.

    Returns EARLY_CLOSE (13:00 ET) for half-days, MARKET_CLOSE (16:00 ET)
    for normal sessions. Call is_trading_day() before calling this; behaviour
    is undefined for non-trading days.
    """
    year = d.year
    if year not in _HALF_DAYS:
        raise ValueError(
            f"calendar covers {_FIRST_YEAR}–{_LAST_YEAR}; got {year}"
        )
    return EARLY_CLOSE if d in _HALF_DAYS[year] else MARKET_CLOSE
