from __future__ import annotations

import datetime as dt

import pandas as pd

LOCAL_TZ = "Europe/Berlin"
GATE_CLOSURE_LOCAL_HOUR = 12  # EPEX day-ahead gate closure: 12:00 local time on D-1


def _local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local (Europe/Berlin) calendar-day midnight for each UTC timestamp."""
    return index.tz_convert(LOCAL_TZ).normalize()


def gate_closure_for_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Gate closure (UTC) for each target timestamp: 12:00 Europe/Berlin on the
    calendar day before the timestamp's local delivery day.

    Mirrors Fold.gate_closure from walkforward.py and coincides with it on the
    gapless hourly grid produced in step 2.1 (cross-checked by test). DST-safe
    via DateOffset; 12:00 is never in the ambiguous DST window.
    """
    prev_local_day = _local_day(index) - pd.DateOffset(days=1)
    return (prev_local_day + pd.Timedelta(hours=GATE_CLOSURE_LOCAL_HOUR)).tz_convert("UTC")


def local_day_bounds(date: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Local midnight of date and of the following day (end exclusive)."""
    start = pd.Timestamp(date, tz=LOCAL_TZ)
    end = pd.Timestamp(date + dt.timedelta(days=1), tz=LOCAL_TZ)
    return start, end


def local_window_bounds(
    date: dt.date, *, days_back_start: int, days_back_end: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Local midnight bounds of a multi-day window anchored on date.

    Window is [date - days_back_start, date - days_back_end) in calendar
    days, both ends localized independently -- not via Timedelta(days=N) or
    DateOffset chaining from a single point, so a window spanning a DST
    transition still lands on the correct local midnight at each end
    (Timestamp - Timedelta(hours=24*N) would be off by an hour whenever the
    window crosses a transition).

    days_back_end may be negative to reach a day *after* date.
    """
    start_date = date - dt.timedelta(days=days_back_start)
    end_date = date - dt.timedelta(days=days_back_end)
    start = pd.Timestamp(start_date, tz=LOCAL_TZ)
    end = pd.Timestamp(end_date, tz=LOCAL_TZ)
    return start, end


def expected_timestamp_count(
    start: pd.Timestamp, end: pd.Timestamp, resolution_minutes: int
) -> int:
    """Number of resolution_minutes-spaced timestamps in [start, end).

    Safe to call with any tz-aware start/end (e.g. from local_day_bounds or
    local_window_bounds): Timestamp subtraction always yields the actual
    elapsed real time regardless of how each endpoint's tzinfo was
    constructed, so a DST-transition window's 23h/25h falls out automatically.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be tz-aware")
    elapsed = end - start
    return int(elapsed / pd.Timedelta(minutes=resolution_minutes))
