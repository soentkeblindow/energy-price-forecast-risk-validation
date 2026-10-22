"""Unit tests for the calendar-window helpers in common/market_time.py
(local_day_bounds, local_window_bounds, expected_timestamp_count).

Renamed from Repo 2's test_ops_windows.py: those three functions moved from
ops/windows.py into common/market_time.py (Sprint 7.0, step 4.3), next to
the pre-existing gate_closure_for_index / _local_day. next_delivery_day
stays in Repo 2 only (it is live-submission logic, not a pure calendar
helper), so its tests are not part of this copy.

The consistency test at the bottom is this file's actual purpose: it is the
only thing keeping local_day_bounds and market_time's private _local_day
from silently drifting apart now that both live in the same module.
"""

import datetime as dt

import pandas as pd
import pytest

from energy_price_forecast.common.market_time import (
    _local_day,
    expected_timestamp_count,
    local_day_bounds,
    local_window_bounds,
    month_start_local,
)

# ---------------------------------------------------------------------------
# local_day_bounds / expected_timestamp_count
# ---------------------------------------------------------------------------


def test_local_day_bounds_normal_day() -> None:
    start, end = local_day_bounds(dt.date(2026, 8, 20))
    assert start.isoformat() == "2026-08-20T00:00:00+02:00"
    assert end.isoformat() == "2026-08-21T00:00:00+02:00"
    assert expected_timestamp_count(start, end, 15) == 96
    assert expected_timestamp_count(start, end, 60) == 24


def test_local_day_bounds_spring_forward_dst() -> None:
    # 2026-03-29: 02:00 -> 03:00 CEST, 23h day.
    start, end = local_day_bounds(dt.date(2026, 3, 29))
    assert start.isoformat() == "2026-03-29T00:00:00+01:00"
    assert end.isoformat() == "2026-03-30T00:00:00+02:00"
    assert expected_timestamp_count(start, end, 15) == 92
    assert expected_timestamp_count(start, end, 60) == 23


def test_local_day_bounds_fall_back_dst() -> None:
    # 2026-10-25: 03:00 -> 02:00 CET, 25h day.
    start, end = local_day_bounds(dt.date(2026, 10, 25))
    assert start.isoformat() == "2026-10-25T00:00:00+02:00"
    assert end.isoformat() == "2026-10-26T00:00:00+01:00"
    assert expected_timestamp_count(start, end, 15) == 100
    assert expected_timestamp_count(start, end, 60) == 25


def test_expected_timestamp_count_rejects_naive_bounds() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        expected_timestamp_count(pd.Timestamp("2026-08-20"), pd.Timestamp("2026-08-21"), 60)


# ---------------------------------------------------------------------------
# local_window_bounds
# ---------------------------------------------------------------------------


def test_local_window_bounds_matches_two_independent_day_bounds() -> None:
    date = dt.date(2026, 8, 20)
    start, end = local_window_bounds(date, days_back_start=2, days_back_end=1)
    assert start == local_day_bounds(dt.date(2026, 8, 18))[0]
    assert end == local_day_bounds(dt.date(2026, 8, 19))[0]


def test_local_window_bounds_supports_negative_days_back_end_for_forward_reach() -> None:
    # A forward-reaching window: D-1 to D+1.
    date = dt.date(2026, 8, 20)
    start, end = local_window_bounds(date, days_back_start=1, days_back_end=-1)
    assert start == local_day_bounds(dt.date(2026, 8, 19))[0]
    assert end == local_day_bounds(dt.date(2026, 8, 21))[0]


def test_local_window_bounds_90_days_across_dst_lands_on_local_midnight() -> None:
    # Window ending 2026-11-01 (CET, after the fall-back transition), starting
    # 2026-08-03 (CEST, before it): the window genuinely straddles the
    # transition, so naive Timestamp(days=90) subtraction from `end` is off
    # by an hour; per-endpoint localization is not.
    date = dt.date(2026, 11, 1)
    start, end = local_window_bounds(date, days_back_start=90, days_back_end=0)

    assert start == pd.Timestamp(dt.date(2026, 8, 3), tz="Europe/Berlin")
    assert end == pd.Timestamp(date, tz="Europe/Berlin")

    naive_start = end - pd.Timedelta(days=90)
    assert naive_start != start
    assert naive_start.isoformat() == "2026-08-03T01:00:00+02:00"  # off by 1h, not local midnight


# ---------------------------------------------------------------------------
# month_start_local (Sprint 7 Step 7.5, Teil 0: beta re-estimation anchor)
# ---------------------------------------------------------------------------


def test_month_start_local_normal_month() -> None:
    anchor = month_start_local(pd.Timestamp("2026-08-20", tz="Europe/Berlin"))
    assert anchor.isoformat() == "2026-08-01T00:00:00+02:00"


def test_month_start_local_same_for_every_day_in_month() -> None:
    first = month_start_local(pd.Timestamp("2026-10-01", tz="Europe/Berlin"))
    mid = month_start_local(pd.Timestamp("2026-10-17", tz="Europe/Berlin"))
    last = month_start_local(pd.Timestamp("2026-10-31", tz="Europe/Berlin"))
    assert first == mid == last


def test_month_start_local_across_fall_back_dst() -> None:
    # The fall-back transition (2026-10-25) falls *inside* October, not on a
    # month boundary: D=2026-10-28 is already CET (+01:00), but its own
    # month's anchor (2026-10-01) is still CEST (+02:00) -- the anchor's
    # offset must reflect the 1st's own position, not D's.
    d = pd.Timestamp("2026-10-28", tz="Europe/Berlin")
    assert d.isoformat() == "2026-10-28T00:00:00+01:00"
    anchor = month_start_local(d)
    assert anchor.isoformat() == "2026-10-01T00:00:00+02:00"
    assert anchor.tz_convert("UTC").isoformat() == "2026-09-30T22:00:00+00:00"


def test_month_start_local_anchor_offset_differs_from_days_own_offset_across_spring_forward() -> (
    None
):
    # D=2026-03-30 is CEST (+02:00, after the 2026-03-29 spring-forward); its
    # month anchor (2026-03-01) is still CET (+01:00) -- confirms the anchor
    # is computed from its own calendar position, not by reusing D's offset
    # (which a tz-aware Timestamp.replace(day=1) would do).
    d = pd.Timestamp("2026-03-30", tz="Europe/Berlin")
    anchor = month_start_local(d)
    assert anchor.isoformat() == "2026-03-01T00:00:00+01:00"


# ---------------------------------------------------------------------------
# Consistency against market_time._local_day (the point of this file)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date",
    [
        dt.date(2026, 8, 20),  # normal day
        dt.date(2026, 3, 29),  # spring-forward
        dt.date(2026, 10, 25),  # fall-back
    ],
)
def test_local_day_bounds_agrees_with_market_time_local_day(date: dt.date) -> None:
    """local_day_bounds' local midnight for `date` must equal what
    _local_day floors any UTC timestamp within that day down to.

    If this ever fails, the two helpers disagree on what "local day" means
    and any window arithmetic built on either is no longer trustworthy.
    """
    expected_start, _ = local_day_bounds(date)

    # Any UTC instant within the local day must floor back to expected_start,
    # not just the boundary itself -- exercise midnight and mid-day.
    for offset_hours in (0, 12):
        probe_utc = expected_start.tz_convert("UTC") + pd.Timedelta(hours=offset_hours)
        floored = _local_day(pd.DatetimeIndex([probe_utc]))[0]
        assert floored == expected_start
