"""Similar-Day reference-day selection and hour-mapping, ported from
``sbl-energy-forecast``'s ``arena/load_patch.py`` (commit ``b21b55a``,
2026-09-26), for the `price_similar_day` feature
(docs/seasonal_window_similar_day_spec.md, section 2.1).

``LoadPatchReference``/``_last_sunday_before``/``_candidate_for``/
``choose_reference_day``/``_local_hour_index``/``_canonical_24h``/
``build_patched_load`` are unchanged content-wise from the source -- the
only code difference is the ``local_day_bounds`` import, which comes from
this repo's own ``market_time`` module instead of Repo 2's ``ops.windows``
(confirmed byte-identical implementations in both repos, so this is a pure
drop-in). ``build_patched_load`` is generic over any hourly Series despite
its load-specific name; it is reused here unchanged for a price series.

``build_similar_day_feature``, ``is_holiday_de``, and ``is_complete_price_day``
below are Repo-1-specific: they are not in Repo 2, and wire the ported
functions into this repo's Availability/Feature contract and data.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import holidays
import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ, local_day_bounds
from .availability import Availability, Feature, knowledge_time

MAX_LOAD_PATCH_WEEKS_BACK: Final[int] = 4  # not measured, see spec 2.2 (sbl-energy-forecast)

_DST_HOUR = 2  # the only local hour ever duplicated/missing on a DST transition day


@dataclass(frozen=True)
class LoadPatchReference:
    """The reference day the rule settled on, and the escalation trail
    that led there."""

    reference_day: dt.date
    weeks_back: int  # 0 = first reference of the rule, 1 = one week further, ...
    skipped: tuple[tuple[dt.date, str], ...]  # (day, "holiday" | "incomplete")


def _last_sunday_before(day: dt.date) -> dt.date:
    """Most recent Sunday strictly before ``day`` -- never ``day`` itself,
    even if ``day`` happens to be a Sunday."""
    days_back = (day.weekday() - 6) % 7
    if days_back == 0:
        days_back = 7
    return day - dt.timedelta(days=days_back)


def _candidate_for(target_day: dt.date, *, is_holiday_target: bool, weeks_back: int) -> dt.date:
    """The single candidate day for one escalation step. ``weeks_back`` is
    the step index, not literally "weeks before target_day" for the
    Tue-Fri chain's own first step."""
    if is_holiday_target:
        return _last_sunday_before(target_day) - dt.timedelta(weeks=weeks_back)

    weekday = target_day.weekday()  # Mon=0 .. Sun=6
    if weekday in (5, 6, 0):  # Sat, Sun, Mon: D-7, D-14, ...
        return target_day - dt.timedelta(weeks=weeks_back + 1)

    # Tue-Fri: D-1, then D-7, D-14, ...
    if weeks_back == 0:
        return target_day - dt.timedelta(days=1)
    return target_day - dt.timedelta(weeks=weeks_back)


def choose_reference_day(
    target_day: dt.date,
    *,
    is_holiday: Callable[[dt.date], bool],
    is_complete: Callable[[dt.date], bool],
) -> LoadPatchReference | None:
    """Tue-Fri D-1, then D-7, D-14 ...; Mon/Sat/Sun D-7, D-14 ...; holiday
    target: last Sunday, then the Sunday before ... A holiday reference is
    skipped unless it is a Sunday; an incomplete reference is always
    skipped. None if nothing usable within MAX_LOAD_PATCH_WEEKS_BACK."""
    is_holiday_target = is_holiday(target_day)
    skipped: list[tuple[dt.date, str]] = []

    for weeks_back in range(MAX_LOAD_PATCH_WEEKS_BACK + 1):
        day = _candidate_for(target_day, is_holiday_target=is_holiday_target, weeks_back=weeks_back)

        if is_holiday(day) and day.weekday() != 6:  # a Sunday is always usable, even if a holiday
            skipped.append((day, "holiday"))
            continue
        if not is_complete(day):
            skipped.append((day, "incomplete"))
            continue

        return LoadPatchReference(reference_day=day, weeks_back=weeks_back, skipped=tuple(skipped))

    return None


def _local_hour_index(day: dt.date) -> pd.DatetimeIndex:
    start, end = local_day_bounds(day)
    return pd.date_range(start, end, freq="h", inclusive="left")


def _canonical_24h(local_index: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Normalises one local day's values (23, 24, or 25 of them) onto a
    fixed 24-slot array indexed by local hour-of-day, resolving the
    reference day's own DST oddity -- the only local hour ever affected is
    2 o'clock."""
    local_hours = local_index.hour.to_numpy()
    canonical = np.empty(24, dtype=float)
    for hour in range(24):
        positions = np.flatnonzero(local_hours == hour)
        if hour == _DST_HOUR and len(positions) == 0:
            # Reference is a 23h day: the missing 2 o'clock value comes from 1 o'clock.
            fallback = np.flatnonzero(local_hours == _DST_HOUR - 1)
            canonical[hour] = values[fallback[0]]
        else:
            # Normal day: exactly one match. Reference is a 25h day: two matches
            # at hour 2 (the repeated wall-clock hour) -- the first one wins.
            canonical[hour] = values[positions[0]]
    return canonical


def build_patched_load(
    reference_load: pd.Series,
    reference_day: dt.date,
    target_day: dt.date,
) -> pd.Series:
    """Map the reference day's hourly series onto the target day by local
    wall-clock hour. Output has exactly the target day's number of local
    hours. Generic over any hourly Series (name unchanged from the source;
    the logic is not load-specific)."""
    ref_index = _local_hour_index(reference_day)
    ref_values = reference_load.reindex(ref_index.tz_convert("UTC")).to_numpy()
    canonical = _canonical_24h(ref_index, ref_values)

    target_index = _local_hour_index(target_day)
    target_hours = target_index.hour.to_numpy()
    out_values = canonical[target_hours]

    return pd.Series(out_values, index=target_index.tz_convert("UTC"), name=reference_load.name)


def build_similar_day_feature(
    price: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    is_holiday: Callable[[dt.date], bool],
    is_complete: Callable[[dt.date], bool],
) -> Feature:
    """``price_similar_day``: for each local delivery day in
    ``target_index``, the feiertagsbereinigt reference day's price, mapped
    onto the target day by local wall-clock hour (spec section 2.1).

    One ``choose_reference_day`` call per local delivery day, not per hour.
    Days where no reference is found within
    ``MAX_LOAD_PATCH_WEEKS_BACK`` get NaN for all their hours (LightGBM
    handles NaN).

    Knowledge time = DA_FIXED knowledge time of the reference day's own
    index -- the reference day is always strictly before the target day by
    construction of ``choose_reference_day``, so this is always well before
    any target day's gate closure (the existing ``price_lag_24h`` already
    proves the closest case, Tue-Fri's D-1 reference, passes).
    """
    local_days = pd.DatetimeIndex(target_index).tz_convert(LOCAL_TZ).normalize()
    unique_days = pd.DatetimeIndex(local_days.unique()).sort_values()

    value_parts: list[pd.Series] = []
    kt_parts: list[pd.Series] = []
    for day_ts in unique_days:
        day = day_ts.date()
        ref = choose_reference_day(day, is_holiday=is_holiday, is_complete=is_complete)
        day_target_index = target_index[local_days == day_ts]
        if ref is None:
            # No reference found: NaN carries no information, so it is
            # always known (DETERMINISTIC sentinel) -- never a leakage risk.
            value_parts.append(pd.Series(np.nan, index=day_target_index))
            kt_parts.append(
                pd.Series(
                    knowledge_time(Availability.DETERMINISTIC, day_target_index),
                    index=day_target_index,
                )
            )
            continue
        patched = build_patched_load(price, ref.reference_day, day)
        value_parts.append(patched.reindex(day_target_index))
        # Knowledge time comes from the REFERENCE day's own source hours,
        # not the target day's -- take the max (latest, safe) DA_FIXED
        # instant over the reference day's own hours and apply it to every
        # target hour that day's value was mapped onto (same max-composition
        # principle as availability.combine()).
        ref_source_index = _local_hour_index(ref.reference_day).tz_convert("UTC")
        day_kt = knowledge_time(Availability.DA_FIXED, ref_source_index).max()
        kt_parts.append(pd.Series(day_kt, index=day_target_index))

    values = pd.concat(value_parts).reindex(target_index).rename("price_similar_day")
    kt = pd.concat(kt_parts).reindex(target_index)
    return Feature("price_similar_day", values, kt)


def is_holiday_de(day: dt.date) -> bool:
    """Nationwide-DE holiday check for a single date -- same source
    ``common/features/calendar.py``'s own ``is_holiday`` feature uses
    (``holidays.country_holidays("DE", ...)``), not the two additional
    regional holidays ``is_regional_holiday`` tracks separately.
    """
    return day in holidays.country_holidays("DE", years=[day.year])


def is_complete_price_day(price: pd.Series) -> Callable[[dt.date], bool]:
    """Returns an ``is_complete`` callable for ``choose_reference_day``: a
    day is complete if every one of its local hours (23/24/25 across DST,
    via ``_local_hour_index``) has a non-NaN value in ``price``.
    """

    def _is_complete(day: dt.date) -> bool:
        local_index = _local_hour_index(day).tz_convert("UTC")
        return bool(price.reindex(local_index).notna().all())

    return _is_complete
