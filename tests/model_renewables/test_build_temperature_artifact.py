import datetime as dt

import numpy as np
import pandas as pd

from energy_price_forecast.common.data.weather_client import run_init_for_target_day
from energy_price_forecast.common.data.weather_grid import GRID_POINTS, expected_columns
from energy_price_forecast.model_renewables.features.renewables_forecast import (
    _target_hours_for_day,
)
from scripts.model_renewables.build_temperature_artifact import (
    _CENTRAL_POINTS,
    _NORTH_POINTS,
    _SOUTH_LAND_POINTS,
    TEMPERATURE_COLUMNS,
    build_temperature_artifact,
)


def _hours_needed_for_day(day: dt.date) -> pd.DatetimeIndex:
    hours = _target_hours_for_day(day)
    return pd.DatetimeIndex(hours.union(hours + pd.Timedelta(hours=1)))


def _weather_for_days(days: list[dt.date], value: float = 0.0) -> pd.DataFrame:
    cols = expected_columns()
    frames = []
    for day in days:
        run_init = run_init_for_target_day(day)
        valid_times = _hours_needed_for_day(day)
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([run_init] * len(valid_times), tz="UTC"), valid_times],
            names=["run_init_utc", "valid_time_utc"],
        )
        frames.append(pd.DataFrame({c: value for c in cols}, index=idx, dtype="float64"))
    return pd.concat(frames)


def test_region_point_counts_are_7_3_5() -> None:
    assert len(_NORTH_POINTS) == 7
    assert len(_CENTRAL_POINTS) == 3
    assert len(_SOUTH_LAND_POINTS) == 5
    assert all(p.kind != "offshore" for p in _SOUTH_LAND_POINTS)
    assert {p.region for p in _NORTH_POINTS} == {"north"}
    assert {p.region for p in _CENTRAL_POINTS} == {"central"}
    assert {p.region for p in _SOUTH_LAND_POINTS} == {"south"}


def test_region_means_on_synthetic_grid() -> None:
    day = dt.date(2025, 6, 15)
    weather = _weather_for_days([day])

    # Distinct, constant-over-time value per point (same value at H and H+1,
    # so the instantaneous H/H+1 averaging _align_point_variables applies
    # doesn't change it -- isolates the region-mean computation under test).
    point_value = {p.point_id: float(i) for i, p in enumerate(GRID_POINTS)}
    for point_id, value in point_value.items():
        weather[f"{point_id}__temperature_2m"] = value

    temperature, excluded = build_temperature_artifact([day], weather)
    assert excluded == ()
    assert list(temperature.columns) == list(TEMPERATURE_COLUMNS)

    expected_north = np.mean([point_value[p.point_id] for p in _NORTH_POINTS])
    expected_central = np.mean([point_value[p.point_id] for p in _CENTRAL_POINTS])
    expected_south = np.mean([point_value[p.point_id] for p in _SOUTH_LAND_POINTS])

    np.testing.assert_allclose(temperature["temperature_north"].to_numpy(), expected_north)
    np.testing.assert_allclose(temperature["temperature_central"].to_numpy(), expected_central)
    np.testing.assert_allclose(temperature["temperature_south"].to_numpy(), expected_south)


def test_incomplete_run_excludes_whole_day() -> None:
    good_day = dt.date(2025, 6, 15)
    bad_day = dt.date(2025, 6, 16)
    weather = _weather_for_days([good_day, bad_day])

    # Knock out one point/hour on bad_day only -- assign_runs' own rule
    # (common/model_renewables/features/renewables_forecast.py) must exclude
    # the *entire* delivery day, not just that one hour. Must match on both
    # index levels: good_day's and bad_day's needed-hour ranges share their
    # boundary valid_time_utc (good_day's "+1h" extension point is bad_day's
    # first target hour), so matching on valid_time_utc alone would hit a
    # row in *both* days' (run_init_utc, valid_time_utc) frames.
    bad_run_init = run_init_for_target_day(bad_day)
    bad_first_hour = _hours_needed_for_day(bad_day)[0]
    weather.loc[(bad_run_init, bad_first_hour), "sh_west__temperature_2m"] = np.nan

    temperature, excluded = build_temperature_artifact([good_day, bad_day], weather)
    assert excluded == (bad_day,)

    # Day = run_init + 1 day (run_init_for_target_day's inverse) -- avoids
    # any local-midnight/DST ambiguity a UTC valid_time_utc -> local-date
    # conversion would risk near day boundaries.
    present_days = {
        (run_init + pd.Timedelta(days=1)).date()
        for run_init in temperature.index.get_level_values("run_init_utc").unique()
    }
    assert bad_day not in present_days
    assert good_day in present_days
