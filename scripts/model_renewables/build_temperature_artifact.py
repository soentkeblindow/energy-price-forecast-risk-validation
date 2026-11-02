"""Temperature artifact for `hourly_model_v2.1` (Sprint 7 Step 7.4a): three
region-mean 2m-temperature columns (temperature_north/central/south),
reconstructed from the existing ECMWF IFS weather archive the same way the
renewables wind/solar features already are.

Reuses model_renewables.features.renewables_forecast.assign_runs directly:
same D-1 00Z run assignment (common.data.weather_client.run_init_for_target_day),
same time alignment, same whole-day exclusion rule on missing/incomplete
weather coverage -- no new alignment or NaN logic here, just a region mean
on top of assign_runs' already time-aligned per-point temperature_2m
columns. Region membership is weather_grid.GRID_POINTS' own `region` field
(north=7, central=3, south=3 onshore + 2 solar-only=5 land points, offshore
excluded).

    uv run python scripts/model_renewables/build_temperature_artifact.py

Full rebuild every run (no --store-new merge machinery -- this file is
tiny, three float32 columns, cheap to regenerate wholesale), written to
data/processed/temperature_regions.parquet by default. Read by
scripts/model_risk_regimes/run_price_model.py's --include-extra-features
temperature flag.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_weather
from energy_price_forecast.common.data.weather_grid import GRID_POINTS, GridPoint
from energy_price_forecast.model_renewables.features.renewables_forecast import assign_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_WEATHER_PATH = Path("data/interim/weather_ifs_run00.parquet")
_OUT_PATH = Path("data/processed/temperature_regions.parquet")

_NORTH_POINTS: tuple[GridPoint, ...] = tuple(p for p in GRID_POINTS if p.region == "north")
_CENTRAL_POINTS: tuple[GridPoint, ...] = tuple(p for p in GRID_POINTS if p.region == "central")
_SOUTH_LAND_POINTS: tuple[GridPoint, ...] = tuple(
    p for p in GRID_POINTS if p.region == "south" and p.kind != "offshore"
)
assert len(_NORTH_POINTS) == 7
assert len(_CENTRAL_POINTS) == 3
assert len(_SOUTH_LAND_POINTS) == 5

_REGION_POINTS: dict[str, tuple[GridPoint, ...]] = {
    "north": _NORTH_POINTS,
    "central": _CENTRAL_POINTS,
    "south": _SOUTH_LAND_POINTS,
}
TEMPERATURE_COLUMNS: tuple[str, ...] = tuple(f"temperature_{region}" for region in _REGION_POINTS)


def _default_day_range(weather: pd.DataFrame) -> tuple[dt.date, dt.date]:
    """First/last local delivery day with a D-1 00Z run in ``weather`` --
    the inverse of ``run_init_for_target_day`` (day = run_init + 1 day).
    """
    run_init = weather.index.get_level_values("run_init_utc")
    start_day = (run_init.min() + pd.Timedelta(days=1)).date()
    end_day = (run_init.max() + pd.Timedelta(days=1)).date()
    return start_day, end_day


def region_temperature_means(aligned: pd.DataFrame) -> pd.DataFrame:
    """Mean ``temperature_2m`` over each region's land grid points, from
    ``assign_runs``' already time-aligned per-point output.
    """
    out: dict[str, object] = {}
    for region, points in _REGION_POINTS.items():
        cols = [f"{p.point_id}__temperature_2m" for p in points]
        out[f"temperature_{region}"] = aligned[cols].to_numpy().mean(axis=1)
    return pd.DataFrame(out, index=aligned.index)


def build_temperature_artifact(
    target_days: Sequence[dt.date], weather: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[dt.date, ...]]:
    """Returns (temperature, excluded_days): ``temperature`` is indexed by
    (run_init_utc, valid_time_utc), float32, columns
    ``temperature_north/central/south``; ``excluded_days`` lists every day
    whose weather run was missing or incomplete (``assign_runs``' own rule).
    """
    aligned, excluded = assign_runs(target_days, weather)
    if aligned.empty:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="float32") for c in TEMPERATURE_COLUMNS}, index=aligned.index
        )
        return empty, excluded
    return region_temperature_means(aligned).astype("float32"), excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-path", type=Path, default=_WEATHER_PATH)
    parser.add_argument("--out-path", type=Path, default=_OUT_PATH)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    args = parser.parse_args()

    weather = load_interim_weather(args.weather_path)
    default_start, default_end = _default_day_range(weather)
    start = args.start or default_start
    end = args.end or default_end
    n_days = (end - start).days + 1
    target_days = [start + dt.timedelta(days=i) for i in range(n_days)]

    logger.info(
        "building temperature artifact for %s..%s (%d candidate day(s))",
        start,
        end,
        len(target_days),
    )

    temperature, excluded = build_temperature_artifact(target_days, weather)
    if excluded:
        logger.warning(
            "%d day(s) excluded (missing/incomplete weather run): %s", len(excluded), excluded
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    temperature.to_parquet(args.out_path)
    logger.info("wrote %d rows to %s", len(temperature), args.out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
