import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.common.data.entsoe_client import (
    AREA_DE_LU,
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast,
)
from energy_price_forecast.common.data.quarterhourly import validate_delivery_day_slot_counts

logger = logging.getLogger(__name__)

_COMMODITY_COLUMNS = ["ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"]
# Originally 4 days = 96 hours (bridges weekends and the longest common holiday
# gap, Good Friday to Easter Monday). Raised to 14 days (336 hours) to match
# the live pipeline (sbl-energy-forecast), which measured real gaps up to 5
# days in the full commodity history and widened the margin further as
# headroom against a future outage -- a behaviour change, not a bug fix, kept
# as a parameter here (not hardcoded) so Project 1 can revert to the original
# 4-day value if the regression report shows this matters for its results.
_COMMODITY_FFILL_LIMIT_HOURS = 14 * 24


def time_limited_ffill(series: pd.Series, *, limit_hours: int) -> pd.Series:
    """Forward-fill honoring a real wall-clock time limit, not a row count.

    ``Series.ffill(limit=N)`` counts ROWS, which silently means a different
    real time span depending on the index's own resolution: day_ahead_price
    (and therefore this merged frame's own index) has been quarter-hourly
    since 2025-09-30, so a row-count limit tuned for an hourly index covers
    only a quarter of the intended real time span from that date onward.

    Correct regardless of the index's own resolution: every NaN is filled
    from the most recent real (non-NaN) value, then reverted to NaN wherever
    that value's own age exceeds ``limit_hours``.
    """
    filled = series.ffill()
    last_valid_time = series.index.to_series().where(series.notna()).ffill()
    age = series.index.to_series() - last_valid_time
    return filled.where(age <= pd.Timedelta(hours=limit_hours))


def _log_fetch(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.info("%s: 0 rows (empty)", name)
    else:
        logger.info("%s: %d rows (%s to %s)", name, len(df), df.index.min(), df.index.max())
    return df


def load_all_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
    *,
    commodity_ffill_limit_hours: int = _COMMODITY_FFILL_LIMIT_HOURS,
) -> pd.DataFrame:
    """Load and merge all data sources into a single hourly DataFrame.

    Calls all six ENTSO-E fetchers and both commodities fetchers, then merges
    them on the hourly UTC index using an outer join. The day-ahead price
    column is required: rows where the price is missing are dropped.

    Commodity prices (TTF gas, EUA CO2) come at daily granularity and are
    forward-filled with a real wall-clock time limit (``time_limited_ffill``,
    not a row count -- see that function). This bridges weekends and typical
    holiday gaps but leaves multi-day outages visible as NaN for downstream
    data quality analysis. This is the only resampling done by the loader;
    all other forward-fill or interpolation decisions are deferred to
    feature engineering (Sprint 2).

    Parameters
    ----------
    start : pd.Timestamp
        Start of the requested time range. Converted to UTC at entry.
    end : pd.Timestamp
        End of the requested time range (inclusive). Converted to UTC.
    area : str, default "DE_LU"
        ENTSO-E area code. Currently only DE_LU is fully supported; other
        areas would require neighbor-list adjustments in entsoe_client.py.
    commodity_ffill_limit_hours : int, default 336 (14 days)
        Forward-fill limit for the commodity columns, in real hours.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with UTC DatetimeIndex. Columns include the
        day-ahead price (target), all load and renewable forecasts/actuals,
        all generation types, all six neighbor scheduled exchanges and
        physical flows, plus the two commodity prices forward-filled to
        hourly. Columns missing from individual fetchers are not synthesised.
    """
    start_utc = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    end_utc = end.tz_convert("UTC") if end.tzinfo is not None else end.tz_localize("UTC")

    if start_utc >= end_utc:
        raise ValueError(f"start must be before end, got start={start_utc!r}, end={end_utc!r}")

    frames = [
        _log_fetch(
            "fetch_day_ahead_prices",
            fetch_day_ahead_prices(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_load",
            fetch_load(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_wind_solar_forecast",
            fetch_wind_solar_forecast(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_generation_by_type",
            fetch_generation_by_type(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_scheduled_exchanges",
            fetch_scheduled_exchanges(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_cross_border_flows",
            fetch_cross_border_flows(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_ttf_gas",
            fetch_ttf_gas(start_utc, end_utc),
        ),
        _log_fetch(
            "fetch_eua_co2",
            fetch_eua_co2(start_utc, end_utc),
        ),
    ]

    df = pd.concat(frames, axis=1, join="outer")

    # Forward-fill commodity columns only (daily → hourly granularity).
    # All other gaps remain as NaN so EDA can see real data-quality issues.
    for col in _COMMODITY_COLUMNS:
        if col in df.columns:
            df[col] = time_limited_ffill(df[col], limit_hours=commodity_ffill_limit_hours)

    # Rows without a target price are unusable for modelling; drop them early
    # so they don't distort predictor gap-visualisation in EDA.
    df = df[df["day_ahead_price"].notna()]

    dt_index = pd.DatetimeIndex(df.index)
    assert dt_index.tz is not None, "merged index must be timezone-aware"
    assert str(dt_index.tz) == "UTC", f"merged index timezone must be UTC, got {dt_index.tz!r}"

    logger.info("merged result: %d rows × %d columns", len(df), len(df.columns))

    return df


# ---------------------------------------------------------------------------
# Interim layer — normalised hourly grid
# ---------------------------------------------------------------------------

_INTERIM_PATH = Path("data/interim/hourly.parquet")

# Timestamp of the resolution break in the raw data (hourly → 15-min).
_BREAK_TS = pd.Timestamp("2025-09-30 22:00", tz="UTC")


def build_interim_hourly(
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path = _INTERIM_PATH,
    area: str = AREA_DE_LU,
    *,
    commodity_ffill_limit_hours: int = _COMMODITY_FFILL_LIMIT_HOURS,
) -> pd.DataFrame:
    """Fetch raw data, normalise to an hourly grid, and persist as Parquet.

    The resulting DataFrame is written to `path` and also returned so the
    caller can inspect it without a second read. If the parent directory does
    not exist it is created automatically.

    After writing, logs daily mean load for the two days on either side of
    the resolution break as a continuity sanity check.
    """
    from energy_price_forecast.common.data.normalize import to_hourly

    raw = load_all_data(start, end, area, commodity_ffill_limit_hours=commodity_ffill_limit_hours)
    hourly = to_hourly(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(path)
    logger.info("wrote %d hourly rows to %s", len(hourly), path)

    # Continuity sanity: daily mean load around the resolution break.
    window_start = _BREAK_TS - pd.Timedelta("2D")
    window_end = _BREAK_TS + pd.Timedelta("2D")
    if "load_actual" in hourly.columns:
        snippet = hourly.loc[window_start:window_end, "load_actual"]
        daily = snippet.resample("D").mean()
        logger.info(
            "load_actual daily mean around resolution break (MW):\n%s",
            daily.to_string(),
        )

    return hourly


def load_interim_hourly(path: Path = _INTERIM_PATH) -> pd.DataFrame:
    """Load the normalised hourly DataFrame from Parquet.

    Raises FileNotFoundError if the file does not exist (run
    build_interim_hourly first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Interim Parquet not found at {path!r}. Run build_interim_hourly() to generate it."
        )
    df = pd.read_parquet(path)
    logger.info("loaded %d hourly rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Processed layer — engineered feature matrix
# ---------------------------------------------------------------------------

_FEATURES_PATH = Path("data/processed/features.parquet")


def load_processed_features(
    path: str | Path = _FEATURES_PATH,
) -> pd.DataFrame:
    """Load the engineered feature matrix (X only, no target) from step 2.3.

    The matrix carries a UTC DatetimeIndex on a regular hourly grid, already
    warm-up-trimmed. NaN is expected only in the EUA-CO2 region (pre-Oct-2021,
    by design D4); model-side imputation happens in the LassoForecaster.

    Fails loudly with a pointer to scripts/model_risk_regimes/build_features.py if the file is
    missing -- never silently return an empty frame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"processed features not found at {path} -- run scripts/model_risk_regimes/build_features.py first"
        )
    df = pd.read_parquet(path)
    logger.info("loaded %d feature rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Interim layer — native quarter-hourly day-ahead prices
# ---------------------------------------------------------------------------

_QUARTERHOURLY_PATH = Path("data/interim/quarterhourly_prices.parquet")


def load_interim_quarterhourly(path: Path = _QUARTERHOURLY_PATH) -> pd.DataFrame:
    """Load the native (unresampled) quarter-hourly day-ahead price series.

    Raises FileNotFoundError if the file does not exist (run
    data.quarterhourly.build_quarterhourly_prices() first). Validates every
    local delivery day's slot count before returning -- see
    quarterhourly.validate_delivery_day_slot_counts; a day with a gap raises
    rather than being silently skipped or interpolated.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Quarter-hourly Parquet not found at {path!r}. "
            "Run data.quarterhourly.build_quarterhourly_prices() to generate it."
        )
    df = pd.read_parquet(path)
    validate_delivery_day_slot_counts(pd.DatetimeIndex(df.index))
    logger.info("loaded %d quarter-hourly rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Interim layer — raw ECMWF IFS weather artefact
# ---------------------------------------------------------------------------

_WEATHER_PATH = Path("data/interim/weather_ifs_run00.parquet")


def load_interim_weather(path: Path = _WEATHER_PATH) -> pd.DataFrame:
    """Load the raw weather artefact (scripts/model_renewables/build_weather_artefact.py).

    Raises FileNotFoundError if the file does not exist. Validates the
    artefact's shape -- MultiIndex level names, column count, dtype -- before
    returning, rather than passing a structurally wrong frame downstream.
    Does not re-check the exact column set against
    weather_grid.expected_columns(): that's already enforced file-by-file
    when build_weather_artefact.py reads each cached run
    (data/_weather_cache.py::read_cached_run), so a merged artefact with the
    wrong shape here means something outside that pipeline touched the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Weather Parquet not found at {path!r}. "
            "Run scripts/model_renewables/build_weather_artefact.py to generate it."
        )
    df = pd.read_parquet(path)

    index_names = list(df.index.names)
    if index_names != ["run_init_utc", "valid_time_utc"]:
        raise ValueError(
            f"weather artefact at {path!r} has index names {index_names!r}, "
            "expected ['run_init_utc', 'valid_time_utc']"
        )
    if len(df.columns) != 162:
        raise ValueError(
            f"weather artefact at {path!r} has {len(df.columns)} columns, expected 162"
        )
    bad_dtypes = {col: dtype for col, dtype in df.dtypes.items() if dtype != "float32"}
    if bad_dtypes:
        raise ValueError(f"weather artefact at {path!r} has non-float32 columns: {bad_dtypes}")

    logger.info("loaded %d weather rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Processed layer — renewables reconstruction artefact
# ---------------------------------------------------------------------------

_RENEWABLES_PREDICTIONS_PATH = Path("data/processed/renewables_forecast_rolling365_l2.parquet")


def load_renewables_predictions(path: Path = _RENEWABLES_PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the headline-variant renewables reconstruction artefact
    (model_renewables/evaluation/renewables_walkforward.run_renewables_backtest's
    output, written by scripts/model_renewables/run_renewables_model.py --store-new).

    Raises FileNotFoundError if the file does not exist. Only the headline
    variant is a valid feature source -- the two comparison variants
    (expanding_l2, rolling365_quantile) are for the DM-test comparison, not
    for feeding the price model.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Renewables predictions Parquet not found at {path!r}. "
            "Run scripts/model_renewables/run_renewables_model.py --store-new to generate it."
        )
    df = pd.read_parquet(path)

    index_names = list(df.index.names)
    if index_names != ["run_init_utc", "valid_time_utc"]:
        raise ValueError(
            f"renewables predictions artefact at {path!r} has index names {index_names!r}, "
            "expected ['run_init_utc', 'valid_time_utc']"
        )

    logger.info("loaded %d renewables prediction rows from %s", len(df), path)
    return df
