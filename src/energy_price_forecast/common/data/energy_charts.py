"""Energy-Charts client.

Covers two Energy-Charts endpoints this project uses:

- ``public_power_forecast`` (load/solar/wind_onshore/wind_offshore day-ahead
  forecasts): ``fetch_series_range`` pulls the four series over an
  arbitrary past date range and raises (fail-fast) on any non-200 status,
  since past history has no "not yet published" outcome. ``probe_series``
  measures, once per call, whether a day-ahead forecast series is fully
  available for a given delivery day at the moment of the probe -- never
  writes to the store, never feeds a feature. Every outcome except a
  genuine API-behavior anomaly is logged and treated as a successful
  measurement: a forecast that is not yet published, a rate limit, or a
  transport error are exactly what this probe exists to observe, not
  failures of the probe itself.

  Both raise ``EnergyChartsProbeAnomalyError`` for the same structural
  anomaly: a response claiming ``deprecated=true``, or one that echoes back
  a different ``production_type``/``forecast_type`` than requested.

- ``/price`` (day-ahead price, ``bzn=DE-LU``): ``fetch_price_range``, used
  for the historical bulk pull. Raises ``EnergyChartsPriceAnomalyError`` on
  ``deprecated=true`` or an unexpected ``unit`` -- this endpoint's own,
  smaller echo surface (no production_type/forecast_type field to compare
  against the request).

Both endpoints share ``EnergyChartsRateLimitedError`` (HTTP 429, honour
``Retry-After``) and ``merge_existing_with_fresh`` (existing-wins merge with
a byte-identical post-merge verification of every pre-existing non-NaN
cell).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from energy_price_forecast.common.market_time import (
    expected_timestamp_count,
    gate_closure_for_index,
    local_day_bounds,
)

_ENDPOINT: Final = "https://api.energy-charts.info/public_power_forecast"
_COUNTRY: Final = "de"
_FORECAST_TYPE: Final = "day-ahead"
_TIMEOUT_S: Final = 30.0
_RESOLUTION_MINUTES: Final = 15

# The four series a residual-load-forecast backup would need: residual_load
# = load - wind_onshore - wind_offshore - solar. Order matches load first --
# the series expected to arrive on time.
SERIES: Final[tuple[str, ...]] = ("load", "solar", "wind_onshore", "wind_offshore")

_PROBE_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "probe_timestamp_utc",
    "run_id",
    "target_day",
    "minutes_to_gate_closure",
    "production_type",
    "forecast_type",
    "echoed_production_type",
    "echoed_forecast_type",
    "n_values",
    "coverage_ratio",
    "first_ts",
    "last_ts",
    "http_status",
    "error_kind",
    "deprecated",
)


class EnergyChartsProbeAnomalyError(RuntimeError):
    """A genuine API-behavior anomaly, not a "not yet published" result:
    ``deprecated=true``, or the response echoes a different production_type
    / forecast_type than requested."""


class EnergyChartsRateLimitedError(RuntimeError):
    """The endpoint returned HTTP 429. ``retry_after_s`` is the server's own
    ``Retry-After`` value -- honour the header, never assume a fixed wait,
    falling back to ``_DEFAULT_RETRY_AFTER_S`` only if the header is absent
    or unparseable. Callers of ``fetch_series_range`` decide whether and how
    long to wait; this module never sleeps on its own."""

    def __init__(self, retry_after_s: float) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(f"rate limited by {_ENDPOINT}, Retry-After={retry_after_s}s")


_DEFAULT_RETRY_AFTER_S: Final = 30.0


def _parse_retry_after(response: requests.Response) -> float:
    header = response.headers.get("Retry-After")
    if header is None:
        return _DEFAULT_RETRY_AFTER_S
    try:
        return float(header)
    except ValueError:
        return _DEFAULT_RETRY_AFTER_S


def fetch_series_range(production_type: str, start: dt.date, end: dt.date) -> pd.Series:
    """Raw day-ahead forecast values for ``production_type`` over the local
    calendar-date range [start, end] (both ends inclusive, per the
    endpoint's own daily-format start/end convention), as a UTC-indexed
    ``pd.Series`` -- ``None`` values kept as NaN, not dropped or clipped to
    any expected grid. Grid measurement is the caller's job, on the raw
    data this returns.

    Raises ``EnergyChartsRateLimitedError`` on HTTP 429 (caller decides how
    long to wait), ``requests.HTTPError`` on any other non-200 status
    (fail-fast: unlike the probe, a gap in already-past history is not a
    valid "not yet published" outcome), and ``EnergyChartsProbeAnomalyError``
    on ``deprecated=true`` or a mismatched echoed production_type/
    forecast_type -- the same anomaly class ``probe_series`` raises, same
    endpoint, same failure semantics.
    """
    if production_type not in SERIES:
        raise ValueError(f"production_type must be one of {SERIES}, got {production_type!r}")

    params = {
        "country": _COUNTRY,
        "production_type": production_type,
        "forecast_type": _FORECAST_TYPE,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    response = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_S)

    if response.status_code == 429:
        raise EnergyChartsRateLimitedError(_parse_retry_after(response))
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    echoed_production_type = payload.get("production_type", "")
    echoed_forecast_type = payload.get("forecast_type", "")

    if payload.get("deprecated"):
        raise EnergyChartsProbeAnomalyError(
            f"{production_type} [{start}..{end}]: endpoint reports deprecated=true"
        )
    if echoed_production_type != production_type or echoed_forecast_type != _FORECAST_TYPE:
        raise EnergyChartsProbeAnomalyError(
            f"{production_type} [{start}..{end}]: response echoes production_type="
            f"{echoed_production_type!r} forecast_type={echoed_forecast_type!r}, requested "
            f"production_type={production_type!r} forecast_type={_FORECAST_TYPE!r}"
        )

    unix_seconds = payload.get("unix_seconds", [])
    forecast_values = payload.get("forecast_values", [])
    if len(unix_seconds) != len(forecast_values):
        raise EnergyChartsProbeAnomalyError(
            f"{production_type} [{start}..{end}]: unix_seconds length {len(unix_seconds)} != "
            f"forecast_values length {len(forecast_values)}"
        )

    index = pd.DatetimeIndex(pd.to_datetime(unix_seconds, unit="s", utc=True), name="timestamp")
    values = [float(v) if v is not None else float("nan") for v in forecast_values]
    return pd.Series(values, index=index, name=production_type, dtype="float64")


def _request_params(production_type: str, target_day: dt.date) -> dict[str, str]:
    return {
        "country": _COUNTRY,
        "production_type": production_type,
        "forecast_type": _FORECAST_TYPE,
        # Daily format, both ends the same day: the default (no start/end)
        # behaviour sometimes covers only today, not the delivery day this
        # probe actually needs to measure -- explicit start/end is the only
        # way to ask the exact question this probe exists to answer.
        "start": target_day.isoformat(),
        "end": target_day.isoformat(),
    }


def probe_series(
    production_type: str,
    target_day: dt.date,
    *,
    run_id: str,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    """One measurement of one series for one delivery day.

    ``as_of`` is the probe instant (UTC, tz-aware) -- both the timestamp
    recorded in the row and the instant ``minutes_to_gate_closure`` is
    measured from, so a caller pacing several calls can pass a fresh
    ``as_of`` per call rather than one shared instant for the whole run.

    Raises ``EnergyChartsProbeAnomalyError`` for a structural anomaly. Every
    other outcome -- 404 (not yet published), 429 (rate limited), 5xx / a
    transport error, or a non-JSON 200 body -- is recorded in the returned
    row and this returns normally.
    """
    if production_type not in SERIES:
        raise ValueError(f"production_type must be one of {SERIES}, got {production_type!r}")
    if as_of.tzinfo is None:
        raise ValueError(f"as_of must be tz-aware, got a naive timestamp: {as_of!r}")

    day_start, day_end = local_day_bounds(target_day)
    gate_closure_utc = gate_closure_for_index(pd.DatetimeIndex([day_start]))[0]
    expected_slots = expected_timestamp_count(
        day_start.tz_convert("UTC"), day_end.tz_convert("UTC"), _RESOLUTION_MINUTES
    )

    row: dict[str, object] = {
        "probe_timestamp_utc": as_of.isoformat(),
        "run_id": run_id,
        "target_day": target_day.isoformat(),
        "minutes_to_gate_closure": round((gate_closure_utc - as_of).total_seconds() / 60.0, 1),
        "production_type": production_type,
        "forecast_type": _FORECAST_TYPE,
        "echoed_production_type": "",
        "echoed_forecast_type": "",
        "n_values": 0,
        "coverage_ratio": 0.0,
        "first_ts": "",
        "last_ts": "",
        "http_status": "",
        "error_kind": "",
        "deprecated": "",
    }

    params = _request_params(production_type, target_day)
    try:
        response = requests.get(_ENDPOINT, params=params, timeout=_TIMEOUT_S)
    except (requests.ConnectionError, requests.Timeout) as exc:
        row["error_kind"] = f"transport_error:{exc.__class__.__name__}"
        return row

    row["http_status"] = str(response.status_code)

    if response.status_code == 429:
        row["error_kind"] = "rate_limited"
        return row
    if response.status_code == 404:
        # Not-yet-published and "outside the forecast horizon" return the
        # identical 404, no way to tell them apart from the response alone
        # -- both are the measurement itself, not a failure.
        row["error_kind"] = "not_available"
        return row
    if response.status_code >= 500:
        row["error_kind"] = "server_error"
        return row
    if response.status_code != 200:
        row["error_kind"] = f"http_{response.status_code}"
        return row

    try:
        payload: dict[str, Any] = response.json()
    except ValueError:
        row["error_kind"] = "malformed_200_body"
        return row

    row["echoed_production_type"] = payload.get("production_type", "")
    row["echoed_forecast_type"] = payload.get("forecast_type", "")
    row["deprecated"] = str(payload.get("deprecated", ""))

    if payload.get("deprecated"):
        raise EnergyChartsProbeAnomalyError(f"{production_type}: endpoint reports deprecated=true")
    if (
        row["echoed_production_type"] != production_type
        or row["echoed_forecast_type"] != _FORECAST_TYPE
    ):
        raise EnergyChartsProbeAnomalyError(
            f"{production_type}: response echoes production_type="
            f"{row['echoed_production_type']!r} forecast_type={row['echoed_forecast_type']!r}, "
            f"requested production_type={production_type!r} forecast_type={_FORECAST_TYPE!r}"
        )

    unix_seconds = payload.get("unix_seconds", [])
    forecast_values = payload.get("forecast_values", [])
    if len(unix_seconds) != len(forecast_values):
        raise EnergyChartsProbeAnomalyError(
            f"{production_type}: unix_seconds length {len(unix_seconds)} != "
            f"forecast_values length {len(forecast_values)}"
        )

    n_values = 0
    if unix_seconds:
        timestamps = pd.DatetimeIndex(pd.to_datetime(unix_seconds, unit="s", utc=True))
        day_start_utc = day_start.tz_convert("UTC")
        day_end_utc = day_end.tz_convert("UTC")
        in_range_mask = (timestamps >= day_start_utc) & (timestamps < day_end_utc)
        non_null_mask = pd.Series(forecast_values).notna().to_numpy()
        valid_mask = in_range_mask & non_null_mask

        n_values = int(valid_mask.sum())
        row["n_values"] = n_values
        row["coverage_ratio"] = round(n_values / expected_slots, 4) if expected_slots else 0.0
        if valid_mask.any():
            valid_timestamps = timestamps[valid_mask]
            row["first_ts"] = valid_timestamps.min().isoformat()
            row["last_ts"] = valid_timestamps.max().isoformat()

    # A response that answered (200, well-formed, correctly echoed) but
    # doesn't fully cover the target day is a genuine partial-coverage
    # measurement, not a code-level anomaly -- flagged in error_kind, not
    # raised.
    if n_values < expected_slots:
        row["error_kind"] = "partial_coverage"

    return row


def append_probe_row(row: dict[str, object], log_path: Path) -> None:
    """Append one row to the probe log, migrating the header if the column
    set has grown since the file was last written on disk.

    Columns come from ``row``'s own keys (their insertion order, which
    ``probe_series`` already builds to match ``_PROBE_LOG_COLUMNS``), NOT a
    hardcoded column list -- forcing a fixed column list here would silently
    drop any key the migration below exists to preserve.
    """
    frame = pd.DataFrame([row])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        frame.to_csv(log_path, index=False)
        return

    existing = pd.read_csv(log_path)
    if list(existing.columns) == list(frame.columns):
        frame.to_csv(log_path, mode="a", header=False, index=False)
        return

    union_columns = list(dict.fromkeys([*existing.columns, *frame.columns]))
    existing = existing.reindex(columns=union_columns)
    frame = frame.reindex(columns=union_columns)
    pd.concat([existing, frame], ignore_index=True).to_csv(log_path, index=False)


# ---------------------------------------------------------------------------
# ``/price`` -- day-ahead price, bzn=DE-LU
# ---------------------------------------------------------------------------

_PRICE_ENDPOINT: Final = "https://api.energy-charts.info/price"
_PRICE_BZN: Final = "DE-LU"
_PRICE_EXPECTED_UNIT: Final = "EUR / MWh"
PRICE_COLUMN: Final = "day_ahead_price_ec"


class EnergyChartsPriceAnomalyError(RuntimeError):
    """A genuine API-behavior anomaly on the ``/price`` endpoint:
    ``deprecated=true``, or an unexpected ``unit`` -- the same anomaly class
    ``EnergyChartsProbeAnomalyError`` is for ``public_power_forecast``,
    applied to this endpoint's own, smaller echo surface."""


def fetch_price_range(start: dt.date, end: dt.date, *, bzn: str = _PRICE_BZN) -> pd.Series:
    """Raw day-ahead prices over local calendar-date range [start, end]
    (both ends inclusive), UTC-indexed, ``None`` kept as NaN. Raises
    ``EnergyChartsRateLimitedError`` on 429, ``requests.HTTPError`` on any
    other non-200, ``EnergyChartsPriceAnomalyError`` on ``deprecated=true``
    or an unexpected unit."""
    params = {"bzn": bzn, "start": start.isoformat(), "end": end.isoformat()}
    response = requests.get(_PRICE_ENDPOINT, params=params, timeout=_TIMEOUT_S)

    if response.status_code == 429:
        raise EnergyChartsRateLimitedError(_parse_retry_after(response))
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    if payload.get("deprecated"):
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: endpoint reports deprecated=true"
        )
    unit = payload.get("unit")
    if unit != _PRICE_EXPECTED_UNIT:
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: unexpected unit {unit!r}, expected {_PRICE_EXPECTED_UNIT!r}"
        )

    unix_seconds = payload.get("unix_seconds", [])
    prices = payload.get("price", [])
    if len(unix_seconds) != len(prices):
        raise EnergyChartsPriceAnomalyError(
            f"price [{start}..{end}]: unix_seconds length {len(unix_seconds)} != "
            f"price length {len(prices)}"
        )

    index = pd.DatetimeIndex(pd.to_datetime(unix_seconds, unit="s", utc=True), name="timestamp")
    values = [float(v) if v is not None else float("nan") for v in prices]
    return pd.Series(values, index=index, name=PRICE_COLUMN, dtype="float64")


# ---------------------------------------------------------------------------
# Shared existing-wins merge
# ---------------------------------------------------------------------------


def merge_existing_with_fresh(
    existing_full: pd.DataFrame | None, fresh: pd.Series, *, column: str
) -> pd.DataFrame:
    """Merge ``fresh`` into ``existing_full`` (the whole on-disk series for
    ``column``), existing wins, verified byte-identical for every
    pre-existing non-NaN cell. ``existing_full=None`` (nothing on disk yet)
    returns ``fresh`` as-is."""
    fresh_frame = fresh.to_frame(name=column)
    if existing_full is None:
        return fresh_frame

    overlap = existing_full.index.intersection(fresh_frame.index)
    real_overlap = existing_full.loc[overlap, column].dropna()
    if not real_overlap.empty:
        disagreeing = ~fresh_frame.loc[real_overlap.index, column].eq(real_overlap)
        if disagreeing.any():
            first = disagreeing[disagreeing].index[0]
            raise ValueError(
                f"{int(disagreeing.sum())} timestamp(s) already have a real value that "
                f"disagrees with the freshly fetched one -- refusing to overwrite; first: {first}"
            )

    merged = existing_full.combine_first(fresh_frame)
    existing_notna = existing_full[column].dropna()
    if not merged.loc[existing_notna.index, column].equals(existing_notna):
        raise RuntimeError(
            "an existing non-NaN value changed during the merge -- this must never happen, "
            "aborting without writing"
        )
    return merged
