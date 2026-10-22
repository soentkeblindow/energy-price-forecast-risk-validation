"""Hourly French (FR) day-ahead price series, Sprint 7 Step 7.5 Hauptteil
(docs/sprint7_step7_5_spec.md, Phase 0).

Built the identical way this repo's own German day_ahead_price hourly
column is built (data/normalize.py::to_hourly_vwap): native-resolution
price, volume-weighted by the SAME country's own load_actual (falling back
to load_forecast_day_ahead for still-missing weights), collapsed to
hourly. On the pre-15-min era the weight cancels and this is a
pass-through, so one call covers the whole requested range without
needing FR's own quarter-hour cutover date hardcoded here -- Phase 0's
coverage check verifies that date empirically instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import DATA_INTERIM
from .entsoe_client import fetch_day_ahead_prices, fetch_load
from .normalize import to_hourly_vwap

logger = logging.getLogger(__name__)

AREA_FR = "FR"
_PATH = DATA_INTERIM / "fr_day_ahead_hourly.parquet"


def build_fr_day_ahead_hourly(
    start: pd.Timestamp, end: pd.Timestamp, path: Path = _PATH, area: str = AREA_FR
) -> pd.DataFrame:
    """Fetch FR day-ahead price and load (native resolution), collapse to
    hourly via to_hourly_vwap, and persist as a single-column
    (``fr_day_ahead_price``) hourly Parquet.
    """
    price = fetch_day_ahead_prices(start, end, area)["day_ahead_price"]
    load = fetch_load(start, end, area)
    hourly_price = to_hourly_vwap(
        price, load["load_actual"], weight_fallback=load.get("load_forecast_day_ahead")
    )
    out = hourly_price.rename("fr_day_ahead_price").to_frame()

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    logger.info(
        "wrote %d hourly FR day-ahead price rows to %s (%s -> %s)",
        len(out),
        path,
        out.index.min() if not out.empty else None,
        out.index.max() if not out.empty else None,
    )
    return out


def load_fr_day_ahead_hourly(path: Path = _PATH) -> pd.DataFrame:
    """Load the persisted hourly FR day-ahead price series.

    Raises FileNotFoundError if the file does not exist (run
    build_fr_day_ahead_hourly() first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FR day-ahead hourly Parquet not found at {path!r}. "
            "Run fr_prices.build_fr_day_ahead_hourly() to generate it."
        )
    df = pd.read_parquet(path)
    logger.info("loaded %d hourly FR day-ahead price rows from %s", len(df), path)
    return df
