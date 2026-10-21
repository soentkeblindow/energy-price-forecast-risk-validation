"""Oracle fundamentals substitution for the O-RE/O-ALL rungs of 7.3's
error-decomposition ladder (docs/residual_error_decomposition_hourly_spec.md,
section 3.1/4).

Deliberately lives in ``model_risk_regimes``, not ``common`` (spec section
4 point 3 / section 8): the dependency rule (``tests/test_dependency_rules.py``)
then structurally forbids ``model_post_crisis_resolution`` -- the path into
the live model -- from importing this Oracle-only code. The leakage
framework itself (``common/features/availability.py``) is untouched (spec
section 4 point 2); this module does plain post-hoc column overwrites on
already-built, already-leakage-checked frames.

Substitution scope (design decision, spec section 4 points 1 and 4 read
together): GLOBAL, not per-fold/per-target-day. Every hourly row across the
whole dataset gets its fundamentals columns replaced once, before any
walk-forward backtest runs -- not a per-fold "swap just today's row" done
inside the backtest loop. This way ``x_train`` and ``x_test`` both see Ist-
Werte in every fold automatically ("Training und Vorhersage beide mit Ist-
Werten", point 4), without a backtest-loop special case; "Zieltag" in point
1 describes the per-hour replacement RULE (value at hour h is replaced by
the actual at hour h), not a fold-local restriction.

O-RE: wind/solar/residual-load/renewable-share from actuals, load stays on
its ENTSO-E forecast. O-ALL: load is also replaced by its actual. Both are
oracle leakage BY DESIGN -- never a live-system candidate, measurement only.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from energy_price_forecast.common.features.merit_order import (
    renewable_share_from_components,
    residual_load_from_components,
)

OracleRung = Literal["o_re", "o_all"]

# The 5 fundamentals columns both oracle rungs always replace (spec table
# 3.1); O-ALL additionally replaces load_forecast_day_ahead (see
# _oracle_load below).
ORACLE_SWAPPED_COLUMNS: tuple[str, ...] = (
    "wind_onshore_forecast",
    "wind_offshore_forecast",
    "solar_forecast",
    "residual_load_forecast",
    "renewable_share_forecast",
)


def _oracle_load(df: pd.DataFrame, rung: OracleRung) -> pd.Series:
    """O-RE keeps the ENTSO-E load forecast; O-ALL uses the actual (spec
    table 3.1: "Last des Zieltags" column)."""
    return df["load_actual"] if rung == "o_all" else df["load_forecast_day_ahead"]


def build_oracle_feature_frame(
    features: pd.DataFrame, df: pd.DataFrame, rung: OracleRung
) -> pd.DataFrame:
    """An L1-shaped feature frame (``features``) with its fundamentals
    columns globally replaced by their actual-value counterparts, sourced
    from ``df`` (the raw ``hourly_merit_order.parquet``-shaped frame with
    ``gen_wind_onshore``/``gen_wind_offshore``/``gen_solar``/``load_actual``).

    ``gen_wind_onshore``/``gen_wind_offshore``/``gen_solar`` (actual
    aggregated generation, from ENTSO-E's generation-by-type query) are the
    actual-wind/solar source, not a ``wind_onshore_actual`` column -- that
    name stopped being populated after commit e37b2cb (2026-06-02) removed
    it as a duplicate of these same ``gen_*`` columns. Using the retired name
    here silently read all-NaN for every hour after 2026-05-31.

    Only ``ORACLE_SWAPPED_COLUMNS`` (plus ``load_forecast_day_ahead`` for
    ``rung="o_all"``) change; every other column of ``features`` -- calendar,
    price lags, TTF -- is returned unchanged (asserted by
    ``tests/model_risk_regimes/test_oracle_fundamentals.py``).
    """
    load = _oracle_load(df, rung)
    won, woff, solar = df["gen_wind_onshore"], df["gen_wind_offshore"], df["gen_solar"]
    updates: dict[str, pd.Series] = {
        "wind_onshore_forecast": won,
        "wind_offshore_forecast": woff,
        "solar_forecast": solar,
        "residual_load_forecast": residual_load_from_components(load, won, woff, solar),
        "renewable_share_forecast": renewable_share_from_components(load, won, woff, solar),
    }
    if rung == "o_all":
        updates["load_forecast_day_ahead"] = load
    aligned = {col: series.reindex(features.index) for col, series in updates.items()}
    return features.assign(**aligned)


def build_oracle_long_history_frame(
    long_history: pd.DataFrame, df: pd.DataFrame, rung: OracleRung
) -> pd.DataFrame:
    """Oracle counterpart of
    ``common.features.merit_order.build_merit_order_frame``'s output, for
    variant B's beta-estimation long-history side channel.

    7.3 section 3.4's clarification: beta for O-RE/O-ALL is estimated on the
    SAME (oracle) data the tree trains on, not just at the target day -- so
    ``residual_load``/``load`` are replaced across the whole history, not
    only for the current fold's test day.
    """
    load = _oracle_load(df, rung)
    won, woff, solar = df["gen_wind_onshore"], df["gen_wind_offshore"], df["gen_solar"]
    residual = residual_load_from_components(load, won, woff, solar)

    out = long_history.copy()
    out["residual_load"] = residual.reindex(long_history.index)
    out["load"] = load.reindex(long_history.index)
    return out
