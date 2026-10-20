"""Merit-order gas pass-through: short-run marginal cost (SRMC) and residual-load
(RL) binning shared by the structural measurement (Phase 1) and the variant
comparison (Phase 2) of
docs/merit_order_extrapolation_structure_measurement_spec.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..market_time import LOCAL_TZ
from .config import FeatureConfig
from .fundamentals import build_commodity_features, build_forecast_fundamentals

# Clean Spark Spread market convention (UK/ICE) for a gas-fired plant's
# short-run marginal cost. Physical cross-check: 56.6 t CO2/TJ (EU ETS standard
# factor for natural gas, lower heating value) = 0.204 t CO2/MWh gas, divided
# by GAS_EFFICIENCY = 0.415 -- close to CO2_INTENSITY_EL below (the ~10%
# difference is the lower- vs. upper-heating-value convention gas is usually
# traded on). GAS_EFFICIENCY is a pure scaling factor for srmc_gas_only /
# srmc_gas_full: any value of eta gives the same regression fit (beta*x/eta =
# (beta/eta)*x), so it only affects how readable beta is (beta ~= 1 means "the
# marginal plant has a typical gas efficiency").
GAS_EFFICIENCY = 0.4913
CO2_INTENSITY_EL = 0.411  # t CO2 per MWh electricity


def srmc_gas_only(ttf: pd.Series) -> pd.Series:
    """Gas-only short-run marginal cost of a gas plant, EUR/MWh electricity.

    Primary SRMC definition for 2019-2021-10, where EUA is not yet available.
    """
    return ttf / GAS_EFFICIENCY


def srmc_gas_full(ttf: pd.Series, eua: pd.Series) -> pd.Series:
    """Short-run marginal cost of a gas plant including carbon cost, EUR/MWh electricity.

    Secondary SRMC definition, only meaningful from 2021-10 onward (EUA data start).
    """
    return ttf / GAS_EFFICIENCY + CO2_INTENSITY_EL * eua


# Residual-load (RL) bin edges, MW. Frozen from the deciles of
# residual_load_forecast over 2019-01-01..2020-12-31 (see
# docs/merit_order_extrapolation_structure_measurement_log.md, Phase 1), plus
# an explicit edge at RL=0: the lowest decile (~19.2 GW) does not separate the
# rare RL<0 hours (0.11% of 2019-2020) on its own, so a dedicated bin is added
# per the spec rule. Not recomputed at run time -- a later change in the
# underlying data must not silently shift these boundaries.
RL_BIN_EDGES_MW: tuple[float, ...] = (
    float("-inf"),
    0.0,
    19194.795,
    25552.934,
    29652.111,
    32979.196,
    36063.885,
    39024.152,
    42378.031,
    46003.602,
    50769.293,
    float("inf"),
)

RL_BIN_LABELS: tuple[int, ...] = tuple(range(len(RL_BIN_EDGES_MW) - 1))


def rl_bin(residual_load: pd.Series) -> pd.Series:
    """Bin residual load (MW) into the frozen RL bins.

    Bin 0 is RL < 0; bin ``len(RL_BIN_LABELS) - 1`` is the highest decile and
    up. Half-open on the right edge, so a value exactly on a boundary falls
    into the upper bin.
    """
    return pd.cut(
        residual_load,
        bins=RL_BIN_EDGES_MW,
        labels=RL_BIN_LABELS,
        right=False,
    )


def rl_relative(residual_load: pd.Series, load: pd.Series) -> pd.Series:
    """Residual load as a fraction of day-ahead load (spec section 2,
    B-rel sensitivity): ``residual_load / load``.

    Renewables build-out since 2019 shifts the absolute RL distribution, so
    this is the relative-RL counterpart to the (fixed-MW) ``RL_BIN_EDGES_MW``
    binning, meant to stay comparable across years.
    """
    return residual_load / load


# Relative-RL bin edges (fraction of load, dimensionless). Frozen from the
# deciles of residual_load_forecast / load_forecast_day_ahead over the same
# 2019-01-01..2020-12-31 window as RL_BIN_EDGES_MW (n=17,520 valid hours,
# identical 24 NaN hours dropped) -- see
# docs/merit_order_extrapolation_structure_measurement_log.md, B-rel
# sensitivity. Same rule as the absolute edges: the lowest decile
# (~0.367) does not separate the rare RL<0 hours (0.11% of 2019-2020, the
# same hours as for RL_BIN_EDGES_MW since the sign of RL/load matches the
# sign of RL), so a dedicated bin at 0 is added.
RL_BIN_EDGES_RELATIVE: tuple[float, ...] = (
    float("-inf"),
    0.0,
    0.3669691009968795,
    0.47778109214719716,
    0.5558413666390126,
    0.6195223389993966,
    0.6766913007795885,
    0.7284595026250806,
    0.7776308011413019,
    0.8327000943108853,
    0.8910611743250727,
    float("inf"),
)


def rl_bin_relative(residual_load_relative: pd.Series) -> pd.Series:
    """Bin relative residual load (fraction of load) into the frozen
    relative-RL bins (same bin count/semantics as ``rl_bin``).
    """
    return pd.cut(
        residual_load_relative,
        bins=RL_BIN_EDGES_RELATIVE,
        labels=RL_BIN_LABELS,
        right=False,
    )


# Variant B clamps beta into this range (spec section 5.3, step 2): a
# gas-plant pass-through below 0 or above 1.5x the commodity price has no
# physical reading and would only reflect estimation noise in a thin bin.
BETA_CLAMP_MIN = 0.0
BETA_CLAMP_MAX = 1.5


def smooth_beta_isotonic(
    beta_by_bin: pd.Series,
    *,
    lower: float = BETA_CLAMP_MIN,
    upper: float = BETA_CLAMP_MAX,
) -> pd.Series:
    """Smooth per-bin beta estimates into a non-decreasing sequence over bin
    order, clamped to ``[lower, upper]`` (variant B, spec section 5.3, step 2).

    NaN entries (bins with too few observations for a regression) are
    excluded from the isotonic fit itself, but every bin in the input still
    gets a predicted value by interpolation/extrapolation from its
    neighbours -- the caller never has to special-case a gap bin.
    """
    bins = beta_by_bin.index.to_numpy(dtype=float)
    values = beta_by_bin.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        raise ValueError("need at least two non-NaN bin estimates to fit an isotonic smoother")
    iso = IsotonicRegression(y_min=lower, y_max=upper, out_of_bounds="clip")
    iso.fit(bins[valid], values[valid])
    smoothed = iso.predict(bins)
    return pd.Series(smoothed, index=beta_by_bin.index)


def build_merit_order_frame(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Price/residual-load/commodity-lag/crisis-flag frame shared by the
    structural measurement (Phase 1 script) and the variant B forecaster
    (Phase 2, long-history side channel for its own beta estimation).

    Columns: ``price``, ``residual_load``, ``load``, ``ttf_lag``, ``eua_lag``,
    ``is_crisis``. Callers derive RL bins and SRMC on top via ``rl_bin`` /
    ``rl_bin_relative`` / ``srmc_gas_only`` / ``srmc_gas_full`` -- kept out of
    this frame so the RL measure and SRMC definition stay a per-caller
    choice.
    """
    cfg = config if config is not None else FeatureConfig()
    fundamentals = {f.name: f.values for f in build_forecast_fundamentals(df, target_index)}
    commodities = {f.name: f.values for f in build_commodity_features(df, target_index, cfg)}

    frame = pd.DataFrame(
        {
            "price": df["day_ahead_price"],
            "residual_load": fundamentals["residual_load_forecast"],
            "load": fundamentals["load_forecast_day_ahead"],
            "ttf_lag": commodities["ttf_gas_lag_48h"],
            "eua_lag": commodities["eua_co2_lag_48h"],
        },
        index=target_index,
    )
    frame["is_crisis"] = (frame.index >= cfg.crisis_start) & (frame.index < cfg.post_crisis_start)
    return frame


def ttf_out_of_range_flag(
    ttf_lag: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    train_span_days: int,
) -> pd.Series:
    """Per-hour flag: True where ``ttf_lag`` exceeds the max of that feature
    seen in the fold's own rolling ``train_span_days``-day training window
    (spec section 2, "TTF-Out-of-range-Flag").

    ``ttf_lag`` is constant within a local (Europe/Berlin) calendar day
    (daily commodity data, lagged), so this groups by LOCAL day -- not a
    UTC-indexed ``resample("D")``, which would bucket by UTC calendar days
    instead and never line up with the walk-forward harness's own local-day
    boundaries (``walkforward.py``'s ``local_date``), silently producing an
    all-NaN (all-False) flag. The rolling max over the preceding
    ``train_span_days`` days, shifted by one day, matches exactly the day
    range a rolling walk-forward fold's ``train_index`` covers for that
    delivery day (days < D).
    """
    local_day_full = pd.DatetimeIndex(ttf_lag.index).tz_convert(LOCAL_TZ).normalize()
    daily_ttf = ttf_lag.groupby(local_day_full).first()
    train_max = daily_ttf.rolling(train_span_days, min_periods=train_span_days).max().shift(1)

    local_day_target = pd.DatetimeIndex(target_index).tz_convert(LOCAL_TZ).normalize()
    train_max_per_hour = train_max.reindex(local_day_target).to_numpy()
    flag = ttf_lag.reindex(target_index).to_numpy() > train_max_per_hour
    return pd.Series(flag, index=target_index)
