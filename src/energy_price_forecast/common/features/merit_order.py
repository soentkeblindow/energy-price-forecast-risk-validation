"""Merit-order gas pass-through: short-run marginal cost (SRMC) and residual-load
(RL) binning shared by the structural measurement (Phase 1) and the variant
comparison (Phase 2) of
docs/merit_order_extrapolation_structure_measurement_spec.md.

``residual_load_from_components``/``renewable_share_from_components`` are
also reused by the error-decomposition ladder (7.3,
docs/residual_error_decomposition_hourly_spec.md) for its NWP (L2) and
Oracle (O-RE/O-ALL) rungs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..market_time import LOCAL_TZ
from .config import FeatureConfig
from .fundamentals import build_commodity_features, build_forecast_fundamentals

SrmcDefinition = Literal["gas_only", "gas_full"]
RlMeasure = Literal["absolute"]

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


def residual_load_from_components(
    load: pd.Series, wind_onshore: pd.Series, wind_offshore: pd.Series, solar: pd.Series
) -> pd.Series:
    """Residual load from its four components: ``load - wind_onshore -
    wind_offshore - solar``.

    Mirrors ``common.features.fundamentals.build_forecast_fundamentals``'s
    ``residual_load_forecast`` formula exactly, but as plain arithmetic on
    whatever Series the caller passes in -- no ``Feature``/``Availability``
    wrapper (7.3 section 4, point 2: the leakage framework itself is not to
    be touched). Lets 7.3's L2 (NWP-reconstructed components) and Oracle
    (actual components) rungs reuse the one formula instead of each
    reimplementing it.
    """
    return load - wind_onshore - wind_offshore - solar


def renewable_share_from_components(
    load: pd.Series, wind_onshore: pd.Series, wind_offshore: pd.Series, solar: pd.Series
) -> pd.Series:
    """Renewable share from its four components: ``(wind_onshore +
    wind_offshore + solar) / load``. See ``residual_load_from_components``.
    """
    return (wind_onshore + wind_offshore + solar) / load


# Variant B clamps beta into this range (spec section 5.3, step 2): a
# gas-plant pass-through below 0 or above 1.5x the commodity price has no
# physical reading and would only reflect estimation noise in a thin bin.
BETA_CLAMP_MIN = 0.0
BETA_CLAMP_MAX = 1.5


def smooth_beta_isotonic(
    beta_by_bin: pd.Series,
    *,
    lower: float = BETA_CLAMP_MIN,
    upper: float | None = BETA_CLAMP_MAX,
) -> pd.Series:
    """Smooth per-bin beta estimates into a non-decreasing sequence over bin
    order, clamped to ``[lower, upper]`` (variant B, spec section 5.3, step 2).

    ``upper=None`` removes the ceiling entirely (7.3 section 3.4's "Gegenprobe
    ohne obere Grenze"): passed straight through as sklearn's own
    ``IsotonicRegression(y_max=None)`` sentinel for "unbounded", not emulated
    via ``float("inf")`` (which would introduce infinity arithmetic into the
    fit for no reason).

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


# Same pragmatic floor as the Phase 1 structural measurement (not HAC-derived
# here -- this is a point-estimate-only closed-form slope, not a
# standard-error report -- but kept at the same value so a bin only starts
# contributing a beta once it has a comparable amount of evidence).
_MIN_OBS_PER_BIN = 48


def _srmc_for_pool(srmc_definition: SrmcDefinition, ttf: pd.Series, eua: pd.Series) -> pd.Series:
    if srmc_definition == "gas_only":
        return srmc_gas_only(ttf)
    return srmc_gas_full(ttf, eua)


def _bin_pool(rl_measure: RlMeasure, pool: pd.DataFrame) -> pd.Series:
    assert rl_measure == "absolute"
    return rl_bin(pool["residual_load"])


def raw_beta_per_bin(
    long_history: pd.DataFrame,
    cutoff_local: pd.Timestamp,
    *,
    srmc_definition: SrmcDefinition,
    rl_measure: RlMeasure,
    window_days: int | None,
) -> pd.DataFrame:
    """Per-bin closed-form OLS slope of price on SRMC (variant B's beta(RL)),
    estimated from ``long_history`` rows strictly before ``cutoff_local``
    (and, if ``window_days`` is given, no earlier than
    ``cutoff_local - window_days``), crisis-period rows always excluded.

    Returns a frame indexed by ``RL_BIN_LABELS`` with columns ``beta_raw``
    (NaN where the bin has fewer than ``_MIN_OBS_PER_BIN`` valid observations
    or a degenerate SRMC spread) and ``n`` (valid observation count, before
    the ``_MIN_OBS_PER_BIN`` floor).

    Leakage contract: it is the caller's job to pick ``cutoff_local`` so that
    only data known before whatever it will be used for enters the pool --
    see ``merit_order_forecaster.MeritOrderForecaster``'s calendar-month
    update rule.

    Required ``long_history`` columns: ``price``, ``residual_load``,
    ``ttf_lag``, ``eua_lag``, ``is_crisis``.
    """
    required = {"price", "residual_load", "ttf_lag", "eua_lag", "is_crisis"}
    missing = required - set(long_history.columns)
    if missing:
        raise ValueError(f"long_history is missing columns: {sorted(missing)}")

    cutoff_utc = cutoff_local.tz_convert("UTC")
    pool = long_history.loc[long_history.index < cutoff_utc]
    if window_days is not None:
        window_start_local = cutoff_local - pd.DateOffset(days=window_days)
        pool = pool.loc[pool.index >= window_start_local.tz_convert("UTC")]
    pool = pool.loc[~pool["is_crisis"]]

    bin_pool = _bin_pool(rl_measure, pool)
    srmc_pool = _srmc_for_pool(srmc_definition, pool["ttf_lag"], pool["eua_lag"])
    valid = pool["price"].notna() & srmc_pool.notna() & bin_pool.notna()

    rows: dict[int, dict[str, float]] = {}
    for label in RL_BIN_LABELS:
        cell_mask = valid & (bin_pool == label)
        n = int(cell_mask.sum())
        if n < _MIN_OBS_PER_BIN:
            rows[label] = {"beta_raw": float("nan"), "n": n}
            continue
        y = pool.loc[cell_mask, "price"].to_numpy()
        x = srmc_pool.loc[cell_mask].to_numpy()
        x_centered = x - x.mean()
        denom = float((x_centered**2).sum())
        if denom <= 0:
            rows[label] = {"beta_raw": float("nan"), "n": n}
            continue
        beta = float((x_centered * (y - y.mean())).sum() / denom)
        rows[label] = {"beta_raw": beta, "n": n}

    return pd.DataFrame.from_dict(rows, orient="index").reindex(list(RL_BIN_LABELS))


def estimate_beta_per_bin(
    long_history: pd.DataFrame,
    cutoff_local: pd.Timestamp,
    *,
    srmc_definition: SrmcDefinition,
    rl_measure: RlMeasure,
    window_days: int | None,
    clamp_max: float | None,
) -> pd.Series:
    """Isotonic-smoothed beta(RL) (spec section 5.3, step 2), built on
    ``raw_beta_per_bin``'s per-bin closed-form regression. See that
    function's docstring for the leakage contract and required columns.
    """
    raw = raw_beta_per_bin(
        long_history,
        cutoff_local,
        srmc_definition=srmc_definition,
        rl_measure=rl_measure,
        window_days=window_days,
    )
    return smooth_beta_isotonic(raw["beta_raw"], upper=clamp_max)


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
    ``srmc_gas_full`` -- kept out of this frame so that stays a per-caller
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
