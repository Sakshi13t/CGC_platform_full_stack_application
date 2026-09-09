"""
backend/app/preprocessing/physics_features.py
================================================
Foundational physics-derived features for the CGC dataset, computed ONLY
from raw telemetry + assets.csv design values (never from
ground_truth_reference.csv).

Formulas used (as given in the assignment brief, Section 0.3 "The physics
you will actually use"):

    Polytropic head per stage:
        H_p = Z * (R/MW) * T_suct * (n/(n-1)) * [(P_disch/P_suct)^((n-1)/n) - 1]

    Discharge temperature:
        T_disch = T_suct * (P_disch/P_suct)^((k-1)/(k*eta_poly))

    Specific energy consumption (KPI):
        SEC = shaft_power_kw / throughput_tph   [kWh/tonne]

Two things are NOT directly given in the assignment or knowledge-base docs
and are therefore isolated as named, documented assumptions rather than
silently baked into a formula (see config.PhysicsAssumptions and
TECHNICAL_DECISIONS.md):

    1. Compressibility factor Z (assumed 1.0 — ideal gas) — used only by the
       optional polytropic-head feature.
    2. The standard turbomachinery identity connecting the polytropic
       exponent n, isentropic exponent k, and polytropic efficiency:
           (n-1)/n = (k-1)/(k*eta_poly)
       This is what makes the assignment's two given equations consistent;
       it is not restated verbatim in the assignment text.

Every function here is pure (DataFrame in, Series/DataFrame out) and never
mutates its input. Quality of every derived value is propagated from the
quality of its required raw inputs — see `propagate_quality`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .quality import Quality

KELVIN_OFFSET = 273.15

# Physically-plausible bounds used only to catch a derived value that is
# mathematically defined but physically nonsensical (e.g. a negative
# pressure ratio from a validation bug) — NOT used to silently clip values.
PLAUSIBLE_EFFICIENCY_RANGE = (0.0, 1.0)
PLAUSIBLE_PRESSURE_RATIO_MIN = 1.0


# ---------------------------------------------------------------------------
# Quality propagation
# ---------------------------------------------------------------------------

_PRIORITY = {
    Quality.SHUTDOWN.value: 3,
    Quality.BAD_SENSOR.value: 2,
    Quality.MISSING.value: 1,
    Quality.GOOD.value: 0,
    Quality.INVALID_PHYSICS.value: 4,
}


def propagate_quality(*quality_series: pd.Series) -> pd.Series:
    """
    Combine several input quality Series into one output quality Series by
    taking the worst quality present at each row (SHUTDOWN and
    INVALID_PHYSICS outrank BAD_SENSOR, which outranks MISSING, which
    outranks GOOD). All Series must share the same index.
    """
    stacked = pd.concat(quality_series, axis=1)
    worst = stacked.apply(lambda row: max(row, key=lambda q: _PRIORITY[q]), axis=1)
    return worst


def flag_implausible(values: pd.Series, low: float, high: float) -> pd.Series:
    """Boolean mask: True where a defined (non-NaN) value falls outside
    [low, high] — used to elevate a GOOD-quality derived value to
    INVALID_PHYSICS when the *computed number itself* is nonsensical."""
    return values.notna() & ((values < low) | (values > high))


# ---------------------------------------------------------------------------
# Pressure ratio
# ---------------------------------------------------------------------------

def pressure_ratio(df: pd.DataFrame, stage: int) -> pd.Series:
    """P_disch / P_suct for one stage. Direct ratio of two raw columns."""
    return df[config.stage_disch_p(stage)] / df[config.stage_suct_p(stage)]


def overall_pressure_ratio(df: pd.DataFrame) -> pd.Series:
    """Final-stage discharge / first-stage suction — overall compression ratio."""
    return df[config.stage_disch_p(5)] / df[config.stage_suct_p(1)]


# ---------------------------------------------------------------------------
# Polytropic efficiency: observed (backed out from measured T) and the
# design/clean baseline (from assets.csv), plus the discharge-T residual.
# ---------------------------------------------------------------------------

def observed_polytropic_efficiency(df: pd.DataFrame, stage: int, k: pd.Series) -> pd.Series:
    """
    Invert the assignment's discharge-temperature equation to back out an
    'observed' polytropic efficiency directly from measured T_suct, T_disch,
    and the stage pressure ratio:

        eta_poly_obs = (k - 1) * ln(pr) / (k * ln(T_disch_K / T_suct_K))

    `k` is per-row assets.gas_k_cp_cv (joined by train_id), not a fitted
    quantity — this is a direct algebraic inversion of the given formula,
    not a new one.
    """
    pr = pressure_ratio(df, stage)
    t_suct_k = df[config.stage_suct_t(stage)] + KELVIN_OFFSET
    t_disch_k = df[config.stage_disch_t(stage)] + KELVIN_OFFSET
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = (k - 1) * np.log(pr) / (k * np.log(t_disch_k / t_suct_k))
    return eta


def predicted_discharge_temperature_c(
    df: pd.DataFrame, stage: int, k: pd.Series, eta_poly_baseline: pd.Series
) -> pd.Series:
    """
    Predict discharge temperature (deg C) at the CLEAN/DESIGN baseline
    efficiency (assets.design_poly_eff_pct / 100), given measured suction
    temperature and pressure ratio:

        T_disch_pred_K = T_suct_K * pr^((k-1)/(k*eta_poly_baseline))

    This is the "physics-consistency" prediction used for the residual
    check in Section 4 of the assignment. Using the fixed design efficiency
    (rather than the observed one) is deliberate: it lets the residual grow
    both when a sensor drifts AND when the machine fouls, which is exactly
    the dual-purpose cross-check the assignment describes ("catch a
    drifting sensor" using a physics prediction independent of the
    instrument being checked).
    """
    pr = pressure_ratio(df, stage)
    t_suct_k = df[config.stage_suct_t(stage)] + KELVIN_OFFSET
    exponent = (k - 1) / (k * eta_poly_baseline)
    t_disch_pred_k = t_suct_k * np.power(pr, exponent)
    return t_disch_pred_k - KELVIN_OFFSET


def discharge_temperature_residual_c(
    df: pd.DataFrame, stage: int, k: pd.Series, eta_poly_baseline: pd.Series
) -> pd.Series:
    """Measured discharge T minus baseline-predicted discharge T (deg C).
    Near zero at a clean machine with a healthy sensor; grows with fouling
    and/or with a drifting transmitter (Section 4 of the assignment;
    verified in Phase 1 against the CGC-100A stage-3 drift event)."""
    predicted = predicted_discharge_temperature_c(df, stage, k, eta_poly_baseline)
    return df[config.stage_disch_t(stage)] - predicted


# ---------------------------------------------------------------------------
# Specific Energy Consumption
# ---------------------------------------------------------------------------

def specific_energy_consumption(df: pd.DataFrame) -> pd.Series:
    """SEC = shaft_power_kw / throughput_tph [kWh/tonne], per Section 0.3 of
    the assignment and EM-CGC-05 §1. Direct ratio of two raw columns —
    caller is responsible for propagating throughput_tph's quality flag
    onto this feature (see build_physics_features)."""
    return df["shaft_power_kw"] / df["throughput_tph"]


# ---------------------------------------------------------------------------
# Polytropic head (optional; carries an explicit physics_assumption tag)
# ---------------------------------------------------------------------------

def polytropic_head_j_per_kg(
    df: pd.DataFrame, stage: int, k: pd.Series, eta_poly: pd.Series, mw_col: str = "gas_mw"
) -> pd.Series:
    """
    H_p = Z * (R/MW) * T_suct * (n/(n-1)) * [pr^((n-1)/n) - 1]   [J/kg]

    Uses config.PHYSICS.compressibility_factor_z (assumed ideal gas, Z=1)
    and the standard n/eta/k identity (n-1)/n = (k-1)/(k*eta_poly) — both
    named explicitly in config.PhysicsAssumptions and TECHNICAL_DECISIONS.md
    because neither is given verbatim in the assignment/knowledge-base docs.
    """
    if not config.PHYSICS.use_standard_n_eta_k_relation:
        raise NotImplementedError(
            "polytropic_head_j_per_kg requires the standard n/eta/k relation; "
            "set config.PHYSICS.use_standard_n_eta_k_relation=True to enable it."
        )
    pr = pressure_ratio(df, stage)
    t_suct_k = df[config.stage_suct_t(stage)] + KELVIN_OFFSET
    mw_kg_per_mol = df[mw_col] / 1000.0
    m = (k - 1) / (k * eta_poly)  # == (n-1)/n under the standard identity
    r_over_mw = config.PHYSICS.universal_gas_constant_j_per_molk / mw_kg_per_mol
    with np.errstate(divide="ignore", invalid="ignore"):
        head = (
            config.PHYSICS.compressibility_factor_z
            * r_over_mw
            * t_suct_k
            * (1.0 / m)
            * (np.power(pr, m) - 1.0)
        )
    return head


# ---------------------------------------------------------------------------
# Surge-margin threshold flag (SP-CGC-04 §2, documented minimum = 8%)
# ---------------------------------------------------------------------------

def below_min_surge_margin(df: pd.DataFrame) -> pd.Series:
    """True where surge_margin_pct is below the documented minimum
    operating margin (8%, SP-CGC-04 §2). Direct threshold application of a
    documented standard, not a new physics formula."""
    return df["surge_margin_pct"] < config.THRESHOLDS.surge_margin_min_pct


# ---------------------------------------------------------------------------
# Orchestration: build the full physics feature block with quality columns
# ---------------------------------------------------------------------------

def build_physics_features(
    df: pd.DataFrame,
    assets: pd.DataFrame,
    quality_frame: pd.DataFrame,
    stages: Iterable[int] = config.STAGE_NUMBERS,
    include_polytropic_head: bool = True,
) -> pd.DataFrame:
    """
    Compute the foundational physics features for every row and propagate
    quality from their raw inputs plus the shared `is_shutdown` flag and
    any known BAD_SENSOR windows in `quality_frame` (see quality.py).

    Returns a new DataFrame keyed by (ts, train_id) — never mutates `df`.
    """
    merged = df.merge(
        assets[["train_id", "gas_k_cp_cv", "design_poly_eff_pct"]], on="train_id", how="left"
    )
    k = merged["gas_k_cp_cv"]
    eta_design = merged["design_poly_eff_pct"] / 100.0

    is_shutdown = quality_frame.set_index(["ts", "train_id"]).loc[
        list(zip(merged["ts"], merged["train_id"])), "is_shutdown"
    ].values
    shutdown_quality = pd.Series(
        np.where(is_shutdown, Quality.SHUTDOWN.value, Quality.GOOD.value), index=merged.index
    )

    out = pd.DataFrame({"ts": merged["ts"], "train_id": merged["train_id"]})

    # Per-stage: pressure ratio, observed eta_poly, discharge-T residual
    for stage in stages:
        pr = pressure_ratio(merged, stage)
        out[f"pressure_ratio_s{stage}"] = pr

        # raw-input quality for this stage's four sensors
        stage_cols = [
            config.stage_suct_p(stage),
            config.stage_disch_p(stage),
            config.stage_suct_t(stage),
            config.stage_disch_t(stage),
        ]
        stage_missing = pd.Series(
            np.where(merged[stage_cols].isna().any(axis=1), Quality.MISSING.value, Quality.GOOD.value),
            index=merged.index,
        )
        quality_inputs = [shutdown_quality, stage_missing]

        disch_t_col = config.stage_disch_t(stage)
        if f"{disch_t_col}_quality" in quality_frame.columns:
            fault_q = quality_frame.set_index(["ts", "train_id"]).loc[
                list(zip(merged["ts"], merged["train_id"])), f"{disch_t_col}_quality"
            ].values
            quality_inputs.append(pd.Series(fault_q, index=merged.index))

        eta_obs = observed_polytropic_efficiency(merged, stage, k)
        out[f"polytropic_eff_observed_s{stage}"] = eta_obs
        residual = discharge_temperature_residual_c(merged, stage, k, eta_design)
        out[f"discharge_temp_residual_c_s{stage}"] = residual

        stage_quality = propagate_quality(*quality_inputs)
        implausible = flag_implausible(eta_obs, *PLAUSIBLE_EFFICIENCY_RANGE) | (pr.notna() & (pr < PLAUSIBLE_PRESSURE_RATIO_MIN))
        stage_quality = stage_quality.mask(implausible & (stage_quality == Quality.GOOD.value), Quality.INVALID_PHYSICS.value)
        out[f"stage{stage}_physics_quality"] = stage_quality

        if include_polytropic_head:
            out[f"polytropic_head_jkg_s{stage}"] = polytropic_head_j_per_kg(merged, stage, k, eta_obs)

    out["overall_pressure_ratio"] = overall_pressure_ratio(merged)

    # SEC: quality follows throughput_tph_quality (BAD/MISSING/SHUTDOWN) plus shaft_power missingness
    sec = specific_energy_consumption(merged)
    out["sec_kwh_per_t"] = sec
    throughput_quality = quality_frame.set_index(["ts", "train_id"]).loc[
        list(zip(merged["ts"], merged["train_id"])), "throughput_tph_quality"
    ].values
    shaft_power_missing = pd.Series(
        np.where(merged["shaft_power_kw"].isna(), Quality.MISSING.value, Quality.GOOD.value), index=merged.index
    )
    out["sec_quality"] = propagate_quality(
        pd.Series(throughput_quality, index=merged.index), shaft_power_missing, shutdown_quality
    )

    out["below_min_surge_margin"] = below_min_surge_margin(merged)

    return out
