"""
data_schema.py
==============
Static schema definitions for the Cracked Gas Compressor (CGC) dataset.

Purpose: a single source of truth for (a) expected dtypes, (b) physically
plausible ranges used for validation/flagging, and (c) the INPUT / TARGET /
EXCLUDE role of every column, so that model-building code (built in a later
phase) imports this instead of re-deriving it.

This module contains NO training code and performs NO I/O on import.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ColumnRole(str, Enum):
    INPUT = "input"                # safe to feed to a model at inference time
    TARGET_SOURCE = "target_source"  # used only to construct a training target/label
    IDENTIFIER = "identifier"      # key / grouping column, not a feature
    EXCLUDE = "exclude"            # must never be used as a feature (leakage or no info)
    DERIVED_INPUT = "derived_input"  # engineered physics feature, safe as input


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    role: ColumnRole
    unit: Optional[str] = None
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# compressor_telemetry.csv  — 45 columns, hourly, 2 trains, the model INPUT file
# ---------------------------------------------------------------------------
TELEMETRY_SCHEMA = [
    ColumnSpec("ts", "datetime64[ns]", ColumnRole.IDENTIFIER, notes="hourly timestamp, per-train index"),
    ColumnSpec("train_id", "category", ColumnRole.IDENTIFIER, notes="CGC-100A | CGC-200B"),
    ColumnSpec("unit", "category", ColumnRole.EXCLUDE, notes="1:1 with train_id — redundant identifier, drop or fold into train_id encoding"),
    ColumnSpec("feed_mode", "category", ColumnRole.INPUT, notes="ethane | mixed | naphtha — one-hot or ordinal-by-diene encode"),
    ColumnSpec("gas_mw", "float64", ColumnRole.INPUT, unit="g/mol", valid_min=15, valid_max=40),
    ColumnSpec("diene_ppm", "float64", ColumnRole.INPUT, unit="ppm", valid_min=0, valid_max=400, notes="fouling precursor concentration; correlates with feed_mode"),
    ColumnSpec("ambient_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=-20, valid_max=55),
    ColumnSpec("cw_supply_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=45),
    ColumnSpec("throughput_tph", "float64", ColumnRole.INPUT, unit="t/h", valid_min=0, valid_max=400,
               notes="GENUINE SENSOR FAULT: frozen at 243.85 t/h for 432 consecutive hours on CGC-200B "
                     "(2024-04-30→2024-05-17), matching the events.csv sensor_fault record. Mark this window "
                     "BAD via a quality/status flag and do not trust throughput-derived physics features "
                     "(SEC, residuals) computed from it during that window. Preserve the raw frozen values "
                     "as-is — do not overwrite or interpolate them. Separately NaN during the "
                     "2025-08-13→18 shutdown (train not running, not a fault)."),
    ColumnSpec("speed_rpm", "float64", ColumnRole.INPUT, unit="rpm", valid_min=0, valid_max=13000,
               notes="cross-check against assets.min_gov_speed_rpm / trip_speed_rpm per train"),
    ColumnSpec("shaft_power_kw", "float64", ColumnRole.INPUT, unit="kW", valid_min=0, valid_max=40000),
    ColumnSpec("surge_margin_pct", "float64", ColumnRole.INPUT, unit="%", valid_min=-100, valid_max=100,
               notes="can be legitimately negative during an actual surge excursion (observed on CGC-100A 2024-07-14) — do not clip to >=0"),
    ColumnSpec("recycle_valve_pct", "float64", ColumnRole.INPUT, unit="%", valid_min=0, valid_max=100),
    ColumnSpec("wash_oil_rate_kgph", "float64", ColumnRole.INPUT, unit="kg/h", valid_min=0, valid_max=500, notes="mitigation action"),
    ColumnSpec("antifoulant_ppm", "float64", ColumnRole.INPUT, unit="ppm", valid_min=0, valid_max=30, notes="mitigation action"),
    ColumnSpec("run_hours_since_wash", "float64", ColumnRole.INPUT, unit="h", valid_min=0, valid_max=4000,
               notes="strong, legitimate (observable) predictor — NOT leakage, but note it is highly collinear "
                     "with rul_hours_to_next_wash_true (corr ≈ -0.98) by construction of the simulation"),
    ColumnSpec("vib_de_um", "float64", ColumnRole.INPUT, unit="µm", valid_min=0, valid_max=200, notes="drive-end vibration; bearing-wear precursor"),
    ColumnSpec("vib_nde_um", "float64", ColumnRole.INPUT, unit="µm", valid_min=0, valid_max=200,
               notes="~80/62 sporadic single-hour NaNs per train shared with lube_oil_temp_c and seal_gas_dp_bar — a correlated instrumentation dropout, not shutdown-related"),
    ColumnSpec("vib_axial_um", "float64", ColumnRole.INPUT, unit="µm", valid_min=0, valid_max=200),
    ColumnSpec("brg_de_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=150),
    ColumnSpec("brg_nde_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=150),
    ColumnSpec("brg_thrust_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=150),
    ColumnSpec("lube_oil_press_bar", "float64", ColumnRole.INPUT, unit="bar", valid_min=0, valid_max=10),
    ColumnSpec("lube_oil_temp_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=100),
    ColumnSpec("seal_gas_dp_bar", "float64", ColumnRole.INPUT, unit="bar", valid_min=0, valid_max=5),
    # Per-stage pressures/temperatures (5 stages)
    *[
        spec
        for i in range(1, 6)
        for spec in [
            ColumnSpec(f"p_s{i}_suct_bara", "float64",
                       ColumnRole.EXCLUDE if i == 1 else ColumnRole.INPUT,
                       unit="bara", valid_min=0.5, valid_max=45,
                       notes="NOT A SENSOR FAULT — a fixed upstream pressure setpoint, zero variance across "
                             "every available reading: 1.32 bara for all 17,520 hours on CGC-100A, 1.28 bara "
                             "for all 17,376 non-null hours on CGC-200B (remaining 144 = the shutdown NaN "
                             "block). A naive consecutive-run scan reports 14,160h for CGC-200B only because "
                             "it resets across that NaN gap — the value never actually changes. Matches "
                             "assets.design_suct_bara exactly. Exclude from model features; do NOT impute "
                             "or otherwise modify the raw values." if i == 1 else ""),
            ColumnSpec(f"p_s{i}_disch_bara", "float64", ColumnRole.INPUT, unit="bara", valid_min=0.5, valid_max=45),
            ColumnSpec(f"t_s{i}_suct_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=80),
            ColumnSpec(f"t_s{i}_disch_c", "float64", ColumnRole.INPUT, unit="°C", valid_min=0, valid_max=160,
                       notes="stage-3 discharge T on CGC-100A drifted +3-4°C 2024-10-27→2024-12-25 "
                             "with pressure ratio unchanged — classic sensor-drift signature to be caught "
                             "by the physics residual, not modeled as a real process change" if i == 3 else ""),
        ]
    ],
]

# ---------------------------------------------------------------------------
# ground_truth_reference.csv — simulation LATENT truth. VALIDATION / TARGET
# CONSTRUCTION ONLY. Every column here is ColumnRole.TARGET_SOURCE or EXCLUDE
# and must never appear in an inference-time feature vector.
# ---------------------------------------------------------------------------
GROUND_TRUTH_SCHEMA = [
    ColumnSpec("ts", "datetime64[ns]", ColumnRole.IDENTIFIER),
    ColumnSpec("train_id", "category", ColumnRole.IDENTIFIER),
    ColumnSpec("fouling_index_true", "float64", ColumnRole.TARGET_SOURCE, valid_min=0, valid_max=1,
               notes="LEAKAGE if used as a feature. Use only to label/validate a fouling or SEC-excess model."),
    ColumnSpec("polytropic_eff_true", "float64", ColumnRole.TARGET_SOURCE, valid_min=0, valid_max=1,
               notes="LEAKAGE if used as a feature. Compare your computed (from raw P/T/flow) efficiency against this for validation only."),
    ColumnSpec("specific_energy_kwh_per_t_true", "float64", ColumnRole.TARGET_SOURCE, unit="kWh/t",
               notes="LEAKAGE if used as a feature. NaN during the CGC-200B 2025-08-13→18 shutdown (no throughput -> SEC undefined)."),
    ColumnSpec("health_index_true", "float64", ColumnRole.TARGET_SOURCE, valid_min=0, valid_max=1,
               notes="LEAKAGE if used as a feature. Composite mechanical+performance health score, candidate anomaly/RUL label source."),
    ColumnSpec("rul_hours_to_next_wash_true", "float64", ColumnRole.TARGET_SOURCE, unit="h", valid_min=0,
               notes="LEAKAGE if used as a feature. Right-censored: NaN after the last wash of each train's series "
                     "(961 rows on CGC-100A tail, 2293 on CGC-200B tail) because no future wash exists to count down to."),
]

# ---------------------------------------------------------------------------
# events.csv — labels for anomaly-detection evaluation (precision/recall),
# and for building intervention-aware split boundaries. Not a per-row feature
# join target unless explicitly windowed/joined by the pipeline.
# ---------------------------------------------------------------------------
EVENTS_SCHEMA = [
    ColumnSpec("train_id", "category", ColumnRole.IDENTIFIER),
    ColumnSpec("ts_start", "datetime64[ns]", ColumnRole.IDENTIFIER),
    ColumnSpec("ts_end", "datetime64[ns]", ColumnRole.IDENTIFIER),
    ColumnSpec("event_type", "category", ColumnRole.TARGET_SOURCE,
               notes="online_wash | surge | sensor_fault | process_upset | degradation | trip | unscheduled_maintenance — "
                     "use to build anomaly ground-truth windows and wash-trigger labels"),
    ColumnSpec("severity", "category", ColumnRole.TARGET_SOURCE, notes="planned | warning | critical"),
    ColumnSpec("detail", "str", ColumnRole.EXCLUDE, notes="free text, human-readable only"),
]

# ---------------------------------------------------------------------------
# assets.csv — static per-train design specs and protective limits. Safe as
# input (join on train_id) since these are known a priori, not sensed.
# ---------------------------------------------------------------------------
ASSETS_SCHEMA = [
    ColumnSpec("train_id", "category", ColumnRole.IDENTIFIER),
    ColumnSpec("unit", "category", ColumnRole.EXCLUDE, notes="redundant with train_id"),
    ColumnSpec("service", "category", ColumnRole.EXCLUDE, notes="constant across both trains — no information"),
    ColumnSpec("oem", "category", ColumnRole.EXCLUDE, notes="constant across both trains — no information"),
    ColumnSpec("driver", "category", ColumnRole.EXCLUDE, notes="constant across both trains — no information"),
    ColumnSpec("num_stages", "int64", ColumnRole.EXCLUDE, notes="constant (5) — no information for this dataset"),
    ColumnSpec("design_speed_rpm", "int64", ColumnRole.DERIVED_INPUT, unit="rpm", notes="use to normalise speed_rpm per train"),
    ColumnSpec("min_gov_speed_rpm", "int64", ColumnRole.DERIVED_INPUT, unit="rpm", notes="operating envelope bound"),
    ColumnSpec("trip_speed_rpm", "int64", ColumnRole.DERIVED_INPUT, unit="rpm", notes="protective limit for RUL/anomaly logic"),
    ColumnSpec("design_flow_tph", "float64", ColumnRole.DERIVED_INPUT, unit="t/h", notes="use to normalise throughput_tph per train"),
    ColumnSpec("design_suct_bara", "float64", ColumnRole.DERIVED_INPUT, unit="bara", notes="matches the constant p_s1_suct_bara observed in telemetry"),
    ColumnSpec("design_disch_bara", "float64", ColumnRole.DERIVED_INPUT, unit="bara"),
    ColumnSpec("design_poly_eff_pct", "float64", ColumnRole.DERIVED_INPUT, unit="%", notes="clean/baseline efficiency reference for fouling-excess-energy calc"),
    ColumnSpec("gas_k_cp_cv", "float64", ColumnRole.DERIVED_INPUT, notes="ratio of specific heats, used in polytropic relations"),
    ColumnSpec("install_date", "datetime64[ns]", ColumnRole.EXCLUDE, notes="outside the 2024-2025 telemetry window; no variation to exploit"),
    ColumnSpec("surge_control_line_pct", "float64", ColumnRole.DERIVED_INPUT, unit="%", notes="protective limit for surge-margin logic"),
]


def as_lookup(schema):
    """Return {column_name: ColumnSpec} for quick access."""
    return {c.name: c for c in schema}


def input_columns(schema):
    return [c.name for c in schema if c.role in (ColumnRole.INPUT, ColumnRole.DERIVED_INPUT)]


def excluded_columns(schema):
    return [c.name for c in schema if c.role == ColumnRole.EXCLUDE]


def target_source_columns(schema):
    return [c.name for c in schema if c.role == ColumnRole.TARGET_SOURCE]


if __name__ == "__main__":
    # tiny self-check: no column appears in both an input schema and as a
    # ground-truth target-source column (this is the leakage guard in code form)
    tele_inputs = set(input_columns(TELEMETRY_SCHEMA))
    gt_targets = set(target_source_columns(GROUND_TRUTH_SCHEMA))
    overlap = tele_inputs & gt_targets
    assert not overlap, f"LEAKAGE: columns present in both telemetry inputs and ground-truth targets: {overlap}"
    print(f"OK: {len(tele_inputs)} telemetry input columns, {len(gt_targets)} ground-truth target-source "
          f"columns, zero overlap.")
