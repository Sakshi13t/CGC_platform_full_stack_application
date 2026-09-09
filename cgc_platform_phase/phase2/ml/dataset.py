"""
backend/app/ml/dataset.py
==========================
Phase 2 (Classical ML) — assembles the leakage-safe feature/target matrix
for forecasting a near-future degradation/performance KPI.

Target
------
specific_energy_kwh_per_t_true, H=24 hours ahead, per train.

SEC (specific energy consumption, kWh/tonne) is the assignment's own
headline energy KPI (Section 0.3 / EM-CGC-05) and rises monotonically with
fouling between washes, which is exactly the "near-future
degradation/performance KPI" Phase 2 asks for. The *_true column (from
ground_truth_reference.csv) is used ONLY as the label — never as a feature,
per the hard rule in the assignment brief. 24h was chosen as the forecast
horizon because it is short enough to be actionable (a shift lead can react
within a day) while long enough that hour-to-hour sensor noise averages out;
it is a stated assumption, not something the data dictated.

Feature set (all computed from information available at time t; nothing
here is derived from ground_truth_reference.csv):
  - Raw telemetry inputs (per data_schema.py's INPUT role, minus
    EXCLUDED_FROM_FEATURES) — feed_mode is one-hot encoded.
  - Physics-derived features from physics_features.py (pressure ratios,
    observed polytropic efficiency per stage, discharge-T residuals per
    stage, overall pressure ratio, the MEASURED/noisy sec_kwh_per_t, the
    below-min-surge-margin flag).
  - Causal rolling features from features.py (24h/168h mean/std/min/max +
    rate-of-change for vib_de_um, brg_de_temp_c, surge_margin_pct,
    run_hours_since_wash).

Rows excluded from training/evaluation (documented, not silently dropped):
  - Any row where the CURRENT timestep is a shutdown row (is_shutdown) —
    the machine is not running, so "predict SEC 24h from now" is not a
    meaningful operating question at that instant.
  - Any row where the target is NaN (this already captures both the
    144-row shutdown gap in ground truth and the last 24h of each train's
    series, which has no t+24 label).

Splits: assigned by the CURRENT timestep using the existing chronological
boundaries in config.SPLIT_BOUNDARIES (splits.assign_split) — never a
random split, per the assignment's leakage rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..preprocessing import config, features, physics_features, preprocessing, quality, splits

FORECAST_HORIZON_HOURS = 24
TARGET_COLUMN_TRUE = "specific_energy_kwh_per_t_true"
TARGET_NAME = f"sec_true_t_plus_{FORECAST_HORIZON_HOURS}h"

ROLLING_COLUMNS = ["vib_de_um", "brg_de_temp_c", "surge_margin_pct", "run_hours_since_wash"]

# Raw telemetry columns used directly as features (INPUT role per
# data_schema.py, minus config.EXCLUDED_FROM_FEATURES, minus the
# identifiers). feed_mode is categorical and one-hot encoded separately.
RAW_NUMERIC_FEATURE_COLUMNS = [
    "gas_mw", "diene_ppm", "ambient_temp_c", "cw_supply_temp_c",
    "throughput_tph", "speed_rpm", "shaft_power_kw", "surge_margin_pct",
    "recycle_valve_pct", "wash_oil_rate_kgph", "antifoulant_ppm",
    "run_hours_since_wash",
    "vib_de_um", "vib_nde_um", "vib_axial_um",
    "brg_de_temp_c", "brg_nde_temp_c", "brg_thrust_temp_c",
    "lube_oil_press_bar", "lube_oil_temp_c", "seal_gas_dp_bar",
] + [
    f"{p}_s{i}_{suffix}"
    for i in range(1, 6)
    for p, suffix in [("p", "disch_bara"), ("t", "suct_c"), ("t", "disch_c")]
] + [f"p_s{i}_suct_bara" for i in range(2, 6)]  # p_s1_suct_bara excluded (constant setpoint)

RAW_CATEGORICAL_FEATURE_COLUMNS = ["feed_mode"]


@dataclass
class MLDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    naive_proxy_train: pd.Series  # current measured sec_kwh_per_t, aligned to X_train — for the naive baseline
    naive_proxy_val: pd.Series
    naive_proxy_test: pd.Series
    metadata: dict = field(default_factory=dict)


def _shift_target_per_train(ground_truth: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return a frame keyed by (ts, train_id) whose target column holds the
    *_true value observed `horizon` hours later on the SAME train. Uses a
    plain per-train shift(-horizon) on the hourly grid (both trains have a
    complete, evenly-spaced hourly index — confirmed in DATASET_PROFILE.md
    — so a row-count shift is equivalent to a time-based shift here)."""
    gt = ground_truth.sort_values(["train_id", "ts"]).reset_index(drop=True)
    gt[TARGET_NAME] = gt.groupby("train_id")[TARGET_COLUMN_TRUE].shift(-horizon)
    return gt[["ts", "train_id", TARGET_NAME]]


def build_feature_frame(
    telemetry: pd.DataFrame, quality_frame: pd.DataFrame, physics: pd.DataFrame, rolling: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the full (ts, train_id)-keyed feature frame from raw
    telemetry + already-built physics/rolling blocks. No ground-truth
    column is ever read in this function."""
    df = telemetry[["ts", "train_id"] + RAW_NUMERIC_FEATURE_COLUMNS + RAW_CATEGORICAL_FEATURE_COLUMNS].copy()
    df = df.merge(physics, on=["ts", "train_id"], how="left", suffixes=("", "_physics"))
    df = df.merge(rolling, on=["ts", "train_id"], how="left", suffixes=("", "_roll"))
    df = df.merge(quality_frame[["ts", "train_id", "is_shutdown"]], on=["ts", "train_id"], how="left")

    # feed_mode plus any physics/quality-flag columns that came back as
    # strings (e.g. stage{i}_physics_quality, sec_quality — the Quality
    # enum values from quality.py) all need one-hot encoding; identified
    # generically rather than hand-listed so a future new quality column
    # doesn't silently leak through as an unencoded string.
    protected = {"ts", "train_id"}
    categorical_cols = [
        c for c in df.columns
        if c not in protected
        and (pd.api.types.is_object_dtype(df[c])
             or pd.api.types.is_string_dtype(df[c])
             or isinstance(df[c].dtype, pd.CategoricalDtype))
    ]
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    return df


def load_ml_dataset(horizon: int = FORECAST_HORIZON_HOURS) -> MLDataset:
    telemetry, ground_truth, events, assets = preprocessing.load_raw()
    quality_frame = preprocessing.build_quality(telemetry, events)
    physics = preprocessing.build_physics(telemetry, assets, quality_frame)
    rolling = preprocessing.build_rolling_features(telemetry, quality_frame)

    features_frame = build_feature_frame(telemetry, quality_frame, physics, rolling)
    target_frame = _shift_target_per_train(ground_truth, horizon)

    full = features_frame.merge(target_frame, on=["ts", "train_id"], how="left")
    split_series = splits.assign_split(telemetry)
    full = full.merge(
        pd.DataFrame({"ts": telemetry["ts"], "train_id": telemetry["train_id"], "split": split_series}),
        on=["ts", "train_id"], how="left",
    )

    n_before = len(full)
    full = full[~full["is_shutdown"].astype(bool)]
    n_after_shutdown_drop = len(full)
    full = full.dropna(subset=[TARGET_NAME])
    n_after_target_drop = len(full)

    non_feature_cols = {"ts", "train_id", "is_shutdown", "split", TARGET_NAME}
    feature_columns = [c for c in full.columns if c not in non_feature_cols]

    # Any remaining NaNs in engineered features (rolling windows warming up
    # right after a shutdown boundary, or a physics feature undefined
    # during the known BAD_SENSOR window) are left as NaN here — no blind
    # fillna in this module. XGBoost handles NaN splits natively; the
    # RandomForest baseline needs a train-only-fit imputer, applied in
    # ml/models.py (not here) so the imputation strategy is a modeling
    # choice, not baked into the dataset.

    def _split(name: str):
        mask = full["split"] == name
        X = full.loc[mask, feature_columns].reset_index(drop=True)
        y = full.loc[mask, TARGET_NAME].reset_index(drop=True)
        naive = full.loc[mask, "sec_kwh_per_t"].reset_index(drop=True)
        return X, y, naive

    X_train, y_train, naive_train = _split("train")
    X_val, y_val, naive_val = _split("validation")
    X_test, y_test, naive_test = _split("test")

    metadata = {
        "target": TARGET_NAME,
        "target_source_column": TARGET_COLUMN_TRUE,
        "forecast_horizon_hours": horizon,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "rows_total_telemetry": n_before,
        "rows_after_shutdown_drop": n_after_shutdown_drop,
        "rows_after_target_dropna": n_after_target_drop,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "naive_baseline_definition": (
            "current measured (noisy) sec_kwh_per_t physics feature at time t, "
            "used unchanged as the forecast for specific_energy_kwh_per_t_true at t+H "
            "(persistence assumption: 'tomorrow looks like today')."
        ),
    }

    return MLDataset(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_columns=feature_columns,
        naive_proxy_train=naive_train, naive_proxy_val=naive_val, naive_proxy_test=naive_test,
        metadata=metadata,
    )
