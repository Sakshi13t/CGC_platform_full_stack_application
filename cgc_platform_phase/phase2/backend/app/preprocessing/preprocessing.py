"""
backend/app/preprocessing/preprocessing.py
===========================================
Top-level Phase 2 orchestrator. Wires together quality.py, physics_features.py,
features.py, splits.py, and leakage_checks.py into one reproducible run that:

  1. Loads the raw CSVs READ-ONLY (never writes back to data/).
  2. Builds the quality-flag frame (is_shutdown, <col>_quality).
  3. Builds the physics feature block (pressure ratios, observed efficiency,
     discharge-T residuals, SEC, polytropic head), with quality propagated.
  4. Builds causal, boundary-safe rolling/rate features for a documented
     column list.
  5. Assigns the chronological train/validation/test split.
  6. Runs the leakage audit and raises if anything fails.
  7. Writes ONLY derived artifacts to outputs/processed and
     outputs/diagnostics — never touches data/.

Run as a script:
    python -m backend.app.preprocessing.preprocessing
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config, features, leakage_checks, physics_features, quality, splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all four raw CSVs, parse timestamps, sort per train. Read-only:
    this function never writes to config.DATA_TIMESERIES_DIR / DATA_REFERENCE_DIR."""
    telemetry = pd.read_csv(config.TELEMETRY_CSV)
    telemetry["ts"] = pd.to_datetime(telemetry["ts"])
    telemetry = telemetry.sort_values(["train_id", "ts"]).reset_index(drop=True)

    ground_truth = pd.read_csv(config.GROUND_TRUTH_CSV)
    ground_truth["ts"] = pd.to_datetime(ground_truth["ts"])

    events = quality.load_events(config.EVENTS_CSV)
    assets = pd.read_csv(config.ASSETS_CSV)
    return telemetry, ground_truth, events, assets


def build_quality(telemetry: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return the quality frame keyed by (ts, train_id): is_shutdown plus
    per-fault-column BAD_SENSOR/quality columns. See quality.py."""
    logger.info("Building quality-flag frame...")
    return quality.build_quality_frame(telemetry, events)


def build_physics(telemetry: pd.DataFrame, assets: pd.DataFrame, quality_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the physics feature block. See physics_features.py."""
    logger.info("Building physics-derived features...")
    return physics_features.build_physics_features(telemetry, assets, quality_frame)


def build_rolling_features(telemetry: pd.DataFrame, quality_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build the documented rolling/rate feature block for a fixed set of
    high-value degradation/anomaly-precursor columns (vibration, bearing
    temperature, surge margin, run-hours-since-wash), at the window sizes
    in config.ROLLING_WINDOWS_HOURS, per train and reset at shutdown
    boundaries (see features.py).
    """
    logger.info("Building rolling/rate feature block...")
    columns = [
        "vib_de_um",
        "brg_de_temp_c",
        "surge_margin_pct",
        "run_hours_since_wash",
    ]
    is_shutdown_by_key = quality_frame.set_index(["ts", "train_id"])["is_shutdown"]

    pieces = []
    for train_id, g in telemetry.groupby("train_id", sort=False):
        g = g.sort_values("ts")
        is_shutdown = is_shutdown_by_key.loc[list(zip(g["ts"], g["train_id"]))]
        is_shutdown.index = g.index
        block = features.build_rolling_feature_block(
            g, is_shutdown, columns, config.ROLLING_WINDOWS_HOURS
        )
        block["ts"] = g["ts"]
        block["train_id"] = g["train_id"]
        pieces.append(block)
    return pd.concat(pieces, axis=0).reset_index(drop=True)


def build_interpolated_columns(telemetry: pd.DataFrame, quality_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative interpolation for the three columns confirmed (Phase 1) to
    share a short, isolated, non-shutdown, non-fault dropout pattern
    (config.INTERPOLATABLE_COLUMNS), bridging gaps of at most
    config.MAX_INTERPOLATION_GAP_HOURS hours. Returns a new DataFrame with
    `<col>_imputed` (the interpolated series) and `<col>_is_imputed` (bool
    flag) for each such column — the ORIGINAL raw column is untouched and
    is not included in this output.
    """
    logger.info("Building conservative short-gap interpolation columns...")
    is_shutdown_by_key = quality_frame.set_index(["ts", "train_id"])["is_shutdown"]
    out_pieces = []
    for train_id, g in telemetry.groupby("train_id", sort=False):
        g = g.sort_values("ts")
        is_shutdown = is_shutdown_by_key.loc[list(zip(g["ts"], g["train_id"]))]
        is_shutdown.index = g.index
        block = pd.DataFrame({"ts": g["ts"], "train_id": g["train_id"]})
        for col in config.INTERPOLATABLE_COLUMNS:
            eligible = quality.isolated_dropout_mask(
                g, col, is_shutdown, config.MAX_INTERPOLATION_GAP_HOURS
            )
            interpolated = g[col].interpolate(method="linear", limit=config.MAX_INTERPOLATION_GAP_HOURS)
            # Only accept the interpolated value where the gap was eligible
            # (isolated, short, non-shutdown); elsewhere keep the raw value
            # (which may still be NaN — that is the correct, honest state).
            filled = g[col].where(~eligible, interpolated)
            block[f"{col}_imputed"] = filled
            block[f"{col}_is_imputed"] = eligible
        out_pieces.append(block)
    return pd.concat(out_pieces, axis=0).reset_index(drop=True)


def get_predictive_feature_columns(telemetry: pd.DataFrame) -> list[str]:
    """
    The documented, explicit list of columns considered for predictive ML
    features downstream (Phase 3+) — i.e. every telemetry column EXCEPT
    identifiers and config.EXCLUDED_FROM_FEATURES. This function reads
    columns from data, but the EXCLUSION list itself is a fixed, reviewed
    constant (config.py), not inferred at runtime, per DATASET_PROFILE.md §17.
    """
    identifier_cols = {"ts", "train_id"}
    excluded = set(config.EXCLUDED_FROM_FEATURES)
    return [c for c in telemetry.columns if c not in identifier_cols and c not in excluded]


def run(write_outputs: bool = True) -> dict[str, pd.DataFrame]:
    """
    Execute the full Phase 2 pipeline and return all derived frames in a
    dict (also optionally persisted to config.PROCESSED_DIR /
    config.DIAGNOSTICS_DIR). Raises leakage_checks.LeakageError if any
    audit check fails — the pipeline does not produce partial/unsafe
    outputs on a failed audit.
    """
    telemetry, ground_truth, events, assets = load_raw()

    quality_frame = build_quality(telemetry, events)
    physics = build_physics(telemetry, assets, quality_frame)
    rolling = build_rolling_features(telemetry, quality_frame)
    interpolated = build_interpolated_columns(telemetry, quality_frame)

    split_series = splits.assign_split(telemetry)

    logger.info("Running leakage audit...")
    feature_columns = get_predictive_feature_columns(telemetry)
    audit_messages = leakage_checks.run_all_checks(
        feature_columns=feature_columns,
        df=telemetry,
        is_shutdown=quality_frame["is_shutdown"],  # quality_frame is row-aligned to telemetry (same index)
        quality_frame=quality_frame,
        split_series=split_series,
    )
    for m in audit_messages:
        logger.info(m)

    results = {
        "quality_frame": quality_frame,
        "physics_features": physics,
        "rolling_features": rolling,
        "interpolated_columns": interpolated,
        "split_assignment": pd.DataFrame({"ts": telemetry["ts"], "train_id": telemetry["train_id"], "split": split_series}),
    }

    if write_outputs:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        config.DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        for name, frame in results.items():
            path = config.PROCESSED_DIR / f"{name}.csv"
            frame.to_csv(path, index=False)
            logger.info(f"Wrote {path} ({len(frame)} rows)")

        with open(config.DIAGNOSTICS_DIR / "leakage_audit.log", "w") as f:
            f.write("\n".join(audit_messages) + "\n")

    return results


if __name__ == "__main__":
    run()
