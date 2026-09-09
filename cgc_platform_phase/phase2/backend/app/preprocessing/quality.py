"""
backend/app/preprocessing/quality.py
=====================================
Reusable data-quality flag mechanism for CGC telemetry.

Design principle (per Phase 2 spec): raw value + quality flag, never a
silently-overwritten raw value. Nothing in this module mutates the raw
DataFrame's original sensor columns — it only adds new columns.

Quality states (minimum set required):
    GOOD             measurement present, not shutdown, not a known fault
    MISSING          NaN, not attributable to shutdown or a known fault window
    BAD_SENSOR       known instrument fault (frozen/drifting transmitter)
    SHUTDOWN         machine not running (row falls in an identified outage)
    INVALID_PHYSICS  a *derived* physics feature whose required inputs were
                     not GOOD, or whose computed value is outside physically
                     possible bounds

Only two hand-picked columns get an explicit, persisted per-column quality
flag in this dataset (`throughput_tph`, `t_s3_disch_c`) because those are the
only two columns with a *documented, known* BAD_SENSOR window (events.csv).
Every other column's row-level trustworthiness is fully captured by the
shared `is_shutdown` flag plus a generic on-demand missingness classifier —
adding a `<col>_quality` column for all 41 telemetry columns would not add
information beyond that (see Phase-2 spec: "do not necessarily force every
column into every category if a simpler design is cleaner").
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from . import config


class Quality(str, Enum):
    GOOD = "GOOD"
    MISSING = "MISSING"
    BAD_SENSOR = "BAD_SENSOR"
    SHUTDOWN = "SHUTDOWN"
    INVALID_PHYSICS = "INVALID_PHYSICS"


def load_events(events_csv=config.EVENTS_CSV) -> pd.DataFrame:
    """Load events.csv with parsed timestamps. Read-only."""
    events = pd.read_csv(events_csv)
    events["ts_start"] = pd.to_datetime(events["ts_start"])
    events["ts_end"] = pd.to_datetime(events["ts_end"])
    return events


def compute_shutdown_mask(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series aligned to df.index: True where the row falls
    inside a documented shutdown window for that row's train_id.

    A "shutdown" is defined, per DATASET_PROFILE.md §5/§8, as an
    `unscheduled_maintenance` event window in events.csv (the only event
    type in this dataset representing the machine actually being down).
    We deliberately do NOT infer shutdowns purely from NaN density, because
    that would conflate shutdown with the unrelated BAD_SENSOR / isolated-
    dropout patterns — the event log is the authoritative source for "the
    machine was not running."
    """
    mask = pd.Series(False, index=df.index)
    shutdown_events = events[events["event_type"].isin(config.SHUTDOWN_EVENT_TYPES)]
    for _, ev in shutdown_events.iterrows():
        window = (
            (df["train_id"] == ev["train_id"])
            & (df["ts"] >= ev["ts_start"])
            & (df["ts"] <= ev["ts_end"])
        )
        mask |= window
    return mask


def compute_bad_sensor_mask(
    df: pd.DataFrame, events: pd.DataFrame, column: str
) -> pd.Series:
    """
    Return a boolean Series aligned to df.index: True where `column` is
    known-unreliable because it falls inside a `sensor_fault` event window
    that the documented `detail` text maps to this column
    (config.SENSOR_FAULT_COLUMN_HINTS).
    """
    mask = pd.Series(False, index=df.index)
    fault_events = events[events["event_type"] == "sensor_fault"]
    for _, ev in fault_events.iterrows():
        affected = config.SENSOR_FAULT_COLUMN_HINTS.get(ev["detail"], [])
        if column not in affected:
            continue
        window = (
            (df["train_id"] == ev["train_id"])
            & (df["ts"] >= ev["ts_start"])
            & (df["ts"] <= ev["ts_end"])
        )
        mask |= window
    return mask


def classify_column_quality(
    df: pd.DataFrame,
    column: str,
    is_shutdown: pd.Series,
    events: pd.DataFrame,
) -> pd.Series:
    """
    Generic per-value quality classifier for one column:
      SHUTDOWN    if is_shutdown is True for that row
      BAD_SENSOR  elif the row falls in a known sensor_fault window for
                  this column (regardless of whether the value is NaN or a
                  frozen numeric value — a frozen 243.85 is present but bad)
      MISSING     elif the raw value is NaN
      GOOD        otherwise
    """
    bad_sensor = compute_bad_sensor_mask(df, events, column)
    values = df[column]

    quality = pd.Series(Quality.GOOD.value, index=df.index)
    quality[values.isna()] = Quality.MISSING.value
    quality[bad_sensor] = Quality.BAD_SENSOR.value
    quality[is_shutdown] = Quality.SHUTDOWN.value  # shutdown takes priority
    return quality


def build_quality_frame(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full set of quality columns for the telemetry DataFrame:
      - is_shutdown (bool)
      - <col>_quality for every column named in config.SENSOR_FAULT_COLUMN_HINTS' values

    Returns a NEW DataFrame containing only `ts`, `train_id`, and the
    quality columns, meant to be joined back onto the raw telemetry rather
    than mutating it in place.
    """
    is_shutdown = compute_shutdown_mask(df, events)
    out = pd.DataFrame({"ts": df["ts"], "train_id": df["train_id"], "is_shutdown": is_shutdown})

    fault_columns = sorted({c for cols in config.SENSOR_FAULT_COLUMN_HINTS.values() for c in cols})
    for col in fault_columns:
        if col in df.columns:
            out[f"{col}_quality"] = classify_column_quality(df, col, is_shutdown, events)
    return out


def isolated_dropout_mask(
    df: pd.DataFrame, column: str, is_shutdown: pd.Series, max_gap_hours: int
) -> pd.Series:
    """
    True where `column` is NaN, NOT part of a shutdown, and the contiguous
    run of NaNs (per train, in time order) is <= max_gap_hours long.

    Used to decide which missing values are safe, conservative candidates
    for short-gap interpolation (DATASET_PROFILE.md §5's "sporadic single-
    hour dropout" pattern), as opposed to a shutdown or an extended,
    unexplained gap that should stay missing.
    """
    result = pd.Series(False, index=df.index)
    for train_id, g in df.groupby("train_id"):
        g = g.sort_values("ts")
        na = g[column].isna() & ~is_shutdown.loc[g.index]
        run_id = (~na).cumsum()
        run_lengths = na.groupby(run_id).transform("sum")
        eligible = na & (run_lengths <= max_gap_hours)
        result.loc[g.index] = eligible
    return result
