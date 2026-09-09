"""
backend/app/preprocessing/splits.py
====================================
Chronological (never random) dataset splitting for CGC telemetry.

Split boundaries are a single named constant (config.SPLIT_BOUNDARIES),
documented in DATASET_PROFILE.md §19:

    train:      2024-01-01 00:00:00 -> 2025-06-30 23:00:00  (~18 months)
    validation: 2025-07-01 00:00:00 -> 2025-09-30 23:00:00
    test:       2025-10-01 00:00:00 -> 2025-12-30 23:00:00

The same three windows are applied independently to BOTH trains (i.e. the
split is by calendar time, not by train) so that a model evaluated on
"validation" is always being asked about a time period strictly after
everything it was fitted on, for both CGC-100A and CGC-200B.

Every "fit on training data only" requirement in Phase 2 (scalers,
imputers, baselines, physics clean-baseline references) should call
`train_only(df)` to get its fitting data.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from . import config


class SplitBounds(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive


def get_split_bounds() -> dict[str, SplitBounds]:
    """Parse config.SPLIT_BOUNDARIES into Timestamp pairs, once."""
    return {
        name: SplitBounds(pd.Timestamp(start), pd.Timestamp(end))
        for name, (start, end) in config.SPLIT_BOUNDARIES.items()
    }


def assign_split(df: pd.DataFrame, ts_col: str = "ts") -> pd.Series:
    """
    Return a Series of {"train", "validation", "test", None} aligned to
    df.index, based purely on `ts_col` against config.SPLIT_BOUNDARIES.
    A timestamp outside all three windows (there should be none, given
    the dataset's known span) is assigned None rather than silently
    dropped or coerced into the nearest split.
    """
    bounds = get_split_bounds()
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, b in bounds.items():
        mask = (df[ts_col] >= b.start) & (df[ts_col] <= b.end)
        result[mask] = name
    return result


def split_frame(df: pd.DataFrame, ts_col: str = "ts") -> dict[str, pd.DataFrame]:
    """Split `df` into {'train': ..., 'validation': ..., 'test': ...} sub-frames
    by calendar time. Rows outside all three windows (none expected for this
    dataset) are dropped from every returned frame and reported via the
    'unassigned' key so callers can detect an unexpected boundary problem."""
    split_col = assign_split(df, ts_col=ts_col)
    out = {name: df.loc[split_col == name].copy() for name in config.SPLIT_BOUNDARIES}
    out["unassigned"] = df.loc[split_col.isna()].copy()
    return out


def train_only(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Convenience accessor: rows belonging to the 'train' split only. Use
    this — and only this — as the fitting data for any scaler, imputer,
    or clean-baseline reference."""
    bounds = get_split_bounds()["train"]
    mask = (df[ts_col] >= bounds.start) & (df[ts_col] <= bounds.end)
    return df.loc[mask].copy()


def assert_no_leakage_across_splits(
    fit_max_ts: pd.Timestamp, applied_min_ts: pd.Timestamp, context: str = ""
) -> None:
    """
    Raise AssertionError if a transform "fitted" using data up to
    `fit_max_ts` is about to be applied to data starting before that same
    timestamp — i.e. if the fitting window and the application window
    overlap in time. Intended as a guard at the point a scaler/imputer's
    `.transform()` (or equivalent) is called on validation/test data.
    """
    if applied_min_ts <= fit_max_ts:
        raise AssertionError(
            f"Leakage guard tripped{f' ({context})' if context else ''}: "
            f"a transform fitted using data up to {fit_max_ts} is being applied "
            f"to data starting at {applied_min_ts}, which is not strictly after "
            f"the fitting window."
        )


class TrainOnlyStandardScaler:
    """
    Minimal, dependency-free standard scaler that can ONLY be fit on a
    DataFrame slice the caller has already restricted to the 'train'
    split (see `train_only`). This class does not know about ML models —
    it exists purely as the Phase-2 infrastructure piece requested by the
    spec ("prepare the pipeline so transformations ... are fitted ONLY on
    the training period"), for later phases to reuse.

    `fit_max_ts` is recorded at fit time and `transform` refuses to run on
    data that starts at or before it, closing the most common accidental-
    leakage path (fitting on the full series, or re-fitting per split).
    """

    def __init__(self, columns: list[str]):
        self.columns = columns
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._fit_max_ts: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame, ts_col: str = "ts") -> "TrainOnlyStandardScaler":
        self._mean = train_df[self.columns].mean()
        self._std = train_df[self.columns].std().replace(0, 1.0)
        self._fit_max_ts = train_df[ts_col].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted mean/std to `df`. This method does NOT itself
        decide whether `df` is in-sample (the training split, expected to
        be transformed too) or out-of-sample (validation/test) — both are
        legitimate calls. Before transforming validation/test data,
        callers should call `assert_no_leakage_across_splits(scaler.fit_max_ts,
        val_or_test_df['ts'].min())` explicitly to confirm the split
        boundary is respected; see splits_smoke-test in tests/ for the
        pattern.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("TrainOnlyStandardScaler.transform() called before fit().")
        out = df.copy()
        out[self.columns] = (df[self.columns] - self._mean) / self._std
        return out

    @property
    def fit_max_ts(self) -> pd.Timestamp:
        if self._fit_max_ts is None:
            raise RuntimeError("TrainOnlyStandardScaler has not been fit yet.")
        return self._fit_max_ts
