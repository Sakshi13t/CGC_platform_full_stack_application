"""
backend/app/preprocessing/leakage_checks.py
============================================
Executable leakage audit for the Phase 2 pipeline. Every check here is a
function that returns True/False (or raises) — not a comment or a manual
review note — so it can run in CI / tests / a pre-flight script.

Each function corresponds to one bullet in the Phase 2 spec's "Leakage
audit" section and, transitively, to a risk named in DATASET_PROFILE.md §16.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

from . import config
from .splits import assert_no_leakage_across_splits, get_split_bounds


class LeakageError(AssertionError):
    """Raised by any check in this module that fails."""


def check_no_ground_truth_columns_in_features(feature_columns: Iterable[str]) -> None:
    """*_true columns must never appear in a feature list."""
    offenders = set(feature_columns) & set(config.GROUND_TRUTH_TARGET_COLUMNS)
    if offenders:
        raise LeakageError(f"Ground-truth target columns found in feature list: {sorted(offenders)}")


def check_no_event_text_columns_in_features(feature_columns: Iterable[str]) -> None:
    """events.csv free-text / label columns must never appear in a feature list."""
    offenders = set(feature_columns) & set(config.EVENT_NON_FEATURE_COLUMNS + ["event_type", "severity"])
    if offenders:
        raise LeakageError(f"events.csv label/text columns found in feature list: {sorted(offenders)}")


def check_p_s1_suct_bara_excluded(feature_columns: Iterable[str]) -> None:
    """p_s1_suct_bara (constant-by-design, zero predictive info) must never
    be used as a predictive feature, per DATASET_PROFILE.md §9/§17."""
    if "p_s1_suct_bara" in set(feature_columns):
        raise LeakageError("p_s1_suct_bara must be excluded from predictive model features.")


def check_redundant_identifier_excluded(feature_columns: Iterable[str]) -> None:
    """'unit' is 1:1 redundant with train_id and should not double as a feature."""
    if "unit" in set(feature_columns):
        raise LeakageError("'unit' is redundant with train_id and should be excluded from features.")


def check_rolling_feature_no_future_leakage(
    build_feature: Callable[[pd.Series], pd.Series],
    raw_series: pd.Series,
    n_perturbations: int = 5,
) -> None:
    """
    Concrete, direct causality test for a feature-builder function.

    For `n_perturbations` randomly-chosen future indices, this recomputes
    the feature with that single future raw value drastically changed and
    asserts every feature value at or before that index is IDENTICAL to
    the original. A causal (backward-looking-only) feature builder must
    satisfy this by construction; a builder that leaks future information
    (e.g. via `center=True`, a backward `.shift(-1)`, or similar) will
    fail it.

    `build_feature` must be a pure function: pd.Series -> pd.Series of the
    same length/index, e.g. `lambda s: features.rolling_mean(s, 24, is_shutdown)`.
    """
    import numpy as np

    baseline = build_feature(raw_series)
    n = len(raw_series)
    if n < 20:
        raise ValueError("check_rolling_feature_no_future_leakage needs >=20 rows to be meaningful.")

    rng = np.random.default_rng(0)
    # only perturb indices with room for at least 5 prior rows to check
    candidate_positions = range(5, n)
    perturb_positions = rng.choice(list(candidate_positions), size=min(n_perturbations, n - 5), replace=False)

    for pos in perturb_positions:
        perturbed_raw = raw_series.copy()
        original_val = perturbed_raw.iloc[pos]
        perturbed_raw.iloc[pos] = (original_val if pd.notna(original_val) else 0.0) + 1.0e9

        perturbed_feature = build_feature(perturbed_raw)
        before = baseline.iloc[:pos]
        before_perturbed = perturbed_feature.iloc[:pos]

        mismatch = ~(before.eq(before_perturbed) | (before.isna() & before_perturbed.isna()))
        if mismatch.any():
            bad_idx = before.index[mismatch][0]
            raise LeakageError(
                f"Feature builder leaks future information: perturbing raw_series at position "
                f"{pos} changed the feature value at earlier position {bad_idx} "
                f"(before={before.loc[bad_idx]!r}, after={before_perturbed.loc[bad_idx]!r})."
            )


def check_scaler_not_fit_on_full_series(scaler_fit_max_ts: pd.Timestamp) -> None:
    """A fitted scaler's fit_max_ts must fall at or before the end of the
    documented 'train' split — never later (which would mean it was fit
    on validation/test/full-series data)."""
    train_end = get_split_bounds()["train"].end
    if scaler_fit_max_ts > train_end:
        raise LeakageError(
            f"Scaler appears fit using data after the train split ends "
            f"(fit_max_ts={scaler_fit_max_ts} > train end={train_end})."
        )


def check_no_random_split_used(split_series: pd.Series, ts: pd.Series) -> None:
    """
    Structural check that a split assignment is chronological, not random:
    within each split label, the assigned rows' timestamps must form a
    single contiguous time range that does not interleave with another
    split's range (a random split would scatter 'train'/'validation'/'test'
    labels across the full timeline; a chronological split partitions it
    into blocks).
    """
    frame = pd.DataFrame({"split": split_series, "ts": ts}).dropna(subset=["split"])
    bounds = frame.groupby("split")["ts"].agg(["min", "max"]).sort_values("min")
    ranges = list(bounds.itertuples(index=True))
    for a, b in zip(ranges, ranges[1:]):
        if a.max > b.min:
            raise LeakageError(
                f"Split ranges for '{a.Index}' and '{b.Index}' overlap "
                f"({a.Index} ends {a.max}, {b.Index} starts {b.min}) — "
                f"this is inconsistent with a chronological split."
            )


def check_shutdown_rows_flagged(df: pd.DataFrame, is_shutdown: pd.Series) -> None:
    """The known CGC-200B 2025-08-13->18 shutdown window must be entirely
    covered by is_shutdown == True (not partially missed)."""
    window = (
        (df["train_id"] == "CGC-200B")
        & (df["ts"] >= "2025-08-13 00:00:00")
        & (df["ts"] <= "2025-08-18 23:00:00")
    )
    if not is_shutdown[window].all():
        missed = int((~is_shutdown[window]).sum())
        raise LeakageError(f"{missed} rows inside the known CGC-200B shutdown window are not flagged is_shutdown.")


def check_throughput_fault_window_flagged(quality_frame: pd.DataFrame) -> None:
    """The known CGC-200B frozen-throughput fault window (2024-04-30 -> 2024-05-17)
    must be flagged BAD_SENSOR in throughput_tph_quality, not treated as GOOD."""
    window = (
        (quality_frame["train_id"] == "CGC-200B")
        & (quality_frame["ts"] >= "2024-04-30 00:00:00")
        & (quality_frame["ts"] <= "2024-05-17 23:00:00")
    )
    col = "throughput_tph_quality"
    if col not in quality_frame.columns:
        raise LeakageError(f"'{col}' column missing from quality_frame — cannot verify fault window is flagged.")
    bad = quality_frame.loc[window, col].eq("BAD_SENSOR")
    if not bad.all():
        n_not_bad = int((~bad).sum())
        raise LeakageError(
            f"{n_not_bad} rows inside the known throughput sensor-fault window are not flagged BAD_SENSOR."
        )


def run_all_checks(
    feature_columns: Iterable[str],
    df: pd.DataFrame,
    is_shutdown: pd.Series,
    quality_frame: pd.DataFrame,
    split_series: pd.Series | None = None,
) -> list[str]:
    """Run every static/structural leakage check and return a list of
    human-readable PASS messages. Raises LeakageError on the first failure
    (fail-fast), which is the desired behaviour for a pipeline gate."""
    messages = []

    check_no_ground_truth_columns_in_features(feature_columns)
    messages.append("PASS: no *_true columns in feature list")

    check_no_event_text_columns_in_features(feature_columns)
    messages.append("PASS: no events.csv label/text columns in feature list")

    check_p_s1_suct_bara_excluded(feature_columns)
    messages.append("PASS: p_s1_suct_bara excluded from feature list")

    check_redundant_identifier_excluded(feature_columns)
    messages.append("PASS: 'unit' excluded from feature list")

    check_shutdown_rows_flagged(df, is_shutdown)
    messages.append("PASS: known shutdown window fully flagged is_shutdown")

    check_throughput_fault_window_flagged(quality_frame)
    messages.append("PASS: known throughput sensor-fault window fully flagged BAD_SENSOR")

    if split_series is not None:
        check_no_random_split_used(split_series, df["ts"])
        messages.append("PASS: split assignment is chronological (non-overlapping ranges)")

    return messages
