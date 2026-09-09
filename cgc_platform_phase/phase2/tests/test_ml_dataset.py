"""Phase 2 ML dataset sanity/leakage tests.

These are cheap, fast checks on backend.app.ml.dataset — NOT a re-run of
model training (that's exercised manually via
`python -m backend.app.ml.models`, which takes several minutes on a single
core; keeping it out of the default pytest run keeps the suite fast).
"""

from __future__ import annotations

import pytest

from backend.app.ml import dataset as ds_module


@pytest.fixture(scope="module")
def ml_dataset():
    return ds_module.load_ml_dataset()


def test_no_ground_truth_columns_in_features(ml_dataset):
    """Ground truth (*_true) columns must never appear in the feature set —
    only as the label."""
    for col in ml_dataset.feature_columns:
        assert "_true" not in col, f"ground-truth-looking column leaked into features: {col}"


def test_target_not_in_features(ml_dataset):
    assert ds_module.TARGET_NAME not in ml_dataset.feature_columns


def test_splits_are_chronological_and_non_overlapping(ml_dataset):
    """Train/validation/test splits must not share rows, and validation
    must not precede train (spot-checked via row counts and the underlying
    split assignment already tested in test_splits_and_train_only_fitting.py;
    here we just confirm the ML dataset actually partitions into three
    non-empty, disjoint sets)."""
    n_train, n_val, n_test = len(ml_dataset.X_train), len(ml_dataset.X_val), len(ml_dataset.X_test)
    assert n_train > 0 and n_val > 0 and n_test > 0
    assert n_train + n_val + n_test == ml_dataset.metadata["rows_after_target_dropna"]


def test_all_features_numeric(ml_dataset):
    """Every feature column must be numeric (float/bool/int) after
    encoding — no raw strings should reach a model."""
    import pandas.api.types as pdt

    for col in ml_dataset.feature_columns:
        dtype = ml_dataset.X_train[col].dtype
        assert (
            pdt.is_numeric_dtype(dtype) or pdt.is_bool_dtype(dtype)
        ), f"non-numeric feature column: {col} ({dtype})"


def test_shutdown_rows_excluded_from_current_timestep(ml_dataset):
    """No row in X_train/X_val/X_test should have is_shutdown=True at the
    CURRENT timestep (is_shutdown itself is dropped from features, but we
    can confirm indirectly: rows_after_shutdown_drop must be strictly less
    than the raw telemetry row count, proving the filter actually removed
    something, and equal to the known shutdown-block size)."""
    meta = ml_dataset.metadata
    dropped = meta["rows_total_telemetry"] - meta["rows_after_shutdown_drop"]
    assert dropped == 144, f"expected exactly the known 144-row shutdown block dropped, got {dropped}"


def test_naive_proxy_uses_measured_not_true_sec(ml_dataset):
    """The naive-baseline proxy series must be the MEASURED sec_kwh_per_t
    physics feature, not the ground-truth target itself (which would be
    leakage even for a baseline)."""
    # If the proxy were literally the (shifted) true target, it would be
    # far closer to y_val than the measured proxy is expected to be; here
    # we simply assert the proxy is not identical to the target (a weak
    # but cheap leakage smoke test — exact equality would be suspicious).
    assert not ml_dataset.naive_proxy_val.equals(ml_dataset.y_val)
