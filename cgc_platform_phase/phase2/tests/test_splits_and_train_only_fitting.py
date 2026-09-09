"""Definition-of-Done items: chronological split works correctly; scalers/
imputers/transformations are fitted only on the training split."""

import pandas as pd
import pytest

from backend.app.preprocessing import config, leakage_checks, splits


def test_split_boundaries_are_chronological_and_non_overlapping(telemetry):
    split_series = splits.assign_split(telemetry)
    leakage_checks.check_no_random_split_used(split_series, telemetry["ts"])  # should not raise

    bounds = splits.get_split_bounds()
    assert bounds["train"].end < bounds["validation"].start
    assert bounds["validation"].end < bounds["test"].start


def test_split_frame_partitions_every_row(telemetry):
    parts = splits.split_frame(telemetry)
    total = sum(len(df) for name, df in parts.items() if name != "unassigned")
    assert total + len(parts["unassigned"]) == len(telemetry)
    assert len(parts["unassigned"]) == 0  # dataset span is fully covered by the 3 windows


def test_train_only_returns_strictly_pre_validation_data(telemetry):
    train_df = splits.train_only(telemetry)
    bounds = splits.get_split_bounds()
    assert train_df["ts"].max() <= bounds["train"].end
    assert train_df["ts"].min() >= bounds["train"].start
    assert len(train_df) < len(telemetry)


def test_scaler_fits_only_on_train_split_and_flags_leakage_on_full_series(telemetry):
    train_df = splits.train_only(telemetry)
    scaler = splits.TrainOnlyStandardScaler(columns=["shaft_power_kw", "vib_de_um"])
    scaler.fit(train_df)

    # fitting on the train-only slice must not exceed the train split end
    leakage_checks.check_scaler_not_fit_on_full_series(scaler.fit_max_ts)  # should not raise

    # simulate the leakage bug: fitting on the FULL series (spans into test)
    bad_scaler = splits.TrainOnlyStandardScaler(columns=["shaft_power_kw"])
    bad_scaler.fit(telemetry)  # the whole 2-year series, not train_only(...)
    with pytest.raises(leakage_checks.LeakageError):
        leakage_checks.check_scaler_not_fit_on_full_series(bad_scaler.fit_max_ts)


def test_scaler_transform_values_are_train_stats_based(telemetry):
    train_df = splits.train_only(telemetry)
    parts = splits.split_frame(telemetry)
    scaler = splits.TrainOnlyStandardScaler(columns=["shaft_power_kw"])
    scaler.fit(train_df)

    val_transformed = scaler.transform(parts["validation"])
    # Recompute expected values by hand from TRAIN stats and confirm they
    # match — i.e. the scaler used train_df's mean/std, not validation's.
    train_mean = train_df["shaft_power_kw"].mean()
    train_std = train_df["shaft_power_kw"].std()
    expected_first = (parts["validation"]["shaft_power_kw"].iloc[0] - train_mean) / train_std
    assert val_transformed["shaft_power_kw"].iloc[0] == pytest.approx(expected_first)

    # And confirm it is NOT validation's own mean/std (guards against a
    # copy-paste bug where the scaler silently refits on whatever it's given).
    val_mean = parts["validation"]["shaft_power_kw"].mean()
    assert train_mean != pytest.approx(val_mean)


def test_assert_no_leakage_across_splits_helper():
    fit_max = pd.Timestamp("2025-06-30 23:00:00")
    # Applying to data starting strictly after fit_max: fine.
    splits.assert_no_leakage_across_splits(fit_max, pd.Timestamp("2025-07-01 00:00:00"))
    # Applying to data starting at/before fit_max: must raise.
    with pytest.raises(AssertionError):
        splits.assert_no_leakage_across_splits(fit_max, pd.Timestamp("2025-06-30 23:00:00"))
    with pytest.raises(AssertionError):
        splits.assert_no_leakage_across_splits(fit_max, pd.Timestamp("2024-01-01 00:00:00"))
