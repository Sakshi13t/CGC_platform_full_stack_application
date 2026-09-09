"""Definition-of-Done items: p_s1_suct_bara excluded, throughput fault
flagged BAD, *_true columns can never enter the feature list."""

import pytest

from backend.app.preprocessing import config, leakage_checks, preprocessing


def test_p_s1_suct_bara_excluded_from_features(telemetry):
    feature_columns = preprocessing.get_predictive_feature_columns(telemetry)
    assert "p_s1_suct_bara" not in feature_columns
    with pytest.raises(leakage_checks.LeakageError):
        leakage_checks.check_p_s1_suct_bara_excluded(feature_columns + ["p_s1_suct_bara"])


def test_ground_truth_true_columns_can_never_enter_feature_list(telemetry):
    feature_columns = preprocessing.get_predictive_feature_columns(telemetry)
    leakage_checks.check_no_ground_truth_columns_in_features(feature_columns)  # should not raise

    for leaky_col in config.GROUND_TRUTH_TARGET_COLUMNS:
        with pytest.raises(leakage_checks.LeakageError):
            leakage_checks.check_no_ground_truth_columns_in_features(feature_columns + [leaky_col])


def test_event_text_columns_excluded(telemetry):
    feature_columns = preprocessing.get_predictive_feature_columns(telemetry)
    leakage_checks.check_no_event_text_columns_in_features(feature_columns)  # should not raise
    with pytest.raises(leakage_checks.LeakageError):
        leakage_checks.check_no_event_text_columns_in_features(feature_columns + ["detail"])


def test_throughput_fault_window_flagged_bad(pipeline_results):
    quality_frame = pipeline_results["quality_frame"]
    leakage_checks.check_throughput_fault_window_flagged(quality_frame)  # should not raise

    window = (
        (quality_frame["train_id"] == "CGC-200B")
        & (quality_frame["ts"] >= "2024-04-30 00:00:00")
        & (quality_frame["ts"] <= "2024-05-17 23:00:00")
    )
    assert (quality_frame.loc[window, "throughput_tph_quality"] == "BAD_SENSOR").all()
    assert window.sum() == 432


def test_p_s1_suct_bara_not_labeled_bad_anywhere(pipeline_results):
    """p_s1_suct_bara must never receive a BAD_SENSOR quality label — it is
    not a fault, and it has no <col>_quality column at all (by design,
    since it's not in config.SENSOR_FAULT_COLUMN_HINTS)."""
    quality_frame = pipeline_results["quality_frame"]
    assert "p_s1_suct_bara_quality" not in quality_frame.columns


def test_throughput_fault_check_fails_on_corrupted_frame(pipeline_results):
    quality_frame = pipeline_results["quality_frame"].copy()
    window = (quality_frame["train_id"] == "CGC-200B") & (
        quality_frame["ts"] == "2024-05-01 00:00:00"
    )
    quality_frame.loc[window, "throughput_tph_quality"] = "GOOD"
    with pytest.raises(leakage_checks.LeakageError):
        leakage_checks.check_throughput_fault_window_flagged(quality_frame)


def test_run_all_checks_passes_on_real_pipeline_output(telemetry, pipeline_results):
    feature_columns = preprocessing.get_predictive_feature_columns(telemetry)
    messages = leakage_checks.run_all_checks(
        feature_columns=feature_columns,
        df=telemetry,
        is_shutdown=pipeline_results["quality_frame"]["is_shutdown"],
        quality_frame=pipeline_results["quality_frame"],
        split_series=pipeline_results["split_assignment"]["split"],
    )
    assert len(messages) == 7
    assert all(m.startswith("PASS") for m in messages)
