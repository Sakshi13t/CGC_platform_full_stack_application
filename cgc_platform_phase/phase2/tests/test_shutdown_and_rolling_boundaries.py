"""Definition-of-Done items: shutdown rows identified correctly; rolling
windows do not cross shutdowns (or train boundaries)."""

import numpy as np
import pandas as pd

from backend.app.preprocessing import features


def test_shutdown_rows_identified_correctly(pipeline_results):
    q = pipeline_results["quality_frame"]
    window = (
        (q["train_id"] == "CGC-200B")
        & (q["ts"] >= "2025-08-13 00:00:00")
        & (q["ts"] <= "2025-08-18 23:00:00")
    )
    assert window.sum() == 144
    assert q.loc[window, "is_shutdown"].all()
    # and nothing outside that window (for CGC-200B) is mis-flagged True,
    # except possibly other legitimate shutdowns — there are none in this
    # dataset, so is_shutdown must be exactly this one 144-row block.
    assert q["is_shutdown"].sum() == 144


def test_rolling_mean_is_nan_for_shutdown_rows():
    n = 48
    is_shutdown = pd.Series([False] * 20 + [True] * 6 + [False] * (n - 26))
    series = pd.Series(np.arange(n, dtype=float))
    result = features.rolling_mean(series, window_hours=12, is_shutdown=is_shutdown, min_periods=1)
    assert result[is_shutdown].isna().all()


def test_rolling_mean_resets_after_shutdown_does_not_include_preshutdown_values():
    n = 40
    is_shutdown = pd.Series([False] * 15 + [True] * 5 + [False] * (n - 20))
    # Pre-shutdown values are huge; post-shutdown values are small.
    series = pd.Series([1000.0] * 15 + [np.nan] * 5 + [1.0] * (n - 20))
    result = features.rolling_mean(series, window_hours=24, is_shutdown=is_shutdown, min_periods=1)
    first_post_shutdown_idx = 20
    # If the window incorrectly spanned the shutdown, this value would be
    # pulled up toward 1000; because it must reset, it should stay near 1.0.
    assert result.iloc[first_post_shutdown_idx] < 2.0


def test_rate_of_change_is_nan_across_shutdown_boundary():
    n = 10
    is_shutdown = pd.Series([False, False, False, True, True, False, False, False, False, False])
    series = pd.Series([10.0, 11.0, 12.0, np.nan, np.nan, 500.0, 501.0, 502.0, 503.0, 504.0])
    result = features.rate_of_change(series, is_shutdown)
    # The first row after the shutdown must NOT be a diff against the last
    # pre-shutdown value (12.0 -> 500.0 would be a huge, meaningless jump).
    assert pd.isna(result.iloc[5])
    # But once inside the new segment, ordinary diffs resume.
    assert result.iloc[6] == 1.0


def test_rolling_features_never_cross_train_boundary(pipeline_results, telemetry):
    """Structural check: apply_per_train guarantees this by construction,
    but assert it end-to-end on the real pipeline output by confirming the
    first row of each train's series has a NaN 168h-rolling feature (it
    cannot have 168 hours of history from a DIFFERENT train)."""
    roll = pipeline_results["rolling_features"]
    for train_id, g in roll.groupby("train_id"):
        first_row = g.sort_values("ts").iloc[0]
        assert pd.isna(first_row["vib_de_um_roll_mean_168h"])
