"""Definition-of-Done item: rolling features do not use future values."""

import numpy as np
import pandas as pd

from backend.app.preprocessing import features, leakage_checks


def _make_series(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n).cumsum())


def test_rolling_mean_is_causal():
    series = _make_series()
    is_shutdown = pd.Series(False, index=series.index)

    def build(s):
        return features.rolling_mean(s, window_hours=12, is_shutdown=is_shutdown, min_periods=1)

    leakage_checks.check_rolling_feature_no_future_leakage(build, series, n_perturbations=8)


def test_rolling_std_is_causal():
    series = _make_series(seed=1)
    is_shutdown = pd.Series(False, index=series.index)

    def build(s):
        return features.rolling_std(s, window_hours=12, is_shutdown=is_shutdown, min_periods=2)

    leakage_checks.check_rolling_feature_no_future_leakage(build, series, n_perturbations=8)


def test_rate_of_change_is_causal():
    series = _make_series(seed=2)
    is_shutdown = pd.Series(False, index=series.index)

    def build(s):
        return features.rate_of_change(s, is_shutdown=is_shutdown)

    leakage_checks.check_rolling_feature_no_future_leakage(build, series, n_perturbations=8)


def test_a_deliberately_noncausal_feature_is_caught():
    """Sanity-check the checker itself: a centered rolling mean (which DOES
    peek at future values) must be flagged as leaking."""
    series = _make_series(seed=3)

    def build_leaky(s):
        return s.rolling(window=12, center=True, min_periods=1).mean()

    try:
        leakage_checks.check_rolling_feature_no_future_leakage(build_leaky, series, n_perturbations=8)
    except leakage_checks.LeakageError:
        return  # expected
    raise AssertionError("Centered rolling mean should have been caught as non-causal, but wasn't.")
