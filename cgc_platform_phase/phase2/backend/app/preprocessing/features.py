"""
backend/app/preprocessing/features.py
======================================
Reusable, leakage-safe temporal feature builders (rolling stats, rate of
change, delta) for CGC telemetry.

Leakage-safety rules enforced by every function in this module:

  1. All windows are CAUSAL (backward-looking only): a feature value at
     time t is a function of rows with timestamp <= t. `pandas.rolling`
     with default arguments is already causal — this module never uses
     `center=True` and never shifts a feature backward in time.
  2. All windows are computed PER TRAIN. A rolling window never mixes rows
     from CGC-100A and CGC-200B.
  3. All windows RESET across a shutdown boundary. A rolling statistic
     computed on the first hour after the CGC-200B 2025-08-13->18 outage
     will not include any pre-shutdown hours.
  4. Nothing here fabricates values across a gap — a rolling mean over a
     window that contains fewer than `min_periods` valid (non-NaN,
     non-shutdown) observations returns NaN rather than a value computed
     from a partial/contaminated window.

These functions operate on a single-train, time-sorted DataFrame slice by
design (see `apply_per_train`), so callers get boundary-safety "for free"
by construction rather than by remembering to group before rolling.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd


def apply_per_train(
    df: pd.DataFrame,
    fn: Callable[[pd.DataFrame], pd.DataFrame],
    train_col: str = "train_id",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Group `df` by `train_col`, sort each group by `ts_col`, apply `fn` to
    each group independently, and concatenate the results back together
    in original row order.

    This is the single choke point that guarantees every feature builder
    in this module is train-boundary-safe: `fn` never sees more than one
    train's rows in a single call.
    """
    pieces = [fn(g.sort_values(ts_col)) for _, g in df.groupby(train_col, sort=False)]
    result = pd.concat(pieces, axis=0)
    return result.reindex(df.index)


def _shutdown_reset_groups(is_shutdown: pd.Series) -> pd.Series:
    """
    Return an integer "segment id" Series that increments every time the
    shutdown status flips (running -> shutdown, or shutdown -> running),
    so that `groupby(segment_id)` on the resulting groups never lets a
    rolling/diff computation span across a shutdown boundary.
    """
    shutdown_int = is_shutdown.astype(int)
    return shutdown_int.ne(shutdown_int.shift(fill_value=shutdown_int.iloc[0])).cumsum()


def rolling_mean(
    series: pd.Series,
    window_hours: int,
    is_shutdown: pd.Series,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Causal rolling mean over `window_hours` hourly rows, computed within a
    single train's time-sorted series, reset at each shutdown boundary,
    and forced to NaN for any row that is itself inside a shutdown (a
    shutdown row has no meaningful "current" telemetry value to trend).

    `series`, `is_shutdown` must share the same (already train-sorted)
    index. Use via `apply_per_train` for multi-train DataFrames.
    """
    min_periods = min_periods or max(1, window_hours // 2)
    segment = _shutdown_reset_groups(is_shutdown)
    result = (
        series.groupby(segment)
        .apply(lambda s: s.rolling(window=window_hours, min_periods=min_periods).mean())
    )
    result.index = result.index.droplevel(0)
    result = result.reindex(series.index)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def rolling_std(
    series: pd.Series,
    window_hours: int,
    is_shutdown: pd.Series,
    min_periods: int | None = None,
) -> pd.Series:
    """Causal rolling standard deviation. Same boundary-safety as `rolling_mean`."""
    min_periods = min_periods or max(2, window_hours // 2)
    segment = _shutdown_reset_groups(is_shutdown)
    result = (
        series.groupby(segment)
        .apply(lambda s: s.rolling(window=window_hours, min_periods=min_periods).std())
    )
    result.index = result.index.droplevel(0)
    result = result.reindex(series.index)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def rolling_min(series: pd.Series, window_hours: int, is_shutdown: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Causal rolling minimum. Same boundary-safety as `rolling_mean`."""
    min_periods = min_periods or max(1, window_hours // 2)
    segment = _shutdown_reset_groups(is_shutdown)
    result = series.groupby(segment).apply(lambda s: s.rolling(window=window_hours, min_periods=min_periods).min())
    result.index = result.index.droplevel(0)
    result = result.reindex(series.index)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def rolling_max(series: pd.Series, window_hours: int, is_shutdown: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Causal rolling maximum. Same boundary-safety as `rolling_mean`."""
    min_periods = min_periods or max(1, window_hours // 2)
    segment = _shutdown_reset_groups(is_shutdown)
    result = series.groupby(segment).apply(lambda s: s.rolling(window=window_hours, min_periods=min_periods).max())
    result.index = result.index.droplevel(0)
    result = result.reindex(series.index)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def rate_of_change(series: pd.Series, is_shutdown: pd.Series, periods: int = 1) -> pd.Series:
    """
    Causal first difference: value[t] - value[t - periods]. NaN across a
    shutdown boundary (a rate computed from "last good reading before the
    outage" to "first reading after it" would be a multi-day average
    disguised as an hourly rate, which is misleading) and NaN for any row
    that is itself inside a shutdown.
    """
    segment = _shutdown_reset_groups(is_shutdown)
    result = series.groupby(segment).diff(periods=periods)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def pct_change(series: pd.Series, is_shutdown: pd.Series, periods: int = 1) -> pd.Series:
    """
    Causal percentage change, same boundary rules as `rate_of_change`.
    Only physically meaningful for strictly-positive quantities (e.g.
    throughput, pressure) — caller should not apply this to a signal that
    can legitimately cross zero (e.g. surge_margin_pct).
    """
    segment = _shutdown_reset_groups(is_shutdown)
    result = series.groupby(segment).pct_change(periods=periods)
    result[is_shutdown.astype(bool)] = np.nan
    return result


def build_rolling_feature_block(
    df: pd.DataFrame,
    is_shutdown: pd.Series,
    columns: Iterable[str],
    window_hours: Iterable[int],
) -> pd.DataFrame:
    """
    Convenience orchestrator: for a train-and-time-sorted `df` (already
    the result of one `apply_per_train` group, i.e. single train), build
    rolling mean/std and a 1-hour rate-of-change for every (column,
    window) pair. Returns a new DataFrame of feature columns only.

    Column naming: `<col>_roll_mean_<w>h`, `<col>_roll_std_<w>h`,
    `<col>_rate_1h`.
    """
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[f"{col}_rate_1h"] = rate_of_change(df[col], is_shutdown)
        for w in window_hours:
            out[f"{col}_roll_mean_{w}h"] = rolling_mean(df[col], w, is_shutdown)
            out[f"{col}_roll_std_{w}h"] = rolling_std(df[col], w, is_shutdown)
    return out
