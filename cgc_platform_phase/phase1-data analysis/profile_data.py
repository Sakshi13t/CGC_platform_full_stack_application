"""
profile_data.py
================
Reproducible data-profiling script for the Cracked Gas Compressor (CGC)
reliability/energy-optimization dataset.

This script performs READ-ONLY analysis. It does not train any model and
does not modify the source CSVs. It prints a structured report to stdout
and also writes a few small derived CSVs into ./profile_outputs/ for
inspection (e.g. gap list, event overlap table).

Usage:
    python scripts/profile_data.py --data-dir /path/to/uploads

Inputs expected in --data-dir:
    compressor_telemetry.csv
    ground_truth_reference.csv
    events.csv
    assets.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 60)


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_data(data_dir):
    tele = pd.read_csv(os.path.join(data_dir, "compressor_telemetry.csv"))
    gt = pd.read_csv(os.path.join(data_dir, "ground_truth_reference.csv"))
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    assets = pd.read_csv(os.path.join(data_dir, "assets.csv"))

    for df, cols in [
        (tele, ["ts"]),
        (gt, ["ts"]),
        (events, ["ts_start", "ts_end"]),
    ]:
        for c in cols:
            df[c] = pd.to_datetime(df[c])

    return tele, gt, events, assets


def profile_shape(name, df):
    print(f"\n--- {name} ---")
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes)


def profile_timestamp_coverage(tele):
    hr("TIMESTAMP COVERAGE & FREQUENCY (per train)")
    for train, g in tele.groupby("train_id"):
        g = g.sort_values("ts")
        span_start, span_end = g["ts"].min(), g["ts"].max()
        n_rows = len(g)
        expected_hours = int((span_end - span_start).total_seconds() // 3600) + 1
        deltas = g["ts"].diff().dropna()
        modal_delta = deltas.mode().iloc[0] if not deltas.empty else None
        print(f"\n[{train}]")
        print(f"  span: {span_start} -> {span_end}")
        print(f"  n_rows: {n_rows}  expected_hourly_rows_if_no_gaps: {expected_hours}")
        print(f"  missing_hours (expected - actual): {expected_hours - n_rows}")
        print(f"  modal timestep: {modal_delta}")
        print(f"  distinct timestep values (top 5):\n{deltas.value_counts().head(5)}")


def profile_duplicates_and_order(tele):
    hr("DUPLICATES & OUT-OF-ORDER TIMESTAMPS (per train)")
    for train, g in tele.groupby("train_id"):
        dup_ts = g["ts"].duplicated().sum()
        g_sorted_idx = g["ts"].sort_values().index
        is_sorted_as_given = list(g.index) == list(g_sorted_idx)
        # out-of-order = original ordering not monotonic increasing
        monotonic = g["ts"].is_monotonic_increasing
        print(f"[{train}] duplicate timestamps: {dup_ts} | already sorted ascending in file: {monotonic}")
    dup_full_rows = tele.duplicated().sum()
    print(f"\nfully duplicated rows (all columns) across dataset: {dup_full_rows}")
    dup_key = tele.duplicated(subset=["ts", "train_id"]).sum()
    print(f"duplicated (ts, train_id) keys across dataset: {dup_key}")


def profile_gaps(tele, out_dir):
    hr("SHUTDOWN / GAP PERIODS (per train, gap > 1h)")
    all_gaps = []
    for train, g in tele.groupby("train_id"):
        g = g.sort_values("ts")
        deltas = g["ts"].diff()
        gap_mask = deltas > pd.Timedelta(hours=1)
        gap_rows = g.loc[gap_mask.fillna(False)]
        for idx in gap_rows.index:
            pos = g.index.get_loc(idx)
            prev_ts = g.iloc[pos - 1]["ts"]
            this_ts = g.loc[idx, "ts"]
            gap_len = this_ts - prev_ts
            all_gaps.append(
                {
                    "train_id": train,
                    "gap_start_after": prev_ts,
                    "gap_end_before": this_ts,
                    "gap_duration": gap_len,
                }
            )
        print(f"[{train}] number of gaps > 1h: {gap_mask.sum()}")
    gaps_df = pd.DataFrame(all_gaps)
    if not gaps_df.empty:
        gaps_df = gaps_df.sort_values(["train_id", "gap_start_after"])
        print("\nGap detail:")
        print(gaps_df.to_string(index=False))
        gaps_df.to_csv(os.path.join(out_dir, "detected_gaps.csv"), index=False)
    else:
        print("No gaps > 1h detected.")
    return gaps_df


def profile_missing_values(tele, gt):
    hr("MISSING VALUES")
    print("\n-- compressor_telemetry.csv --")
    miss = tele.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("No NaNs in any column (note: gaps show up as ABSENT ROWS, not NaNs — see gap analysis).")
    else:
        print(miss)
    print("\n-- ground_truth_reference.csv --")
    miss_gt = gt.isna().sum()
    miss_gt = miss_gt[miss_gt > 0]
    print("No NaNs." if miss_gt.empty else miss_gt)


def profile_constant_or_stuck(tele, out_dir):
    hr("CONSTANT / STUCK (FROZEN) SENSOR DETECTION")
    numeric_cols = tele.select_dtypes(include=[np.number]).columns.tolist()
    stuck_report = []
    for train, g in tele.groupby("train_id"):
        g = g.sort_values("ts")
        for col in numeric_cols:
            vals = g[col]
            # a "stuck" run = consecutive identical values for a long stretch
            same_as_prev = vals.diff().eq(0)
            # find max run length of consecutive identical values
            run_id = (~same_as_prev).cumsum()
            run_lengths = same_as_prev.groupby(run_id).sum() + 1
            max_run = run_lengths.max() if len(run_lengths) else 1
            if max_run >= 24:  # stuck for >= 24 consecutive hours
                # locate it
                longest_run_id = run_lengths.idxmax()
                run_positions = g.index[run_id == longest_run_id]
                start_ts = g.loc[run_positions, "ts"].min()
                end_ts = g.loc[run_positions, "ts"].max()
                stuck_val = g.loc[run_positions, col].iloc[0]
                stuck_report.append(
                    {
                        "train_id": train,
                        "column": col,
                        "max_consecutive_identical_hours": int(max_run),
                        "value": stuck_val,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    }
                )
    stuck_df = pd.DataFrame(stuck_report).sort_values(
        "max_consecutive_identical_hours", ascending=False
    )
    if not stuck_df.empty:
        print(stuck_df.to_string(index=False))
        stuck_df.to_csv(os.path.join(out_dir, "stuck_sensor_candidates.csv"), index=False)
    else:
        print("No column found stuck for >=24 consecutive hours.")

    # also flag globally-constant columns (zero variance across whole train)
    hr("GLOBALLY CONSTANT COLUMNS (zero variance within a train)")
    for train, g in tele.groupby("train_id"):
        zero_var = [c for c in numeric_cols if g[c].nunique(dropna=True) <= 1]
        print(f"[{train}] zero-variance columns: {zero_var if zero_var else 'none'}")


def profile_impossible_values(tele, assets):
    hr("PHYSICALLY IMPOSSIBLE / SUSPICIOUS VALUES")
    checks = []

    def check(mask, label):
        n = int(mask.sum())
        if n:
            checks.append((label, n))

    check(tele["throughput_tph"] < 0, "negative throughput_tph")
    check(tele["speed_rpm"] < 0, "negative speed_rpm")
    check(tele["shaft_power_kw"] < 0, "negative shaft_power_kw")
    check(tele["surge_margin_pct"] < 0, "negative surge_margin_pct (would imply surge)")
    check(tele["lube_oil_press_bar"] <= 0, "non-positive lube_oil_press_bar")
    check(tele["recycle_valve_pct"] < 0, "negative recycle_valve_pct")
    check(tele["recycle_valve_pct"] > 100, "recycle_valve_pct > 100")
    check(tele["antifoulant_ppm"] < 0, "negative antifoulant_ppm")
    check(tele["run_hours_since_wash"] < 0, "negative run_hours_since_wash")

    for i in range(1, 6):
        ps, pd_ = f"p_s{i}_suct_bara", f"p_s{i}_disch_bara"
        check(tele[pd_] <= tele[ps], f"stage {i}: discharge P <= suction P")
        ts_, td_ = f"t_s{i}_suct_c", f"t_s{i}_disch_c"
        check(tele[td_] <= tele[ts_], f"stage {i}: discharge T <= suction T")
        check(tele[ps] <= 0, f"stage {i}: non-positive suction pressure")

    # cross-train pressure continuity: stage i discharge should ~= stage i+1 suction (interstage cooling drops T not P much)
    for i in range(1, 5):
        pd_i, ps_ip1 = f"p_s{i}_disch_bara", f"p_s{i+1}_suct_bara"
        diff = (tele[pd_i] - tele[ps_ip1]).abs()
        check(diff > 0.5, f"stage {i}->{i+1}: discharge/suction pressure mismatch > 0.5 bara")

    # speed vs asset trip/min governor speed
    merged = tele.merge(assets[["train_id", "min_gov_speed_rpm", "trip_speed_rpm", "design_speed_rpm"]], on="train_id")
    check(merged["speed_rpm"] > merged["trip_speed_rpm"], "speed_rpm exceeds asset trip_speed_rpm")
    check(merged["speed_rpm"] < merged["min_gov_speed_rpm"] * 0.5, "speed_rpm far below min governor speed (<50%)")

    if checks:
        for label, n in checks:
            print(f"  {label}: {n} rows")
    else:
        print("  No impossible-value violations found in the checks run.")

    hr("RANGE / DISTRIBUTION SUMMARY (numeric columns)")
    numeric_cols = tele.select_dtypes(include=[np.number]).columns.tolist()
    desc = tele[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    print(desc.to_string())


def profile_relationship_to_ground_truth(tele, gt):
    hr("TELEMETRY <-> GROUND TRUTH RELATIONSHIP")
    merged = tele.merge(gt, on=["ts", "train_id"], how="inner", suffixes=("", "_gt"))
    print(f"telemetry rows: {len(tele)}, ground_truth rows: {len(gt)}, joined rows: {len(merged)}")
    if len(merged) != len(tele) or len(merged) != len(gt):
        print("WARNING: telemetry and ground_truth_reference do not have identical (ts, train_id) keys.")
    # correlation between a few observable telemetry signals and latent truth
    candidate_cols = [
        "run_hours_since_wash",
        "vib_de_um",
        "brg_de_temp_c",
        "surge_margin_pct",
        "shaft_power_kw",
        "throughput_tph",
    ]
    gt_cols = [
        "fouling_index_true",
        "polytropic_eff_true",
        "specific_energy_kwh_per_t_true",
        "health_index_true",
        "rul_hours_to_next_wash_true",
    ]
    present_candidates = [c for c in candidate_cols if c in merged.columns]
    present_gt = [c for c in gt_cols if c in merged.columns]
    corr = merged[present_candidates + present_gt].corr()
    print("\nCorrelation matrix (observable telemetry proxies vs latent ground truth):")
    print(corr.loc[present_candidates, present_gt].to_string())


def profile_events(tele, events):
    hr("EVENTS / ANOMALIES SUMMARY")
    print(events["event_type"].value_counts())
    print("\nBy train and type:")
    print(events.groupby(["train_id", "event_type"]).size())
    print("\nEvent durations (hours):")
    dur = (events["ts_end"] - events["ts_start"]).dt.total_seconds() / 3600
    print(dur.describe())
    print("\nFull event log:")
    print(events.to_string(index=False))


def profile_assets(assets, tele):
    hr("ASSET SPECIFICATIONS / LIMITS")
    print(assets.to_string(index=False))
    hr("TELEMETRY VS ASSET DESIGN LIMITS (spot-check on speed and throughput)")
    for _, row in assets.iterrows():
        train = row["train_id"]
        g = tele[tele["train_id"] == train]
        print(f"\n[{train}]")
        print(f"  design_speed_rpm={row['design_speed_rpm']}  observed speed_rpm range: "
              f"[{g['speed_rpm'].min():.1f}, {g['speed_rpm'].max():.1f}]")
        print(f"  trip_speed_rpm={row['trip_speed_rpm']}  rows >= trip speed: {(g['speed_rpm'] >= row['trip_speed_rpm']).sum()}")
        print(f"  design_flow_tph={row['design_flow_tph']}  observed throughput range: "
              f"[{g['throughput_tph'].min():.1f}, {g['throughput_tph'].max():.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/mnt/user-data/uploads")
    ap.add_argument("--out-dir", default="./profile_outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tele, gt, events, assets = load_data(args.data_dir)

    hr("BASIC SHAPES & DTYPES")
    profile_shape("compressor_telemetry.csv", tele)
    profile_shape("ground_truth_reference.csv", gt)
    profile_shape("events.csv", events)
    profile_shape("assets.csv", assets)

    profile_timestamp_coverage(tele)
    profile_duplicates_and_order(tele)
    profile_gaps(tele, args.out_dir)
    profile_missing_values(tele, gt)
    profile_constant_or_stuck(tele, args.out_dir)
    profile_impossible_values(tele, assets)
    profile_relationship_to_ground_truth(tele, gt)
    profile_events(tele, events)
    profile_assets(assets, tele)

    hr("DONE")


if __name__ == "__main__":
    main()
