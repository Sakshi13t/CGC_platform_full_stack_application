# DATASET_PROFILE.md

Profiling report for the Cracked Gas Compressor (CGC) reliability & energy-optimization dataset.
Scope of this document: **data understanding only**. No model is trained here; no API/frontend
code is built. This is the required precursor to Sections 3–9 of the assignment.

Reproduce everything below with:
```
python scripts/profile_data.py --data-dir /path/to/uploads --out-dir ./profile_outputs
```

---

## 1. Files & shapes

| File | Rows | Columns | Role |
|---|---|---|---|
| `compressor_telemetry.csv` | 35,040 | 45 | **Model input** — sensor & operating telemetry |
| `ground_truth_reference.csv` | 35,040 | 7 | **Validation / target construction ONLY — never a feature** |
| `events.csv` | 22 | 6 | Labeled incidents — anomaly-detection evaluation, wash-trigger labels |
| `assets.csv` | 2 | 16 | Static per-train design specs & protective limits |

35,040 = 2 trains × 17,520 hourly rows = 2 years (2024-01-01 00:00 → 2025-12-30 23:00) per train, exactly as stated in the assignment brief.

## 2. Columns & data types

- `compressor_telemetry.csv`: 1 timestamp (`ts`), 2 identifier/categorical strings (`train_id`, `unit`), 1 categorical (`feed_mode`), and 41 numeric (`float64`) sensor/operating columns — gas composition, throughput, speed, power, surge margin, recycle valve, mitigation actions (wash-oil rate, antifoulant ppm), run-hours-since-wash, vibration (DE/NDE/axial), bearing temps (DE/NDE/thrust), lube oil (pressure/temp), seal-gas dP, and 5 stages × (suction P, discharge P, suction T, discharge T).
- `ground_truth_reference.csv`: `ts`, `train_id`, and 5 latent-truth floats (`fouling_index_true`, `polytropic_eff_true`, `specific_energy_kwh_per_t_true`, `health_index_true`, `rul_hours_to_next_wash_true`).
- `events.csv`: `train_id`, `ts_start`, `ts_end` (datetimes), `event_type`/`severity` (categorical), `detail` (free text).
- `assets.csv`: `train_id`, several categorical/constant descriptors, and numeric design limits (speeds, flows, pressures, efficiency, `gas_k_cp_cv`), plus `install_date`.

Full per-column schema, roles, units and validation ranges are codified in `data_schema.py`.

## 3. Timestamp coverage & frequency

- Both trains span **2024-01-01 00:00:00 → 2025-12-30 23:00:00**, hourly (`freq='1h'`), with **exactly 17,520 rows each** and **zero missing hourly rows** (no row-level gaps — every clock hour has a row for every train).
- Modal timestep is 1h with no deviation in the row cadence.

**Important nuance:** "no missing rows" does not mean "no shutdown." The simulator kept a row for every hour but set most sensor columns to `NaN` during the one true shutdown window in the data (see §7). Row presence and process availability are two different things here — a downstream pipeline must not confuse "a row exists" with "the machine was running."

## 4. Train identifiers

- `train_id` ∈ {`CGC-100A`, `CGC-200B`}, perfectly balanced (17,520 rows each).
- `unit` is 1:1 with `train_id` (`CGC-100A` ↔ `Ethylene Unit 100`, `CGC-200B` ↔ `Ethylene Unit 200`) — redundant identifier.
- `assets.csv` has exactly one row per train and joins cleanly on `train_id`.

## 5. Missing values

**`compressor_telemetry.csv`** — two distinct, unrelated patterns:

1. **A single 6-day (144-hour) block of NaNs across ~28 sensor columns** on `CGC-200B`, from **2025-08-13 00:00 → 2025-08-18 23:00**. This lines up exactly with the `unscheduled_maintenance` event in `events.csv` ("Bearing replacement; train down 6 days") following the high-vibration `trip` at 2025-08-13 00:00–02:00. This is a genuine shutdown: the machine was not running, so throughput/speed/power/vibration/pressures/temperatures are correctly undefined rather than fabricated. **Do not forward-fill or interpolate across this window** — treat it as a distinct "not running" state.
2. **Sporadic, single-hour dropouts shared identically across exactly three columns** — `vib_nde_um`, `lube_oil_temp_c`, `seal_gas_dp_bar` — 80 timestamps on `CGC-100A` and 62 (+ the 144 shutdown hours, 206 total) on `CGC-200B`, scattered randomly across the full 2-year history with no shutdown association. Every affected timestamp is missing on **all three** of these columns simultaneously and no others — consistent with an intermittent instrumentation/historian channel-group dropout rather than three independent sensor failures. Safe to interpolate (short, isolated, non-shutdown) with a `is_imputed` flag.

**`ground_truth_reference.csv`**:
- `specific_energy_kwh_per_t_true`: 144 NaNs — the same CGC-200B shutdown block (SEC is undefined with zero/near-zero throughput).
- `rul_hours_to_next_wash_true`: 961 NaNs at the tail of `CGC-100A` (from 2025-11-20 23:00, right after its last recorded wash) and 2,293 NaNs at the tail of `CGC-200B` (from 2025-09-26 11:00, right after its last recorded wash). This is **right-censoring**: RUL-to-next-wash is undefined once there is no future wash event left in the series to count down to. This truncates usable RUL-labeled training data well before the end of each series and must be handled explicitly (see §14).
- `fouling_index_true`, `polytropic_eff_true`, `health_index_true` have **no** NaNs, including during the shutdown — the latent physical/mechanical state is still defined (and simply frozen) while the machine is down.

## 6. Duplicates

- Zero fully-duplicated rows in `compressor_telemetry.csv`.
- Zero duplicated `(ts, train_id)` keys in either `compressor_telemetry.csv` or `ground_truth_reference.csv`.
- Zero duplicated timestamps within either train's series.

## 7. Out-of-order timestamps

- None. Each train's rows are already strictly increasing in `ts` as stored in the file (`is_monotonic_increasing == True` for both trains). A production pipeline should still explicitly `sort_values(['train_id','ts'])` and assert monotonicity rather than trust file order, but this dataset requires no reordering.

## 8. Shutdown / gap periods

- No **row-level** time gaps (§3). The one true shutdown is represented as **NaN-filled rows, not absent rows** — `CGC-200B`, 2025-08-13 00:00 → 2025-08-18 23:00 (144 h), matching the `trip` → `unscheduled_maintenance` pair in `events.csv`.
- Practical implication: a gap-detection routine that only checks for missing timestamps will **not** find this shutdown. Detection must additionally check for NaN blocks / non-operating state (e.g., `throughput_tph.isna()` or `speed_rpm == 0`).

## 9. Constant / stuck sensors

- **`p_s1_suct_bara` is constant across every available reading on both trains** — `1.32` bara for all 17,520 hours on CGC-100A, and `1.28` bara for all 17,376 non-null hours on CGC-200B (the remaining 144 hours are `NaN`, coinciding with the shutdown block below — the value is never anything other than 1.28 when it is measured at all). Both constants match `assets.design_suct_bara` exactly.
  - Note on the run-length figures: a naive "longest consecutive identical value" scan reports 14,160 hours for CGC-200B, not 17,376 — that is an artifact of the scan resetting its counter across the `NaN` shutdown gap, not evidence the value ever changed. Before *and* after the shutdown the value is 1.28 throughout.
  - **This is NOT a sensor fault.** It is a fixed upstream pressure setpoint held by a control loop outside this compressor train, not something this equipment measures dynamically. **Do not impute or otherwise modify these raw values.**
  - **Treat it as zero-variance / non-predictive and exclude it from model features** (or reduce it to the per-train constant already captured via `train_id`/`assets.design_suct_bara` if a fixed offset is ever wanted).
- **`throughput_tph` on `CGC-200B` is frozen at exactly 243.85 t/h for 432 consecutive hours** (2024-04-30 → 2024-05-17), matching the `sensor_fault` event in `events.csv` ("Suction flow transmitter stuck (frozen value)"). This **is a genuine sensor fault**, distinct in kind from the `p_s1_suct_bara` case above.
  - **Mark those 432 hourly observations as BAD/unreliable** via an explicit quality/status flag (e.g. `throughput_tph_quality = 'stuck'`), and **do not trust throughput-derived physics features (SEC, physics residuals, etc.) computed from `throughput_tph` during this window.**
  - **Preserve the raw frozen values as-is** — the flag communicates unreliability; the pipeline should not silently overwrite or interpolate over them without that flag surviving downstream.
- No other column is stuck for ≥24 consecutive hours on either train.

## 10. Suspicious / impossible values

Checks run (see `scripts/profile_data.py::profile_impossible_values`): negative throughput/speed/power, out-of-range recycle valve %, non-positive suction pressures, discharge P ≤ suction P per stage, discharge T ≤ suction T per stage, inter-stage pressure continuity (stage *i* discharge vs stage *i+1* suction), speed above trip limit / implausibly below governor minimum.

- **Only one flag fired**: `surge_margin_pct < 0` on **11 rows**, all on `CGC-100A` between **2024-07-14 00:00 and 16:00**. These line up exactly with the four `surge` (critical) events in `events.csv` on that date. A negative surge margin during a real surge excursion is **physically correct, not a data error** — it must not be clipped to zero or treated as invalid.
- No negative throughput/speed/power/pressure, no stage where discharge pressure ≤ suction pressure, no discontinuous inter-stage pressures, and no speed excursion above `trip_speed_rpm` for either train (max observed speed stays ~700–900 rpm below trip speed on both trains).
- No dataset-wide out-of-physical-range values were found on the 41 numeric columns; full min/1%/25%/50%/75%/99%/max ranges for every column are in the script output and were sanity-checked against `assets.csv` design values (e.g., stage-5 discharge pressure ranges 34–43 bara around the ~36.5–37.2 bara design discharge pressure — consistent with normal + degraded operation, not a data artifact).

## 11. Telemetry distributions / ranges (highlights)

- `feed_mode`: `mixed` 55.6% (19,464 rows), `ethane` 27.3% (9,552), `naphtha` 17.2% (6,024).
- `diene_ppm` (fouling precursor) tracks feed slate tightly: mean ≈ 45 ppm on ethane, ≈ 110 ppm on mixed, ≈ 210 ppm on naphtha.
- `speed_rpm`: CGC-100A 10,436–10,819 rpm (design 10,600, trip 11,766); CGC-200B 10,708–11,110 rpm (design 10,850, trip 12,043). Both trains operate well inside their governed envelope; no trip-speed excursions in the data.
- `run_hours_since_wash`: 0–2,924 h (up to ~4 months between washes), median ≈ 1,273 h — consistent with the wash cadence visible in `events.csv`.
- Per-stage pressures/temperatures increase monotonically stage-to-stage as expected for a 5-stage centrifugal train (e.g., mean suction/discharge climbs from ~1.3/2.6 bara at stage 1 to ~19.3/38.2 bara at stage 5), consistent with the design ratios in `assets.csv`.

## 12. Relationship between telemetry and ground truth

- `compressor_telemetry.csv` and `ground_truth_reference.csv` join **perfectly 1:1 on `(ts, train_id)`** — 35,040 rows in, 35,040 rows out, no orphans either side.
- Correlation of a few *observable* telemetry signals against the *latent* truth columns (Pearson, full history, both trains pooled):

| | fouling_index_true | polytropic_eff_true | specific_energy_kwh_per_t_true | health_index_true | rul_hours_to_next_wash_true |
|---|---|---|---|---|---|
| `run_hours_since_wash` | 0.81 | −0.82 | 0.20 | −0.81 | **−0.98** |
| `surge_margin_pct` | −0.95 | 0.94 | −0.09 | 0.95 | 0.71 |
| `vib_de_um` | 0.53 | −0.51 | 0.21 | −0.52 | −0.43 |
| `shaft_power_kw` | 0.17 | −0.14 | **0.90** | −0.16 | −0.21 |
| `brg_de_temp_c` | 0.12 | −0.12 | 0.20 | −0.12 | 0.03 |

Reading this: fouling shows up strongly and coherently across `run_hours_since_wash` and `surge_margin_pct` — both fully legitimate, observable inputs — which is reassuring (the simulation is internally consistent and a model trained on real telemetry should be able to recover fouling/health/RUL reasonably well). It also means `run_hours_since_wash` alone is a very strong baseline predictor of `rul_hours_to_next_wash_true` (r ≈ −0.98) purely because the simulation defines RUL as "time until the next wash trigger," which is mechanically close to "elapsed time since the last wash." This is **not leakage** (run_hours_since_wash is a legitimate real-time input), but it does mean a naive model can look deceptively good; a stronger model should be judged on how much it improves over this trivial baseline, and on picking up the feed-slate/fouling-rate variation that a pure run-hours counter cannot see.

## 13. Events / anomalies

22 labeled events across both trains:

| event_type | count | trains | notes |
|---|---|---|---|
| `online_wash` | 12 | both (6 each) | planned, 8h typical duration, "efficiency recovery ~X%" in `detail` |
| `surge` | 4 | CGC-100A only | critical, 2024-07-14, four 3h episodes — matches the 11 negative-surge-margin rows (§10) |
| `sensor_fault` | 2 | one per train | CGC-100A: stage-3 discharge TT drift, 2024-10-27→2024-12-25 (60 days); CGC-200B: frozen flow transmitter, 2024-04-30→2024-05-17 (18 days) — matches the stuck `throughput_tph` block (§9) |
| `process_upset` | 1 | CGC-100A | interstage cooler fouling, elevated stage temps, 2025-04-15→2025-06-03 (50 days) |
| `degradation` | 1 | CGC-200B | drive-end bearing wear ramp, 2025-06-29→2025-08-12 (45 days) |
| `trip` | 1 | CGC-200B | high-vibration trip, 2025-08-13 00:00–02:00 |
| `unscheduled_maintenance` | 1 | CGC-200B | bearing replacement, 6-day outage — the shutdown block in §5/§8 |

**Verified against raw telemetry:**
- **Stage-3 TT drift (CGC-100A)**: mean stage-3 ΔT (discharge − suction) rises from ≈56.7°C in the month before the event to ≈60.4°C during it, while the stage-3 pressure ratio is essentially unchanged (1.992 → 1.991). A pressure-ratio-consistent temperature jump with no corresponding pressure change is the textbook signature of a drifting temperature transmitter, not a real process change — exactly what the physics-consistency residual (§ "leakage/physics" below and Section 4 of the assignment) is designed to catch.
- **Bearing-wear ramp → trip (CGC-200B)**: `vib_de_um` climbs from a weekly mean of ~21.7 µm (2025-06-20) to ~65.0 µm by 2025-08-08, and `brg_de_temp_c` climbs from ~73.4°C to ~86.5°C over the same 7 weeks — a clean, monotonic precursor signal ending in the trip. This is a strong, learnable mechanical-RUL case.
- **Surge cluster (CGC-100A, 2024-07-14)**: confirmed via negative `surge_margin_pct` (§10).
- **Sensor faults**: confirmed via the frozen `throughput_tph` block and the stage-3 T drift above.

`process_upset` and the interstage-cooler-fouling window were not independently re-derived numerically here (left for the EDA notebook) but the event window is available for labeling.

## 14. Asset specifications & limits

`assets.csv` — one row per train:

| | CGC-100A | CGC-200B |
|---|---|---|
| design_speed_rpm | 10,600 | 10,850 |
| min_gov_speed_rpm | 7,419 | 7,594 |
| trip_speed_rpm | 11,766 | 12,043 |
| design_flow_tph | 255.0 | 240.0 |
| design_suct_bara / design_disch_bara | 1.32 / 36.5 | 1.28 / 37.2 |
| design_poly_eff_pct | 81.5 | 80.5 |
| gas_k_cp_cv | 1.2 | 1.2 |
| surge_control_line_pct | 112.0 | 112.0 |
| install_date | 2016-05-01 | 2018-09-01 |

Observed telemetry stays inside these envelopes on both trains (no trip-speed excursions; throughput and pressures track design values with normal ± operational variation). `design_suct_bara` exactly matches the constant `p_s1_suct_bara` noted in §9, confirming that column reflects a fixed upstream setpoint rather than a live measurement.

## 15. Candidate target variables for degradation / RUL

Built from `ground_truth_reference.csv` (validation/label construction only — never joined into the feature matrix used at inference time):

- **Performance degradation KPI** — `polytropic_eff_true` decline or `fouling_index_true` rise, forecastable per-train over a defined horizon (Section 5.1 of the assignment). In production, this target is *constructed* by comparing your own physics-computed efficiency/SEC against the clean baseline — `polytropic_eff_true`/`fouling_index_true` are used only to validate that construction.
- **specific_energy_kwh_per_t_true** — useful as a validation target for a computed-SEC-vs-clean-baseline "excess energy from fouling" KPI; note the 144-hour NaN block during the CGC-200B shutdown.
- **health_index_true** — a natural label source for a composite anomaly/health score if you choose to frame anomaly detection as a supervised health-index regression rather than pure unsupervised detection.
- **rul_hours_to_next_wash_true** — the performance-RUL target named explicitly in Section 5.2. Right-censored at the tail of each train's series (§5) — the last ~2-4 months of each train's data have no valid RUL label and must be excluded from RUL-target training (they can still be used for other targets/unsupervised tasks).
- **Mechanical RUL** (Section 5.2b) has no direct `*_true` column — it must be *constructed* from `events.csv` (hours from each timestamp back-counted to the `trip` event, or forward-looking labeled windows) combined with the observed `vib_de_um` / `brg_de_temp_c` ramp (§13), since only one trip exists in the whole dataset (CGC-200B, 2025-08-13). With a single labeled trip, mechanical-RUL evaluation will necessarily be data-poor — flag this as a limitation and consider a synthetic/rule-based limit-crossing threshold as a secondary label source.

## 16. Potential leakage risks

1. **The `*_true` columns themselves** (`fouling_index_true`, `polytropic_eff_true`, `specific_energy_kwh_per_t_true`, `health_index_true`, `rul_hours_to_next_wash_true`) — explicitly called out in the assignment as a deliberate trap. These must **never** appear in a feature vector; they are validation/label-construction only. Enforced here in code via `data_schema.py`'s `ColumnRole.TARGET_SOURCE` tagging and the leakage self-check at the bottom of that file.
2. **Rolling/lag features computed with a centered or backward-looking-but-improperly-windowed function** that peeks across a wash event or across the shutdown boundary — e.g. a rolling mean of `run_hours_since_wash` computed without resetting at each wash, or a rolling statistic computed over the raw (ungapped) row index that silently averages pre-shutdown and post-shutdown values. All rolling/lag features must be computed **per train**, **time-ordered**, using only past data, and must explicitly reset or gap-flag across the 2025-08-13→18 shutdown and across each wash event.
3. **Scalers/normalizers fit on the full series** (including validation/test time ranges) rather than on the training split only — explicitly called out as mandatory to avoid in Section 6 of the assignment.
4. **`p_s1_suct_bara`** is not a leakage risk in the strict sense (it's an input, not a `*_true` column) but is a near-useless, zero-variance feature that could accidentally encode `train_id` through its two fixed values (1.32 vs 1.28) — worth excluding or explicitly documenting rather than letting a model quietly use it as a train fingerprint.
5. **The frozen `throughput_tph` window** on CGC-200B (§9) — if not flagged as BAD, a model would learn from 432 hours of fabricated-flat sensor data, degrading any feature derived from throughput (SEC, physics residuals) during that window.
6. **Feature/target temporal alignment for RUL** — `rul_hours_to_next_wash_true` at time *t* is computed by the simulator using knowledge of the *future* wash time; this is fine as a training label (that's literally what a RUL target is) but any feature engineering that also uses "time until X" computed from future events (as opposed to past-only run-hours) would be leakage. `run_hours_since_wash` is safe (backward-looking); anything counting forward to the next wash is not.
7. **Train/test split leakage via shuffling** — with only 2 trains and ~2 years each, a naive random row split would put adjacent hours of the same wash/fouling cycle on both sides of the split. A time-based, per-train split is required (see §18).

## 17. Recommended column roles

Full detail in `data_schema.py`; summary:

**Model inputs** (41 telemetry columns + `feed_mode` + engineered physics features):
- All per-stage pressures/temperatures except `p_s1_suct_bara`, all vibration/bearing/lube-oil/seal-gas columns, `throughput_tph` (with the frozen window flagged/excluded), `speed_rpm`, `shaft_power_kw`, `surge_margin_pct`, `recycle_valve_pct`, `wash_oil_rate_kgph`, `antifoulant_ppm`, `run_hours_since_wash`, `gas_mw`, `diene_ppm`, `ambient_temp_c`, `cw_supply_temp_c`, `feed_mode` (encoded).
- Static per-train design values from `assets.csv` (join key `train_id`) for normalization: `design_speed_rpm`, `design_flow_tph`, `design_suct_bara`, `design_disch_bara`, `design_poly_eff_pct`, `gas_k_cp_cv`, `min_gov_speed_rpm`, `trip_speed_rpm`, `surge_control_line_pct`.
- Engineered physics features (Section 4 of the assignment): pressure ratios per stage, polytropic head/efficiency, computed SEC, fouling-attributable excess energy, surge margin vs. surge-line reference, and the discharge-temperature physics residual.

**Targets** (constructed from, or validated against, `ground_truth_reference.csv` — never fed back as inputs): a degradation KPI (efficiency/SEC drift), performance RUL (`rul_hours_to_next_wash_true`, tail-censored), mechanical RUL (constructed from `events.csv` + vibration/bearing trend, since only one trip exists), and anomaly labels (from `events.csv` event windows).

**Excluded from modeling**:
- All five `*_true` columns in `ground_truth_reference.csv` (leakage).
- `unit` (redundant with `train_id`).
- `p_s1_suct_bara` (zero-variance, no information).
- `service`, `oem`, `driver`, `num_stages`, `install_date` from `assets.csv` (constant across both trains in this dataset, or outside the telemetry time window).
- `detail` from `events.csv` (free text, not a feature).

## 18. Required preprocessing

1. **Per-train time indexing**: `sort_values(['train_id','ts'])`, set a per-train `DatetimeIndex`, assert monotonicity and hourly frequency (already true here, but assert rather than assume in a real pipeline).
2. **Shutdown handling**: explicitly detect the CGC-200B 2025-08-13→18 block (via NaN density or `throughput_tph.isna()`/near-zero `speed_rpm`) and mark it as a distinct `is_running = False` state. Never forward-fill through it; either drop it from training windows or carry it as an explicit gap that resets rolling/lag features.
3. **Frozen-sensor handling**: flag the CGC-200B `throughput_tph` 2024-04-30→2024-05-17 block as `BAD`/unreliable via a status column (matches `events.csv`); exclude that window's throughput-derived features (SEC, physics residual) from training. Keep the raw 243.85 t/h values in place — the flag marks unreliability, it does not license overwriting or interpolating the raw column.
4. **Sensor-drift handling**: flag the CGC-100A stage-3 discharge-T 2024-10-27→2024-12-25 window as `BAD`/`SUSPECT` using the physics residual (predicted vs. measured T rising while pressure ratio is flat); exclude or correct before using stage-3 efficiency/SEC in that window.
5. **Physics cross-checks as a validation layer, not silent correction**: compute per-stage pressure ratio, polytropic head/efficiency, and the discharge-T residual; flag rather than auto-correct out-of-tolerance rows.
6. **Random single-hour dropouts** (`vib_nde_um`, `lube_oil_temp_c`, `seal_gas_dp_bar`): short, isolated, non-shutdown — safe to interpolate (e.g., linear, ≤1h gap) with an `is_imputed` flag retained as a feature.
7. **Exclude `p_s1_suct_bara` from the feature set** (zero-variance setpoint, not a fault) **without modifying its raw values** — a per-train constant already available via `assets.design_suct_bara` if a reference is ever needed.
8. **Categorical encoding**: `feed_mode` (3 levels) — one-hot or an ordinal encoding by mean `diene_ppm`/fouling rate if a monotonic relationship is assumed; `train_id` as a model input only if the model is meant to generalize across trains (otherwise train per-train models, or include it purely for a shared model with a per-train offset/embedding).
9. **Per-train, per-split scaling**: fit any scaler/normalizer on the training split only, per train (or with `train_id` as an explicit covariate), never on validation/test or on the full series.
10. **Rolling/lag/rate-of-change features**: compute per train, time-ordered, using only past data; reset (or gap-flag) at the shutdown boundary and, where physically meaningful (e.g., `run_hours_since_wash`-based rollups), at each wash event.
11. **RUL label truncation**: drop rows with `rul_hours_to_next_wash_true = NaN` (the right-censored tails, §5) when training the performance-RUL target specifically; those rows remain usable for other targets.

## 19. Recommended time-based split strategy

- **No random row shuffling** — this is time-series data with strong autocorrelation and a physical wash/fouling cycle; a random split would leak adjacent in-cycle hours across train/test.
- **Split chronologically, per train**, e.g.:
  - Train: 2024-01-01 → 2025-06-30 (~18 months)
  - Validation: 2025-07-01 → 2025-09-30 (captures the CGC-100A wash on 2025-07-27 and the CGC-200B bearing-wear-ramp→trip→shutdown sequence — genuinely useful for validating both a degradation forecaster and an anomaly/RUL detector on a real event)
  - Test (holdout): 2025-10-01 → 2025-12-30 (captures the final CGC-100A wash on 2025-11-20 and CGC-200B's final wash on 2025-09-26 falls just before this window, so test skews toward "post-recovery, re-fouling" behavior for CGC-200B — worth stating explicitly as a limitation)
- Alternative: a **walk-forward / rolling-origin** scheme (as explicitly required for the degradation forecaster in Section 5.1) — repeatedly train on an expanding or sliding window and validate on the next block, sliding forward in time; report metrics averaged across folds rather than a single train/val/test cut.
- **Respect the shutdown and RUL right-censoring** when drawing split boundaries: don't let a training window's last few days bleed into the CGC-200B shutdown without labeling it, and don't include right-censored (NaN-RUL) tail rows in a supervised RUL training or evaluation set.
- Fit all preprocessing (scalers, encoders, physics-baseline "clean" efficiency reference) on the **training split only**, then apply unchanged to validation/test.

---

## Appendix: files produced by this profiling pass

- `DATASET_PROFILE.md` — this report.
- `data_schema.py` — machine-readable column roles/ranges + a leakage self-check (`python data_schema.py`).
- `scripts/profile_data.py` — reproducible profiling script; also writes `profile_outputs/detected_gaps.csv` and `profile_outputs/stuck_sensor_candidates.csv`.
