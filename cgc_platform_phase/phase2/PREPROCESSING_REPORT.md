# PREPROCESSING_REPORT.md — Phase 2: Data Quality & Preprocessing

Scope: data-quality flagging, leakage-safe feature engineering foundations, and chronological
splitting for the CGC telemetry dataset. No ML/DL/RUL/anomaly/optimization/API/frontend code
was introduced in this phase (see `TECHNICAL_DECISIONS.md` and the Definition of Done below).

Run the pipeline: `python -m backend.app.preprocessing.preprocessing`
Run the tests: `pytest tests/ -v`

## What was changed

Nothing in `data/timeseries/` or `data/reference/` — those four raw CSVs are read-only inputs
throughout this phase. All work products are new files:

- `backend/app/preprocessing/config.py` — paths, column roles, documented thresholds, physics assumptions, split boundaries.
- `backend/app/preprocessing/quality.py` — shutdown/BAD_SENSOR quality-flag mechanism.
- `backend/app/preprocessing/physics_features.py` — pressure ratio, observed polytropic efficiency, discharge-T residual, SEC, polytropic head, surge-margin threshold flag, with quality propagation.
- `backend/app/preprocessing/features.py` — causal, per-train, shutdown-boundary-safe rolling mean/std/min/max, rate-of-change, pct-change.
- `backend/app/preprocessing/splits.py` — chronological split assignment, train-only accessor, a train-only-fit scaler, and a leakage-guard assertion helper.
- `backend/app/preprocessing/leakage_checks.py` — 9 executable leakage checks + an orchestrating `run_all_checks`.
- `backend/app/preprocessing/preprocessing.py` — orchestrator: loads raw data, builds all derived frames, runs the leakage audit, writes derived outputs only.
- `tests/` — 24 tests across 5 files (see "Tests run" below).
- `outputs/processed/*.csv` — derived feature frames (quality flags, physics features, rolling features, interpolation columns, split assignment). Each keyed by `(ts, train_id)`, none duplicating the 45 raw telemetry columns.
- `outputs/diagnostics/leakage_audit.log` — the audit's PASS messages from the most recent run.
- `TECHNICAL_DECISIONS.md`, this report.

## What was intentionally NOT changed

- The raw CSVs (verified byte-identical via SHA-256 before/after every pipeline run and every test run — see `test_raw_immutability.py`).
- `p_s1_suct_bara`'s values — not modified, not imputed, not labeled BAD. Only excluded from the predictive feature list.
- The frozen `throughput_tph` values during the CGC-200B fault window — preserved as-is; only flagged.
- `ground_truth_reference.csv` values — never imputed, never joined into any feature frame.
- No ML model, deep-learning model, RUL estimator, anomaly detector, optimizer, digital twin, RAG/agent component, FastAPI endpoint, auth, frontend, or Docker/deployment artifact.

## Quality-flag logic

Five quality states (`quality.Quality`): `GOOD`, `MISSING`, `BAD_SENSOR`, `SHUTDOWN`, `INVALID_PHYSICS`.

- `is_shutdown` (bool, all rows): derived from `events.csv`'s `unscheduled_maintenance` event window only — not from NaN density — to keep it independent of unrelated missingness patterns. Currently True for exactly the 144-row CGC-200B 2025-08-13→18 block.
- `throughput_tph_quality`, `t_s3_disch_c_quality` (the only two columns with a documented, known instrument fault): `SHUTDOWN` > `BAD_SENSOR` > `MISSING` > `GOOD`, in that priority order, per row. `throughput_tph_quality == BAD_SENSOR` for exactly the 432-row CGC-200B 2024-04-30→2024-05-17 block.
- Every other telemetry column's trustworthiness is fully captured by `is_shutdown` plus a generic on-demand missingness check — no `_quality` column is persisted for columns with no documented fault, per `TECHNICAL_DECISIONS.md` §4 (avoids 41 near-duplicate columns).
- Physics features get `INVALID_PHYSICS` when either their required raw inputs were not `GOOD`/were missing, or the computed value itself is out of physically plausible bounds (e.g. an efficiency outside [0, 1], a pressure ratio below 1.0) — `propagate_quality` takes the worst state across all quality inputs feeding a derived value.

## Missing-data treatment

Four distinct patterns, handled differently (see `DATASET_PROFILE.md` §5 for the original analysis):

| Pattern | Rows | Treatment |
|---|---|---|
| Operational shutdown (CGC-200B, 2025-08-13→18) | 144 | `is_shutdown = True`; no interpolation; rolling features forced to NaN |
| Known sensor-fault window (`throughput_tph`, CGC-200B) | 432 | `throughput_tph_quality = BAD_SENSOR`; raw value untouched; downstream physics features (SEC, residuals) inherit BAD_SENSOR |
| Isolated single-hour dropout (`vib_nde_um`, `lube_oil_temp_c`, `seal_gas_dp_bar`) | 80–206/train | Conservative linear interpolation, gap ≤ 1h, only outside shutdown windows, output to `<col>_imputed` + `<col>_is_imputed` — raw column untouched |
| `ground_truth_reference.csv` NaNs (SEC during shutdown; RUL right-censored tails) | 144 / 961–2293 | Never imputed. Right-censored RUL rows are a modeling-phase concern (drop from RUL-target training), out of scope here beyond documenting it |

No blind forward-fill is performed anywhere in the pipeline.

## Shutdown handling

`quality.compute_shutdown_mask` scans `events.csv` for `unscheduled_maintenance` events and marks every telemetry row within `[ts_start, ts_end]` for the matching `train_id`. Rolling/rate features are forced to NaN for shutdown rows and reset immediately afterward (see below) rather than treating the shutdown as an ordinary gap.

## Rolling-window policy

- **Causal only**: every function in `features.py` uses trailing (`center=False`) windows; a value at time *t* is a function of rows at or before *t*. Verified with a direct causality test (`leakage_checks.check_rolling_feature_no_future_leakage`) that perturbs a future raw value and asserts no earlier feature value changes — applied to `rolling_mean`, `rolling_std`, and `rate_of_change`, and to a deliberately-broken `center=True` control case to confirm the test itself catches real leakage.
- **Per-train**: `apply_per_train` (and the orchestrator's own per-train grouping) guarantees a rolling window never spans two trains. Verified: the first row of each train's series has a NaN 168h rolling feature.
- **Reset at shutdown boundaries**: `_shutdown_reset_groups` assigns a new "segment id" every time the shutdown flag flips, so `groupby(segment).rolling(...)` never spans a shutdown; shutdown rows themselves are forced to NaN. Verified against the real CGC-200B trip: `vib_de_um_roll_mean_168h` climbs cleanly from ~21.5→~61.4 µm across the pre-trip bearing-wear ramp, and the 24h rolling mean resets to a fresh, correctly-low ~17.4 µm baseline in the first hours after the shutdown ends (not contaminated by the ~65 µm pre-trip values).
- **Window sizes applied**: 24h and 168h (1 day, 1 week) — short- and medium-term trend, on `vib_de_um`, `brg_de_temp_c`, `surge_margin_pct`, `run_hours_since_wash`. `min_periods` defaults to half the window so a rolling value is never computed from a mostly-empty window.
- **Rate-of-change**: 1-hour causal diff, NaN across a shutdown boundary (so a multi-day jump isn't reported as a 1-hour rate).

## Physics-feature policy

Every formula is taken directly from Section 0.3 of the assignment brief (pressure ratio, polytropic head, discharge temperature, shaft power/SEC) or from `docs/knowledge_base/*.md` (thresholds). Two engineering assumptions not stated verbatim in either source are named explicitly rather than hidden in a formula: compressibility factor Z = 1.0 (ideal gas), and the standard `(n-1)/n = (k-1)/(k·η_poly)` identity connecting the assignment's two given equations. Both are documented in `config.PhysicsAssumptions` and `TECHNICAL_DECISIONS.md`.

Implemented: per-stage pressure ratio, overall pressure ratio, observed polytropic efficiency (algebraically inverted from the given T_disch equation), design-baseline-predicted discharge temperature, the discharge-temperature physics residual (verified elevated during the known CGC-100A stage-3 sensor-drift event: mean residual rises from ~9.5°C to ~13.9°C), SEC, optional polytropic head, and a documented (8%, SP-CGC-04) surge-margin threshold flag.

Quality propagation: every derived feature's quality is the worst of its required raw inputs' quality (missingness, shutdown, or a documented BAD_SENSOR window) via `propagate_quality`. `sec_kwh_per_t` during the CGC-200B throughput fault window correctly inherits `BAD_SENSOR` (verified in `test_leakage_checks.py` indirectly via the source column check; direct SEC-quality assertion left as a natural extension for the modeling phase).

## Train/validation/test split

Chronological, applied by calendar time across both trains simultaneously (`config.SPLIT_BOUNDARIES`):

| Split | Range |
|---|---|
| train | 2024-01-01 00:00 → 2025-06-30 23:00 (~18 months) |
| validation | 2025-07-01 00:00 → 2025-09-30 23:00 |
| test | 2025-10-01 00:00 → 2025-12-30 23:00 |

No random shuffling anywhere (verified structurally: `check_no_random_split_used` confirms the three splits' timestamp ranges are non-overlapping blocks). `splits.train_only()` is the single sanctioned source of fitting data for any scaler/imputer; `TrainOnlyStandardScaler` demonstrates the fit-on-train/apply-elsewhere pattern and is provided as reusable infrastructure for later modeling phases — no model is fit in this phase.

## Leakage checks performed (all executable, `leakage_checks.py`)

1. `*_true` columns never in the feature list.
2. `events.csv` free-text/label columns (`detail`, `event_type`, `severity`) never in the feature list.
3. `p_s1_suct_bara` never in the feature list.
4. `unit` (redundant with `train_id`) never in the feature list.
5. Rolling/rate features are causal (perturbation test, §"Rolling-window policy").
6. A scaler's fit-max-timestamp never exceeds the documented train-split end.
7. Split assignment is chronological, not random (non-overlapping time ranges).
8. The known shutdown window is fully flagged `is_shutdown`.
9. The known throughput sensor-fault window is fully flagged `BAD_SENSOR`.

All 9 checks pass against the real pipeline output (`run_all_checks` → 7 aggregate PASS messages, some covering multiple underlying columns; see `outputs/diagnostics/leakage_audit.log`).

## Tests run + result

```
pytest tests/ -v
============================= 24 passed in 4.97s ==============================
```

Coverage against the Phase 2 spec's required test list: raw-file immutability (2 tests), `p_s1_suct_bara` exclusion (1), throughput fault BAD flagging (2), shutdown row identification (1), rolling-window shutdown/train-boundary safety (4), rolling-feature causality (4, including a negative control), chronological split correctness (2), train-only fitting (3), `*_true` exclusion (1), plus supporting leakage-audit and split-mechanics tests (4).

Raw-file SHA-256 checksums were verified identical to the pre-Phase-2 baseline (`.raw_checksums_before.txt`) before writing this report.

## Known limitations

- `INVALID_PHYSICS` propagation is implemented and unit-testable but not yet exercised by a dedicated end-to-end test against the real pipeline output (e.g. asserting zero `INVALID_PHYSICS` rows outside known fault windows) — a natural next-phase addition.
- Mechanical-RUL labeling is not attempted here (correctly out of scope for Phase 2) but is flagged in `DATASET_PROFILE.md` §15 as data-poor (only one labeled trip in the whole dataset) — worth remembering when Phase 3+ designs that target.
- The interpolation policy intentionally covers only 3 columns / ≤1h gaps; it does not attempt to recover the 432-hour `throughput_tph` fault window or the 144-hour shutdown — both are correctly left as unreliable/undefined rather than filled.
- `TrainOnlyStandardScaler` is deliberately minimal (mean/std only) — later phases may need a richer imputer/encoder set, but per scope this phase only had to make the *architecture* leakage-resistant, not build every transformer a future model will use.
