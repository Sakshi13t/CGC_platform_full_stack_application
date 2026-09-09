# TECHNICAL_DECISIONS.md — Phase 2 (Data Quality & Preprocessing)

Decisions relevant to Phase 2 only. Phase 1's decisions live in `DATASET_PROFILE.md`.

1. **p_s1_suct_bara**: excluded from `EXCLUDED_FROM_FEATURES` (config.py) rather than dropped from the dataset or given a quality flag. Not modified, not imputed, no `_quality` column created for it (it is not a fault). Preserved unchanged in every derived output.

2. **throughput_tph fault window** (CGC-200B, 2024-04-30→2024-05-17): resolved via `quality.SENSOR_FAULT_COLUMN_HINTS`, which maps the exact `events.csv` `detail` text to the affected column name. This keeps the mapping auditable (grep-able back to the source event) instead of inferring "which column broke" from free text at runtime.

3. **Shutdown definition**: `is_shutdown` is derived strictly from `events.csv`'s `unscheduled_maintenance` event type — not from NaN density — so it can never be confused with the unrelated BAD_SENSOR / isolated-dropout patterns that also produce NaNs.

4. **Quality granularity**: only two columns (`throughput_tph`, `t_s3_disch_c`) get a persisted `<col>_quality` column, because those are the only two with a documented, known BAD_SENSOR window. Every other column's trustworthiness is fully captured by the shared `is_shutdown` flag plus on-demand missingness. Adding 41 near-identical `_quality` columns was judged to add noise, not information (Phase 2 spec explicitly permits this: "do not necessarily force every column into every category if a simpler design is cleaner").

5. **Physics assumptions not given verbatim in the assignment/knowledge base** are isolated into `config.PhysicsAssumptions` and named explicitly rather than folded into a formula:
   - Compressibility factor Z = 1.0 (ideal gas) — used only by the optional polytropic-head feature.
   - The standard turbomachinery identity `(n-1)/n = (k-1)/(k·η_poly))`, needed to make the assignment's two given equations (head-via-n, T_disch-via-k-and-η) mutually consistent.
   All documented thresholds actually used (vibration/bearing/lube-oil/surge/wash-trigger limits) were verified against `docs/knowledge_base/*.md` line-by-line before being encoded in `config.DocumentedThresholds` — none are invented.

6. **Discharge-temperature residual baseline**: computed against the train's *design* polytropic efficiency (`assets.design_poly_eff_pct`), not the *observed* one, so the residual grows under both a genuine efficiency drop (fouling) and an instrument drift — matching the assignment's stated purpose for this check ("catch a drifting sensor" via a physics prediction independent of the instrument itself).

7. **Rolling-feature scope**: implemented generically and reusably (`features.py`) but only *applied* in the orchestrator to four columns most relevant to degradation/anomaly precursors (`vib_de_um`, `brg_de_temp_c`, `surge_margin_pct`, `run_hours_since_wash`) at two window sizes (24h, 168h), rather than exhaustively across all 41 telemetry columns — kept to what Phase 2 needs to hand off, not a speculative feature-store dump.

8. **Interpolation scope**: limited to the three columns confirmed in Phase 1 to share the isolated single-hour dropout pattern (`vib_nde_um`, `lube_oil_temp_c`, `seal_gas_dp_bar`), gap ≤ 1 hour, linear method, and only where the gap is provably outside a shutdown window. Ground-truth (`*_true`) columns are never imputed (spec requirement).

9. **Chronological split**: applied by calendar time across both trains simultaneously (not a per-train split with different boundaries), per `DATASET_PROFILE.md` §19. `TrainOnlyStandardScaler` is provided as reusable infrastructure for later phases, not as a model-fitting step in Phase 2 itself.

10. **No file bloat**: derived CSVs under `outputs/processed/` contain only new/derived columns (plus `ts`, `train_id` join keys) — never a copy of the 45 raw telemetry columns — to avoid duplicating the ~10 MB raw file across five outputs.
