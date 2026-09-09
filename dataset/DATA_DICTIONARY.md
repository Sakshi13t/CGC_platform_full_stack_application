# Data Dictionary — Cracked Gas Compressor Dataset

All data is **synthetic** and generated from a physics-based simulation (`generate_data.py`,
seed `20260908`). Two 5-stage steam-turbine-driven cracked gas compressor (CGC) trains are
simulated at **hourly** resolution over **24 months** (2024-01-01 to 2025-12-31).

## Files

| File | Grain | Rows | Purpose |
|---|---|---|---|
| `timeseries/compressor_telemetry.csv` | 1 row / train / hour | 35,040 | Main sensor & operating telemetry (model INPUT) |
| `timeseries/ground_truth_reference.csv` | 1 row / train / hour | 35,040 | **Simulation ground truth** — see warning below |
| `reference/events.csv` | 1 row / event | 14 | Washes, surges, trips, sensor faults, maintenance |
| `reference/assets.csv` | 1 row / train | 2 | Design specs & protective limits |
| `reference/compressor_curves.csv` | 1 row / flow point | 26 | Performance map / surge-line reference |
| `reference/feed_slate_daily.csv` | 1 row / train / day | 1,460 | Daily feed composition summary |
| `knowledge_base/*.md` | 7 documents | — | Engineering docs for the RAG task |

> ### ⚠️ Label-leakage warning — read this
> `ground_truth_reference.csv` contains the simulation's **latent** states
> (`fouling_index_true`, `polytropic_eff_true`, `specific_energy_kwh_per_t_true`,
> `health_index_true`, `rul_hours_to_next_wash_true`). In a real plant these do **not**
> exist as measurements — fouling is never directly observed. Use this file **only** to
> validate models you build from the telemetry, and to construct honest training targets.
> Feeding `*_true` columns in as model features is leakage and will be treated as a
> disqualifying error in evaluation. Part of the assignment is demonstrating that you
> understand this distinction.

## `compressor_telemetry.csv`

| Column | Unit | Notes |
|---|---|---|
| `ts` | ISO timestamp | Hourly |
| `train_id` | — | `CGC-100A` or `CGC-200B` |
| `unit` | — | Ethylene unit label |
| `feed_mode` | — | `ethane` / `mixed` / `naphtha` (drives MW & fouling) |
| `gas_mw` | g/mol | Cracked-gas molecular weight |
| `diene_ppm` | ppm | Diene loading — primary fouling driver |
| `ambient_temp_c` | °C | Seasonal |
| `cw_supply_temp_c` | °C | Cooling-water supply temperature |
| `throughput_tph` | t/h | Cracked-gas mass flow |
| `speed_rpm` | rpm | Turbine/compressor speed |
| `shaft_power_kw` | kW | Driver shaft power |
| `surge_margin_pct` | % | Distance to surge limit (control signal) |
| `recycle_valve_pct` | % | Anti-surge (spillback) valve position |
| `wash_oil_rate_kgph` | kg/h | Wash-oil injection |
| `antifoulant_ppm` | ppm | Antifoulant dosing |
| `run_hours_since_wash` | h | Resets to 0 at each online wash |
| `vib_de_um` / `vib_nde_um` / `vib_axial_um` | µm | Drive-end / non-drive-end / axial vibration |
| `brg_de_temp_c` / `brg_nde_temp_c` / `brg_thrust_temp_c` | °C | Bearing metal temperatures |
| `lube_oil_press_bar` / `lube_oil_temp_c` | bar / °C | Lube-oil system |
| `seal_gas_dp_bar` | bar | Seal-gas differential pressure |
| `p_s{1..5}_suct_bara` | bar a | Per-stage suction pressure |
| `p_s{1..5}_disch_bara` | bar a | Per-stage discharge pressure |
| `t_s{1..5}_suct_c` | °C | Per-stage suction (interstage-cooled) temperature |
| `t_s{1..5}_disch_c` | °C | Per-stage discharge temperature |

**Derivable quantities (you compute these):** per-stage & overall pressure ratio,
polytropic head, polytropic efficiency, specific energy consumption (SEC = shaft_power/throughput),
and physics cross-checks (predicted vs measured discharge temperature).

## Known, intentionally-embedded phenomena

- **Fouling cycles**: efficiency decays ~10–15 points across each ~100–120-day run, then a
  wash partially recovers it. Naphtha campaigns foul faster than ethane.
- **Seasonality**: ambient & cooling-water temperature follow an annual cycle affecting duty.
- **CGC-100A**: a stage-3 discharge-temperature **calibration drift**, a **surge event cluster**,
  and an **interstage-cooler fouling** episode.
- **CGC-200B**: a **drive-end bearing wear ramp** ending in a **high-vibration trip** and 6-day
  unscheduled maintenance outage (NaNs during the outage), and a **stuck flow transmitter** period.
- **Random data-quality issues**: scattered missing values and occasional sensor spikes.

## `events.csv`
`train_id, ts_start, ts_end, event_type, severity, detail`.
`event_type` ∈ {online_wash, surge, trip, unscheduled_maintenance, sensor_fault,
process_upset, degradation}.

## `assets.csv`
Design speed, min governor & trip speed, design flow, suction/discharge pressure, design
polytropic efficiency, gas k (Cp/Cv), install date, surge control line.
