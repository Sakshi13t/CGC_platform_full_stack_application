"""
Synthetic data generator for a refinery Cracked Gas Centrifugal Compressor (CGC).

Two 5-stage, steam-turbine-driven cracked gas compressor trains in an ethylene
plant are simulated at HOURLY resolution over 24 months. The generator embeds
real compression thermodynamics (polytropic head & power), a physically-motivated
fouling model (polymer/coke deposition driven by diene concentration and
temperature), an anti-surge margin model, condition-monitoring signals
(vibration, bearing temps, lube oil), operator mitigation actions (wash-oil and
antifoulant injection, online washes) and several INJECTED FAULTS so that the
downstream ML/DL/anomaly/optimization/RUL tasks are non-trivial.

Everything is reproducible from SEED. Ground-truth latent states (fouling index,
health, remaining useful life) are emitted in a *separate* reference file and are
clearly flagged as simulation artefacts that would NOT exist in production — this
is deliberate, so the assignment can test whether candidates understand label
leakage.
"""

import numpy as np
import pandas as pd

SEED = 20260908
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Time base
# ----------------------------------------------------------------------------
START = pd.Timestamp("2024-01-01 00:00:00")
DAYS = 730
HOURS = DAYS * 24
idx = pd.date_range(START, periods=HOURS, freq="h")
hours = np.arange(HOURS)
doy = idx.dayofyear.to_numpy()

# ----------------------------------------------------------------------------
# Physical constants / helpers
# ----------------------------------------------------------------------------
R_UNIV = 8.314462          # J/mol/K
KELVIN = 273.15

def poly_head_kj_per_kg(p_ratio, T_suct_K, MW, n_exp):
    """Polytropic head [kJ/kg] for an ideal-ish gas.
    H = Z*(R/MW)*T1 * (n/(n-1)) * (rp^((n-1)/n) - 1)
    """
    Z = 0.97
    Rgas = R_UNIV / (MW / 1000.0)            # J/kg/K
    exp = (n_exp - 1.0) / n_exp
    H = Z * Rgas * T_suct_K * (n_exp / (n_exp - 1.0)) * (p_ratio ** exp - 1.0)
    return H / 1000.0                         # kJ/kg

def disch_temp_K(p_ratio, T_suct_K, k_exp, eta_poly):
    """Discharge temperature from polytropic relation."""
    n_over = (k_exp - 1.0) / (k_exp * eta_poly)   # (n-1)/n = (k-1)/(k*eta)
    return T_suct_K * (p_ratio ** n_over)

# ----------------------------------------------------------------------------
# Train / asset design specs
# ----------------------------------------------------------------------------
TRAINS = {
    "CGC-100A": dict(
        unit="Ethylene Unit 100", n_stages=5, design_speed=10600,
        design_flow_tph=255.0, design_disch_bara=36.5, design_eff=0.815,
        p_suct_bara=1.32, oem="TurboMax", install="2016-05-01",
        wash_interval_days=118, k_gas=1.20,
    ),
    "CGC-200B": dict(
        unit="Ethylene Unit 200", n_stages=5, design_speed=10850,
        design_flow_tph=240.0, design_disch_bara=37.2, design_eff=0.805,
        p_suct_bara=1.28, oem="TurboMax", install="2018-09-01",
        wash_interval_days=104, k_gas=1.20,
    ),
}

# ----------------------------------------------------------------------------
# Shared exogenous drivers (ambient / cooling water — seasonal)
# ----------------------------------------------------------------------------
season = np.sin(2 * np.pi * (doy - 110) / 365.25)          # peak ~ late July
ambient = 22.0 + 9.0 * season + rng.normal(0, 1.4, HOURS)   # deg C
cw_supply = 27.0 + 6.0 * season + rng.normal(0, 0.8, HOURS) # cooling water supply

# ----------------------------------------------------------------------------
# Feed-slate campaigns (drives molecular weight & fouling propensity)
#   ethane  -> light gas, low dienes, low fouling
#   naphtha -> heavier gas, high dienes, high fouling
# ----------------------------------------------------------------------------
def build_feed_slate(train_offset):
    """Piecewise campaigns of 10-45 days each."""
    mode = np.empty(HOURS, dtype=object)
    mw = np.empty(HOURS)
    diene = np.empty(HOURS)
    t = 0
    r = np.random.default_rng(SEED + train_offset)
    while t < HOURS:
        span_days = int(r.integers(10, 46))
        span = min(span_days * 24, HOURS - t)
        pick = r.choice(["ethane", "mixed", "naphtha"], p=[0.35, 0.4, 0.25])
        if pick == "ethane":
            base_mw, base_di = 22.5, 45
        elif pick == "mixed":
            base_mw, base_di = 26.0, 110
        else:
            base_mw, base_di = 29.5, 210
        mode[t:t + span] = pick
        mw[t:t + span] = base_mw + r.normal(0, 0.4, span)
        diene[t:t + span] = np.clip(base_di + r.normal(0, 10, span), 5, None)
        t += span
    # smooth transitions a little
    mw = pd.Series(mw).rolling(6, min_periods=1, center=True).mean().to_numpy()
    diene = pd.Series(diene).rolling(6, min_periods=1, center=True).mean().to_numpy()
    return mode, mw, diene

# ----------------------------------------------------------------------------
# Per-train simulation
# ----------------------------------------------------------------------------
def simulate_train(name, spec, train_offset):
    n_stages = spec["n_stages"]
    k = spec["k_gas"]
    eta_clean = spec["design_eff"]
    p_suct1 = spec["p_suct_bara"]
    p_disch_final = spec["design_disch_bara"]

    feed_mode, gas_mw, diene_ppm = build_feed_slate(train_offset)

    # ------- Throughput: demand pattern + weekly rhythm + planned rate cuts ---
    base_flow = spec["design_flow_tph"]
    weekly = 3.0 * np.sin(2 * np.pi * hours / (24 * 7))
    slow = 6.0 * np.sin(2 * np.pi * hours / (24 * 60) + train_offset)
    throughput = base_flow + weekly + slow + rng.normal(0, 2.0, HOURS)
    # occasional planned rate cuts (~ turndown to 82-88%)
    r = np.random.default_rng(SEED + 7 + train_offset)
    n_cuts = 9
    for _ in range(n_cuts):
        s = int(r.integers(0, HOURS - 96))
        dur = int(r.integers(24, 96))
        throughput[s:s + dur] *= r.uniform(0.82, 0.90)
    throughput = np.clip(throughput, base_flow * 0.7, base_flow * 1.06)

    # ------- Speed tracks flow (turbine control) -----------------------------
    speed = spec["design_speed"] * (0.9 + 0.11 * (throughput / base_flow)) \
        + rng.normal(0, 25, HOURS)

    # ------- Mitigation actions (operator policy) ----------------------------
    antifoulant = np.full(HOURS, 6.0)                       # ppm baseline
    antifoulant += (diene_ppm > 150) * 6.0                  # ramp up on heavy feed
    antifoulant += rng.normal(0, 0.5, HOURS)
    antifoulant = np.clip(antifoulant, 2, 22)
    wash_oil = np.clip(120 + 0.35 * diene_ppm + rng.normal(0, 12, HOURS), 60, 400)

    # ------- Fouling dynamics (sequential) -----------------------------------
    # dF/dt driven by dienes & interstage temp; mitigated by antifoulant/wash.
    fouling = np.zeros(HOURS)
    run_hours = np.zeros(HOURS)
    wash_events = []            # (start_ts_index, kind)
    F = 0.0
    rh = 0.0
    wash_interval_h = spec["wash_interval_days"] * 24
    next_wash = wash_interval_h + int(r.integers(-240, 240))
    a_foul = 1.75e-4
    for i in range(HOURS):
        temp_factor = np.exp(0.020 * (cw_supply[i] - 27.0))   # hotter CW -> more fouling
        diene_factor = (diene_ppm[i] / 100.0) ** 1.15
        mitig = (1.0 - 0.028 * (antifoulant[i] - 6.0)) * (1.0 - 0.0006 * (wash_oil[i] - 120))
        mitig = np.clip(mitig, 0.45, 1.2)
        dF = a_foul * diene_factor * temp_factor * mitig * (throughput[i] / base_flow)
        F = F + dF
        rh += 1.0
        # scheduled online wash resets most of the fouling
        do_wash = (i >= next_wash) and (F > 0.05)
        if do_wash:
            recovery = r.uniform(0.72, 0.9)
            F = F * (1.0 - recovery)
            wash_events.append((i, "online_wash", recovery))
            rh = 0.0
            next_wash = i + wash_interval_h + int(r.integers(-240, 240))
        fouling[i] = F
        run_hours[i] = rh
    fouling = np.clip(fouling, 0, 0.65)

    # ------- Polytropic efficiency degrades with fouling ---------------------
    eta = eta_clean - 0.22 * fouling + rng.normal(0, 0.003, HOURS)
    eta = np.clip(eta, 0.60, eta_clean + 0.005)
    n_exp = k / (k - (k - 1) / eta_clean)   # placeholder; recompute per-stage below

    # ------- Per-stage thermodynamics ----------------------------------------
    # equal pressure ratio per stage, small stage dP losses; interstage cooling
    total_ratio = p_disch_final / p_suct1
    stage_ratio = total_ratio ** (1.0 / n_stages)
    p_suct = np.zeros((HOURS, n_stages))
    p_disch = np.zeros((HOURS, n_stages))
    t_suct = np.zeros((HOURS, n_stages))
    t_disch = np.zeros((HOURS, n_stages))

    T_in1 = ambient * 0.0 + 38.0 + 0.25 * (ambient - 22.0) + rng.normal(0, 0.6, HOURS)
    head_total = np.zeros(HOURS)
    for i in range(HOURS):
        p_lo = p_suct1
        Tin = T_in1[i] + KELVIN
        # fouling narrows flow path -> slightly higher effective ratio & temp rise
        foul_ratio_bump = 1.0 + 0.05 * fouling[i]
        H_sum = 0.0
        for s in range(n_stages):
            rp = stage_ratio * foul_ratio_bump * (1 + rng.normal(0, 0.004))
            p_hi = p_lo * rp
            eta_stage = eta[i] * (1 - 0.01 * s)      # rear stages slightly lower
            Tout = disch_temp_K(rp, Tin, k, eta_stage)
            H = poly_head_kj_per_kg(rp, Tin, gas_mw[i], k / (k - (k - 1) / eta_stage))
            H_sum += H
            p_suct[i, s] = p_lo
            p_disch[i, s] = p_hi
            t_suct[i, s] = Tin - KELVIN
            t_disch[i, s] = Tout - KELVIN
            # interstage cooler brings gas back near CW+approach (except last stage)
            if s < n_stages - 1:
                Tin = (cw_supply[i] + 8.0 + 0.3 * fouling[i] * 20 + rng.normal(0, 0.5)) + KELVIN
            p_lo = p_hi * (1 - 0.012)                # interstage dP
        head_total[i] = H_sum

    # ------- Power & specific energy -----------------------------------------
    m_dot_kg_s = throughput * 1000.0 / 3600.0
    # shaft power = m*H/eta_mech; head already includes per-stage eta
    mech_eff = 0.985
    shaft_kw = (m_dot_kg_s * head_total) / mech_eff        # kJ/s = kW
    shaft_kw += rng.normal(0, 90, HOURS)
    specific_energy = shaft_kw / throughput                 # kWh per tonne

    # ------- Surge margin -----------------------------------------------------
    # margin shrinks with fouling and at low flow; recycle valve opens to protect
    surge_margin = 22.0 - 16.0 * fouling - 9.0 * (1 - throughput / base_flow) \
        + rng.normal(0, 0.8, HOURS)
    recycle_valve = np.clip(6 + np.maximum(0, 12 - surge_margin) * 3.5
                            + rng.normal(0, 1.0, HOURS), 0, 100)
    surge_margin = surge_margin + 0.12 * recycle_valve      # opening valve restores margin
    surge_margin = np.clip(surge_margin, -3, 35)

    # ------- Condition monitoring --------------------------------------------
    vib_de = 18 + 6 * fouling * 4 + 0.5 * np.abs(rng.normal(0, 1, HOURS)) \
        + 0.4 * (throughput / base_flow - 1) * 20
    vib_nde = 16 + 5 * fouling * 3 + 0.5 * np.abs(rng.normal(0, 1, HOURS))
    vib_axial = 12 + 4 * fouling * 2 + 0.4 * np.abs(rng.normal(0, 1, HOURS))
    brg_de_temp = 62 + 8 * (throughput / base_flow) + 0.4 * (ambient - 22) \
        + rng.normal(0, 1.2, HOURS)
    brg_nde_temp = 60 + 7 * (throughput / base_flow) + rng.normal(0, 1.2, HOURS)
    brg_thrust_temp = 66 + 10 * fouling + rng.normal(0, 1.3, HOURS)
    lube_press = np.clip(2.4 + rng.normal(0, 0.05, HOURS), 2.0, 2.8)
    lube_temp = 45 + 0.35 * (ambient - 22) + rng.normal(0, 0.8, HOURS)
    seal_gas_dp = np.clip(0.55 + rng.normal(0, 0.03, HOURS), 0.4, 0.75)

    # =========================================================================
    # INJECTED FAULTS & EVENTS
    # =========================================================================
    events = []
    for (i, kind, rec) in wash_events:
        events.append(dict(train_id=name, ts_start=idx[i],
                           ts_end=idx[min(i + 8, HOURS - 1)], event_type=kind,
                           severity="planned",
                           detail=f"Online wash, efficiency recovery ~{rec*100:.0f}%"))

    if name == "CGC-100A":
        # (1) Slow sensor drift on stage-3 discharge temperature (calibration)
        d0, d1 = 24 * 300, 24 * 360
        drift = np.linspace(0, 9.0, d1 - d0)
        t_disch[d0:d1, 2] += drift
        events.append(dict(train_id=name, ts_start=idx[d0], ts_end=idx[d1 - 1],
                           event_type="sensor_fault", severity="warning",
                           detail="Stage-3 discharge TT calibration drift (+0..9 C)"))
        # (2) Surge event cluster during a low-flow upset
        s0 = 24 * 195
        for j in range(4):
            a = s0 + j * 5
            surge_margin[a:a + 3] -= 14
            vib_de[a:a + 3] += 55
            vib_axial[a:a + 3] += 40
            recycle_valve[a:a + 3] = 100
            events.append(dict(train_id=name, ts_start=idx[a], ts_end=idx[a + 3],
                               event_type="surge", severity="critical",
                               detail="Anti-surge excursion during feed upset"))
        # (3) Cooling water fouling episode -> higher interstage temps
        c0, c1 = 24 * 470, 24 * 520
        t_disch[c0:c1, :] += 4.0
        specific_energy[c0:c1] *= 1.03
        events.append(dict(train_id=name, ts_start=idx[c0], ts_end=idx[c1 - 1],
                           event_type="process_upset", severity="warning",
                           detail="Interstage cooler fouling; elevated stage temps"))

    if name == "CGC-200B":
        # (1) DE bearing wear ramp -> vibration climbs -> unscheduled maintenance
        b0 = 24 * 545
        b1 = 24 * 590
        ramp = np.linspace(0, 42, b1 - b0)
        vib_de[b0:b1] += ramp
        brg_de_temp[b0:b1] += np.linspace(0, 14, b1 - b0)
        events.append(dict(train_id=name, ts_start=idx[b0], ts_end=idx[b1 - 1],
                           event_type="degradation", severity="warning",
                           detail="Drive-end bearing wear; rising 1x vibration"))
        # trip + unscheduled maintenance shutdown
        trip = b1
        dur = 24 * 6
        for arr in (throughput, shaft_kw, speed, surge_margin, vib_de, vib_nde,
                    vib_axial, specific_energy, head_total):
            arr[trip:trip + dur] = np.nan
        events.append(dict(train_id=name, ts_start=idx[trip], ts_end=idx[trip + 2],
                           event_type="trip", severity="critical",
                           detail="High-vibration trip on drive-end bearing"))
        events.append(dict(train_id=name, ts_start=idx[trip], ts_end=idx[trip + dur - 1],
                           event_type="unscheduled_maintenance", severity="critical",
                           detail="Bearing replacement; train down 6 days"))
        # post-repair reset
        vib_de[trip + dur:] = 17 + 0.5 * np.abs(rng.normal(0, 1, HOURS - (trip + dur)))
        brg_de_temp[trip + dur:] -= 6
        # (2) Stuck flow transmitter (throughput frozen for a period)
        f0, f1 = 24 * 120, 24 * 138
        throughput[f0:f1] = throughput[f0]
        events.append(dict(train_id=name, ts_start=idx[f0], ts_end=idx[f1 - 1],
                           event_type="sensor_fault", severity="warning",
                           detail="Suction flow transmitter stuck (frozen value)"))

    # ------- Random small data-quality issues (missing / spikes) -------------
    miss = rng.random(HOURS) < 0.004
    for col in [vib_nde, seal_gas_dp, lube_temp]:
        col[miss] = np.nan
    spike = rng.random(HOURS) < 0.0015
    brg_nde_temp[spike] += rng.normal(0, 25, spike.sum())

    # =========================================================================
    # Ground-truth latent references (SIMULATION ONLY — flagged, separate file)
    # =========================================================================
    health = np.clip(1.0 - fouling / 0.6, 0, 1)
    # RUL = hours until next wash OR next critical event
    rul = np.full(HOURS, np.nan)
    wash_idx = [e[0] for e in wash_events]
    for i in range(HOURS):
        future = [w for w in wash_idx if w >= i]
        if future:
            rul[i] = future[0] - i

    # ------- Assemble main telemetry frame -----------------------------------
    df = pd.DataFrame({
        "ts": idx, "train_id": name, "unit": spec["unit"],
        "feed_mode": feed_mode, "gas_mw": np.round(gas_mw, 3),
        "diene_ppm": np.round(diene_ppm, 1),
        "ambient_temp_c": np.round(ambient, 2),
        "cw_supply_temp_c": np.round(cw_supply, 2),
        "throughput_tph": np.round(throughput, 2),
        "speed_rpm": np.round(speed, 1),
        "shaft_power_kw": np.round(shaft_kw, 1),
        "surge_margin_pct": np.round(surge_margin, 2),
        "recycle_valve_pct": np.round(recycle_valve, 1),
        "wash_oil_rate_kgph": np.round(wash_oil, 1),
        "antifoulant_ppm": np.round(antifoulant, 2),
        "run_hours_since_wash": np.round(run_hours, 0),
        "vib_de_um": np.round(vib_de, 2),
        "vib_nde_um": np.round(vib_nde, 2),
        "vib_axial_um": np.round(vib_axial, 2),
        "brg_de_temp_c": np.round(brg_de_temp, 2),
        "brg_nde_temp_c": np.round(brg_nde_temp, 2),
        "brg_thrust_temp_c": np.round(brg_thrust_temp, 2),
        "lube_oil_press_bar": np.round(lube_press, 3),
        "lube_oil_temp_c": np.round(lube_temp, 2),
        "seal_gas_dp_bar": np.round(seal_gas_dp, 3),
    })
    for s in range(n_stages):
        df[f"p_s{s+1}_suct_bara"] = np.round(p_suct[:, s], 3)
        df[f"p_s{s+1}_disch_bara"] = np.round(p_disch[:, s], 3)
        df[f"t_s{s+1}_suct_c"] = np.round(t_suct[:, s], 2)
        df[f"t_s{s+1}_disch_c"] = np.round(t_disch[:, s], 2)

    # blank out tripped period in stage sensors too
    if name == "CGC-200B":
        trip = 24 * 590
        dur = 24 * 6
        stage_cols = [c for c in df.columns if c.startswith(("p_s", "t_s"))]
        df.loc[trip:trip + dur - 1, stage_cols] = np.nan

    ref = pd.DataFrame({
        "ts": idx, "train_id": name,
        "fouling_index_true": np.round(fouling, 5),
        "polytropic_eff_true": np.round(eta, 5),
        "specific_energy_kwh_per_t_true": np.round(specific_energy, 3),
        "health_index_true": np.round(health, 4),
        "rul_hours_to_next_wash_true": rul,
    })

    ev = pd.DataFrame(events).sort_values("ts_start").reset_index(drop=True)
    return df, ref, ev

# ----------------------------------------------------------------------------
# Run both trains
# ----------------------------------------------------------------------------
frames, refs, evs = [], [], []
for off, (nm, sp) in enumerate(TRAINS.items()):
    d, r_, e = simulate_train(nm, sp, off)
    frames.append(d); refs.append(r_); evs.append(e)

telemetry = pd.concat(frames, ignore_index=True)
reference = pd.concat(refs, ignore_index=True)
events = pd.concat(evs, ignore_index=True)

# ----------------------------------------------------------------------------
# Reference tables
# ----------------------------------------------------------------------------
assets = pd.DataFrame([
    dict(train_id=k, unit=v["unit"], service="Cracked Gas Compressor",
         oem=v["oem"], driver="Extraction-condensing steam turbine",
         num_stages=v["n_stages"], design_speed_rpm=v["design_speed"],
         min_gov_speed_rpm=int(v["design_speed"]*0.7),
         trip_speed_rpm=int(v["design_speed"]*1.11),
         design_flow_tph=v["design_flow_tph"],
         design_suct_bara=v["p_suct_bara"], design_disch_bara=v["design_disch_bara"],
         design_poly_eff_pct=v["design_eff"]*100, gas_k_cp_cv=v["k_gas"],
         install_date=v["install"], surge_control_line_pct=112.0)
    for k, v in TRAINS.items()
])

# Compressor performance map / surge line (per train, reference speed)
curve_rows = []
for k, v in TRAINS.items():
    for flow_pct in np.arange(55, 116, 5):
        head_pct = 135 - 0.0085 * (flow_pct ** 2)          # falling head curve
        eff_pct = v["design_eff"]*100 - 0.02*(flow_pct-100)**2
        surge_flow = 62.0                                   # % of design flow at surge
        curve_rows.append(dict(train_id=k, ref_speed_rpm=v["design_speed"],
                               flow_pct_design=float(flow_pct),
                               poly_head_pct_design=round(head_pct, 2),
                               poly_eff_pct=round(eff_pct, 2),
                               surge_flow_pct_design=surge_flow))
curves = pd.DataFrame(curve_rows)

# Daily feed slate summary (derived from hourly)
fs = (telemetry.assign(date=telemetry["ts"].dt.date)
      .groupby(["train_id", "date"])
      .agg(feed_mode=("feed_mode", lambda s: s.mode().iloc[0]),
           gas_mw=("gas_mw", "mean"),
           diene_ppm=("diene_ppm", "mean"),
           throughput_tph=("throughput_tph", "mean"))
      .reset_index())

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
base = "cgc_assignment/dataset"
telemetry.to_csv(f"{base}/timeseries/compressor_telemetry.csv", index=False)
reference.to_csv(f"{base}/timeseries/ground_truth_reference.csv", index=False)
events.to_csv(f"{base}/reference/events.csv", index=False)
assets.to_csv(f"{base}/reference/assets.csv", index=False)
curves.to_csv(f"{base}/reference/compressor_curves.csv", index=False)
fs.to_csv(f"{base}/reference/feed_slate_daily.csv", index=False)

print("telemetry rows:", len(telemetry), "cols:", telemetry.shape[1])
print("reference rows:", len(reference))
print("events rows:", len(events))
print("\nEvent type counts:\n", events["event_type"].value_counts())
print("\nTelemetry columns:\n", list(telemetry.columns))
print("\nSample stats (CGC-100A):")
a = telemetry[telemetry.train_id == "CGC-100A"]
print(a[["throughput_tph","shaft_power_kw","surge_margin_pct","vib_de_um",
         "t_s5_disch_c"]].describe().round(2))
print("\nNaN counts (telemetry):", int(telemetry.isna().sum().sum()))
