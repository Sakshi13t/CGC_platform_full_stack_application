# Cracked Gas Compressor (CGC) Dataset

Synthetic, physics-based dataset accompanying the technical hiring assignment
**"AI-Powered Cracked Gas Compressor Reliability & Energy-Optimization Platform"**.

## Contents
```
dataset/
├── DATA_DICTIONARY.md              # field-by-field reference (READ FIRST)
├── timeseries/
│   ├── compressor_telemetry.csv    # 35,040 rows — model INPUT
│   └── ground_truth_reference.csv  # simulation latent truth (validation only)
├── reference/
│   ├── events.csv                  # washes, surges, trips, faults
│   ├── assets.csv                  # design specs & protective limits
│   ├── compressor_curves.csv       # performance map / surge line
│   └── feed_slate_daily.csv        # daily feed composition
└── knowledge_base/                 # 7 engineering docs for the RAG task
    ├── 01_reliability_condition_monitoring_standard.md
    ├── 02_cgc_operating_procedure.md
    ├── 03_antifoulant_online_wash_program.md
    ├── 04_surge_protection_philosophy.md
    ├── 05_energy_management_policy.md
    ├── 06_maintenance_strategy_rcm.md
    └── 07_incident_reports.md
```

## Quick start
```python
import pandas as pd
tel = pd.read_csv("timeseries/compressor_telemetry.csv", parse_dates=["ts"])
a = tel[tel.train_id == "CGC-100A"].set_index("ts").sort_index()

# Specific energy consumption (the energy KPI)
a["sec_kwh_per_t"] = a["shaft_power_kw"] / a["throughput_tph"]

# Overall pressure ratio
a["pr_overall"] = a["p_s5_disch_bara"] / a["p_s1_suct_bara"]
```

## Important
- The `*_true` columns in `ground_truth_reference.csv` are **not measurements**. Use them
  only to validate/label; never as model inputs. See the leakage warning in the data dictionary.
- Two trains are provided so you can demonstrate cross-asset generalisation (train on one,
  test on the other) and fleet-level modelling.
- Regenerate deterministically with `python generate_data.py` (seed `20260908`).
