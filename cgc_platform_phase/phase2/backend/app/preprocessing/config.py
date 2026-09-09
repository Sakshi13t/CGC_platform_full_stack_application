"""
backend/app/preprocessing/config.py
====================================
Single source of truth for paths, column roles, and physical constants/
thresholds used by the Phase 2 preprocessing pipeline.

Every threshold or constant here is sourced from one of:
  - the raw dataset itself (data_schema.py / DATASET_PROFILE.md, Phase 1), or
  - the knowledge-base documents shipped with the assignment
    (docs/knowledge_base/*.md), or
  - assets.csv (per-train design values).

Nothing here is an invented physical constant. Where a genuine engineering
assumption is unavoidable (ideal-gas compressibility factor, the standard
polytropic n/eta/k relation), it is called out explicitly in
TECHNICAL_DECISIONS.md and exposed here as a named, overridable constant
rather than hidden inside a formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (no hard-coded absolute paths — resolved relative to the repo root)
# ---------------------------------------------------------------------------
# REPO_ROOT here means "this phase's working directory" (cgc_platform_phase/
# phase2) — outputs/ stays anchored here. The raw dataset, however, lives at
# the true git-repo root under dataset/ (added after the initial Phase 2
# handoff), not under phase2/data/ as originally assumed. PROJECT_ROOT points
# there explicitly so the mismatch is visible rather than silently patched.
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_TIMESERIES_DIR = PROJECT_ROOT / "dataset" / "timeseries"
DATA_REFERENCE_DIR = PROJECT_ROOT / "dataset" / "reference"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

TELEMETRY_CSV = DATA_TIMESERIES_DIR / "compressor_telemetry.csv"
# ground_truth_reference.csv was added under dataset/timeseries/, not
# dataset/reference/ — path reflects the actual file location, not the name.
GROUND_TRUTH_CSV = DATA_TIMESERIES_DIR / "ground_truth_reference.csv"
EVENTS_CSV = DATA_REFERENCE_DIR / "events.csv"
ASSETS_CSV = DATA_REFERENCE_DIR / "assets.csv"
COMPRESSOR_CURVES_CSV = DATA_REFERENCE_DIR / "compressor_curves.csv"
FEED_SLATE_CSV = DATA_REFERENCE_DIR / "feed_slate_daily.csv"

TRAIN_IDS = ("CGC-100A", "CGC-200B")

# ---------------------------------------------------------------------------
# Column groups (kept explicit rather than inferred from dtype, per Phase 2
# code-quality requirement to avoid fragile string hacks)
# ---------------------------------------------------------------------------
STAGE_NUMBERS = (1, 2, 3, 4, 5)

def stage_suct_p(i: int) -> str: return f"p_s{i}_suct_bara"
def stage_disch_p(i: int) -> str: return f"p_s{i}_disch_bara"
def stage_suct_t(i: int) -> str: return f"t_s{i}_suct_c"
def stage_disch_t(i: int) -> str: return f"t_s{i}_disch_c"

# Columns that must NEVER be used as predictive model inputs, for any reason.
# Union of: (a) ground_truth_reference.csv latent-truth columns (leakage trap
# named explicitly in the assignment), (b) redundant identifiers, (c) columns
# confirmed in Phase 1 to carry zero predictive information.
GROUND_TRUTH_TARGET_COLUMNS = [
    "fouling_index_true",
    "polytropic_eff_true",
    "specific_energy_kwh_per_t_true",
    "health_index_true",
    "rul_hours_to_next_wash_true",
]

# p_s1_suct_bara: NOT a sensor fault (see DATASET_PROFILE.md §9). It is a
# fixed upstream setpoint, constant across every non-null reading on both
# trains (1.32 CGC-100A / 1.28 CGC-200B, matching assets.design_suct_bara).
# Per explicit instruction: do not modify it, do not impute it, do not label
# it BAD — simply exclude it from the ML feature list while preserving it
# unchanged in the raw and derived datasets for traceability.
EXCLUDED_FROM_FEATURES = [
    "unit",              # redundant with train_id
    "p_s1_suct_bara",    # constant-by-design setpoint, zero predictive info
]

# Free-text / non-feature columns from events.csv
EVENT_NON_FEATURE_COLUMNS = ["detail"]

# ---------------------------------------------------------------------------
# Known data-quality windows, sourced from events.csv + Phase-1 verification
# against raw telemetry (DATASET_PROFILE.md §5, §9, §13). These are read as
# data (from events.csv) by quality.py; the constants below exist only to
# name the specific event_type/column mapping so the mapping is explicit and
# auditable rather than inferred by string-matching on free text.
# ---------------------------------------------------------------------------

# event_type -> quality outcome. "unscheduled_maintenance" and the "trip"
# instant both fall inside what we call SHUTDOWN (machine not running).
SHUTDOWN_EVENT_TYPES = ("unscheduled_maintenance",)

# event_type -> which specific telemetry column(s) it invalidates, and how.
# Only sensor_fault events map to BAD_SENSOR; every other event_type is
# either a real (trustworthy) process condition (surge, trip) or a real
# planned intervention (online_wash) and does not mark any sensor BAD.
#
# The mapping keys off the human-readable `detail` text already present in
# events.csv (documented, not fabricated) to decide WHICH column(s) a given
# sensor_fault event affects, since events.csv does not carry a column name.
SENSOR_FAULT_COLUMN_HINTS = {
    "Suction flow transmitter stuck (frozen value)": ["throughput_tph"],
    "Stage-3 discharge TT calibration drift (+0..9 C)": ["t_s3_disch_c"],
}

# ---------------------------------------------------------------------------
# Physical / process thresholds sourced from the knowledge-base documents
# (docs/knowledge_base/*.md). Cited inline.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentedThresholds:
    # RS-CGC-01 §2 — vibration limits (um, shaft relative)
    vib_alert_um: float = 38.0
    vib_alarm_um: float = 50.0
    vib_trip_um: float = 62.0
    axial_alarm_um: float = 45.0
    axial_trip_um: float = 60.0

    # RS-CGC-01 §3 — bearing metal temperature (degC)
    brg_temp_alarm_c: float = 85.0
    brg_temp_trip_c: float = 95.0
    brg_temp_rate_of_rise_c_per_hr: float = 3.0

    # RS-CGC-01 §4 — lube oil / seal gas
    lube_oil_press_alarm_low_bar: float = 2.0
    lube_oil_press_trip_low_bar: float = 1.8
    lube_oil_temp_alarm_high_c: float = 52.0
    seal_gas_dp_min_bar: float = 0.40

    # RS-CGC-01 §6 — data-quality rule: a sensor frozen for more than this
    # duration must be flagged BAD. Used as the generic stuck-sensor
    # threshold for any column not already covered by a known events.csv
    # sensor_fault window.
    stuck_sensor_max_minutes: float = 30.0

    # SP-CGC-04 §2 — surge control line / minimum margin
    surge_control_line_pct_of_sll: float = 112.0
    surge_margin_min_pct: float = 8.0

    # RS-CGC-03 §3 — online wash trigger criteria (documented; used only to
    # describe/label data in this phase, not to trigger any action)
    wash_trigger_poly_eff_drop_pp: float = 4.0
    wash_trigger_disch_temp_rise_c: float = 10.0
    wash_trigger_sec_rise_pct: float = 8.0
    wash_trigger_min_surge_margin_pct: float = 10.0


THRESHOLDS = DocumentedThresholds()

# ---------------------------------------------------------------------------
# Physics-formula assumptions. Everything not directly measurable/derivable
# from the assignment's own equations + assets.csv is called out here by
# name so it is impossible to use silently. See TECHNICAL_DECISIONS.md.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicsAssumptions:
    # Universal gas constant (J / (mol*K)) — a physical constant, not an
    # engineering assumption.
    universal_gas_constant_j_per_molk: float = 8.314462618

    # Compressibility factor Z. The assignment's H_p formula includes Z but
    # no value or correlation for Z is given anywhere in the assignment doc
    # or knowledge base. We assume ideal-gas behaviour (Z = 1.0) ONLY for the
    # optional polytropic-head feature, and label every value derived using
    # it with a `physics_assumption` tag so it is never mistaken for a
    # directly-measured or directly-specified quantity.
    compressibility_factor_z: float = 1.0

    # Standard turbomachinery relation between the polytropic exponent n,
    # the isentropic exponent k = cp/cv (assets.gas_k_cp_cv), and polytropic
    # efficiency eta_poly:
    #     (n-1)/n = (k-1)/(k * eta_poly)
    # This is the standard textbook identity that makes the assignment's two
    # given equations (polytropic head using n, and discharge temperature
    # using k and eta_poly) mutually consistent. It is not stated verbatim in
    # the assignment text, so it is named explicitly here rather than baked
    # silently into a formula.
    use_standard_n_eta_k_relation: bool = True


PHYSICS = PhysicsAssumptions()

# ---------------------------------------------------------------------------
# Chronological split boundaries (DATASET_PROFILE.md §19). Applied per train.
# Inclusive on both ends; validation/test start immediately after the prior
# period's last timestamp.
# ---------------------------------------------------------------------------

SPLIT_BOUNDARIES = {
    "train": ("2024-01-01 00:00:00", "2025-06-30 23:00:00"),
    "validation": ("2025-07-01 00:00:00", "2025-09-30 23:00:00"),
    "test": ("2025-10-01 00:00:00", "2025-12-30 23:00:00"),
}

# ---------------------------------------------------------------------------
# Rolling-feature policy
# ---------------------------------------------------------------------------

ROLLING_WINDOWS_HOURS = (24, 168)  # 1 day, 7 days — short- and medium-term trend
MAX_INTERPOLATION_GAP_HOURS = 1    # only ever bridge single-hour dropouts
INTERPOLATABLE_COLUMNS = [
    # The three columns confirmed in Phase 1 to share a single-hour,
    # non-shutdown, non-fault dropout pattern (DATASET_PROFILE.md §5).
    "vib_nde_um",
    "lube_oil_temp_c",
    "seal_gas_dp_bar",
]
