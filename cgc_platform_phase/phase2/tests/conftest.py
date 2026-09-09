"""Shared fixtures for the Phase 2 preprocessing test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.preprocessing import config, preprocessing, quality  # noqa: E402


@pytest.fixture(scope="session")
def raw_checksums_before():
    path = REPO_ROOT / ".raw_checksums_before.txt"
    text = path.read_text()
    checksums = {}
    for line in text.strip().splitlines():
        digest, filename = line.split(maxsplit=1)
        checksums[filename.strip()] = digest.strip()
    return checksums


@pytest.fixture(scope="session")
def telemetry() -> pd.DataFrame:
    df = pd.read_csv(config.TELEMETRY_CSV)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values(["train_id", "ts"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    return quality.load_events(config.EVENTS_CSV)


@pytest.fixture(scope="session")
def assets() -> pd.DataFrame:
    return pd.read_csv(config.ASSETS_CSV)


@pytest.fixture(scope="session")
def pipeline_results():
    """Run the full Phase 2 pipeline once per test session (no file writes)."""
    return preprocessing.run(write_outputs=False)
