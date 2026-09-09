"""Definition-of-Done item: raw CSV files remain untouched."""

import hashlib
from pathlib import Path

from backend.app.preprocessing import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_raw_files_byte_for_byte_unchanged(raw_checksums_before):
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path, expected_digest in raw_checksums_before.items():
        actual_digest = _sha256(repo_root / rel_path)
        assert actual_digest == expected_digest, f"{rel_path} has been modified since Phase 1!"


def test_pipeline_run_does_not_modify_raw_files(raw_checksums_before, pipeline_results):
    """Running the full pipeline (which already executed via the
    `pipeline_results` fixture) must not have touched the raw CSVs."""
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path, expected_digest in raw_checksums_before.items():
        actual_digest = _sha256(repo_root / rel_path)
        assert actual_digest == expected_digest, f"Pipeline run modified {rel_path}!"
