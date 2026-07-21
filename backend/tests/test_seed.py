"""The build pipeline: correctness and determinism."""
import json
import subprocess
import sys

from src.paths import BACKEND_ROOT, CHUNKS_PATH, DISTANCE_MATRIX_PATH, SQLITE_PATH
from src.tools import catalog


def test_artifacts_are_created():
    assert SQLITE_PATH.exists()
    assert DISTANCE_MATRIX_PATH.exists()
    assert CHUNKS_PATH.exists()


def test_expected_venue_counts():
    meta = catalog.dataset_meta()
    assert meta["restaurants"] == 60
    assert meta["attractions"] == 25


def test_dataset_meta_carries_the_disclaimer():
    """The data is synthetic; the API must say so. Non-negotiable."""
    meta = catalog.dataset_meta()
    assert "fictional" in meta["data_disclaimer"].lower()
    assert len(meta["data_version"]) == 12


def test_seed_is_deterministic():
    """Same CSVs in, byte-identical artefacts out."""
    before_matrix = DISTANCE_MATRIX_PATH.read_bytes()
    before_chunks = CHUNKS_PATH.read_bytes()
    before_version = catalog.dataset_meta()["data_version"]

    result = subprocess.run(
        [sys.executable, "-m", "scripts.seed"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert DISTANCE_MATRIX_PATH.read_bytes() == before_matrix
    assert CHUNKS_PATH.read_bytes() == before_chunks

    catalog.dataset_meta.cache_clear()
    assert catalog.dataset_meta()["data_version"] == before_version


def test_every_pair_has_a_distance(venues):
    matrix = json.loads(DISTANCE_MATRIX_PATH.read_text())
    n = len(venues)
    pairs = sum(len(v) for v in matrix.values())
    assert pairs == n * (n - 1) // 2


def test_chunks_cover_every_venue(venues):
    lines = CHUNKS_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(venues)
    chunk = json.loads(lines[0])
    assert {"id", "text", "metadata"} <= chunk.keys()
    assert chunk["id"] in venues


def test_chunk_text_mentions_allergens_when_present(venues):
    """RAG answers can only be grounded in what the chunk actually says."""
    lines = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    by_id = {c["id"]: c for c in lines}
    peanut = next(v for v in venues.values() if v["is_trap"] == "peanut_risk")
    assert "peanut" in by_id[peanut["venue_id"]]["text"].lower()


def test_seed_rejects_a_broken_dataset(tmp_path, monkeypatch):
    """Validation must actually fail the build, not warn and continue."""
    from scripts import seed

    broken = [
        {
            "venue_id": "x1",
            "kind": "restaurant",
            "name": "Nowhere Cafe",
            "neighbourhood": "Atlantis",
            "lat": 0.0,  # not in Calgary
            "lon": 0.0,
            "category": "cafe",
            "slot_types": "lunch",
            "price_band": "$",
            "cost_per_person": -5,  # negative
            "duration_min": 60,
            "closed_days": "funday",  # not a weekday
            "open_time": "9am",  # not HH:MM
            "close_time": "17:00",
            "allergens_present": "",
            "dietary_options": "",
            "rating": 9.0,  # out of range
            "is_trap": "",
            "description": "",
        }
    ]
    errors = seed.validate(broken)
    joined = " ".join(errors)
    assert "latitude" in joined
    assert "longitude" in joined
    assert "negative cost" in joined
    assert "unknown closed_day" in joined
    assert "not HH:MM" in joined
    assert "rating" in joined
    # and the missing traps are reported too
    assert "peanut_risk" in joined
