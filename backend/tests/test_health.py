"""API surface tests — no network, no API keys required."""
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_readiness_shape():
    r = client.get("/readiness")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "llm_providers",
        "llm_configured",
        "embeddings_configured",
        "vector_db_configured",
        "default_tier",
    ):
        assert key in body


def test_dataset_meta_reports_the_real_catalogue():
    r = client.get("/dataset/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["cities"] == ["Calgary"]
    assert body["restaurants"] == 60
    assert body["attractions"] == 25
    assert "japanese" in body["cuisines"]
    assert "Downtown Core" in body["neighbourhoods"]


def test_dataset_meta_surfaces_the_synthetic_data_disclaimer():
    """Users must be told the venues are not real. This is a product
    requirement, not a nicety — so it is a test, not a docstring."""
    body = client.get("/dataset/meta").json()
    assert "fictional" in body["data_disclaimer"].lower()


def test_readiness_reports_dataset_state():
    assert client.get("/readiness").json()["dataset_ready"] is True


def test_tier_endpoints_declared_but_not_implemented():
    assert client.post("/chat").status_code == 501
    r = client.post(
        "/itinerary",
        json={"preferences": {"city": "Calgary", "days": 2}, "tier": 2},
    )
    assert r.status_code == 501


def test_preferences_validation_rejects_bad_input():
    r = client.post(
        "/itinerary",
        json={"preferences": {"city": "Calgary", "days": 99}, "tier": 2},
    )
    assert r.status_code == 422  # days capped at 3
