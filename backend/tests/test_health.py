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


def test_chat_answers_a_grounded_question():
    """/chat is the Tier 0 copilot (M3). No keys -> local retriever + mock."""
    r = client.post("/chat", json={"question": "where can I get ramen for lunch?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["refused"] is False
    assert body["citations"]
    assert body["retriever"] == "local"  # no Upstash configured in tests


def test_chat_refuses_off_topic():
    r = client.post("/chat", json={"question": "how do I rebuild a car gearbox"})
    assert r.status_code == 200
    assert r.json()["refused"] is True


def test_chat_validates_question_length():
    assert client.post("/chat", json={"question": "x"}).status_code == 422


def test_readiness_reports_rag_retriever_mode():
    assert client.get("/readiness").json()["rag_retriever"] == "local"


def test_itinerary_streams_a_full_run():
    """/itinerary is now a live SSE stream (Tier 1). Parse the frames and
    check the pipeline ran end to end. No keys -> mock mode automatically."""
    import json

    with client.stream(
        "POST",
        "/itinerary",
        json={"preferences": {"city": "Calgary", "days": 2, "cuisines": ["italian"]}, "tier": 1},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    kinds = [e["event"] for e in events]
    assert kinds[0] == "planner_start"
    assert kinds[-1] == "final"
    final = events[-1]
    assert final["tier"] == 1
    assert len(final["days"]) == 2
    assert final["validation"]["ok"]


def test_preferences_validation_rejects_bad_input():
    r = client.post(
        "/itinerary",
        json={"preferences": {"city": "Calgary", "days": 99}, "tier": 2},
    )
    assert r.status_code == 422  # days capped at 3
