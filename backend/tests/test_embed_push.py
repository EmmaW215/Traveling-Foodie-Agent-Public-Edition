"""embed_push payload assembly — the pure logic, no network.

The actual embedding + upsert require Gemini and Upstash and run in the
embed-push workflow. Here we pin the record shape the upsert depends on: the id,
the vector, and the chunk text carried in metadata so the copilot can ground on
it after retrieval.
"""
from scripts import embed_push


def test_build_vectors_carries_text_in_metadata():
    chunks = [
        {"id": "r001", "text": "Bow Valley Breakfast Co is a cafe.", "metadata": {"name": "Bow Valley"}},
        {"id": "a003", "text": "Island Park Loop is a park.", "metadata": {"name": "Island Park"}},
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    records = embed_push.build_vectors(chunks, vectors)

    assert [r["id"] for r in records] == ["r001", "a003"]
    assert records[0]["vector"] == [0.1, 0.2]
    # text must ride along in metadata for grounding after retrieval
    assert records[0]["metadata"]["text"] == "Bow Valley Breakfast Co is a cafe."
    assert records[0]["metadata"]["name"] == "Bow Valley"


def test_build_vectors_requires_matching_lengths():
    """A vector/chunk count mismatch must fail loudly, not silently truncate."""
    import pytest

    with pytest.raises(ValueError):
        embed_push.build_vectors([{"id": "x", "text": "t", "metadata": {}}], [[0.1], [0.2]])
