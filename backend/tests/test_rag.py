"""Tier 0 RAG copilot — retrieval, grounding, and the venue-exists guard.

Deterministic: local lexical retriever + MockLLM, so no keys and no network.
The M3 exit criteria are asserted here: S1-S3 answered with citations, and the
unknown-venue guard fires on an adversarial answer.
"""
import json

import pytest

from src.agents.mock import MockLLM, MockResponse
from src.rag import copilot
from src.rag.retriever import Hit, LocalRetriever


@pytest.fixture()
def retriever():
    return LocalRetriever()


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ramen_query_retrieves_the_ramen_bar(retriever):
    hits = await retriever.retrieve("where can I get ramen", k=5)
    assert hits
    assert hits[0].id == "r012"  # East Village Ramen Bar ranks first


@pytest.mark.asyncio
async def test_attraction_intent_ranks_an_attraction_first(retriever):
    """The lexical weakness the intent boost fixes: 'attractions downtown'
    must not rank a downtown restaurant first."""
    hits = await retriever.retrieve("what free attractions are near downtown", k=5)
    assert hits
    assert hits[0].metadata["kind"] == "attraction"


@pytest.mark.asyncio
async def test_off_topic_query_retrieves_nothing(retriever):
    """No lexical overlap -> empty, which lets the copilot refuse."""
    assert await retriever.retrieve("how do I rebuild a car gearbox", k=5) == []


@pytest.mark.asyncio
async def test_gibberish_retrieves_nothing(retriever):
    assert await retriever.retrieve("zzzxqw qwlkjq", k=5) == []


@pytest.mark.asyncio
async def test_retrieval_is_deterministic(retriever):
    a = await retriever.retrieve("italian dinner in mission", k=5)
    b = await retriever.retrieve("italian dinner in mission", k=5)
    assert [h.id for h in a] == [h.id for h in b]


# ---------------------------------------------------------------------------
# M3 exit criterion 1: S1-S3 answered with citations
# ---------------------------------------------------------------------------
S1 = "Where can I get good ramen for lunch, and roughly what does it cost?"
S2 = "I have a peanut allergy — which restaurants are safe for dinner?"
S3 = "What free attractions are there near downtown?"


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [S1, S2, S3])
async def test_standard_scenarios_are_answered_with_citations(retriever, question):
    result = await copilot.answer_question(question, model=MockLLM(), retriever=retriever)
    assert result["grounded"] is True
    assert result["refused"] is False
    assert result["citations"], "a grounded answer must cite at least one venue"
    for c in result["citations"]:
        assert c["venue_id"] and c["name"]


@pytest.mark.asyncio
async def test_peanut_allergy_never_cites_the_peanut_venue(retriever):
    """The safety fix: a peanut-allergy question must not surface the peanut
    restaurant, even though lexical search matches 'peanut' straight to it."""
    result = await copilot.answer_question(S2, model=MockLLM(), retriever=retriever)
    cited_ids = {c["venue_id"] for c in result["citations"]}
    assert "r008" not in cited_ids  # Peanut Garden Thai
    assert "r008" not in result["sources"]


# ---------------------------------------------------------------------------
# M3 exit criterion 2: the unknown-venue guard fires
# ---------------------------------------------------------------------------
class CitesFakeVenue:
    """A model that answers confidently about a venue not in the catalogue."""

    async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
        return MockResponse(
            text=json.dumps(
                {
                    "answerable": True,
                    "answer": "You should try The Gilded Bison Chophouse, it's superb.",
                    "cited_venue_ids": ["r999_fake"],
                }
            )
        )


@pytest.mark.asyncio
async def test_guard_refuses_an_invented_venue(retriever):
    result = await copilot.answer_question(
        "recommend somewhere fancy for dinner", model=CitesFakeVenue(), retriever=retriever
    )
    assert result["refused"] is True
    assert result["grounded"] is False
    assert not result["citations"]
    assert "gilded bison" not in result["answer"].lower()  # the invented name is not surfaced


class CitesUnretrievedRealVenue:
    """Cites a real venue that was NOT in the retrieved context — also invalid."""

    def __init__(self, unretrieved_id: str):
        self.unretrieved_id = unretrieved_id

    async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
        return MockResponse(
            text=json.dumps(
                {"answerable": True, "answer": "Go here.", "cited_venue_ids": [self.unretrieved_id]}
            )
        )


@pytest.mark.asyncio
async def test_guard_rejects_a_real_venue_that_was_not_retrieved(retriever):
    """Grounding means citing from the context, not just naming a real place."""
    hits = await retriever.retrieve("ramen", k=3)
    retrieved = {h.id for h in hits}
    a_real_but_unretrieved = next(
        vid for vid in ("a017", "r049", "r022") if vid not in retrieved
    )
    result = await copilot.answer_question(
        "ramen", model=CitesUnretrievedRealVenue(a_real_but_unretrieved), retriever=retriever
    )
    assert result["refused"] is True


# ---------------------------------------------------------------------------
# Refusal on empty retrieval
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_off_topic_question_is_refused(retriever):
    result = await copilot.answer_question(
        "how do I rebuild a car gearbox", model=MockLLM(), retriever=retriever
    )
    assert result["refused"] is True
    assert not result["citations"]


@pytest.mark.asyncio
async def test_model_marking_unanswerable_is_refused(retriever):
    """Even when retrieval finds chunks, if the model judges the question
    unanswerable from them, its message is surfaced as a refusal.

    The query retrieves restaurants (so the model is actually invoked), but the
    model decides the guide can't answer the specific ask."""

    class SaysUnanswerable:
        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            return MockResponse(
                text=json.dumps(
                    {
                        "answerable": False,
                        "answer": "The guide doesn't cover Michelin ratings.",
                        "cited_venue_ids": [],
                    }
                )
            )

    # "sushi" matches real chunks, so retrieval is non-empty and the model runs.
    result = await copilot.answer_question(
        "do any sushi spots have a Michelin star", model=SaysUnanswerable(), retriever=retriever
    )
    assert result["refused"] is True
    assert "doesn't cover" in result["answer"]


# ---------------------------------------------------------------------------
# The correction pass
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bad_citation_is_repaired_on_the_second_try(retriever):
    """First reply cites a fake id, the repair reply cites a valid retrieved one."""
    hits = await retriever.retrieve("ramen", k=3)
    good_id = hits[0].id

    class FixesOnRetry:
        def __init__(self):
            self.calls = 0

        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            self.calls += 1
            vid = "r999_fake" if self.calls == 1 else good_id
            return MockResponse(
                text=json.dumps({"answerable": True, "answer": "Here.", "cited_venue_ids": [vid]})
            )

    model = FixesOnRetry()
    result = await copilot.answer_question("ramen", model=model, retriever=retriever)
    assert model.calls == 2
    assert result["grounded"] is True
    assert result["citations"][0]["venue_id"] == good_id


# ---------------------------------------------------------------------------
# Hit helper
# ---------------------------------------------------------------------------
def test_hit_serialises_without_the_full_text():
    hit = Hit(id="r001", score=1.234567, text="long text", metadata={"name": "X"})
    d = hit.as_dict()
    assert d["id"] == "r001"
    assert d["score"] == 1.2346
    assert "text" not in d
