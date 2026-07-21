"""Agent-level tests. Deterministic — MockLLM makes every run reproducible.

These cover the pieces in isolation: JSON parsing/repair, the anti-hallucination
guard in the executors, and the Planner's preference-preservation.
"""
import json

import pytest

from src.agents import executors, planner
from src.agents.base import AgentError, run_structured
from src.agents.mock import MockLLM
from src.agents.schemas import PlannerOutput, VenuePick
from src.models import Preferences
from src.tools import catalog


class FixedLLM:
    """Returns a scripted string regardless of input — for parsing tests."""

    def __init__(self, reply: str):
        self.reply = reply

    async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
        from src.agents.mock import MockResponse

        return MockResponse(text=self.reply)


class FlakyLLM:
    """Bad JSON first, valid JSON on the repair attempt."""

    def __init__(self, bad: str, good: str):
        self.replies = [bad, good]

    async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
        from src.agents.mock import MockResponse

        return MockResponse(text=self.replies.pop(0))


# ---------------------------------------------------------------------------
# JSON extraction and repair
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parses_fenced_json():
    model = FixedLLM('```json\n{"venue_id": "r001", "reason": "x", "fallback_id": null}\n```')
    result = await run_structured(model, system="restaurant", user="", schema=VenuePick)
    assert result.venue_id == "r001"


@pytest.mark.asyncio
async def test_parses_json_with_surrounding_prose():
    model = FixedLLM('Sure! Here you go: {"venue_id": "r002", "reason": "y", "fallback_id": null}')
    result = await run_structured(model, system="restaurant", user="", schema=VenuePick)
    assert result.venue_id == "r002"


@pytest.mark.asyncio
async def test_repairs_after_one_bad_reply():
    model = FlakyLLM("not json at all", '{"venue_id": "r003", "reason": "z", "fallback_id": null}')
    result = await run_structured(model, system="restaurant", user="", schema=VenuePick)
    assert result.venue_id == "r003"


@pytest.mark.asyncio
async def test_gives_up_after_repair_attempt():
    model = FlakyLLM("garbage", "still garbage")
    with pytest.raises(AgentError):
        await run_structured(model, system="restaurant", user="", schema=VenuePick)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_planner_preserves_stated_cuisines():
    """A stated preference must never be silently dropped by the model."""

    class DropsCuisines:
        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            from src.agents.mock import MockResponse

            return MockResponse(
                text=json.dumps(
                    {
                        "summary": "s",
                        "cuisines_priority": [],  # model dropped them
                        "pace": "balanced",
                        "notes_for_executors": "",
                    }
                )
            )

    prefs = Preferences(cuisines=["thai", "korean"])
    result = await planner.plan(DropsCuisines(), prefs)
    assert set(result.cuisines_priority) >= {"thai", "korean"}


@pytest.mark.asyncio
async def test_planner_output_shape():
    result = await planner.plan(MockLLM(), Preferences(cuisines=["italian"]))
    assert isinstance(result, PlannerOutput)
    assert "italian" in result.cuisines_priority


# ---------------------------------------------------------------------------
# Executors — the anti-hallucination guard in action
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_executor_rejects_off_list_id_and_falls_back():
    """If the model names a venue that wasn't offered, use the top candidate."""
    candidates = catalog.search(kind="restaurant", slot_type="lunch", limit=4)

    class InventsVenue:
        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            from src.agents.mock import MockResponse

            return MockResponse(
                text=json.dumps(
                    {"venue_id": "r999_fake", "reason": "made up", "fallback_id": None}
                )
            )

    venue, reason, _fb, used_model = await executors.pick_restaurant(
        InventsVenue(), slot="d1_lunch", prefs=Preferences(),
        planner_notes="", candidates=candidates,
    )
    assert venue["venue_id"] == candidates[0]["venue_id"]  # deterministic fallback
    assert used_model is False


@pytest.mark.asyncio
async def test_executor_accepts_valid_pick():
    candidates = catalog.search(kind="restaurant", slot_type="lunch", limit=4)
    target = candidates[2]

    class PicksSecond:
        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            from src.agents.mock import MockResponse

            return MockResponse(
                text=json.dumps(
                    {"venue_id": target["venue_id"], "reason": "great", "fallback_id": None}
                )
            )

    venue, reason, _fb, used_model = await executors.pick_restaurant(
        PicksSecond(), slot="d1_lunch", prefs=Preferences(),
        planner_notes="", candidates=candidates,
    )
    assert venue["venue_id"] == target["venue_id"]
    assert used_model is True


@pytest.mark.asyncio
async def test_executor_with_no_candidates_raises():
    with pytest.raises(AgentError):
        await executors.pick_restaurant(
            MockLLM(), slot="d1_lunch", prefs=Preferences(),
            planner_notes="", candidates=[],
        )
