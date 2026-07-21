"""Tier 1 pipeline — the M2 exit criteria.

Runs in mock mode: deterministic, no keys, so the same request always produces
the same plan and a regression is a real regression.
"""
import pytest

from src.models import Preferences
from src.orchestrator import collect_final, run_tier1

# S1 — the standard hackathon scenario.
S1 = Preferences(
    city="Calgary", days=2, budget_total=500, party_size=2,
    cuisines=["japanese", "italian", "thai"], allergies=["peanut"],
)


async def _events(prefs, **kw):
    return [e async for e in run_tier1(prefs, mock=True, **kw)]


# ---------------------------------------------------------------------------
# M2 exit criterion: a full itinerary comes out, and it is coherent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_s1_produces_a_complete_itinerary():
    final = await collect_final(S1, mock=True)
    assert final["tier"] == 1
    assert len(final["days"]) == 2
    # Two days x (breakfast, am, lunch, pm, dinner) = 10 stops.
    stops = [s for d in final["days"] for s in d["stops"]]
    assert len(stops) == 10


@pytest.mark.asyncio
async def test_itinerary_passes_its_own_validation():
    final = await collect_final(S1, mock=True)
    assert final["validation"]["ok"], final["validation"]["issues"]


@pytest.mark.asyncio
async def test_every_venue_is_distinct():
    """The dry-run bug: no venue may be scheduled twice."""
    final = await collect_final(S1, mock=True)
    ids = [s["venue_id"] for d in final["days"] for s in d["stops"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# The guards hold through the whole pipeline, not just in isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_peanut_allergy_keeps_the_trap_out_of_the_plan():
    final = await collect_final(S1, mock=True)
    ids = {s["venue_id"] for d in final["days"] for s in d["stops"]}
    assert "r008" not in ids  # the peanut_risk trap


@pytest.mark.asyncio
async def test_plan_stays_within_budget():
    final = await collect_final(S1, mock=True)
    assert final["budget"]["over_budget"] is False
    assert final["budget"]["spent"] <= final["budget"]["budget_total"]


@pytest.mark.asyncio
async def test_tight_budget_is_respected():
    """A deliberately tight budget must not be blown by the pipeline."""
    prefs = S1.model_copy(update={"budget_total": 180})
    final = await collect_final(prefs, mock=True)
    assert final["budget"]["spent"] <= 180
    assert final["validation"]["ok"], final["validation"]["issues"]


@pytest.mark.asyncio
async def test_restaurant_slots_hold_restaurants_and_attraction_slots_attractions():
    final = await collect_final(S1, mock=True)
    for day in final["days"]:
        for stop in day["stops"]:
            if "attraction" in stop["slot"]:
                assert stop["kind"] == "attraction"
            else:
                assert stop["kind"] == "restaurant"


@pytest.mark.asyncio
async def test_cuisine_preferences_are_honoured_when_possible():
    """At least one stated cuisine should appear among the restaurants."""
    final = await collect_final(S1, mock=True)
    categories = {
        s["category"] for d in final["days"] for s in d["stops"] if s["kind"] == "restaurant"
    }
    assert categories & {"japanese", "italian", "thai"}


# ---------------------------------------------------------------------------
# Shape, determinism, latency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_one_day_itinerary():
    prefs = S1.model_copy(update={"days": 1})
    final = await collect_final(prefs, mock=True)
    assert len(final["days"]) == 1
    assert len(final["days"][0]["stops"]) == 5


@pytest.mark.asyncio
async def test_pipeline_is_deterministic_in_mock_mode():
    first = await collect_final(S1, mock=True)
    second = await collect_final(S1, mock=True)
    ids_first = [s["venue_id"] for d in first["days"] for s in d["stops"]]
    ids_second = [s["venue_id"] for d in second["days"] for s in d["stops"]]
    assert ids_first == ids_second


@pytest.mark.asyncio
async def test_trace_event_sequence():
    """The UI relies on this event order."""
    events = await _events(S1)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "planner_start"
    assert "planner_done" in kinds
    assert kinds.count("executor_result") == 10
    assert "validation" in kinds
    assert "route_ready" in kinds
    assert kinds[-1] == "final"
    # validation and route come after all executors
    assert kinds.index("validation") > max(
        i for i, k in enumerate(kinds) if k == "executor_result"
    )


@pytest.mark.asyncio
async def test_each_day_has_a_route_with_legs():
    final = await collect_final(S1, mock=True)
    for day in final["days"]:
        route = day["route"]
        assert len(route["stops"]) == 5
        assert len(route["legs"]) == 4
        assert route["total_km"] > 0


@pytest.mark.asyncio
async def test_mock_mode_is_fast():
    """No network in mock mode, so the whole run is milliseconds. This is the
    latency-budget proxy: the pipeline structure itself adds no real delay."""
    final = await collect_final(S1, mock=True)
    assert final["elapsed_ms"] < 2000
    assert final["mock"] is True


@pytest.mark.asyncio
async def test_final_carries_data_version():
    final = await collect_final(S1, mock=True)
    assert final["data_version"]
    assert len(final["data_version"]) == 12
