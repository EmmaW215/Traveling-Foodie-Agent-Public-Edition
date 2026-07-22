"""Tier 2 multi-agent — parallel executors + the bounded Critic loop.

The M4 exit criteria live here: planted conflicts are detected and revised within
≤2 loops, the SLOT_IDS guard rejects an off-vocabulary critic slot, and the trace
is complete. Deterministic in mock mode.
"""
import json

import pytest

from src.agents import critic
from src.agents.critic import BUDGET_SLOT
from src.agents.mock import MockLLM, MockResponse
from src.agents.schemas import CriticIssue
from src.models import Preferences
from src.orchestrator import _slot_to_fix, collect_final, run_tier2
from src.tools import catalog

S1 = Preferences(
    days=2, budget_total=500, party_size=2,
    cuisines=["japanese", "italian", "thai"], allergies=["peanut"],
)


async def _events(prefs, **kw):
    return [e async for e in run_tier2(prefs, mock=True, **kw)]


# ---------------------------------------------------------------------------
# Parallel execution + reconciliation — the headline behaviour
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_dispatch_then_reconcile_to_a_clean_plan():
    """Parallel executors collide (same top venue in several slots); the Critic
    loop reconciles to a fully distinct, valid plan within the bound."""
    final = await collect_final(S1, tier=2, mock=True)
    ids = [s["venue_id"] for d in final["days"] for s in d["stops"]]
    assert len(ids) == 10
    assert len(set(ids)) == 10, "every venue must be distinct after reconciliation"
    assert final["validation"]["ok"], final["validation"]["issues"]
    assert final["revisions"] > 0, "the parallel collision must have needed revision"


@pytest.mark.asyncio
async def test_final_is_tier_2():
    final = await collect_final(S1, tier=2, mock=True)
    assert final["tier"] == 2


@pytest.mark.asyncio
async def test_executors_run_in_parallel():
    events = await _events(S1)
    dispatch = next(e for e in events if e["event"] == "executors_dispatched")
    assert dispatch["parallel"] is True
    assert len(dispatch["slots"]) == 10
    results = [e for e in events if e["event"] == "executor_result"]
    assert len(results) == 10
    assert all(e["parallel"] for e in results)


# ---------------------------------------------------------------------------
# The bound is real
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critic_loop_is_bounded_to_two_iterations():
    events = await _events(S1)
    reviews = [e for e in events if e["event"] == "critic_reviewed"]
    assert 1 <= len(reviews) <= 2
    assert [r["iteration"] for r in reviews] == list(range(1, len(reviews) + 1))


@pytest.mark.asyncio
async def test_a_clean_second_pass_ends_the_loop():
    """Once the Critic finds nothing, the loop stops — the last review is empty."""
    events = await _events(S1)
    reviews = [e for e in events if e["event"] == "critic_reviewed"]
    assert reviews[-1]["issues"] == []


# ---------------------------------------------------------------------------
# Guards hold end to end
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_peanut_trap_never_appears():
    final = await collect_final(S1, tier=2, mock=True)
    ids = {s["venue_id"] for d in final["days"] for s in d["stops"]}
    assert "r008" not in ids


@pytest.mark.asyncio
async def test_plan_stays_within_budget():
    final = await collect_final(S1, tier=2, mock=True)
    assert final["budget"]["over_budget"] is False
    assert final["budget"]["spent"] <= final["budget"]["budget_total"]


@pytest.mark.asyncio
async def test_slots_hold_the_right_kind():
    final = await collect_final(S1, tier=2, mock=True)
    for day in final["days"]:
        for stop in day["stops"]:
            expected = "attraction" if "attraction" in stop["slot"] else "restaurant"
            assert stop["kind"] == expected


@pytest.mark.asyncio
async def test_each_day_has_a_route():
    final = await collect_final(S1, tier=2, mock=True)
    for day in final["days"]:
        assert len(day["route"]["stops"]) == 5
        assert len(day["route"]["legs"]) == 4
        assert day["route"]["total_km"] > 0


@pytest.mark.asyncio
async def test_tier2_is_deterministic_in_mock_mode():
    a = await collect_final(S1, tier=2, mock=True)
    b = await collect_final(S1, tier=2, mock=True)
    ids_a = [s["venue_id"] for d in a["days"] for s in d["stops"]]
    ids_b = [s["venue_id"] for d in b["days"] for s in d["stops"]]
    assert ids_a == ids_b


# ---------------------------------------------------------------------------
# Trace completeness (the UI depends on this order)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_trace_event_sequence():
    kinds = [e["event"] for e in await _events(S1)]
    assert kinds[0] == "planner_start"
    assert "planner_done" in kinds
    assert "executors_dispatched" in kinds
    assert kinds.count("executor_result") == 10
    assert "critic_reviewed" in kinds
    assert "validation" in kinds
    assert "route_ready" in kinds
    assert kinds[-1] == "final"
    # dispatch precedes results precede critic precedes routing
    assert kinds.index("executors_dispatched") < kinds.index("executor_result")
    assert kinds.index("executor_result") < kinds.index("critic_reviewed")
    assert kinds.index("critic_reviewed") < kinds.index("route_ready")


# ---------------------------------------------------------------------------
# THE Critic-loop failure mode: SLOT_IDS guard on the LLM critic
# ---------------------------------------------------------------------------
class CriticInventsSlot:
    """A soft critic that names a slot outside the closed vocabulary."""

    async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
        return MockResponse(
            text=json.dumps(
                {"issues": [{"slot": "day1_lunch", "issue": "repetitive", "suggestion": "vary it"}]}
            )
        )


@pytest.mark.asyncio
async def test_off_vocabulary_critic_slot_is_dropped():
    """An invented slot from the LLM critic must never reach the reviser.
    This is the failure the whole SLOT_IDS-in-code discipline exists to prevent."""
    venues = catalog.all_venues()
    plan = {"d1_lunch": venues["r012"]}
    reasons = {"d1_lunch": "x"}
    kept = await critic.soft_issues(CriticInventsSlot(), plan, reasons, Preferences())
    assert kept == [], "the off-vocabulary slot 'day1_lunch' must be dropped"


@pytest.mark.asyncio
async def test_valid_critic_slot_is_kept():
    class CriticValidSlot:
        async def chat(self, system, user, *, temperature=0.3, max_tokens=800, json_mode=False):
            return MockResponse(
                text=json.dumps(
                    {"issues": [{"slot": "d1_lunch", "issue": "repetitive", "suggestion": ""}]}
                )
            )

    venues = catalog.all_venues()
    kept = await critic.soft_issues(
        CriticValidSlot(), {"d1_lunch": venues["r012"]}, {"d1_lunch": "x"}, Preferences()
    )
    assert len(kept) == 1
    assert kept[0].slot == "d1_lunch"


@pytest.mark.asyncio
async def test_mock_soft_critic_is_silent():
    """MockLLM contributes no soft issues — the deterministic checks carry it."""
    venues = catalog.all_venues()
    kept = await critic.soft_issues(
        MockLLM(), {"d1_lunch": venues["r012"]}, {"d1_lunch": "x"}, Preferences()
    )
    assert kept == []


# ---------------------------------------------------------------------------
# The deterministic Critic detects planted hard conflicts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critique_detects_a_planted_allergy_violation():
    venues = catalog.all_venues()
    peanut = next(v for v in venues.values() if v["is_trap"] == "peanut_risk")
    plan = {"d1_dinner": peanut}
    issues = await critic.critique(
        MockLLM(), plan, {"d1_dinner": "x"},
        prefs=Preferences(allergies=["peanut"]),
        day_weekdays={1: "wed", 2: "thu"},
    )
    assert any(i.issue == "allergy_violation" and i.slot == "d1_dinner" for i in issues)


@pytest.mark.asyncio
async def test_critique_detects_a_budget_overrun():
    venues = catalog.all_venues()
    busters = [v for v in venues.values() if v["is_trap"] == "budget_buster"]
    plan = {"d1_dinner": busters[0], "d2_dinner": busters[-1]}
    issues = await critic.critique(
        MockLLM(), plan, {"d1_dinner": "x", "d2_dinner": "y"},
        prefs=Preferences(budget_total=100, party_size=2),
        day_weekdays={1: "wed", 2: "thu"},
    )
    assert any(i.slot == BUDGET_SLOT and i.issue == "budget_exceeded" for i in issues)


@pytest.mark.asyncio
async def test_clean_plan_yields_no_critic_issues():
    venues = catalog.all_venues()
    plan = {
        "d1_breakfast": next(
            v for v in venues.values() if v["kind"] == "restaurant" and "breakfast" in v["slot_types"]
        ),
        "d1_am_attraction": next(
            v for v in venues.values() if v["kind"] == "attraction" and "am" in v["slot_types"]
        ),
    }
    reasons = {s: "x" for s in plan}
    issues = await critic.critique(
        MockLLM(), plan, reasons, prefs=Preferences(budget_total=500, party_size=2),
        day_weekdays={1: "wed", 2: "thu"},
    )
    assert issues == []


# ---------------------------------------------------------------------------
# The budget->slot mapping the reviser uses
# ---------------------------------------------------------------------------
def test_budget_issue_targets_the_priciest_restaurant():
    venues = catalog.all_venues()
    plan = {
        "d1_lunch": venues["r004"],   # ~$18pp
        "d1_dinner": venues["r049"],  # $132pp — the budget buster
        "d1_am_attraction": venues["a001"],
    }
    issue = CriticIssue(slot=BUDGET_SLOT, issue="budget_exceeded")
    assert _slot_to_fix(issue, plan) == "d1_dinner"


def test_valid_slot_issue_targets_itself():
    issue = CriticIssue(slot="d2_lunch", issue="duplicate_venue")
    assert _slot_to_fix(issue, {"d2_lunch": {"cost_per_person": 1}}) == "d2_lunch"


def test_off_vocabulary_slot_maps_to_nothing():
    issue = CriticIssue(slot="lunchtime", issue="whatever")
    assert _slot_to_fix(issue, {}) is None
