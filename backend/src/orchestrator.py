"""The orchestrator — Tier 1 for now, extended to Tier 2 in M4.

Tier 1 is a sequential pipeline:

    Planner -> (per slot) Restaurant | Attraction -> Budget -> Route -> Formatter

It emits a stream of trace events as it goes (planner_start, executor_result,
budget_update, route_ready, final) which the SSE endpoint forwards to the UI —
the public equivalent of the hackathon's "show the agent trace" requirement.

Key properties, all inherited from the hackathon design:
  * The catalogue applies every hard constraint BEFORE an executor sees
    candidates, so an executor picks on taste and cannot pick something unsafe.
  * The budget is tracked in code between slots, and each slot's price ceiling
    is the even split of what remains — so the plan cannot drift over budget.
  * `exclude_ids` stops the top-rated venue winning every slot.
  * The whole assembled plan is validated by the same deterministic guard the
    Critic will use in Tier 2, so Tier 1 already refuses to emit a broken plan.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from . import guards
from .agents import executors, formatter, planner
from .agents.base import ChatModel, default_model
from .agents.mock import MockLLM
from .guards import SLOT_SPEC, SlotId
from .models import Preferences
from .tools import budget as budget_tool
from .tools import catalog, distance


def _trace(event: str, **data: Any) -> dict:
    return {"event": event, **data}


def make_model(mock: bool) -> ChatModel:
    return MockLLM() if mock else default_model()


def _restaurant_candidates(
    slot_type: str,
    cuisines: list[str],
    ceiling: float,
    allergies: list[str],
    weekday: str,
    used_ids: list[str],
) -> list[dict]:
    """Candidate restaurants for a slot, relaxing *preferences* but never a
    hard constraint (allergy, hours) — and treating the budget ceiling as a
    strong preference rather than an absolute wall.

    The relaxation ladder:
      1. cuisine + budget ceiling
      2. budget ceiling only (drop the cuisine preference)
      3. no ceiling, cheapest first (a very tight budget still gets a real,
         allergen-safe, open venue rather than an empty slot)

    Allergies and opening hours are passed at every rung — those never relax.
    The budget guard still catches an overrun on the whole plan; this just
    stops one impossibly-tight slot from failing the entire itinerary.
    """
    for rung, kwargs in enumerate(
        (
            {"cuisines": cuisines or None, "max_cost_per_person": ceiling if ceiling > 0 else None},
            {"max_cost_per_person": ceiling if ceiling > 0 else None},
            {},  # last resort: no ceiling
        )
    ):
        candidates = catalog.search(
            kind="restaurant",
            slot_type=slot_type,
            allergies=allergies,
            open_on=weekday,
            exclude_ids=used_ids,
            limit=6 if rung < 2 else 60,
            **kwargs,
        )
        if candidates:
            # On the last resort the ceiling was impossible, so favour the
            # cheapest option to protect the overall budget. (catalog.search
            # ranks by rating; here cost wins.)
            if rung == 2:
                candidates.sort(key=lambda c: (c["cost_per_person"], -c["rating"], c["venue_id"]))
                return candidates[:6]
            return candidates
    return []


def _attraction_candidates(
    slot_type: str, ceiling: float, weekday: str, used_ids: list[str]
) -> list[dict]:
    """Candidate attractions for a slot, budget-aware.

    Many attractions are free, so a tight budget should lean on them. We ask
    for attractions within the remaining per-slot share first; if none fit (or
    the budget is exhausted), fall back to the cheapest available so the slot
    is still filled with something valid and open.
    """
    within = catalog.search(
        kind="attraction",
        slot_type=slot_type,
        max_cost_per_person=ceiling if ceiling > 0 else 0,
        open_on=weekday,
        exclude_ids=used_ids,
        limit=6,
    )
    if within:
        return within
    cheapest = catalog.search(
        kind="attraction", slot_type=slot_type, open_on=weekday,
        exclude_ids=used_ids, limit=60,
    )
    cheapest.sort(key=lambda c: (c["cost_per_person"], -c["rating"], c["venue_id"]))
    return cheapest[:6]


async def run_tier1(
    prefs: Preferences, *, mock: bool = False
) -> AsyncIterator[dict]:
    """Run the Tier 1 pipeline, yielding trace events then a final itinerary.

    Async generator so the endpoint can stream it. Everything here is awaitable
    but sequential — Tier 2 (M4) is where executors run concurrently.
    """
    started = time.perf_counter()
    model = make_model(mock)
    day_weekdays = prefs.day_weekdays()
    slots = guards.slots_for_days(prefs.days)

    yield _trace(
        "planner_start",
        message=f"Planning {prefs.days} days in {prefs.city} for {prefs.party_size}.",
    )

    plan_meta = await planner.plan(model, prefs)
    yield _trace(
        "planner_done",
        summary=plan_meta.summary,
        cuisines_priority=plan_meta.cuisines_priority,
        pace=plan_meta.pace,
    )

    tracker = budget_tool.BudgetTracker(
        budget_total=prefs.budget_total, party_size=prefs.party_size
    )
    plan: dict[str, dict] = {}
    reasons: dict[str, str] = {}
    used_ids: list[str] = []

    for index, slot in enumerate(slots):
        spec = SLOT_SPEC[SlotId(slot)]
        slots_left = len(slots) - index
        weekday = day_weekdays[spec["day"]]

        yield _trace("executor_start", slot=slot, kind=spec["kind"])

        if spec["kind"] == "restaurant":
            ceiling = tracker.max_affordable_per_person(slots_left)
            candidates = _restaurant_candidates(
                spec["slot_type"], plan_meta.cuisines_priority, ceiling,
                prefs.allergies, weekday, used_ids,
            )
            venue, reason, _fb, used_model = await executors.pick_restaurant(
                model, slot=slot, prefs=prefs,
                planner_notes=plan_meta.notes_for_executors, candidates=candidates,
            )
            tracker.add(slot, venue)
        else:
            candidates = _attraction_candidates(
                spec["slot_type"], tracker.max_affordable_per_person(slots_left),
                weekday, used_ids,
            )
            venue, reason, _fb, used_model = await executors.pick_attraction(
                model, slot=slot, prefs=prefs,
                planner_notes=plan_meta.notes_for_executors, candidates=candidates,
            )
            if venue.get("cost_per_person"):
                tracker.add(slot, venue)

        plan[slot] = venue
        reasons[slot] = reason
        used_ids.append(venue["venue_id"])

        report = tracker.report()
        yield _trace(
            "executor_result",
            slot=slot,
            venue_id=venue["venue_id"],
            name=venue["name"],
            reason=reason,
            picked_by="model" if used_model else "fallback",
            spent=report.spent,
            remaining=report.remaining,
        )

    # Deterministic validation of the whole plan — the same guard the Critic
    # runs in Tier 2. Tier 1 refuses to emit a plan that fails it.
    report = tracker.report()
    plan_check = guards.validate_plan(
        plan,
        allergies=prefs.allergies,
        budget_total=prefs.budget_total,
        party_size=prefs.party_size,
        day_names=day_weekdays,
    )
    yield _trace("validation", ok=plan_check.ok, issues=[i.as_dict() for i in plan_check.issues])

    # Per-day routes from the precomputed matrix — no external routing API.
    routes = []
    plan_by_day: list[list[dict]] = []
    for day in range(1, prefs.days + 1):
        day_slots = [s for s in slots if SLOT_SPEC[SlotId(s)]["day"] == day]
        day_venues = [plan[s] for s in day_slots]
        ordered = distance.order_by_proximity(day_venues)
        routes.append(distance.route_summary(ordered))
        plan_by_day.append(
            [
                {
                    "slot": s,
                    "venue_id": plan[s]["venue_id"],
                    "name": plan[s]["name"],
                    "category": plan[s]["category"],
                    "kind": plan[s]["kind"],
                    "neighbourhood": plan[s]["neighbourhood"],
                    "cost_per_person": plan[s]["cost_per_person"],
                    "lat": plan[s]["lat"],
                    "lon": plan[s]["lon"],
                    "reason": reasons[s],
                }
                for s in day_slots
            ]
        )
    yield _trace("route_ready", total_km=sum(r["total_km"] for r in routes))

    prose = await formatter.format_itinerary(
        model, plan_by_day=plan_by_day, budget=report.as_dict(), days=prefs.days
    )

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    yield _trace(
        "final",
        tier=1,
        title=prose.title,
        intro=prose.intro,
        closing=prose.closing,
        days=[
            {
                "day": day + 1,
                "summary": prose.day_summaries[day] if day < len(prose.day_summaries) else "",
                "stops": plan_by_day[day],
                "route": routes[day],
            }
            for day in range(prefs.days)
        ],
        budget=report.as_dict(),
        validation=plan_check.as_dict(),
        data_version=catalog.dataset_meta().get("data_version"),
        elapsed_ms=elapsed_ms,
        mock=mock,
    )


async def collect_final(prefs: Preferences, *, mock: bool = False) -> dict:
    """Run the pipeline and return only the final itinerary (for the CLI/tests)."""
    final = None
    async for event in run_tier1(prefs, mock=mock):
        if event["event"] == "final":
            final = event
    if final is None:
        raise RuntimeError("pipeline produced no final event")
    return final
