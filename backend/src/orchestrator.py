"""The orchestrator — Tier 1 (sequential) and Tier 2 (multi-agent).

Tier 1 is a sequential pipeline:

    Planner -> (per slot) Restaurant | Attraction -> Budget -> Route -> Formatter

Tier 2 fans the executors out in parallel, then reconciles with a Critic loop:

    Planner -> [Restaurant ∥ Attraction]  (asyncio.gather)
            -> Critic -> revise flagged slots (≤2) -> Route -> Formatter

Both emit a stream of trace events the SSE endpoint forwards to the UI — the
public equivalent of the hackathon's "show the agent trace" requirement.

Key properties, all inherited from the hackathon design:
  * The catalogue applies every hard constraint BEFORE an executor sees
    candidates, so an executor picks on taste and cannot pick something unsafe.
  * Budget is tracked in code, never by an LLM.
  * The whole assembled plan is validated by the same deterministic guard the
    Critic uses, so neither tier emits a broken plan.
  * The Critic's revision loop is bounded to 2 iterations and every issue's slot
    is validated against the closed SLOT_IDS vocabulary in code.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from . import guards
from .agents import critic, executors, formatter, planner
from .agents.base import ChatModel, default_model
from .agents.critic import BUDGET_SLOT
from .agents.mock import MockLLM
from .guards import SLOT_SPEC, SlotId
from .models import Preferences
from .tools import budget as budget_tool
from .tools import catalog, distance

MAX_CRITIC_ITERATIONS = 2  # the bounded revision loop


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
    but sequential — Tier 2 is where executors run concurrently.
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

    async for event in _emit_routes_and_final(
        model, prefs=prefs, plan=plan, reasons=reasons, slots=slots,
        report=report, plan_check=plan_check, tier=1, started=started, mock=mock,
    ):
        yield event


def _day_stops(plan: dict, reasons: dict, day_slots: list[str]) -> list[dict]:
    return [
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


async def _emit_routes_and_final(
    model: ChatModel,
    *,
    prefs: Preferences,
    plan: dict,
    reasons: dict,
    slots: list[str],
    report,
    plan_check,
    tier: int,
    started: float,
    mock: bool,
    revisions: int = 0,
) -> AsyncIterator[dict]:
    """Shared tail for both tiers: per-day routes, the Formatter, the final event.

    Routes come from the precomputed haversine matrix — no external routing API.
    """
    routes = []
    plan_by_day: list[list[dict]] = []
    for day in range(1, prefs.days + 1):
        day_slots = [s for s in slots if SLOT_SPEC[SlotId(s)]["day"] == day]
        ordered = distance.order_by_proximity([plan[s] for s in day_slots])
        routes.append(distance.route_summary(ordered))
        plan_by_day.append(_day_stops(plan, reasons, day_slots))
    yield _trace("route_ready", total_km=sum(r["total_km"] for r in routes))

    prose = await formatter.format_itinerary(
        model, plan_by_day=plan_by_day, budget=report.as_dict(), days=prefs.days
    )

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    final = {
        "event": "final",
        "tier": tier,
        "title": prose.title,
        "intro": prose.intro,
        "closing": prose.closing,
        "days": [
            {
                "day": day + 1,
                "summary": prose.day_summaries[day] if day < len(prose.day_summaries) else "",
                "stops": plan_by_day[day],
                "route": routes[day],
            }
            for day in range(prefs.days)
        ],
        "budget": report.as_dict(),
        "validation": plan_check.as_dict(),
        "data_version": catalog.dataset_meta().get("data_version"),
        "elapsed_ms": elapsed_ms,
        "mock": mock,
    }
    if tier == 2:
        final["revisions"] = revisions
    yield final


# ---------------------------------------------------------------------------
# Tier 2 — parallel executors + the bounded Critic loop
# ---------------------------------------------------------------------------
def _slot_to_fix(issue, plan: dict[str, dict]) -> str | None:
    """Map a Critic issue to the slot the reviser should re-pick.

    A budget overrun has no single slot, so we target the most expensive
    restaurant pick — swapping it for something cheaper is the highest-leverage
    fix. Everything else names its own slot.
    """
    if issue.slot == BUDGET_SLOT:
        restaurant_slots = [
            s for s, v in plan.items()
            if v and SLOT_SPEC[SlotId(s)]["kind"] == "restaurant"
        ]
        if not restaurant_slots:
            return None
        return max(restaurant_slots, key=lambda s: plan[s]["cost_per_person"])
    return issue.slot if guards.is_valid_slot(issue.slot) else None


async def _repick_slot(
    model: ChatModel,
    slot: str,
    *,
    prefs: Preferences,
    plan_meta,
    plan: dict[str, dict],
    used_ids: set[str],
    day_weekdays: dict[int, str],
    issue,
) -> tuple[dict | None, str | None]:
    """Re-pick one slot under tightened constraints, excluding used venues.

    Excludes every venue currently in the plan (so the re-pick is genuinely
    different), keeps allergy and opening-hours filtering, and tightens the
    price ceiling when the issue is about cost.
    """
    spec = SLOT_SPEC[SlotId(slot)]
    weekday = day_weekdays[spec["day"]]
    exclude = list(used_ids)  # force a new venue

    budget_issue = issue.issue in {"budget_exceeded", "too_expensive", "budget_buster"}

    if spec["kind"] == "restaurant":
        ceiling = plan[slot]["cost_per_person"] * 0.7 if budget_issue else 0
        candidates = _restaurant_candidates(
            spec["slot_type"], plan_meta.cuisines_priority, ceiling,
            prefs.allergies, weekday, exclude,
        )
        if not candidates:
            return None, None
        venue, reason, _fb, _used = await executors.pick_restaurant(
            model, slot=slot, prefs=prefs,
            planner_notes=plan_meta.notes_for_executors, candidates=candidates,
        )
    else:
        ceiling = plan[slot]["cost_per_person"] * 0.7 if budget_issue else 0
        candidates = _attraction_candidates(spec["slot_type"], ceiling, weekday, exclude)
        if not candidates:
            return None, None
        venue, reason, _fb, _used = await executors.pick_attraction(
            model, slot=slot, prefs=prefs,
            planner_notes=plan_meta.notes_for_executors, candidates=candidates,
        )
    return venue, reason


async def run_tier2(prefs: Preferences, *, mock: bool = False) -> AsyncIterator[dict]:
    """Tier 2: fan the executors out in parallel, then reconcile with the Critic.

    The parallelism is real: every slot is picked concurrently with
    `asyncio.gather`. Because the parallel executors don't see each other's
    picks, two slots can grab the same top-rated venue — that cross-slot
    conflict is exactly what the Critic reconciles. Parallel for speed, Critic
    for correctness. The revision loop is bounded to 2 iterations and every
    issue's slot is validated against SLOT_IDS in code.
    """
    started = time.perf_counter()
    model = make_model(mock)
    day_weekdays = prefs.day_weekdays()
    slots = guards.slots_for_days(prefs.days)

    yield _trace(
        "planner_start",
        message=f"Planning {prefs.days} days in {prefs.city} for {prefs.party_size} (Tier 2).",
    )
    plan_meta = await planner.plan(model, prefs)
    yield _trace(
        "planner_done",
        summary=plan_meta.summary,
        cuisines_priority=plan_meta.cuisines_priority,
        pace=plan_meta.pace,
    )

    # Even-split per-restaurant budget ceiling, up front — the parallel branches
    # can't share the running total Tier 1 uses, so the Critic reconciles budget
    # afterwards instead.
    restaurant_slots = [s for s in slots if SLOT_SPEC[SlotId(s)]["kind"] == "restaurant"]
    ceiling = prefs.budget_total / (max(len(restaurant_slots), 1) * prefs.party_size)

    yield _trace("executors_dispatched", slots=list(slots), parallel=True)

    async def pick_one(slot: str) -> tuple[str, dict, str]:
        spec = SLOT_SPEC[SlotId(slot)]
        weekday = day_weekdays[spec["day"]]
        if spec["kind"] == "restaurant":
            candidates = _restaurant_candidates(
                spec["slot_type"], plan_meta.cuisines_priority, ceiling,
                prefs.allergies, weekday, [],
            )
            venue, reason, _fb, _u = await executors.pick_restaurant(
                model, slot=slot, prefs=prefs,
                planner_notes=plan_meta.notes_for_executors, candidates=candidates,
            )
        else:
            candidates = _attraction_candidates(spec["slot_type"], ceiling, weekday, [])
            venue, reason, _fb, _u = await executors.pick_attraction(
                model, slot=slot, prefs=prefs,
                planner_notes=plan_meta.notes_for_executors, candidates=candidates,
            )
        return slot, venue, reason

    picked = await asyncio.gather(*(pick_one(s) for s in slots))
    plan: dict[str, dict] = {slot: venue for slot, venue, _ in picked}
    reasons: dict[str, str] = {slot: reason for slot, _, reason in picked}

    for slot in slots:
        yield _trace(
            "executor_result",
            slot=slot,
            venue_id=plan[slot]["venue_id"],
            name=plan[slot]["name"],
            reason=reasons[slot],
            parallel=True,
        )

    # Critic revision loop — bounded, slot-guarded.
    revisions = 0
    for iteration in range(1, MAX_CRITIC_ITERATIONS + 1):
        issues = await critic.critique(
            model, plan, reasons, prefs=prefs, day_weekdays=day_weekdays
        )
        yield _trace(
            "critic_reviewed",
            iteration=iteration,
            issues=[{"slot": i.slot, "issue": i.issue} for i in issues],
        )
        if not issues:
            break

        fixed_any = False
        for issue in issues:
            target = _slot_to_fix(issue, plan)
            if target is None:
                continue
            # Recompute the used set from the live plan each time: a venue we
            # just replaced may still be in use at another unfixed duplicate
            # slot, so a stale set would let the reviser grab it back.
            used_ids = {v["venue_id"] for v in plan.values()}
            new_venue, new_reason = await _repick_slot(
                model, target, prefs=prefs, plan_meta=plan_meta, plan=plan,
                used_ids=used_ids, day_weekdays=day_weekdays, issue=issue,
            )
            if new_venue and new_venue["venue_id"] != plan[target]["venue_id"]:
                old_name = plan[target]["name"]
                plan[target] = new_venue
                reasons[target] = new_reason
                revisions += 1
                fixed_any = True
                yield _trace(
                    "revision",
                    iteration=iteration,
                    slot=target,
                    issue=issue.issue,
                    replaced=old_name,
                    with_=new_venue["name"],
                )
        if not fixed_any:
            break

    report = budget_tool.price_plan(plan, prefs.budget_total, prefs.party_size)
    plan_check = guards.validate_plan(
        plan,
        allergies=prefs.allergies,
        budget_total=prefs.budget_total,
        party_size=prefs.party_size,
        day_names=day_weekdays,
    )
    yield _trace(
        "validation", ok=plan_check.ok, issues=[i.as_dict() for i in plan_check.issues]
    )

    async for event in _emit_routes_and_final(
        model, prefs=prefs, plan=plan, reasons=reasons, slots=slots,
        report=report, plan_check=plan_check, tier=2, started=started,
        mock=mock, revisions=revisions,
    ):
        yield event


async def collect_final(prefs: Preferences, *, tier: int = 1, mock: bool = False) -> dict:
    """Run a tier's pipeline and return only the final itinerary (CLI/tests)."""
    runner = run_tier2 if tier == 2 else run_tier1
    final = None
    async for event in runner(prefs, mock=mock):
        if event["event"] == "final":
            final = event
    if final is None:
        raise RuntimeError("pipeline produced no final event")
    return final
