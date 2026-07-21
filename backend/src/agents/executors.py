"""Restaurant and Attraction executors.

Each takes a slot and a pre-filtered candidate list, asks the model to pick the
best one on taste, and — critically — verifies the returned id against the
candidates in code. An executor can only ever return a venue that was offered
to it; an invented or off-list id is rejected and the deterministic fallback
(the top-ranked candidate) is used instead.

This is the "hallucinated venues score zero" rule made operational: even if the
model ignores the instruction, the code does not.
"""
from __future__ import annotations

import logging

from .base import AgentError, ChatModel, run_structured
from .prompts import ATTRACTION_SYSTEM, RESTAURANT_SYSTEM
from .schemas import VenuePick

log = logging.getLogger(__name__)


def _candidate_block(candidates: list[dict]) -> str:
    """Render candidates as a compact numbered list for the prompt."""
    lines = []
    for c in candidates:
        cost = f"${c['cost_per_person']:.0f}pp" if c["kind"] == "restaurant" else (
            f"${c['cost_per_person']:.0f}pp" if c["cost_per_person"] else "free"
        )
        extra = c["category"]
        if c["kind"] == "restaurant" and c.get("dietary_options"):
            extra += f", {c['dietary_options'].replace(';', '/')}"
        lines.append(
            f"- id={c['venue_id']} | {c['name']} | {extra} | {cost} "
            f"| {c['neighbourhood']} | rating {c['rating']}"
        )
    return "\n".join(lines)


def _context_block(slot: str, prefs, planner_notes: str) -> str:
    return (
        f"Slot: {slot}\n"
        f"Party size: {prefs.party_size}\n"
        f"Cuisines they like: {', '.join(prefs.cuisines) or 'no strong preference'}\n"
        f"Planner notes: {planner_notes or '(none)'}"
    )


async def _pick(
    model: ChatModel,
    *,
    system: str,
    slot: str,
    prefs,
    planner_notes: str,
    candidates: list[dict],
) -> tuple[dict, str, str | None, bool]:
    """Shared pick logic. Returns (venue, reason, fallback_id, used_model).

    `used_model` is False when we had to fall back to the deterministic top
    candidate — surfaced in the trace so a demo can show when the guard fired.
    """
    if not candidates:
        raise AgentError(f"no candidates for slot {slot}")

    by_id = {c["venue_id"]: c for c in candidates}
    default = candidates[0]  # already rating-sorted by the catalogue

    user = (
        _context_block(slot, prefs, planner_notes)
        + "\n\nCandidates (choose exactly one id from this list):\n"
        + _candidate_block(candidates)
    )

    try:
        pick = await run_structured(
            model, system=system, user=user, schema=VenuePick,
            temperature=0.4, max_tokens=300,
        )
    except AgentError:
        # Model failed entirely — deterministic fallback keeps the pipeline alive.
        log.warning("executor for %s fell back to top candidate", slot)
        return default, f"Top-rated available option for {slot}.", None, False

    assert isinstance(pick, VenuePick)

    # The guarantee: the id must be one we offered. This is the anti-
    # hallucination check, in code, not in the prompt.
    if pick.venue_id not in by_id:
        log.warning(
            "executor for %s returned off-list id %r; using top candidate",
            slot, pick.venue_id,
        )
        return default, f"Top-rated available option for {slot}.", None, False

    fallback_id = pick.fallback_id if pick.fallback_id in by_id else None
    return by_id[pick.venue_id], pick.reason, fallback_id, True


async def pick_restaurant(
    model: ChatModel, *, slot: str, prefs, planner_notes: str, candidates: list[dict]
) -> tuple[dict, str, str | None, bool]:
    return await _pick(
        model, system=RESTAURANT_SYSTEM, slot=slot, prefs=prefs,
        planner_notes=planner_notes, candidates=candidates,
    )


async def pick_attraction(
    model: ChatModel, *, slot: str, prefs, planner_notes: str, candidates: list[dict]
) -> tuple[dict, str, str | None, bool]:
    return await _pick(
        model, system=ATTRACTION_SYSTEM, slot=slot, prefs=prefs,
        planner_notes=planner_notes, candidates=candidates,
    )
