"""The Critic — the Tier 2 ceiling, and the piece with the famous failure mode.

The Critic reviews an assembled plan and returns structured issues that the
revision loop acts on. Two sources, one guard:

  1. Deterministic hard checks (guards.validate_plan): allergy, budget, opening
     hours, wrong kind, duplicates. These are authoritative and always run — a
     model is never trusted to notice an allergen.
  2. An LLM soft critique: repetition, lopsided days, vibe clashes — the
     judgement a person makes. Optional; skipped if there is no real model.

Every issue's `slot` is validated against the closed SLOT_IDS vocabulary IN
CODE. The classic Critic-loop failure is the model emitting "day1_lunch" and the
orchestrator re-planning the wrong slot — or, worse, every slot. An
off-vocabulary slot from the LLM critic is dropped (and logged), never acted on.
"""
from __future__ import annotations

import logging

from .. import guards
from ..guards import SLOT_IDS, PlanReport
from .base import AgentError, ChatModel, run_structured
from .mock import MockLLM
from .prompts import CRITIC_SYSTEM
from .schemas import CriticIssue, CriticOutput

log = logging.getLogger(__name__)

# The slot label the budget issue carries; not a real itinerary slot, but a
# valid target the reviser understands.
BUDGET_SLOT = "budget"
_ACTIONABLE_SLOTS = SLOT_IDS | {BUDGET_SLOT}


def _hard_issues(report: PlanReport) -> list[CriticIssue]:
    """Turn the deterministic validator's findings into Critic issues."""
    return [
        CriticIssue(slot=i.slot, issue=i.issue, suggestion=i.detail) for i in report.issues
    ]


def _plan_digest(plan: dict[str, dict], reasons: dict[str, str]) -> str:
    lines = []
    for slot, venue in plan.items():
        if not venue:
            continue
        cost = f"${venue['cost_per_person']:.0f}pp" if venue.get("cost_per_person") else "free"
        lines.append(
            f"{slot}: {venue['name']} ({venue['category']}, {cost}, {venue['neighbourhood']})"
            f" — {reasons.get(slot, '')}"
        )
    return "\n".join(lines)


async def soft_issues(
    model: ChatModel, plan: dict[str, dict], reasons: dict[str, str], prefs
) -> list[CriticIssue]:
    """LLM critique for taste/coherence. Slots validated against SLOT_IDS.

    A real MockLLM returns no issues (the deterministic checks carry the loop in
    offline mode); a live model adds soft judgement. Either way, any issue whose
    slot is off-vocabulary is dropped here, in code.
    """
    # Skip the LLM pass in pure-offline mode: the mock has nothing to add beyond
    # the deterministic checks, and this keeps the mock trace clean.
    if isinstance(model, MockLLM):
        return []

    user = (
        f"Traveller: party of {prefs.party_size}, cuisines "
        f"{', '.join(prefs.cuisines) or 'no strong preference'}, "
        f"allergies {', '.join(prefs.allergies) or 'none'}.\n\n"
        f"PLAN:\n{_plan_digest(plan, reasons)}"
    )
    system = CRITIC_SYSTEM.replace("{slot_ids}", ", ".join(sorted(SLOT_IDS)))
    try:
        result = await run_structured(
            model, system=system, user=user, schema=CriticOutput,
            temperature=0.2, max_tokens=500,
        )
    except AgentError:
        # A critic that can't produce valid output shouldn't block the plan.
        log.warning("soft critic failed to produce valid output; skipping soft issues")
        return []

    assert isinstance(result, CriticOutput)
    kept: list[CriticIssue] = []
    for issue in result.issues:
        if guards.is_valid_slot(issue.slot):
            kept.append(issue)
        else:
            # THE guard: never act on an invented slot. Log and drop.
            log.warning("critic proposed off-vocabulary slot %r; dropped", issue.slot)
    return kept


async def critique(
    model: ChatModel,
    plan: dict[str, dict],
    reasons: dict[str, str],
    *,
    prefs,
    day_weekdays: dict[int, str],
) -> list[CriticIssue]:
    """Full critique: authoritative hard issues + guarded soft issues.

    Deduplicated by (slot, issue) so the same problem isn't reported twice. Only
    issues with an actionable slot are returned — everything here is safe for the
    revision loop to act on.
    """
    report = guards.validate_plan(
        plan,
        allergies=prefs.allergies,
        budget_total=prefs.budget_total,
        party_size=prefs.party_size,
        day_names=day_weekdays,
    )
    issues = _hard_issues(report)
    issues += await soft_issues(model, plan, reasons, prefs)

    seen: set[tuple[str, str]] = set()
    deduped: list[CriticIssue] = []
    for issue in issues:
        if issue.slot not in _ACTIONABLE_SLOTS:
            continue
        key = (issue.slot, issue.issue)
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    return deduped
