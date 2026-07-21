"""Constraint enforcement — in code, never in a prompt.

This module is the public-edition copy of the hackathon's hard-won rules:

  * A closed SLOT_IDS vocabulary, validated in code. The Critic loop's classic
    failure is the LLM inventing a slot name ("day1_lunch", "lunch_day_1") and
    the orchestrator then re-planning the wrong slot — or every slot.
  * Allergen exclusion is a filter, not an instruction. An LLM told "avoid
    peanuts" will still occasionally recommend the peanut restaurant.
  * Budget arithmetic is Python. LLMs do not add up reliably, and a wrong total
    is the most obvious possible demo failure.
  * Every venue an LLM names must exist in the catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SlotId(str, Enum):
    """The complete, closed set of itinerary slots for a 2-day plan.

    Anything outside this set is a schema drift bug, not a new slot.
    """

    D1_BREAKFAST = "d1_breakfast"
    D1_AM_ATTRACTION = "d1_am_attraction"
    D1_LUNCH = "d1_lunch"
    D1_PM_ATTRACTION = "d1_pm_attraction"
    D1_DINNER = "d1_dinner"
    D2_BREAKFAST = "d2_breakfast"
    D2_AM_ATTRACTION = "d2_am_attraction"
    D2_LUNCH = "d2_lunch"
    D2_PM_ATTRACTION = "d2_pm_attraction"
    D2_DINNER = "d2_dinner"


SLOT_IDS: frozenset[str] = frozenset(s.value for s in SlotId)

# Which venue kind and meal/period each slot expects.
SLOT_SPEC: dict[str, dict[str, str]] = {
    SlotId.D1_BREAKFAST: {"kind": "restaurant", "slot_type": "breakfast", "day": 1},
    SlotId.D1_AM_ATTRACTION: {"kind": "attraction", "slot_type": "am", "day": 1},
    SlotId.D1_LUNCH: {"kind": "restaurant", "slot_type": "lunch", "day": 1},
    SlotId.D1_PM_ATTRACTION: {"kind": "attraction", "slot_type": "pm", "day": 1},
    SlotId.D1_DINNER: {"kind": "restaurant", "slot_type": "dinner", "day": 1},
    SlotId.D2_BREAKFAST: {"kind": "restaurant", "slot_type": "breakfast", "day": 2},
    SlotId.D2_AM_ATTRACTION: {"kind": "attraction", "slot_type": "am", "day": 2},
    SlotId.D2_LUNCH: {"kind": "restaurant", "slot_type": "lunch", "day": 2},
    SlotId.D2_PM_ATTRACTION: {"kind": "attraction", "slot_type": "pm", "day": 2},
    SlotId.D2_DINNER: {"kind": "restaurant", "slot_type": "dinner", "day": 2},
}

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class SlotValidationError(ValueError):
    """An LLM emitted a slot id outside the closed vocabulary."""


def is_valid_slot(slot: str) -> bool:
    return slot in SLOT_IDS


def validate_slot(slot: str) -> str:
    """Return the slot if valid, else raise.

    Callers in the Critic loop should catch this and re-ask the Critic rather
    than re-planning — re-planning on a bad slot name is how you lose a demo.
    """
    if not is_valid_slot(slot):
        raise SlotValidationError(
            f"'{slot}' is not a known slot. Valid slots: {sorted(SLOT_IDS)}"
        )
    return slot


def slots_for_days(days: int) -> list[str]:
    """The ordered slot list for a 1- or 2-day itinerary."""
    if days not in (1, 2):
        raise ValueError("Only 1- or 2-day itineraries are supported.")
    ordered = [s.value for s in SlotId]
    return ordered if days == 2 else ordered[:5]


# ---------------------------------------------------------------------------
# Allergen exclusion
# ---------------------------------------------------------------------------
def venue_has_allergen(venue: dict, allergies: list[str]) -> bool:
    """True if the venue lists any of the traveller's allergens as present.

    Conservative by design: `allergens_present` means "this allergen is used in
    the kitchen", so we exclude on any overlap rather than trying to reason
    about individual dishes. A missed exclusion is a safety issue; an
    over-exclusion just costs us one restaurant option.
    """
    if not allergies:
        return False
    present = {a.strip().lower() for a in venue.get("allergens_present", "").split(";") if a.strip()}
    wanted = {a.strip().lower() for a in allergies if a and a.strip()}
    return bool(present & wanted)


def filter_allergens(venues: list[dict], allergies: list[str]) -> list[dict]:
    """Drop every venue that uses one of the traveller's allergens."""
    return [v for v in venues if not venue_has_allergen(v, allergies)]


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
def is_open_on(venue: dict, day: str) -> bool:
    """Is the venue open on the given weekday ('mon'...'sun')?"""
    day = day.strip().lower()
    if day not in DAY_NAMES:
        raise ValueError(f"'{day}' is not a weekday. Use one of {DAY_NAMES}.")
    closed = {d.strip().lower() for d in venue.get("closed_days", "").split(";") if d.strip()}
    return day not in closed


def filter_open_on(venues: list[dict], day: str) -> list[dict]:
    return [v for v in venues if is_open_on(v, day)]


# ---------------------------------------------------------------------------
# Venue existence — the anti-hallucination guard
# ---------------------------------------------------------------------------
def venue_exists(identifier: str, known_venues: dict[str, dict]) -> bool:
    """Accept a venue_id or an exact (case-insensitive) venue name."""
    if not identifier:
        return False
    ident = identifier.strip()
    if ident in known_venues:
        return True
    lowered = ident.lower()
    return any(v["name"].lower() == lowered for v in known_venues.values())


def resolve_venue(identifier: str, known_venues: dict[str, dict]) -> dict | None:
    """Return the venue for an id or exact name, or None if it is invented."""
    if not identifier:
        return None
    ident = identifier.strip()
    if ident in known_venues:
        return known_venues[ident]
    lowered = ident.lower()
    for venue in known_venues.values():
        if venue["name"].lower() == lowered:
            return venue
    return None


# ---------------------------------------------------------------------------
# Plan-level validation (what the Critic checks, deterministically)
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    slot: str
    issue: str
    detail: str

    def as_dict(self) -> dict:
        return {"slot": self.slot, "issue": self.issue, "detail": self.detail}


@dataclass
class PlanReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"ok": self.ok, "issues": [i.as_dict() for i in self.issues]}


def validate_plan(
    plan: dict[str, dict],
    *,
    allergies: list[str],
    budget_total: float,
    party_size: int,
    day_names: dict[int, str] | None = None,
) -> PlanReport:
    """Deterministic checks over an assembled plan: {slot_id: venue}.

    The Critic agent reasons about *taste* and *coherence*; these are the
    checks that must never depend on a model being in a good mood.
    """
    report = PlanReport()

    for slot, venue in plan.items():
        if not is_valid_slot(slot):
            report.issues.append(Issue(slot, "invalid_slot", f"'{slot}' is not a known slot id"))
            continue
        if venue is None:
            report.issues.append(Issue(slot, "empty_slot", "no venue selected"))
            continue

        spec = SLOT_SPEC[SlotId(slot)]

        if venue.get("kind") != spec["kind"]:
            report.issues.append(
                Issue(slot, "wrong_kind", f"expected a {spec['kind']}, got a {venue.get('kind')}")
            )

        slot_types = {s for s in venue.get("slot_types", "").split(";") if s}
        if spec["slot_type"] not in slot_types:
            report.issues.append(
                Issue(
                    slot,
                    "wrong_slot_type",
                    f"{venue['name']} does not serve {spec['slot_type']}",
                )
            )

        if venue_has_allergen(venue, allergies):
            overlap = sorted(
                {a.lower() for a in allergies}
                & {a for a in venue.get("allergens_present", "").split(";") if a}
            )
            report.issues.append(
                Issue(slot, "allergy_violation", f"{venue['name']} uses {', '.join(overlap)}")
            )

        if day_names:
            day = day_names.get(spec["day"])
            if day and not is_open_on(venue, day):
                report.issues.append(
                    Issue(slot, "closed", f"{venue['name']} is closed on {day}")
                )

    # Repeated venues. Found by an end-to-end dry run in M1: every guard passed
    # while the itinerary sent the traveller to the same ramen bar three times.
    # Technically valid, obviously useless — so it is a plan-level constraint.
    seen: dict[str, str] = {}
    for slot, venue in plan.items():
        if not venue or "venue_id" not in venue:
            continue
        vid = venue["venue_id"]
        if vid in seen:
            report.issues.append(
                Issue(
                    slot,
                    "duplicate_venue",
                    f"{venue['name']} is already scheduled in {seen[vid]}",
                )
            )
        else:
            seen[vid] = slot

    total = sum(
        v["cost_per_person"] * party_size for v in plan.values() if v and "cost_per_person" in v
    )
    if total > budget_total:
        report.issues.append(
            Issue(
                "budget",
                "budget_exceeded",
                f"plan totals ${total:.2f} for {party_size} people, budget is ${budget_total:.2f}",
            )
        )

    return report
