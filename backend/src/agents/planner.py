"""Planner — interprets the request into a trip shape.

It does not pick venues; it decides cuisines priority, pace and guidance the
executors will honour. Separating interpretation from selection is what keeps
the pipeline debuggable: if the picks are wrong you can see whether the Planner
misread the request or an executor chose badly.
"""
from __future__ import annotations

import json

from ..models import Preferences
from .base import ChatModel, run_structured
from .prompts import PLANNER_SYSTEM
from .schemas import PlannerOutput


def _user_prompt(prefs: Preferences) -> str:
    return (
        "Traveller preferences:\n"
        + json.dumps(
            {
                "city": prefs.city,
                "days": prefs.days,
                "budget_total_cad": prefs.budget_total,
                "party_size": prefs.party_size,
                "cuisines": prefs.cuisines,
                "allergies": prefs.allergies,
                "notes": prefs.notes,
            },
            indent=2,
        )
        + "\n\nInterpret this into a trip plan."
    )


async def plan(model: ChatModel, prefs: Preferences) -> PlannerOutput:
    result = await run_structured(
        model,
        system=PLANNER_SYSTEM,
        user=_user_prompt(prefs),
        schema=PlannerOutput,
        temperature=0.2,
        max_tokens=400,
    )
    # The Planner may reorder or narrow cuisines, but the traveller's stated
    # cuisines must not vanish. If the model dropped them, fold them back in —
    # a stated preference is not the model's to discard.
    assert isinstance(result, PlannerOutput)
    if prefs.cuisines:
        have = {c.lower() for c in result.cuisines_priority}
        for cuisine in prefs.cuisines:
            if cuisine.lower() not in have:
                result.cuisines_priority.append(cuisine)
    return result
