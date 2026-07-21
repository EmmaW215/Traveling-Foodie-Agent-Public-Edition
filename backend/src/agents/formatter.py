"""Formatter — writes prose around a finished, validated plan.

It receives the assembled itinerary as facts and only writes the words. It
cannot change a price, a venue or a time, because those are computed in code and
merely handed to it. If the model fails, a deterministic template still produces
a readable itinerary — the demo never shows a blank result.
"""
from __future__ import annotations

import logging

from .base import AgentError, ChatModel, run_structured
from .prompts import FORMATTER_SYSTEM
from .schemas import FormatterOutput

log = logging.getLogger(__name__)


def _facts_block(plan_by_day: list[list[dict]], budget: dict) -> str:
    lines = [f"Budget: ${budget['spent']:.0f} of ${budget['budget_total']:.0f} for the group.\n"]
    for day_num, stops in enumerate(plan_by_day, start=1):
        lines.append(f"Day {day_num}:")
        for stop in stops:
            cost = (
                f"${stop['cost_per_person']:.0f}pp"
                if stop["cost_per_person"]
                else "free"
            )
            lines.append(
                f"  - {stop['slot']}: {stop['name']} ({stop['category']}, {cost}) "
                f"in {stop['neighbourhood']} — {stop['reason']}"
            )
    return "\n".join(lines)


def _template_fallback(plan_by_day: list[list[dict]], budget: dict) -> FormatterOutput:
    """A readable itinerary with no LLM. Used if the Formatter call fails."""
    day_summaries = []
    for day_num, stops in enumerate(plan_by_day, start=1):
        bits = [
            f"{s['name']} ({s['category']})"
            for s in stops
        ]
        day_summaries.append(f"Day {day_num}: " + " → ".join(bits) + ".")
    return FormatterOutput(
        title="Your 2-Day Calgary Food Itinerary",
        intro=(
            f"A plan built around your tastes, keeping the group spend to "
            f"${budget['spent']:.0f} of ${budget['budget_total']:.0f}."
        ),
        day_summaries=day_summaries,
        closing="Enjoy the trip!",
    )


async def format_itinerary(
    model: ChatModel, *, plan_by_day: list[list[dict]], budget: dict, days: int
) -> FormatterOutput:
    user = (
        _facts_block(plan_by_day, budget)
        + f"\n\nWrite the itinerary. Produce exactly {days} day_summaries, in order."
    )
    try:
        result = await run_structured(
            model, system=FORMATTER_SYSTEM, user=user,
            schema=FormatterOutput, temperature=0.6, max_tokens=900,
        )
    except AgentError:
        log.warning("formatter fell back to template")
        return _template_fallback(plan_by_day, budget)

    assert isinstance(result, FormatterOutput)
    # Guarantee one summary per day even if the model produced the wrong count.
    if len(result.day_summaries) != days:
        template = _template_fallback(plan_by_day, budget)
        result.day_summaries = template.day_summaries
    return result
