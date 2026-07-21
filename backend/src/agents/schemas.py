"""Structured contracts between agents.

Every agent hand-off is JSON validated by one of these models. If an LLM returns
something off-contract, Pydantic raises here rather than letting a malformed
plan flow downstream — which is the difference between a caught error and a
demo that quietly produces nonsense.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    """What the Planner returns: how to interpret the request.

    The Planner does NOT pick venues. It decides the shape of the trip and the
    per-slot search intent; the executors do the picking. Keeping the two
    separate is what makes the pipeline debuggable.
    """

    summary: str = Field(..., description="One sentence describing the trip plan.")
    cuisines_priority: list[str] = Field(
        default_factory=list,
        description="Cuisines to favour, most-wanted first. May be empty.",
    )
    pace: str = Field(
        default="balanced",
        description="relaxed | balanced | packed — how much to fit in a day.",
    )
    notes_for_executors: str = Field(
        default="",
        description="Free-text guidance the executors should honour (vibe, must-dos).",
    )


class VenuePick(BaseModel):
    """An executor's choice for one slot.

    `venue_id` MUST be one the executor saw in its tool results. It is checked
    against the catalogue after parsing; an invented id is rejected in code.
    """

    venue_id: str = Field(..., description="A venue_id from the provided candidates.")
    reason: str = Field(..., description="Why this venue, in one or two sentences.")
    fallback_id: str | None = Field(
        default=None, description="A second candidate's venue_id, if offered."
    )


class FormatterOutput(BaseModel):
    """The Formatter's human-facing itinerary text.

    Structured facts (venues, budget, routes) are assembled in code; the
    Formatter only writes prose around them, so it cannot alter a price or a
    venue. It returns a title plus one short paragraph per day.
    """

    title: str = Field(..., description="A short, appealing itinerary title.")
    intro: str = Field(..., description="One or two sentences setting up the trip.")
    day_summaries: list[str] = Field(
        ..., description="One friendly paragraph per day, in order."
    )
    closing: str = Field(default="", description="Optional sign-off sentence.")
