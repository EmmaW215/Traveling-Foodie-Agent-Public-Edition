"""Request/response models shared between the API layer and the orchestrator.

Lives here rather than in main.py so the orchestrator and agents can import it
without pulling in FastAPI — the orchestrator is meant to be runnable from a
CLI and from tests, not only from a web request.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Preferences(BaseModel):
    """The traveller's request. Locked at M0 so the frontend contract is stable."""

    city: str = "Calgary"
    days: int = Field(2, ge=1, le=3)
    budget_total: float = Field(500, gt=0)
    party_size: int = Field(2, ge=1, le=8)
    cuisines: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    notes: str = ""

    def start_weekday(self) -> str:
        """Which weekday day 1 falls on.

        v1 has no calendar, so we default to a weekday when most venues are
        open; a traveller can override via notes handling in a later milestone.
        The point is that the value flows through the opening-hours guard.
        """
        return "wed"

    def day_weekdays(self) -> dict[int, str]:
        order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        start = order.index(self.start_weekday())
        return {d: order[(start + d - 1) % 7] for d in range(1, self.days + 1)}


class ItineraryRequest(BaseModel):
    preferences: Preferences
    tier: int = Field(default=2, ge=0, le=2)
