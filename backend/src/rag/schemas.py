"""Copilot answer contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CopilotAnswer(BaseModel):
    answerable: bool = Field(..., description="False if the dataset can't answer the question.")
    answer: str = Field(..., description="A few sentences, grounded in the context.")
    cited_venue_ids: list[str] = Field(
        default_factory=list, description="Venue ids from the context that back the answer."
    )
