"""A deterministic mock model — the pipeline's offline mode.

Carried over from the hackathon starter kit, where it proved the orchestration
before anyone had a key. It matters for three reasons:

  1. CI runs the full agent pipeline with no secrets.
  2. Tests are deterministic — the same request always produces the same plan,
     so a regression is a real regression, not model variance.
  3. A live demo can fall back to mock mode if every free provider is throttled
     that day. It still exercises every agent, guard and route.

The mock reads the system prompt to decide which agent is asking, then returns
valid JSON for that contract. For picks it deterministically chooses the first
candidate id, which is exactly what the code-level fallback would do — so mock
mode and a totally-failed real call produce the same safe result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class MockResponse:
    text: str
    provider: str = "mock"
    model: str = "mock-deterministic"


class MockLLM:
    """Satisfies the ChatModel protocol without any network."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> MockResponse:
        role = self._role(system)
        self.calls.append(role)
        handler = {
            "copilot": self._copilot,
            "planner": self._planner,
            "restaurant": self._pick,
            "attraction": self._pick,
            "formatter": self._formatter,
        }.get(role, self._pick)
        return MockResponse(text=handler(user))

    @staticmethod
    def _role(system: str) -> str:
        head = system[:40].lower()
        # Copilot first: its prompt also mentions venues, so check it before
        # the executor roles.
        for role in ("copilot", "planner", "restaurant", "attraction", "formatter"):
            if role in head:
                return role
        return "unknown"

    @staticmethod
    def _copilot(user: str) -> str:
        """Ground the answer in the first retrieved venue id. Deterministic."""
        ids = re.findall(r"id=(\S+)", user)
        if not ids:
            return json.dumps(
                {
                    "answerable": False,
                    "answer": "The guide doesn't cover that.",
                    "cited_venue_ids": [],
                }
            )
        return json.dumps(
            {
                "answerable": True,
                "answer": f"Based on the guide, {ids[0]} is a strong match for what you asked.",
                "cited_venue_ids": [ids[0]],
            }
        )

    # -- per-agent deterministic replies ------------------------------------
    @staticmethod
    def _planner(user: str) -> str:
        cuisines: list[str] = []
        match = re.search(r'"cuisines":\s*\[(.*?)\]', user, re.DOTALL)
        if match:
            cuisines = re.findall(r'"([^"]+)"', match.group(1))
        return json.dumps(
            {
                "summary": "A balanced two-day food tour built around your tastes.",
                "cuisines_priority": cuisines,
                "pace": "balanced",
                "notes_for_executors": "Favour variety across the days; keep hops short.",
            }
        )

    @staticmethod
    def _pick(user: str) -> str:
        """Choose the first candidate id — deterministic and always valid."""
        ids = re.findall(r"id=(\S+)", user)
        chosen = ids[0] if ids else ""
        fallback = ids[1] if len(ids) > 1 else None
        return json.dumps(
            {
                "venue_id": chosen,
                "reason": "A strong, well-rated match for this slot and your preferences.",
                "fallback_id": fallback,
            }
        )

    @staticmethod
    def _formatter(user: str) -> str:
        day_count = len(re.findall(r"^Day \d+:", user, re.MULTILINE))
        day_count = max(day_count, 1)
        return json.dumps(
            {
                "title": "Your Calgary Food Itinerary",
                "intro": "Here is a plan tuned to your tastes and budget.",
                "day_summaries": [
                    f"Day {i}: a mix of great food and a couple of stops nearby."
                    for i in range(1, day_count + 1)
                ],
                "closing": "Have a wonderful trip!",
            }
        )
