"""The seam between agents and the model.

Agents depend on this Protocol, not on LLMClient directly. That is what lets
the whole pipeline run offline in mock mode with no API key — the single most
useful property inherited from the hackathon starter kit, because it means the
orchestration logic is tested deterministically in CI and a demo can fall back
to mock mode if every free provider is rate-limited on the day.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from ..llm_client import AllProvidersFailedError, LLMClient

log = logging.getLogger(__name__)


@runtime_checkable
class ChatModel(Protocol):
    """Anything that can answer a chat turn. LLMClient and MockLLM both satisfy it."""

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = ...,
        max_tokens: int = ...,
        json_mode: bool = ...,
    ) -> Any: ...


class AgentError(RuntimeError):
    """An agent could not produce a valid result even after a repair attempt."""


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences and leading/trailing prose.

    Models wrap JSON in fences or add a sentence before it more often than
    JSON-mode would suggest, so we strip fences and, failing that, grab the
    outermost {...}.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
        raw = raw.strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


async def run_structured(
    model: ChatModel,
    *,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> BaseModel:
    """One LLM call that must return JSON matching `schema`.

    Includes one bounded repair attempt: if the first response is not valid
    JSON or fails schema validation, we re-ask once with the error attached.
    Bounded on purpose — an unbounded repair loop is just a slower failure and
    burns the free-tier quota.
    """
    last_error: str | None = None

    for attempt in range(2):  # initial + one repair
        prompt = user
        if last_error:
            prompt = (
                f"{user}\n\nYour previous reply was invalid: {last_error}\n"
                f"Return ONLY valid JSON matching the required shape."
            )
        try:
            response = await model.chat(
                system=system, user=prompt, temperature=temperature,
                max_tokens=max_tokens, json_mode=True,
            )
        except AllProvidersFailedError:
            raise
        except TypeError:
            # A mock without json_mode kwarg; call without it.
            response = await model.chat(system=system, user=prompt)

        text = response.text if hasattr(response, "text") else str(response)

        try:
            data = _extract_json(text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:200]
            log.warning("structured call attempt %s failed: %s", attempt + 1, last_error)

    raise AgentError(
        f"{schema.__name__}: could not get valid output after a repair attempt. "
        f"Last error: {last_error}"
    )


def provider_of(response: Any) -> str:
    """Which provider served a response, for the trace. 'mock' when offline."""
    return getattr(response, "provider", "mock")


def default_model() -> ChatModel:
    """The real client. Agents receive a model by injection; this is the prod default."""
    return LLMClient()
