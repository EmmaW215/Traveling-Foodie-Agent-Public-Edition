"""Provider-agnostic LLM client — the public-edition analogue of the hackathon's
`fuelix_client.py`.

Design rules carried over from the TELUS build:
  * httpx only (no vendor SDKs) — smallest install footprint, works on a
    512 MB Render free instance.
  * One thin wrapper; agents never talk to a provider directly.
  * Fallback chain on 429 / 5xx / timeout: Groq -> Gemini -> OpenRouter.
  * JSON mode helper, because every agent hand-off is structured.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Provider, build_provider_chain, embedding_config, settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class AllProvidersFailedError(RuntimeError):
    """Every provider in the chain refused or errored."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def as_json(self) -> Any:
        """Parse the response as JSON, tolerating ```json fences."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
            raw = raw.strip("`").strip()
        return json.loads(raw)


class LLMClient:
    """Chat completions across a chain of free OpenAI-compatible providers."""

    def __init__(self, providers: list[Provider] | None = None) -> None:
        self.providers = providers if providers is not None else build_provider_chain()
        self._timeout = httpx.Timeout(settings.request_timeout_s)

    # -- introspection ------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.providers)

    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    # -- core call ----------------------------------------------------------
    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.providers:
            raise AllProvidersFailedError(
                "No LLM provider configured. Set at least GROQ_API_KEY or GEMINI_API_KEY."
            )

        errors: list[str] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for provider in self.providers:
                for attempt in range(settings.max_retries_per_provider + 1):
                    try:
                        return await self._call_one(
                            client,
                            provider,
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        )
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        errors.append(f"{provider.name}: HTTP {status}")
                        log.warning(
                            "provider %s returned %s (attempt %s)",
                            provider.name,
                            status,
                            attempt + 1,
                        )
                        if status not in RETRYABLE_STATUS:
                            break  # hard failure -> next provider immediately
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        errors.append(f"{provider.name}: {type(exc).__name__}")
                        log.warning("provider %s transport error: %s", provider.name, exc)

        raise AllProvidersFailedError("; ".join(errors) or "unknown failure")

    async def _call_one(
        self,
        client: httpx.AsyncClient,
        provider: Provider,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            **provider.extra_headers,
        }

        resp = await client.post(
            f"{provider.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=provider.name,
            model=provider.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class EmbeddingClient:
    """Free Gemini embeddings via the OpenAI-compatible endpoint.

    Used at build time (embed_push.py) for the corpus and at runtime for the
    single user query in Tier 0.
    """

    def __init__(self) -> None:
        self.cfg = embedding_config
        self._timeout = httpx.Timeout(settings.request_timeout_s)

    @property
    def configured(self) -> bool:
        return self.cfg.enabled

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.cfg.enabled:
            raise AllProvidersFailedError("GEMINI_API_KEY not set — embeddings unavailable.")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.cfg.base_url.rstrip('/')}/embeddings",
                json={"model": self.cfg.model, "input": texts},
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]
