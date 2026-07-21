"""Central configuration. Everything tunable lives in env vars so model IDs can
change without a redeploy (M0 lesson: free-tier model catalogs move fast).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible chat provider in the fallback chain."""

    name: str
    base_url: str
    api_key: str
    model: str
    # Some gateways want extra headers (OpenRouter attribution, etc.)
    extra_headers: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


# ---------------------------------------------------------------------------
# Provider chain: Groq (primary) -> Gemini (fallback 1) -> OpenRouter (last).
# Model IDs are env-overridable; the defaults below are what we verify at M0.
# ---------------------------------------------------------------------------
def build_provider_chain() -> list[Provider]:
    chain = [
        Provider(
            name="groq",
            base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=_env("GROQ_API_KEY"),
            model=_env("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ),
        Provider(
            name="gemini",
            base_url=_env(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            api_key=_env("GEMINI_API_KEY"),
            model=_env("GEMINI_MODEL", "gemini-2.0-flash"),
        ),
        Provider(
            name="openrouter",
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=_env("OPENROUTER_API_KEY"),
            model=_env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            extra_headers={
                "HTTP-Referer": _env("PUBLIC_APP_URL", "https://localhost:3000"),
                "X-Title": "Traveling Foodie Agent",
            },
        ),
    ]
    return [p for p in chain if p.enabled]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Gemini free embeddings, exposed through its OpenAI-compatible endpoint."""

    base_url: str = _env(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    api_key: str = _env("GEMINI_API_KEY")
    model: str = _env("EMBEDDING_MODEL", "text-embedding-004")
    # Must match the dimension chosen when creating the Upstash Vector index.
    dimensions: int = int(_env("EMBEDDING_DIMENSIONS", "768"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Traveling Foodie Agent — Public Edition"
    version: str = _env("APP_VERSION", "0.1.0-m0")
    # Comma-separated list; the Vercel domain must be in here.
    allowed_origins: tuple = tuple(
        o.strip()
        for o in _env(
            "ALLOWED_ORIGIN",
            "http://localhost:3000,https://traveling-foodie-agent-public-edition.vercel.app",
        ).split(",")
        if o.strip()
    )
    request_timeout_s: float = float(_env("LLM_TIMEOUT_S", "45"))
    max_retries_per_provider: int = int(_env("LLM_RETRIES", "1"))
    default_tier: int = int(_env("DEFAULT_TIER", "2"))
    debug: bool = _env_bool("DEBUG", False)

    # Upstash Vector (used from M3 onward; validated at M0 only for presence).
    upstash_url: str = _env("UPSTASH_VECTOR_REST_URL")
    upstash_token: str = _env("UPSTASH_VECTOR_REST_TOKEN")

    @property
    def upstash_configured(self) -> bool:
        return bool(self.upstash_url and self.upstash_token)


settings = Settings()
embedding_config = EmbeddingConfig()
