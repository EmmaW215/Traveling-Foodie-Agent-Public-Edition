"""Central configuration. Everything tunable lives in env vars so model IDs can
change without a redeploy (M0 lesson: free-tier model catalogs move fast).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    """Read an env var, treating an empty value as absent.

    This is deliberate, not defensive noise. GitHub Actions expands an unset
    repository Variable (`${{ vars.FOO }}`) to an *empty string* rather than
    omitting the variable, so `os.environ.get(key, default)` returns "" and
    silently overrides the default. That cost us a failed M0 smoke run: the
    embedding call went out with `{"model": ""}`. Same trap applies to a
    Render env var left blank.
    """
    value = os.environ.get(key, "").strip()
    return value if value else default


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


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
#
# Model IDs are env-overridable because free-tier catalogs churn. Verify the
# current ones with `python scripts/smoke_test.py` before trusting a default.
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
            # No default: OpenRouter's free lineup rotates and any slug we
            # hardcode will rot. Blank => provider skipped (it is the optional
            # tertiary fallback). `smoke_test.py --list-free` lists live ones.
            model=_env("OPENROUTER_MODEL"),
            extra_headers={
                "HTTP-Referer": _env("PUBLIC_APP_URL", "https://localhost:3000"),
                "X-Title": "Traveling Foodie Agent",
            },
        ),
    ]
    return [p for p in chain if p.enabled]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Free Gemini embeddings, exposed through the OpenAI-compatible endpoint.

    NOTE: text-embedding-004 was shut down on 2026-01-14. gemini-embedding-001
    replaces it; it returns 3072 dims by default and supports truncation to
    768 / 1536 / 3072, so we always send the width explicitly.
    """

    base_url: str = _env(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    api_key: str = _env("GEMINI_API_KEY")
    model: str = _env("EMBEDDING_MODEL", "gemini-embedding-001")
    # Must match the dimension chosen when creating the Upstash Vector index.
    dimensions: int = int(_env("EMBEDDING_DIMENSIONS", "1536"))

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
