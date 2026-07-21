"""FastAPI orchestrator.

M0 established the deploy path and the LLM fallback chain. M1 adds the real
dataset behind /dataset/meta. Tier 0/1/2 endpoints are declared as 501 stubs so
the frontend contract is stable from day one; they land in M2-M4.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .llm_client import AllProvidersFailedError, EmbeddingClient, LLMClient
from .tools import catalog
from .tools.catalog import CatalogUnavailableError

logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
log = logging.getLogger("tfa")

app = FastAPI(title=settings.app_name, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_started_at = time.time()
_llm = LLMClient()
_embedder = EmbeddingClient()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class Preferences(BaseModel):
    """Locked in at M0 so the frontend contract never changes later."""

    city: str = "Calgary"
    days: int = Field(2, ge=1, le=3)
    budget_total: float = Field(500, gt=0)
    party_size: int = Field(2, ge=1, le=8)
    cuisines: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    notes: str = ""


class ItineraryRequest(BaseModel):
    preferences: Preferences
    tier: int = Field(default=settings.default_tier, ge=0, le=2)


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Keep-alive + cold-start probe. Must stay dependency-free and instant."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "uptime_s": round(time.time() - _started_at, 1),
    }


@app.get("/readiness")
async def readiness() -> dict:
    """Which subsystems are wired. Used by the M0 exit checklist and the UI badge."""
    try:
        dataset_ready = bool(catalog.all_venues())
    except CatalogUnavailableError:
        dataset_ready = False

    return {
        "llm_providers": _llm.provider_names(),
        "llm_configured": _llm.configured,
        "embeddings_configured": _embedder.configured,
        "vector_db_configured": settings.upstash_configured,
        "dataset_ready": dataset_ready,
        "default_tier": settings.default_tier,
        "allowed_origins": list(settings.allowed_origins),
    }


@app.get("/dataset/meta")
async def dataset_meta() -> dict:
    """What the agent knows about. The disclaimer ships with the data."""
    try:
        meta = catalog.dataset_meta()
        return {
            "cities": [c.strip() for c in meta.get("cities", "").split(",") if c.strip()],
            "restaurants": meta.get("restaurants", 0),
            "attractions": meta.get("attractions", 0),
            "cuisines": catalog.cuisines_available(),
            "neighbourhoods": catalog.neighbourhoods_available(),
            "data_version": meta.get("data_version", "unknown"),
            "data_disclaimer": meta.get("data_disclaimer", ""),
        }
    except CatalogUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail=f"Dataset not built: {exc}"
        ) from exc


@app.post("/echo")
async def echo(req: EchoRequest) -> dict:
    """M0 proof-of-life: one real LLM round-trip through the fallback chain."""
    try:
        result = await _llm.chat(
            system=(
                "You are the Traveling Foodie Agent, in setup mode. "
                "Reply in one short sentence confirming you are online."
            ),
            user=req.message,
            max_tokens=80,
        )
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=503, detail=f"All LLM providers failed: {exc}") from exc

    return {
        "reply": result.text,
        "served_by": result.provider,
        "model": result.model,
    }


# ---------------------------------------------------------------------------
# Tier endpoints — contract declared now, implemented in later milestones
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat_tier0() -> dict:
    raise HTTPException(status_code=501, detail="Tier 0 RAG copilot lands in M3.")


@app.post("/itinerary")
async def itinerary(req: ItineraryRequest) -> dict:
    raise HTTPException(
        status_code=501,
        detail=f"Tier {req.tier} orchestrator lands in M2 (tier 1) / M4 (tier 2).",
    )
