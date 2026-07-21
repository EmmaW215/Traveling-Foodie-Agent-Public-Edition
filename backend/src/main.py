"""FastAPI orchestrator.

M0 established the deploy path and the LLM fallback chain; M1 added the dataset.
M2 adds the Tier 1 pipeline behind /itinerary as a Server-Sent-Events stream:
the agent trace streams live, then the final itinerary. Tier 0 (/chat) and
Tier 2 remain stubs until M3/M4.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import settings
from .llm_client import AllProvidersFailedError, EmbeddingClient, LLMClient
from .models import ItineraryRequest
from .orchestrator import run_tier1
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


# Preferences and ItineraryRequest now live in src/models.py so the orchestrator
# and CLI can import them without pulling in FastAPI.


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
# Tier endpoints
# ---------------------------------------------------------------------------
def _sse(event: dict) -> str:
    """Encode one trace event as an SSE frame."""
    return f"data: {json.dumps(event)}\n\n"


async def _itinerary_stream(req: ItineraryRequest) -> AsyncIterator[str]:
    """Stream the pipeline's trace events, then a final frame.

    Mock mode kicks in automatically when no provider is configured, so the
    endpoint always produces a complete itinerary — a cold demo with no keys
    still works, it just says so in the trace.
    """
    use_mock = not _llm.configured
    try:
        if req.tier in (0, 2):
            # Tier 0 (M3) and Tier 2 (M4) aren't built yet; run Tier 1 and say so.
            yield _sse({"event": "notice", "message": f"Tier {req.tier} lands later; running Tier 1."})
        async for event in run_tier1(req.preferences, mock=use_mock):
            yield _sse(event)
    except CatalogUnavailableError as exc:
        yield _sse({"event": "error", "detail": f"Dataset not built: {exc}"})
    except Exception as exc:  # noqa: BLE001 — surface any failure to the client, don't hang the stream
        log.exception("itinerary stream failed")
        yield _sse({"event": "error", "detail": str(exc)})


@app.post("/itinerary")
async def itinerary(req: ItineraryRequest) -> StreamingResponse:
    """Tier 1 itinerary as Server-Sent Events: trace frames then a final frame."""
    return StreamingResponse(
        _itinerary_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def chat_tier0() -> dict:
    raise HTTPException(status_code=501, detail="Tier 0 RAG copilot lands in M3.")
