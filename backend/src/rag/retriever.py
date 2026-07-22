"""Retrieval — two backends behind one interface.

  * UpstashRetriever — embeds the query with Gemini, queries Upstash Vector.
    The production path. The corpus is embedded once by scripts/embed_push.py;
    at runtime we only ever embed the single user question (one call, well
    inside the free tier).
  * LocalRetriever — pure-Python TF-IDF over chunks.jsonl. No keys, no network,
    deterministic. It runs in CI and tests, and is the runtime fallback when
    Upstash or embeddings aren't configured — so /chat always works, degrading
    to lexical retrieval rather than failing.

Same principle as MockLLM: the offline path is a real, working implementation,
not a stub. A query for "ramen" genuinely retrieves the ramen bar either way.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from ..config import embedding_config, settings
from ..llm_client import EmbeddingClient
from ..paths import CHUNKS_PATH

log = logging.getLogger(__name__)

# Small stopword set — enough to stop "the/and/with" dominating the score
# without dragging in a dependency.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with your you we our can what where which who how why when
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


@dataclass
class Hit:
    id: str
    score: float
    text: str
    metadata: dict

    def as_dict(self) -> dict:
        return {"id": self.id, "score": round(self.score, 4), "metadata": self.metadata}


@runtime_checkable
class Retriever(Protocol):
    mode: str

    async def retrieve(self, query: str, k: int = 8) -> list[Hit]: ...


class RetrievalUnavailableError(RuntimeError):
    """The corpus could not be loaded — the seed step did not run."""


# ---------------------------------------------------------------------------
# Local lexical retriever
# ---------------------------------------------------------------------------
class LocalRetriever:
    """TF-IDF retrieval over the built chunks. Deterministic, offline.

    Scoring is IDF-weighted term coverage with TF saturation — a small BM25-lite
    that ranks the obviously-relevant chunk first without any tuning. A query
    with no lexical overlap returns nothing, which is what lets the copilot
    refuse an off-topic question instead of answering from a weak match.
    """

    mode = "local"
    MIN_SCORE = 0.5  # below this, treat as "nothing relevant found"

    # Words that signal the person wants an attraction vs. a place to eat. A
    # mild boost on the matching kind stops "attractions near downtown" ranking
    # a downtown restaurant first — the one weakness of pure lexical search that
    # real embeddings handle for free.
    _ATTRACTION_INTENT = frozenset(
        "attraction attractions see visit sightseeing museum park landmark tour "
        "tours gallery walk hike explore things".split()
    )
    _RESTAURANT_INTENT = frozenset(
        "eat food restaurant restaurants lunch dinner breakfast brunch dine meal "
        "meals hungry cuisine cafe coffee bar dessert".split()
    )
    INTENT_BOOST = 1.5

    def __init__(self, chunks: list[dict] | None = None) -> None:
        self._chunks = chunks if chunks is not None else self._load_chunks()
        self._doc_tokens: dict[str, list[str]] = {
            c["id"]: _tokenize(c["text"]) for c in self._chunks
        }
        self._text: dict[str, str] = {c["id"]: c["text"] for c in self._chunks}
        self._meta: dict[str, dict] = {c["id"]: c.get("metadata", {}) for c in self._chunks}
        self._idf = self._compute_idf()

    @staticmethod
    def _load_chunks() -> list[dict]:
        if not CHUNKS_PATH.exists():
            raise RetrievalUnavailableError(
                f"{CHUNKS_PATH} not found. Run `python -m scripts.seed`."
            )
        with CHUNKS_PATH.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._doc_tokens) or 1
        df: dict[str, int] = {}
        for tokens in self._doc_tokens.values():
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        # smoothed idf, never negative
        return {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def _intent_kind(self, q_tokens: set[str]) -> str | None:
        """Which venue kind the query leans toward, if any."""
        attraction = bool(q_tokens & self._ATTRACTION_INTENT)
        restaurant = bool(q_tokens & self._RESTAURANT_INTENT)
        if attraction and not restaurant:
            return "attraction"
        if restaurant and not attraction:
            return "restaurant"
        return None

    async def retrieve(self, query: str, k: int = 8) -> list[Hit]:
        q_set = set(_tokenize(query))
        if not q_set:
            return []

        intent_kind = self._intent_kind(q_set)

        scored: list[Hit] = []
        for vid, tokens in self._doc_tokens.items():
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            score = 0.0
            for qt in q_set:
                freq = counts.get(qt, 0)
                if freq:
                    score += self._idf.get(qt, 0.0) * (freq / (freq + 1.5))
            if score <= 0:
                continue
            if intent_kind and self._meta[vid].get("kind") == intent_kind:
                score *= self.INTENT_BOOST
            scored.append(Hit(id=vid, score=score, text=self._text[vid], metadata=self._meta[vid]))

        scored.sort(key=lambda h: (-h.score, h.id))
        return [h for h in scored[:k] if h.score >= self.MIN_SCORE]


# ---------------------------------------------------------------------------
# Upstash Vector retriever
# ---------------------------------------------------------------------------
class UpstashRetriever:
    """Embed the query with Gemini, query Upstash Vector over REST.

    The corpus vectors are populated once by scripts/embed_push.py; this only
    embeds the single query. Chunk text is stored in Upstash metadata under
    "text" so the copilot has something to ground on without a second round-trip.
    """

    mode = "upstash"

    def __init__(self, embedder: EmbeddingClient | None = None) -> None:
        self.url = settings.upstash_url.rstrip("/")
        self.token = settings.upstash_token
        self.embedder = embedder or EmbeddingClient()
        self._timeout = httpx.Timeout(settings.request_timeout_s)

    async def retrieve(self, query: str, k: int = 8) -> list[Hit]:
        vector = (await self.embedder.embed([query]))[0]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.url}/query",
                json={
                    "vector": vector,
                    "topK": k,
                    "includeMetadata": True,
                },
                headers={"Authorization": f"Bearer {self.token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        hits: list[Hit] = []
        for item in data.get("result", []):
            meta = item.get("metadata") or {}
            hits.append(
                Hit(
                    id=item["id"],
                    score=float(item.get("score", 0.0)),
                    text=meta.get("text", ""),
                    metadata={k2: v for k2, v in meta.items() if k2 != "text"},
                )
            )
        return hits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_retriever() -> Retriever:
    """Upstash when it and embeddings are configured, else the local retriever.

    Reported by /readiness so a demo can see which path is live.
    """
    if settings.upstash_configured and embedding_config.enabled:
        try:
            return UpstashRetriever()
        except Exception:  # noqa: BLE001 — never let retriever construction 500 the endpoint
            log.warning("Upstash retriever unavailable; falling back to local", exc_info=True)
    return LocalRetriever()
