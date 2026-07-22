#!/usr/bin/env python3
"""Embed the corpus and upsert it to Upstash Vector.

    python -m scripts.embed_push            # embed chunks.jsonl -> Upstash
    python -m scripts.embed_push --dry-run  # build payloads, print, upsert nothing

This is a build/CI job, not a runtime path. It runs once (and again only when
the dataset changes), so the corpus vectors live in Upstash and the app only
ever embeds the single user query at runtime — which keeps us inside the free
embedding tier.

Chunk text is stored in Upstash metadata under "text" so the copilot can ground
on it without a second lookup. Requires GEMINI_API_KEY and the two
UPSTASH_VECTOR_REST_* secrets.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import embedding_config, settings  # noqa: E402
from src.llm_client import EmbeddingClient  # noqa: E402
from src.paths import CHUNKS_PATH  # noqa: E402

BATCH = 16  # embed a handful at a time to be gentle on the free tier


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"{CHUNKS_PATH} not found. Run `python -m scripts.seed` first.")
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_vectors(chunks: list[dict], embeddings: list[list[float]]) -> list[dict]:
    """Assemble Upstash upsert records. Text goes into metadata for grounding."""
    records = []
    for chunk, vector in zip(chunks, embeddings, strict=True):
        metadata = dict(chunk.get("metadata", {}))
        metadata["text"] = chunk["text"]
        records.append({"id": chunk["id"], "vector": vector, "metadata": metadata})
    return records


async def embed_all(chunks: list[dict]) -> list[list[float]]:
    embedder = EmbeddingClient()
    if not embedder.configured:
        raise SystemExit("GEMINI_API_KEY is not set — cannot embed the corpus.")

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        got = await embedder.embed([c["text"] for c in batch])
        vectors.extend(got)
        print(f"  embedded {min(start + BATCH, len(chunks))}/{len(chunks)}")
    return vectors


async def upsert(records: list[dict]) -> None:
    url = settings.upstash_url.rstrip("/")
    token = settings.upstash_token
    if not (url and token):
        raise SystemExit("UPSTASH_VECTOR_REST_URL / _TOKEN are not set.")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
        for start in range(0, len(records), BATCH):
            batch = records[start : start + BATCH]
            resp = await client.post(
                f"{url}/upsert",
                json=batch,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            print(f"  upserted {min(start + BATCH, len(records))}/{len(records)}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build payloads but don't upsert.")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding model={embedding_config.model} "
          f"dims={embedding_config.dimensions}")

    vectors = await embed_all(chunks)
    if vectors and len(vectors[0]) != embedding_config.dimensions:
        raise SystemExit(
            f"Embedding width {len(vectors[0])} != EMBEDDING_DIMENSIONS "
            f"{embedding_config.dimensions}. Fix the config or the Upstash index."
        )

    records = build_vectors(chunks, vectors)
    print(f"Built {len(records)} vector records ({len(vectors[0])} dims each).")

    if args.dry_run:
        print("--dry-run: not upserting. First record id:", records[0]["id"])
        return 0

    await upsert(records)
    print(f"Done. {len(records)} vectors are live in Upstash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
