#!/usr/bin/env python3
"""M0 gate: verify every free provider actually answers with the configured model.

Runs in GitHub Actions using repository Secrets — API keys never leave the repo
settings and are never printed in full. Exit code 0 only if at least one chat
provider AND the embedding endpoint work.

Free-tier catalogs churn constantly, so this script is written to tell you
exactly what to change: every line names the model it tried, and an OpenRouter
failure triggers a live lookup of what is actually free right now.

Usage:
    python scripts/smoke_test.py           # chat + embeddings + vector db
    python scripts/smoke_test.py --chat-only
    python scripts/smoke_test.py --list-free   # just list free OpenRouter models
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

TIMEOUT = 45.0

CHAT_PROVIDERS = [
    {
        "name": "groq",
        "key_env": "GROQ_API_KEY",
        "key_prefix": "gsk_",
        "base": os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "model_env": "GROQ_MODEL",
        "model_default": "llama-3.3-70b-versatile",
    },
    {
        "name": "gemini",
        "key_env": "GEMINI_API_KEY",
        "key_prefix": "",
        "base": os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        "model_env": "GEMINI_MODEL",
        "model_default": "gemini-2.0-flash",
    },
    {
        "name": "openrouter",
        "key_env": "OPENROUTER_API_KEY",
        "key_prefix": "sk-or-",
        "base": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model_env": "OPENROUTER_MODEL",
        # No default: the free lineup rotates. Use --list-free to pick one.
        "model_default": "",
    },
]


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    return f"{value[:4]}…{value[-2:]}" if len(value) > 8 else "(set)"


def _key_warning(provider: dict, key: str) -> str:
    """Catch the classic paste-the-wrong-vendor's-key mistake early."""
    prefix = provider["key_prefix"]
    if not (prefix and key) or key.startswith(prefix):
        return ""
    if provider["name"] == "groq" and key.startswith("xai-"):
        return (
            "  <- this looks like an xAI (Grok) key. Groq is a different company; "
            "its keys start with 'gsk_' and come from console.groq.com"
        )
    return f"  <- expected a key starting with '{prefix}'"


def list_free_openrouter_models(limit: int = 12) -> list[str]:
    """Ask OpenRouter which models are free *today*. No API key required."""
    try:
        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []

    free = []
    for model in resp.json().get("data", []):
        pricing = model.get("pricing") or {}
        try:
            is_free = float(pricing.get("prompt", 1)) == 0 and float(
                pricing.get("completion", 1)
            ) == 0
        except (TypeError, ValueError):
            is_free = False
        if is_free:
            free.append(model.get("id", ""))
    return sorted(f for f in free if f)[:limit]


def check_chat(provider: dict) -> tuple[bool, str]:
    key = os.environ.get(provider["key_env"], "").strip()
    if not key:
        return False, "no key configured — skipped"

    model = os.environ.get(provider["model_env"], "").strip() or provider["model_default"]
    if not model:
        return False, (
            f"no model set — set {provider['model_env']} "
            f"(run --list-free to see current options)"
        )

    url = f"{provider['base'].rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 12,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider["name"] == "openrouter":
        headers["X-Title"] = "Traveling Foodie Agent"

    started = time.time()
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f'model="{model}"  transport error: {type(exc).__name__}'

    elapsed = time.time() - started
    if resp.status_code != 200:
        body = resp.text[:200].replace("\n", " ")
        hint = ""
        if resp.status_code == 401:
            hint = _key_warning(provider, key)
        elif resp.status_code == 429 and provider["name"] == "gemini":
            hint = (
                "  <- free-tier quota. If the key is new, it may belong to a project "
                "without free-tier access; create one at aistudio.google.com/apikey"
            )
        return False, f'model="{model}"  HTTP {resp.status_code} — {body}{hint}'

    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        return False, f'model="{model}"  unexpected response shape: {exc}'

    return True, f'model="{model}"  {elapsed:.2f}s  reply="{text[:40]}"'


def check_embeddings() -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False, "GEMINI_API_KEY unset — embeddings unavailable"

    base = os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    model = os.environ.get("EMBEDDING_MODEL", "").strip() or "gemini-embedding-001"
    expected_dims = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))

    payload = {"model": model, "input": ["ramen in Calgary"], "dimensions": expected_dims}

    try:
        resp = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f'model="{model}"  transport error: {type(exc).__name__}'

    if resp.status_code != 200:
        body = resp.text[:200].replace("\n", " ")
        hint = ""
        if "model name format" in body or resp.status_code == 404:
            hint = (
                "  <- check EMBEDDING_MODEL. text-embedding-004 was shut down "
                "2026-01-14; use gemini-embedding-001"
            )
        return False, f'model="{model}"  HTTP {resp.status_code} — {body}{hint}'

    vector = resp.json()["data"][0]["embedding"]
    dims = len(vector)
    if dims != expected_dims:
        return False, (
            f'model="{model}" returned {dims} dims but EMBEDDING_DIMENSIONS={expected_dims}. '
            f"Set EMBEDDING_DIMENSIONS={dims}, or recreate the Upstash index at {dims} dims."
        )
    return True, f'model="{model}"  dims={dims}'


def check_upstash() -> tuple[bool, str, int | None]:
    url = os.environ.get("UPSTASH_VECTOR_REST_URL", "").strip()
    token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "").strip()
    if not (url and token):
        return False, "URL/token not configured — skipped", None

    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/info",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f"transport error: {type(exc).__name__}", None

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} — {resp.text[:180]}", None

    info = resp.json().get("result", {})
    dim = info.get("dimension")
    return True, f"dimension={dim}  vectors={info.get('vectorCount')}", dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-only", action="store_true")
    parser.add_argument("--list-free", action="store_true")
    args = parser.parse_args()

    if args.list_free:
        print("Free OpenRouter models right now:")
        for slug in list_free_openrouter_models(limit=25) or ["(lookup failed)"]:
            print(f"  {slug}")
        return 0

    print("=" * 72)
    print("M0 SMOKE TEST — free-tier provider verification")
    print("=" * 72)

    chat_ok = 0
    openrouter_failed = False
    for provider in CHAT_PROVIDERS:
        ok, detail = check_chat(provider)
        flag = "PASS" if ok else ("SKIP" if "no key" in detail else "FAIL")
        key_display = _mask(os.environ.get(provider["key_env"], ""))
        print(f"[{flag}] chat/{provider['name']:<11} key={key_display}  {detail}")
        chat_ok += int(ok)
        if not ok and provider["name"] == "openrouter" and flag == "FAIL":
            openrouter_failed = True

    if openrouter_failed:
        free_models = list_free_openrouter_models()
        if free_models:
            print("\n       OpenRouter models that are free right now — set OPENROUTER_MODEL")
            print("       to one of these (repository Variables, no code change needed):")
            for slug in free_models:
                print(f"         {slug}")
            print()

    emb_ok = True
    if not args.chat_only:
        ok, detail = check_embeddings()
        emb_ok = ok
        print(f"[{'PASS' if ok else 'FAIL'}] embeddings      {detail}")

        ok_up, detail_up, index_dim = check_upstash()
        flag = "PASS" if ok_up else ("SKIP" if "not configured" in detail_up else "FAIL")
        print(f"[{flag}] upstash-vector  {detail_up}")

        expected = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
        if index_dim is not None and index_dim != expected:
            print(
                f"\n       MISMATCH: Upstash index is {index_dim} dims but "
                f"EMBEDDING_DIMENSIONS={expected}."
                f"\n       gemini-embedding-001 supports 768 / 1536 / 3072, so the usual fix"
                f"\n       is to set EMBEDDING_DIMENSIONS={index_dim} and keep the index.\n"
            )
            emb_ok = False

    print("-" * 72)
    print(f"chat providers passing: {chat_ok}/{len(CHAT_PROVIDERS)}")

    if chat_ok == 0:
        print("\nRESULT: FAIL — no chat provider is usable. Check GROQ_API_KEY / GEMINI_API_KEY.")
        return 1
    if not emb_ok:
        print("\nRESULT: FAIL — chat works but embeddings do not (Tier 0 RAG needs them).")
        return 1

    print("\nRESULT: PASS — M0 provider gate cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
