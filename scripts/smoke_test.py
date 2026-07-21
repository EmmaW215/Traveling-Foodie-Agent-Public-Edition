#!/usr/bin/env python3
"""M0 gate: verify every free provider actually answers with the configured model.

Runs in GitHub Actions using repository Secrets — API keys never leave the repo
settings and are never printed in full.

Gate policy
-----------
  * At least one CHAT provider must work.          -> blocking at M0
  * Embeddings are a Tier 0 / RAG dependency.      -> warning at M0, blocking
    from M3 onward (pass --strict once M3 starts).

Free-tier catalogs churn, so every line reports the model it actually used and
an OpenRouter failure triggers a live lookup of what is free right now.

Usage:
    python scripts/smoke_test.py              # M0 policy
    python scripts/smoke_test.py --strict     # M3+ policy: embeddings required
    python scripts/smoke_test.py --chat-only
    python scripts/smoke_test.py --list-free  # current free OpenRouter models
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

TIMEOUT = 45.0


def env(key: str, default: str = "") -> str:
    """Read an env var, treating empty/whitespace as absent.

    GitHub Actions expands an unset repository Variable to an empty string,
    so os.environ.get(key, default) would return "" and override the default.
    """
    value = os.environ.get(key, "").strip()
    return value if value else default


CHAT_PROVIDERS = [
    {
        "name": "groq",
        "key_env": "GROQ_API_KEY",
        "key_prefix": "gsk_",
        "base": env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "model_env": "GROQ_MODEL",
        "model_default": "llama-3.3-70b-versatile",
    },
    {
        "name": "gemini",
        "key_env": "GEMINI_API_KEY",
        "key_prefix": "",
        "base": env(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        "model_env": "GEMINI_MODEL",
        "model_default": "gemini-2.0-flash",
    },
    {
        "name": "openrouter",
        "key_env": "OPENROUTER_API_KEY",
        "key_prefix": "sk-or-",
        "base": env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model_env": "OPENROUTER_MODEL",
        # No default: the free lineup rotates. Use --list-free to pick one.
        "model_default": "",
    },
]


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    return f"{value[:4]}…{value[-2:]}" if len(value) > 8 else "(set)"


def _key_hint(provider: dict, key: str) -> str:
    """Catch the classic paste-the-wrong-vendor's-key mistake."""
    prefix = provider["key_prefix"]
    if not (prefix and key) or key.startswith(prefix):
        return ""
    if provider["name"] == "groq" and key.startswith("xai-"):
        return (
            "\n         ^ that looks like an xAI (Grok) key. Groq is a different"
            " company — its keys start with 'gsk_' (console.groq.com)"
        )
    return f"\n         ^ expected a key starting with '{prefix}'"


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
            if float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0:
                free.append(model.get("id", ""))
        except (TypeError, ValueError):
            continue
    return sorted(f for f in free if f)[:limit]


def check_chat(provider: dict) -> tuple[bool, str]:
    key = env(provider["key_env"])
    if not key:
        return False, "no key configured — skipped"

    model = env(provider["model_env"], provider["model_default"])
    if not model:
        return False, (
            f"no model set — set {provider['model_env']} "
            f"(run with --list-free to see current options)"
        )

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
        resp = httpx.post(
            f"{provider['base'].rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f'model="{model}"  transport error: {type(exc).__name__}'

    elapsed = time.time() - started
    if resp.status_code != 200:
        body = resp.text[:200].replace("\n", " ")
        hint = ""
        if resp.status_code == 401:
            hint = _key_hint(provider, key)
        elif resp.status_code == 429 and provider["name"] == "gemini":
            hint = (
                "\n         ^ free-tier quota is 0 for this key's project. A key made"
                "\n           in an existing Cloud project often has no free tier —"
                "\n           create one at aistudio.google.com/apikey choosing"
                "\n           'Create API key in new project' (keys start 'AIza')"
            )
        return False, f'model="{model}"  HTTP {resp.status_code} — {body}{hint}'

    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        return False, f'model="{model}"  unexpected response shape: {exc}'

    return True, f'model="{model}"  {elapsed:.2f}s  reply="{text[:40]}"'


def check_embeddings() -> tuple[bool, str]:
    key = env("GEMINI_API_KEY")
    if not key:
        return False, "GEMINI_API_KEY unset — embeddings unavailable"

    base = env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    model = env("EMBEDDING_MODEL", "gemini-embedding-001")
    expected_dims = int(env("EMBEDDING_DIMENSIONS", "1536"))

    try:
        resp = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            json={"model": model, "input": ["ramen in Calgary"], "dimensions": expected_dims},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f'model="{model}"  transport error: {type(exc).__name__}'

    if resp.status_code != 200:
        body = resp.text[:200].replace("\n", " ")
        hint = ""
        if "model name format" in body:
            hint = (
                "\n         ^ the model name was empty or malformed. Leave"
                " EMBEDDING_MODEL unset to use the default."
            )
        elif resp.status_code == 429:
            hint = "\n         ^ same free-tier quota problem as the Gemini chat check above"
        return False, f'model="{model}"  HTTP {resp.status_code} — {body}{hint}'

    dims = len(resp.json()["data"][0]["embedding"])
    if dims != expected_dims:
        return False, (
            f'model="{model}" returned {dims} dims but EMBEDDING_DIMENSIONS={expected_dims}. '
            f"Set EMBEDDING_DIMENSIONS={dims}, or recreate the Upstash index at {dims} dims."
        )
    return True, f'model="{model}"  dims={dims}'


def check_upstash() -> tuple[bool, str, int | None]:
    url = env("UPSTASH_VECTOR_REST_URL")
    token = env("UPSTASH_VECTOR_REST_TOKEN")
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if embeddings do not work (use from M3 onward).",
    )
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
        flag = "PASS" if ok else ("SKIP" if "no key" in detail or "no model" in detail else "FAIL")
        print(f"[{flag}] chat/{provider['name']:<11} key={_mask(env(provider['key_env']))}  {detail}")
        chat_ok += int(ok)
        if flag == "FAIL" and provider["name"] == "openrouter":
            openrouter_failed = True

    if openrouter_failed:
        free_models = list_free_openrouter_models()
        if free_models:
            print("\n       Free on OpenRouter right now — set OPENROUTER_MODEL to one of")
            print("       these (repository Variables; no code change needed):")
            for slug in free_models:
                print(f"         {slug}")

    embeddings_ok = True
    if not args.chat_only:
        print()
        embeddings_ok, detail = check_embeddings()
        print(f"[{'PASS' if embeddings_ok else 'FAIL'}] embeddings      {detail}")

        up_ok, up_detail, index_dim = check_upstash()
        flag = "PASS" if up_ok else ("SKIP" if "not configured" in up_detail else "FAIL")
        print(f"[{flag}] upstash-vector  {up_detail}")

        expected = int(env("EMBEDDING_DIMENSIONS", "1536"))
        if index_dim is not None and index_dim != expected:
            print(
                f"\n       MISMATCH: Upstash index is {index_dim} dims but "
                f"EMBEDDING_DIMENSIONS={expected}."
                f"\n       gemini-embedding-001 supports 768 / 1536 / 3072, so the usual"
                f"\n       fix is EMBEDDING_DIMENSIONS={index_dim} — keep the index."
            )
            embeddings_ok = False

    print("-" * 72)
    print(f"chat providers passing: {chat_ok}/{len(CHAT_PROVIDERS)}")

    if chat_ok == 0:
        print("\nRESULT: FAIL — no chat provider is usable. The agent cannot run.")
        return 1

    if not embeddings_ok:
        if args.strict:
            print("\nRESULT: FAIL — embeddings are required from M3 (--strict).")
            return 1
        print(
            "\nRESULT: PASS (with warning) — M0 gate cleared: chat works, so the"
            "\n        agent pipeline can be built and deployed."
            "\n        WARNING: embeddings are broken. Tier 0 RAG needs them, so this"
            "\n        must be resolved before M3. Not a blocker for M1/M2."
        )
        return 0

    print("\nRESULT: PASS — M0 provider gate cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
