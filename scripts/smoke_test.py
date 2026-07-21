#!/usr/bin/env python3
"""M0 gate: verify every free provider actually answers with the configured model.

Runs in GitHub Actions using repository Secrets — API keys never leave the repo
settings and are never printed. Exit code 0 only if at least one chat provider
AND the embedding endpoint work.

Usage:
    python scripts/smoke_test.py           # chat + embeddings
    python scripts/smoke_test.py --chat-only
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
        "base": os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "model_env": "GROQ_MODEL",
        "model_default": "llama-3.3-70b-versatile",
        "required": False,
    },
    {
        "name": "gemini",
        "key_env": "GEMINI_API_KEY",
        "base": os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        "model_env": "GEMINI_MODEL",
        "model_default": "gemini-2.0-flash",
        "required": False,
    },
    {
        "name": "openrouter",
        "key_env": "OPENROUTER_API_KEY",
        "base": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model_env": "OPENROUTER_MODEL",
        "model_default": "meta-llama/llama-3.3-70b-instruct:free",
        "required": False,
    },
]


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    return f"{value[:4]}…{value[-2:]}" if len(value) > 8 else "(set)"


def check_chat(provider: dict) -> tuple[bool, str]:
    key = os.environ.get(provider["key_env"], "").strip()
    if not key:
        return False, "no key configured — skipped"

    model = os.environ.get(provider["model_env"], "").strip() or provider["model_default"]
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
        return False, f"transport error: {type(exc).__name__}"

    elapsed = time.time() - started
    if resp.status_code != 200:
        body = resp.text[:180].replace("\n", " ")
        return False, f"HTTP {resp.status_code} — {body}"

    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"unexpected response shape: {exc}"

    return True, f'model="{model}"  {elapsed:.2f}s  reply="{text[:40]}"'


def check_embeddings() -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False, "GEMINI_API_KEY unset — embeddings unavailable"

    base = os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
    expected_dims = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

    try:
        resp = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            json={"model": model, "input": ["ramen in Calgary"]},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f"transport error: {type(exc).__name__}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} — {resp.text[:180]}"

    vector = resp.json()["data"][0]["embedding"]
    dims = len(vector)
    if dims != expected_dims:
        return False, (
            f'model="{model}" returned {dims} dims but EMBEDDING_DIMENSIONS={expected_dims}. '
            f"Recreate the Upstash index with {dims} dims, or set EMBEDDING_DIMENSIONS={dims}."
        )
    return True, f'model="{model}"  dims={dims} (matches Upstash index)'


def check_upstash() -> tuple[bool, str]:
    url = os.environ.get("UPSTASH_VECTOR_REST_URL", "").strip()
    token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "").strip()
    if not (url and token):
        return False, "URL/token not configured — skipped"

    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/info",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f"transport error: {type(exc).__name__}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} — {resp.text[:180]}"

    info = resp.json().get("result", {})
    return True, f"dimension={info.get('dimension')}  vectors={info.get('vectorCount')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-only", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("M0 SMOKE TEST — free-tier provider verification")
    print("=" * 68)

    chat_ok = 0
    for provider in CHAT_PROVIDERS:
        ok, detail = check_chat(provider)
        flag = "PASS" if ok else ("SKIP" if "no key" in detail else "FAIL")
        print(f"[{flag}] chat/{provider['name']:<11} key={_mask(os.environ.get(provider['key_env'], ''))}  {detail}")
        chat_ok += int(ok)

    emb_ok = True
    if not args.chat_only:
        ok, detail = check_embeddings()
        emb_ok = ok
        print(f"[{'PASS' if ok else 'FAIL'}] embeddings      {detail}")

        ok_up, detail_up = check_upstash()
        flag = "PASS" if ok_up else ("SKIP" if "not configured" in detail_up else "FAIL")
        print(f"[{flag}] upstash-vector  {detail_up}")

    print("-" * 68)
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
