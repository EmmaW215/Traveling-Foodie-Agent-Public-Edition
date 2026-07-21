"""Fallback-chain behaviour, exercised with a mock transport (no real API calls)."""
import httpx
import pytest

from src.config import Provider
from src.llm_client import AllProvidersFailedError, LLMClient, LLMResponse


def _provider(name: str) -> Provider:
    return Provider(
        name=name,
        base_url=f"https://{name}.example/v1",
        api_key="test-key",
        model=f"{name}-model",
    )


def _ok_payload(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_falls_through_to_second_provider_on_429(monkeypatch):
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "groq.example":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_ok_payload("hello from gemini"))

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    client = LLMClient(providers=[_provider("groq"), _provider("gemini")])
    result = await client.chat(system="s", user="u")

    assert result.provider == "gemini"
    assert result.text == "hello from gemini"
    assert "groq.example" in calls and "gemini.example" in calls


@pytest.mark.asyncio
async def test_raises_when_every_provider_fails(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    client = LLMClient(providers=[_provider("groq"), _provider("gemini")])
    with pytest.raises(AllProvidersFailedError):
        await client.chat(system="s", user="u")


@pytest.mark.asyncio
async def test_no_providers_configured_raises():
    client = LLMClient(providers=[])
    assert client.configured is False
    with pytest.raises(AllProvidersFailedError):
        await client.chat(system="s", user="u")


def test_as_json_strips_code_fences():
    fenced = LLMResponse(text='```json\n{"slot": "d1_lunch"}\n```', provider="p", model="m")
    assert fenced.as_json() == {"slot": "d1_lunch"}

    plain = LLMResponse(text='{"ok": true}', provider="p", model="m")
    assert plain.as_json() == {"ok": True}
