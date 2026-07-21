"""Regression tests for env handling.

The first smoke-test run failed because an env var that was *set but empty*
silently overrode a default. GitHub Actions expands an unset repository
Variable (`${{ vars.FOO }}`) to an empty string rather than omitting it, so
`os.environ.get("FOO", default)` returns "" — not the default. These tests
pin the corrected behaviour.
"""
from src import config


def test_env_treats_empty_string_as_absent(monkeypatch):
    monkeypatch.setenv("TFA_TEST_VAR", "")
    assert config._env("TFA_TEST_VAR", "fallback") == "fallback"


def test_env_treats_whitespace_only_as_absent(monkeypatch):
    monkeypatch.setenv("TFA_TEST_VAR", "   ")
    assert config._env("TFA_TEST_VAR", "fallback") == "fallback"


def test_env_uses_real_value_when_set(monkeypatch):
    monkeypatch.setenv("TFA_TEST_VAR", "actual")
    assert config._env("TFA_TEST_VAR", "fallback") == "actual"


def test_env_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("TFA_TEST_VAR", "  spaced  ")
    assert config._env("TFA_TEST_VAR", "fallback") == "spaced"


def test_env_falls_back_when_var_absent(monkeypatch):
    monkeypatch.delenv("TFA_TEST_VAR", raising=False)
    assert config._env("TFA_TEST_VAR", "fallback") == "fallback"


def test_blank_model_var_does_not_disable_groq(monkeypatch):
    """Deleting GROQ_MODEL must fall back to the default, not break the chain."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_MODEL", "")
    chain = config.build_provider_chain()
    groq = next(p for p in chain if p.name == "groq")
    assert groq.model == "llama-3.3-70b-versatile"


def test_openrouter_skipped_when_no_model_configured(monkeypatch):
    """OpenRouter has no safe default slug, so a blank model means 'skip me'."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    assert "openrouter" not in [p.name for p in config.build_provider_chain()]


def test_openrouter_enabled_when_model_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "some-vendor/some-model:free")
    assert "openrouter" in [p.name for p in config.build_provider_chain()]


def test_provider_without_key_is_excluded(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "some-vendor/some-model:free")
    assert "openrouter" not in [p.name for p in config.build_provider_chain()]
