"""Tests for environment-backed settings defaults."""

from __future__ import annotations

import pytest

from atagia.core.config import Settings


def test_workers_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_WORKERS_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.workers_enabled is False


def test_workers_can_be_enabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.workers_enabled is True


def test_context_cache_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_CONTEXT_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", raising=False)
    monkeypatch.delenv("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", raising=False)

    settings = Settings.from_env()

    assert settings.context_cache_enabled is True
    assert settings.context_cache_min_ttl_seconds == 60
    assert settings.context_cache_max_ttl_seconds == 3600


def test_context_cache_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", "30")
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", "900")

    settings = Settings.from_env()

    assert settings.context_cache_enabled is False
    assert settings.context_cache_min_ttl_seconds == 30
    assert settings.context_cache_max_ttl_seconds == 900


def test_context_cache_settings_reject_inverted_ttl_bounds(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", "120")
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", "60")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_chunking_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_CHUNKING_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_CHUNKING_THRESHOLD_TOKENS", raising=False)

    settings = Settings.from_env()

    assert settings.chunking_enabled is False
    assert settings.chunking_threshold_tokens == 2000


def test_chunking_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CHUNKING_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_CHUNKING_THRESHOLD_TOKENS", "321")

    settings = Settings.from_env()

    assert settings.chunking_enabled is False
    assert settings.chunking_threshold_tokens == 321


def test_chunking_threshold_must_be_positive(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CHUNKING_THRESHOLD_TOKENS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_ephemeral_scoring_hours_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_EPHEMERAL_SCORING_HOURS", "12")

    settings = Settings.from_env()

    assert settings.ephemeral_scoring_hours == 12
