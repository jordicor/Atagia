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


def test_privacy_gate_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_OPF_PRIVACY_FILTER_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_OPF_PRIMARY_URL", raising=False)
    monkeypatch.delenv("ATAGIA_OPF_FALLBACK_URL", raising=False)
    monkeypatch.delenv("ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.opf_privacy_filter_enabled is False
    assert settings.opf_primary_url == "http://127.0.0.1:8008"
    assert settings.opf_fallback_url == "http://127.0.0.1:8008"
    assert settings.privacy_validation_gate_enabled is False


def test_privacy_gate_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_OPF_PRIVACY_FILTER_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_OPF_PRIMARY_URL", "http://primary.example")
    monkeypatch.setenv("ATAGIA_OPF_FALLBACK_URL", "http://fallback.example")
    monkeypatch.setenv("ATAGIA_OPF_TIMEOUT_SECONDS", "1.25")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SOURCE_CHARS", "1234")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB", "7")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("ATAGIA_PRIVACY_VALIDATION_GATE_REFINER_MODEL", "refiner-model")

    settings = Settings.from_env()

    assert settings.opf_privacy_filter_enabled is True
    assert settings.opf_primary_url == "http://primary.example"
    assert settings.opf_fallback_url == "http://fallback.example"
    assert settings.opf_timeout_seconds == pytest.approx(1.25)
    assert settings.privacy_validation_gate_enabled is True
    assert settings.privacy_validation_gate_timeout_seconds == pytest.approx(3.5)
    assert settings.privacy_validation_gate_max_source_chars == 1234
    assert settings.privacy_validation_gate_max_summaries_gated_per_job == 7
    assert settings.privacy_validation_gate_judge_model == "judge-model"
    assert settings.privacy_validation_gate_refiner_model == "refiner-model"


def test_privacy_gate_settings_reject_invalid_timeouts(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_OPF_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_operational_profile_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_OPERATIONAL_PROFILES_PATH", raising=False)
    monkeypatch.delenv("ATAGIA_OPERATIONAL_HIGH_RISK_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_OPERATIONAL_ALLOWED_PROFILES", raising=False)

    settings = Settings.from_env()

    assert settings.operational_profiles_path == "./operational_profiles"
    assert settings.operational_high_risk_enabled is False
    assert settings.operational_allowed_profiles == ("normal", "low_power", "offline")


def test_operational_profile_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILES_PATH", "/tmp/profiles")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_HIGH_RISK_ENABLED", "true")
    monkeypatch.setenv(
        "ATAGIA_OPERATIONAL_ALLOWED_PROFILES",
        "normal,offline,emergency",
    )

    settings = Settings.from_env()

    assert settings.operational_profiles_path == "/tmp/profiles"
    assert settings.operational_high_risk_enabled is True
    assert settings.operational_allowed_profiles == ("normal", "offline", "emergency")


def test_operational_allowed_profiles_reject_blank_entries() -> None:
    with pytest.raises(ValueError):
        Settings(
            sqlite_path=":memory:",
            migrations_path="./migrations",
            manifests_path="./manifests",
            storage_backend="inprocess",
            redis_url="redis://localhost:6379/0",
            llm_provider="openai",
            llm_api_key=None,
            openai_api_key="test-openai-key",
            openrouter_api_key=None,
            llm_base_url=None,
            openrouter_site_url="http://localhost",
            openrouter_app_name="Atagia",
            llm_extraction_model="extract-test-model",
            llm_scoring_model="score-test-model",
            llm_classifier_model="classify-test-model",
            llm_chat_model="reply-test-model",
            service_mode=False,
            service_api_key=None,
            admin_api_key=None,
            workers_enabled=False,
            debug=False,
            operational_allowed_profiles=("normal", " "),
        )


def test_ephemeral_scoring_hours_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_EPHEMERAL_SCORING_HOURS", "12")

    settings = Settings.from_env()

    assert settings.ephemeral_scoring_hours == 12
