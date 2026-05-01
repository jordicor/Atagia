"""Tests for environment-backed settings defaults."""

from __future__ import annotations

import pytest

from atagia.core.config import MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS, Settings


def test_workers_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_WORKERS_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.workers_enabled is False


def test_workers_can_be_enabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.workers_enabled is True


def test_intimacy_model_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_INTIMACY_INGEST_MODEL", "openrouter/z-ai/glm-4.6")
    monkeypatch.setenv(
        "ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL",
        "openrouter/x-ai/grok-4.1-fast",
    )
    monkeypatch.setenv(
        "ATAGIA_LLM_INTIMACY_MODEL__EXTRACTOR",
        "google/gemini-3.1-flash-lite-preview",
    )
    monkeypatch.setenv("ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.llm_intimacy_ingest_model == "openrouter/z-ai/glm-4.6"
    assert settings.llm_intimacy_retrieval_model == "openrouter/x-ai/grok-4.1-fast"
    assert settings.llm_intimacy_component_models == {
        "extractor": "google/gemini-3.1-flash-lite-preview"
    }
    assert settings.llm_intimacy_proactive_routing_enabled is True


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


def test_recent_transcript_budget_defaults_to_policy_with_floor(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", raising=False)

    settings = Settings.from_env()

    assert settings.recent_transcript_budget_tokens is None
    assert settings.effective_recent_transcript_budget_tokens(4000) == 4000
    assert (
        settings.effective_recent_transcript_budget_tokens(512)
        == MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS
    )


def test_recent_transcript_budget_can_be_overridden_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", "30000")

    settings = Settings.from_env()

    assert settings.recent_transcript_budget_tokens == 30000
    assert settings.effective_recent_transcript_budget_tokens(4000) == 30000


def test_recent_transcript_budget_override_is_floored(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", "512")

    settings = Settings.from_env()

    assert (
        settings.effective_recent_transcript_budget_tokens(4000)
        == MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS
    )


def test_recent_transcript_budget_respects_hard_cap(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", "30000")

    settings = Settings.from_env()

    assert settings.effective_recent_transcript_budget_tokens(
        4000,
        hard_cap_tokens=2000,
    ) == 2000


def test_recent_transcript_budget_rejects_non_positive_override(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_benchmark_disable_raw_recent_transcript_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", raising=False)

    settings = Settings.from_env()

    assert settings.benchmark_disable_raw_recent_transcript is False


def test_benchmark_disable_raw_recent_transcript_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")

    settings = Settings.from_env()

    assert settings.benchmark_disable_raw_recent_transcript is True


def test_extraction_watchdog_settings_use_defaults(monkeypatch) -> None:
    for name in (
        "ATAGIA_EXTRACTION_WATCHDOG_ENABLED",
        "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER",
        "ATAGIA_EXTRACTION_WATCHDOG_MIN_ELAPSED_SECONDS",
        "ATAGIA_EXTRACTION_WATCHDOG_MIN_OUTPUT_TOKENS",
        "ATAGIA_EXTRACTION_WATCHDOG_CHECK_INTERVAL_TOKENS",
        "ATAGIA_EXTRACTION_WATCHDOG_MAX_CHECKS",
        "ATAGIA_EXTRACTION_WATCHDOG_LLM_TIMEOUT_SECONDS",
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS",
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.extraction_watchdog_enabled is True
    assert settings.extraction_watchdog_allow_different_provider is False
    assert settings.extraction_watchdog_min_elapsed_seconds == pytest.approx(8.0)
    assert settings.extraction_watchdog_min_output_tokens == 2048
    assert settings.extraction_watchdog_check_interval_tokens == 1024
    assert settings.extraction_watchdog_max_checks == 2
    assert settings.extraction_watchdog_llm_timeout_seconds == pytest.approx(8.0)
    assert settings.extraction_watchdog_bounded_retry_max_items == 8
    assert settings.extraction_watchdog_bounded_retry_max_output_tokens == 4096


def test_extraction_watchdog_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER", "true")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_MIN_ELAPSED_SECONDS", "1.5")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_MIN_OUTPUT_TOKENS", "77")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_CHECK_INTERVAL_TOKENS", "33")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_MAX_CHECKS", "4")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_LLM_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "3")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "2048")

    settings = Settings.from_env()

    assert settings.extraction_watchdog_enabled is False
    assert settings.extraction_watchdog_allow_different_provider is True
    assert settings.extraction_watchdog_min_elapsed_seconds == pytest.approx(1.5)
    assert settings.extraction_watchdog_min_output_tokens == 77
    assert settings.extraction_watchdog_check_interval_tokens == 33
    assert settings.extraction_watchdog_max_checks == 4
    assert settings.extraction_watchdog_llm_timeout_seconds == pytest.approx(2.5)
    assert settings.extraction_watchdog_bounded_retry_max_items == 3
    assert settings.extraction_watchdog_bounded_retry_max_output_tokens == 2048


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("ATAGIA_EXTRACTION_WATCHDOG_MIN_ELAPSED_SECONDS", "-1"),
        ("ATAGIA_EXTRACTION_WATCHDOG_MIN_OUTPUT_TOKENS", "0"),
        ("ATAGIA_EXTRACTION_WATCHDOG_CHECK_INTERVAL_TOKENS", "0"),
        ("ATAGIA_EXTRACTION_WATCHDOG_MAX_CHECKS", "-1"),
        ("ATAGIA_EXTRACTION_WATCHDOG_LLM_TIMEOUT_SECONDS", "0"),
        ("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "0"),
        ("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "512"),
    ],
)
def test_extraction_watchdog_settings_reject_invalid_values(monkeypatch, env_name, value) -> None:
    monkeypatch.setenv(env_name, value)

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

    settings = Settings.from_env()

    assert settings.opf_privacy_filter_enabled is True
    assert settings.opf_primary_url == "http://primary.example"
    assert settings.opf_fallback_url == "http://fallback.example"
    assert settings.opf_timeout_seconds == pytest.approx(1.25)
    assert settings.privacy_validation_gate_enabled is True
    assert settings.privacy_validation_gate_timeout_seconds == pytest.approx(3.5)
    assert settings.privacy_validation_gate_max_source_chars == 1234
    assert settings.privacy_validation_gate_max_summaries_gated_per_job == 7


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
            openai_api_key="test-openai-key",
            openrouter_api_key=None,
            openrouter_site_url="http://localhost",
            openrouter_app_name="Atagia",
            llm_chat_model="reply-test-model",
            service_mode=False,
            service_api_key=None,
            admin_api_key=None,
            workers_enabled=False,
            debug=False,
            operational_allowed_profiles=("normal", " "),
        )


def test_artifact_blob_storage_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_ARTIFACT_BLOB_STORAGE_KIND", raising=False)
    monkeypatch.delenv("ATAGIA_ARTIFACT_BLOB_STORAGE_PATH", raising=False)

    settings = Settings.from_env()

    assert settings.artifact_blob_storage_kind == "sqlite_blob"
    assert settings.artifact_blob_storage_path == "./data/artifact_blobs"


def test_artifact_blob_storage_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_ARTIFACT_BLOB_STORAGE_KIND", "local_file")
    monkeypatch.setenv("ATAGIA_ARTIFACT_BLOB_STORAGE_PATH", "/tmp/atagia-artifacts")

    settings = Settings.from_env()

    assert settings.artifact_blob_storage_kind == "local_file"
    assert settings.artifact_blob_storage_path == "/tmp/atagia-artifacts"


def test_artifact_blob_storage_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        Settings(
            sqlite_path=":memory:",
            migrations_path="./migrations",
            manifests_path="./manifests",
            storage_backend="inprocess",
            redis_url="redis://localhost:6379/0",
            openai_api_key="test-openai-key",
            openrouter_api_key=None,
            openrouter_site_url="http://localhost",
            openrouter_app_name="Atagia",
            llm_chat_model="reply-test-model",
            service_mode=False,
            service_api_key=None,
            admin_api_key=None,
            workers_enabled=False,
            debug=False,
            artifact_blob_storage_kind="remote",
        )


def test_ephemeral_scoring_hours_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_EPHEMERAL_SCORING_HOURS", "12")

    settings = Settings.from_env()

    assert settings.ephemeral_scoring_hours == 12
