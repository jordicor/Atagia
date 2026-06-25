"""Tests for environment-backed settings defaults."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from atagia.core.env import env_bool
from atagia.core.config import (
    MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS,
    Settings,
    default_resource_path,
)


def test_default_resource_paths_exist(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_MIGRATIONS_PATH", raising=False)
    monkeypatch.delenv("ATAGIA_MANIFESTS_PATH", raising=False)
    monkeypatch.delenv("ATAGIA_OPERATIONAL_PROFILES_PATH", raising=False)

    settings = Settings.from_env()

    assert settings.migrations_dir().exists()
    assert settings.manifests_dir().exists()
    assert settings.operational_profiles_dir().exists()
    assert default_resource_path("migrations")


def test_relative_resource_env_paths_work_from_external_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATAGIA_MIGRATIONS_PATH", "./migrations")
    monkeypatch.setenv("ATAGIA_MANIFESTS_PATH", "./manifests")
    monkeypatch.setenv("ATAGIA_OPERATIONAL_PROFILES_PATH", "./operational_profiles")

    settings = Settings.from_env()

    assert settings.migrations_dir().exists()
    assert settings.manifests_dir().exists()
    assert settings.operational_profiles_dir().exists()
    assert settings.migrations_dir() != tmp_path / "migrations"


def test_packaged_resources_are_present() -> None:
    packaged = resources.files("atagia.resources")

    assert packaged.joinpath("migrations", "0001_initial_schema.sql").is_file()
    assert packaged.joinpath("manifests", "personal_assistant.json").is_file()
    assert packaged.joinpath("operational_profiles", "normal.json").is_file()


def test_openai_proxy_and_cors_settings_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_PROXY_MODEL_ID", "atagia-st")
    monkeypatch.setenv("ATAGIA_PROXY_UPSTREAM_MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("ATAGIA_PROXY_DEFAULT_MODE", "companion")
    monkeypatch.setenv(
        "ATAGIA_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000, http://localhost:3000",
    )

    settings = Settings.from_env()

    assert settings.openai_proxy_model_id == "atagia-st"
    assert settings.openai_proxy_upstream_model == "openai/gpt-4.1-mini"
    assert settings.openai_proxy_default_mode == "companion"
    assert settings.cors_allowed_origins == (
        "http://127.0.0.1:8000",
        "http://localhost:3000",
    )


def test_workers_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_WORKERS_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.workers_enabled is False


def test_workers_can_be_enabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.workers_enabled is True


def test_default_language_code_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATAGIA_DEFAULT_LANGUAGE_CODE", raising=False)
    assert Settings.from_env().default_language_code == "en"

    monkeypatch.setenv("ATAGIA_DEFAULT_LANGUAGE_CODE", " ES ")
    assert Settings.from_env().default_language_code == "es"


def test_default_language_code_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_DEFAULT_LANGUAGE_CODE", "jp")

    with pytest.raises(ValueError, match="ATAGIA_DEFAULT_LANGUAGE_CODE"):
        Settings.from_env()


def test_consequence_detector_card_concurrency_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATAGIA_CONSEQUENCE_DETECTOR_CARD_CONCURRENCY", raising=False)
    assert Settings.from_env().consequence_detector_card_concurrency == 2

    monkeypatch.setenv("ATAGIA_CONSEQUENCE_DETECTOR_CARD_CONCURRENCY", "1")
    assert Settings.from_env().consequence_detector_card_concurrency == 1


def test_consequence_detector_card_concurrency_rejects_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_CONSEQUENCE_DETECTOR_CARD_CONCURRENCY", "0")

    with pytest.raises(ValueError, match="consequence_detector_card_concurrency"):
        Settings.from_env()


def test_compactor_summary_card_concurrency_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATAGIA_COMPACTOR_SUMMARY_CARD_CONCURRENCY", raising=False)
    assert Settings.from_env().compactor_summary_card_concurrency == 4

    monkeypatch.setenv("ATAGIA_COMPACTOR_SUMMARY_CARD_CONCURRENCY", "1")
    assert Settings.from_env().compactor_summary_card_concurrency == 1


def test_compactor_summary_card_concurrency_rejects_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_COMPACTOR_SUMMARY_CARD_CONCURRENCY", "0")

    with pytest.raises(ValueError, match="compactor_summary_card_concurrency"):
        Settings.from_env()


def test_episode_synthesis_max_episodes_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES", raising=False)
    assert Settings.from_env().episode_synthesis_max_episodes == 24

    monkeypatch.setenv("ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES", "7")
    assert Settings.from_env().episode_synthesis_max_episodes == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("OFF", False),
    ],
)
def test_env_bool_accepts_documented_tokens(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("ATAGIA_TEST_BOOL", value)

    assert env_bool("ATAGIA_TEST_BOOL", default=not expected) is expected


@pytest.mark.parametrize("default", [True, False])
def test_env_bool_uses_default_for_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch,
    default: bool,
) -> None:
    monkeypatch.delenv("ATAGIA_TEST_BOOL", raising=False)
    assert env_bool("ATAGIA_TEST_BOOL", default=default) is default

    monkeypatch.setenv("ATAGIA_TEST_BOOL", "   ")
    assert env_bool("ATAGIA_TEST_BOOL", default=default) is default


def test_env_bool_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAGIA_TEST_BOOL", "enabled")

    with pytest.raises(ValueError, match="ATAGIA_TEST_BOOL"):
        env_bool("ATAGIA_TEST_BOOL", default=False)


def test_settings_reject_invalid_boolean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "enabled")

    with pytest.raises(ValueError, match="ATAGIA_WORKERS_ENABLED"):
        Settings.from_env()


def test_openai_embedding_base_url_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_OPENAI_BASE_URL", "http://completion.test/v1")
    monkeypatch.setenv(
        "ATAGIA_OPENAI_EMBEDDING_BASE_URL",
        "http://embedding.test/v1",
    )

    settings = Settings.from_env()

    assert settings.openai_base_url == "http://completion.test/v1"
    assert settings.openai_embedding_base_url == "http://embedding.test/v1"


def test_minimax_provider_settings_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_MINIMAX_API_KEY", "minimax-key")
    monkeypatch.setenv("ATAGIA_MINIMAX_BASE_URL", "https://mini.example/v1")

    settings = Settings.from_env()

    assert settings.minimax_api_key == "minimax-key"
    assert settings.minimax_base_url == "https://mini.example/v1"


def test_kimi_provider_settings_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_KIMI_API_KEY", "kimi-key")
    monkeypatch.setenv("ATAGIA_KIMI_BASE_URL", "https://kimi.example/v1")

    settings = Settings.from_env()

    assert settings.kimi_api_key == "kimi-key"
    assert settings.kimi_base_url == "https://kimi.example/v1"


def test_embedding_retrieval_settings_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec")
    monkeypatch.setenv("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "23")
    monkeypatch.setenv("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "5")

    settings = Settings.from_env()

    assert settings.embedding_backend == "sqlite_vec"
    assert settings.embedding_vector_limit_cap == 23
    assert settings.embedding_search_overfetch_multiplier == 5


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "0"),
        ("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "0"),
    ],
)
def test_embedding_retrieval_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
) -> None:
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError):
        Settings.from_env()


def test_initial_context_package_rollout_settings_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_READ_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_REFRESH_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS", "512")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS", "256")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS", "1024")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_BLOCK_MAX_TOKENS", "128")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_MAX_ITEMS", "3")
    monkeypatch.setenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_MAX_OUTPUT_TOKENS", "640")

    settings = Settings.from_env()

    assert settings.initial_context_package_read_enabled is False
    assert settings.initial_context_package_refresh_enabled is False
    assert settings.initial_context_package_curation_enabled is False
    assert settings.initial_context_package_prompt_max_tokens == 512
    assert settings.initial_context_package_profile_max_tokens == 256
    assert settings.initial_context_package_total_max_tokens == 1024
    assert settings.initial_context_package_curated_block_max_tokens == 128
    assert settings.initial_context_package_curated_max_items == 3
    assert settings.initial_context_package_curation_max_output_tokens == 640


@pytest.mark.parametrize(
    "env_name",
    [
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_BLOCK_MAX_TOKENS",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_MAX_ITEMS",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_MAX_OUTPUT_TOKENS",
    ],
)
def test_initial_context_package_rollout_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    monkeypatch.setenv(env_name, "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_worker_circuit_breaker_settings_use_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED", raising=False)
    monkeypatch.delenv("ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", raising=False)
    monkeypatch.delenv("ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO", raising=False)

    settings = Settings.from_env()

    assert settings.worker_circuit_breaker_enabled is True
    assert settings.worker_circuit_breaker_failure_threshold == 20
    assert settings.worker_circuit_breaker_window_seconds == 180
    assert settings.worker_circuit_breaker_min_failure_ratio == 0.8


def test_worker_circuit_breaker_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "7")
    monkeypatch.setenv("ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS", "45")
    monkeypatch.setenv("ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO", "0.6")

    settings = Settings.from_env()

    assert settings.worker_circuit_breaker_enabled is False
    assert settings.worker_circuit_breaker_failure_threshold == 7
    assert settings.worker_circuit_breaker_window_seconds == 45
    assert settings.worker_circuit_breaker_min_failure_ratio == 0.6


def test_worker_circuit_breaker_rejects_invalid_settings(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_llm_run_guard_settings_use_defaults(monkeypatch) -> None:
    for name in (
        "ATAGIA_LLM_RUN_GUARD_ENABLED",
        "ATAGIA_LLM_RUN_GUARD_MODE",
        "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_CALLS",
        "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
        "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
        "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
        "ATAGIA_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_CALLS",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.llm_run_guard_enabled is True
    assert settings.llm_run_guard_mode == "enforce"
    assert settings.llm_run_guard_max_total_calls is None
    assert settings.llm_run_guard_max_total_failed_calls == 80
    assert settings.llm_run_guard_max_failed_call_ratio == 0.50
    assert settings.llm_run_guard_max_failed_ratio_per_purpose == 0.50
    assert settings.llm_run_guard_max_consecutive_failures_per_purpose == 8
    assert settings.bulk_ingest_llm_run_guard_max_total_calls == 10000
    assert settings.bulk_ingest_llm_run_guard_max_total_failed_calls == 40
    assert settings.bulk_ingest_llm_run_guard_max_failed_call_ratio == 0.20
    assert settings.bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose == 0.30
    assert settings.bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose == 4
    assert settings.bulk_ingest_llm_run_guard_max_wall_time_seconds == 14400.0


def test_llm_run_guard_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_MODE", "audit")
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_CALLS", "123")
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS", "7")
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO", "0.25")
    monkeypatch.setenv("ATAGIA_LLM_RUN_GUARD_MAX_REPORTED_COST_USD", "3.50")
    monkeypatch.setenv("ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_CALLS", "456")
    monkeypatch.setenv(
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
        "2",
    )
    monkeypatch.setenv(
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS",
        "900",
    )

    settings = Settings.from_env()

    assert settings.llm_run_guard_enabled is False
    assert settings.llm_run_guard_mode == "audit"
    assert settings.llm_run_guard_max_total_calls == 123
    assert settings.llm_run_guard_max_total_failed_calls == 7
    assert settings.llm_run_guard_max_failed_call_ratio == 0.25
    assert settings.llm_run_guard_max_reported_cost_usd == 3.50
    assert settings.bulk_ingest_llm_run_guard_max_total_calls == 456
    assert (
        settings.bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose
        == 2
    )
    assert settings.bulk_ingest_llm_run_guard_max_wall_time_seconds == 900.0


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("ATAGIA_LLM_RUN_GUARD_MODE", "panic"),
        ("ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS", "0"),
        ("ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO", "1.1"),
        ("ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS", "0"),
    ],
)
def test_llm_run_guard_rejects_invalid_settings(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
) -> None:
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError):
        Settings.from_env()


def test_intimacy_model_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_INTIMACY_INGEST_MODEL", "openrouter/z-ai/glm-4.6")
    monkeypatch.setenv(
        "ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL",
        "openrouter/x-ai/grok-4.1-fast",
    )
    monkeypatch.setenv(
        "ATAGIA_LLM_INTIMACY_MODEL__EXTRACTOR",
        "google/gemini-3.1-flash-lite",
    )
    monkeypatch.setenv("ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.llm_intimacy_ingest_model == "openrouter/z-ai/glm-4.6"
    assert settings.llm_intimacy_retrieval_model == "openrouter/x-ai/grok-4.1-fast"
    assert settings.llm_intimacy_component_models == {
        "extractor": "google/gemini-3.1-flash-lite"
    }
    assert settings.llm_intimacy_proactive_routing_enabled is True


def test_structured_output_repair_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED", "true")
    monkeypatch.setenv(
        "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL",
        "anthropic/claude-opus-4-7",
    )

    settings = Settings.from_env()

    assert settings.llm_structured_output_retry_attempts == 2
    assert settings.llm_structured_output_rescue_enabled is True
    assert settings.llm_structured_output_rescue_model == "anthropic/claude-opus-4-7"


def test_structured_output_rescue_uses_default_model_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED", "true")
    monkeypatch.delenv("ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.llm_structured_output_rescue_enabled is True
    assert settings.llm_structured_output_rescue_model == "anthropic/claude-opus-4-7"


def test_llm_technical_recovery_settings_use_defaults(monkeypatch) -> None:
    for name in (
        "ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED",
        "ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS",
        "ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED",
        "ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS",
        "ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS",
        "ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS",
        "ATAGIA_LLM_RUNAWAY_MAX_CHECKS",
        "ATAGIA_LLM_RUNAWAY_HARD_ABORT_MIN_OUTPUT_TOKENS",
        "ATAGIA_LLM_RUNAWAY_MIN_REPEAT_COUNT",
        "ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS",
        "ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO",
        "ATAGIA_LLM_RUNAWAY_HARD_OUTPUT_INPUT_RATIO",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.llm_technical_recovery_enabled is True
    assert settings.llm_output_limit_retry_attempts == 1
    assert settings.llm_runaway_watchdog_enabled is True
    assert settings.llm_runaway_min_elapsed_seconds == pytest.approx(8.0)
    assert settings.llm_runaway_min_output_tokens == 2048
    assert settings.llm_runaway_check_interval_tokens == 1024
    assert settings.llm_runaway_max_checks == 2
    assert settings.llm_runaway_hard_abort_min_output_tokens == 4096
    assert settings.llm_runaway_min_repeat_count == 3
    assert settings.llm_runaway_min_repeat_ratio_tokens == pytest.approx(0.12)
    assert settings.llm_runaway_output_input_ratio == pytest.approx(12.0)
    assert settings.llm_runaway_hard_output_input_ratio == pytest.approx(8.0)


def test_llm_technical_recovery_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS", "0.5")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS", "99")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS", "44")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MAX_CHECKS", "5")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_HARD_ABORT_MIN_OUTPUT_TOKENS", "777")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_COUNT", "4")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS", "0.2")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO", "9.0")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_HARD_OUTPUT_INPUT_RATIO", "6.0")

    settings = Settings.from_env()

    assert settings.llm_technical_recovery_enabled is False
    assert settings.llm_output_limit_retry_attempts == 2
    assert settings.llm_runaway_watchdog_enabled is False
    assert settings.llm_runaway_min_elapsed_seconds == pytest.approx(0.5)
    assert settings.llm_runaway_min_output_tokens == 99
    assert settings.llm_runaway_check_interval_tokens == 44
    assert settings.llm_runaway_max_checks == 5
    assert settings.llm_runaway_hard_abort_min_output_tokens == 777
    assert settings.llm_runaway_min_repeat_count == 4
    assert settings.llm_runaway_min_repeat_ratio_tokens == pytest.approx(0.2)
    assert settings.llm_runaway_output_input_ratio == pytest.approx(9.0)
    assert settings.llm_runaway_hard_output_input_ratio == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS", "-1"),
        ("ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS", "-0.1"),
        ("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS", "0"),
        ("ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS", "0"),
        ("ATAGIA_LLM_RUNAWAY_MAX_CHECKS", "-1"),
        ("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS", "1.1"),
        ("ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO", "0"),
    ],
)
def test_llm_technical_recovery_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
) -> None:
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError):
        Settings.from_env()


def test_answer_postcondition_guard_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_ANSWER_POSTCONDITION_RETRY_MAX_OUTPUT_TOKENS", "16384")
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE", "proactive")
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE_PROMPT_VARIANT", "template_v1")

    settings = Settings.from_env()

    assert settings.answer_postcondition_guard_enabled is True
    assert settings.answer_postcondition_retry_max_output_tokens == 16384
    assert settings.answer_stance == "proactive"
    assert settings.answer_stance_prompt_variant == "template_v1"


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
    monkeypatch.delenv("ATAGIA_DISABLE_CHUNKING_EXTRACTION", raising=False)
    monkeypatch.delenv("ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS", raising=False)

    settings = Settings.from_env()

    assert settings.disable_chunking_extraction is False
    assert settings.chunking_extraction_threshold_tokens == 2048


def test_chunking_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_DISABLE_CHUNKING_EXTRACTION", "true")
    monkeypatch.setenv("ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS", "321")

    settings = Settings.from_env()

    assert settings.disable_chunking_extraction is True
    assert settings.chunking_extraction_threshold_tokens == 321


def test_chunking_threshold_must_be_positive(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS", "0")

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

    assert (
        settings.effective_recent_transcript_budget_tokens(
            4000,
            hard_cap_tokens=2000,
        )
        == 2000
    )


def test_recent_transcript_budget_rejects_non_positive_override(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_context_envelope_defaults_to_structural_4k_budget(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS", raising=False)
    monkeypatch.delenv("ATAGIA_CONTEXT_ENVELOPE_RATIOS", raising=False)

    settings = Settings.from_env()

    assert settings.context_envelope_budget_tokens == 4096
    assert settings.context_envelope_ratios["retrieved_context"] == 0.67
    assert settings.context_envelope_ratios["recent_transcript"] == 0.20


def test_context_envelope_budget_and_ratios_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS", "4096")
    monkeypatch.setenv(
        "ATAGIA_CONTEXT_ENVELOPE_RATIOS",
        "instructions=1,current_turn=1,retrieved_context=6,recent_transcript=2",
    )

    settings = Settings.from_env()

    assert settings.context_envelope_budget_tokens == 4096
    assert settings.context_envelope_ratios == {
        "instructions": 1.0,
        "current_turn": 1.0,
        "retrieved_context": 6.0,
        "recent_transcript": 2.0,
    }


def test_context_envelope_rejects_invalid_ratio_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_CONTEXT_ENVELOPE_RATIOS", "retrieved_context=-1")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_benchmark_disable_raw_recent_transcript_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", raising=False)

    settings = Settings.from_env()

    assert settings.benchmark_disable_raw_recent_transcript is False


def test_anthropic_request_timeout_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS", "45.5")

    settings = Settings.from_env()

    assert settings.anthropic_request_timeout_seconds == pytest.approx(45.5)


def test_anthropic_request_timeout_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_worker_transient_defer_budget_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT", "5")
    monkeypatch.setenv("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_AGE_SECONDS", "1800.5")

    settings = Settings.from_env()

    assert settings.worker_transient_defer_max_count == 5
    assert settings.worker_transient_defer_max_age_seconds == pytest.approx(1800.5)


def test_worker_transient_defer_budget_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT", "0")

    with pytest.raises(ValueError):
        Settings.from_env()

    monkeypatch.setenv("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT", "1")
    monkeypatch.setenv("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_AGE_SECONDS", "0")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_benchmark_disable_raw_recent_transcript_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT", "true")

    settings = Settings.from_env()

    assert settings.benchmark_disable_raw_recent_transcript is True


def test_extraction_watchdog_settings_use_defaults(monkeypatch) -> None:
    for name in (
        "ATAGIA_EXTRACTION_WATCHDOG_ENABLED",
        "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER",
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS",
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.extraction_watchdog_enabled is True
    assert settings.extraction_watchdog_allow_different_provider is False
    assert settings.extraction_watchdog_bounded_retry_max_items == 8
    assert settings.extraction_watchdog_bounded_retry_max_output_tokens == 8192


def test_extraction_watchdog_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER", "true")
    monkeypatch.setenv("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "3")
    monkeypatch.setenv(
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "16384"
    )

    settings = Settings.from_env()

    assert settings.extraction_watchdog_enabled is False
    assert settings.extraction_watchdog_allow_different_provider is True
    assert settings.extraction_watchdog_bounded_retry_max_items == 3
    assert settings.extraction_watchdog_bounded_retry_max_output_tokens == 16384


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "0"),
        ("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "8191"),
    ],
)
def test_extraction_watchdog_settings_reject_invalid_values(
    monkeypatch, env_name, value
) -> None:
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
    monkeypatch.setenv(
        "ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB", "7"
    )

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

    assert settings.operational_profiles_dir().exists()
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


def test_llm_debug_io_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO", "true")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_DIR", "/tmp/atagia-llm-debug")
    monkeypatch.setenv(
        "ATAGIA_DEBUG_LLM_IO_PURPOSES", "applicability_scoring,need_detection"
    )
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_RAW", "yes")
    monkeypatch.setenv("ATAGIA_DEBUG_LLM_IO_MAX_CHARS", "1234")

    settings = Settings.from_env()

    assert settings.llm_debug_io_enabled is True
    assert settings.llm_debug_io_dir == "/tmp/atagia-llm-debug"
    assert settings.llm_debug_io_purposes == ("applicability_scoring", "need_detection")
    assert settings.llm_debug_io_raw is True
    assert settings.llm_debug_io_max_chars == 1234


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
