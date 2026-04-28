"""Tests for provider-qualified LLM model resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from atagia.services.model_resolution import (
    COMPONENTS_BY_ID,
    ModelResolutionError,
    OPENROUTER_FLASH_LITE_MODEL,
    parse_embedding_model_spec,
    parse_model_spec,
    provider_qualified_model,
    required_provider_slugs,
    resolve_component,
    resolve_component_model,
    validate_required_provider_keys,
)


@dataclass(slots=True)
class ResolutionSettings:
    llm_forced_global_model: str | None = None
    llm_ingest_model: str | None = None
    llm_retrieval_model: str | None = None
    llm_chat_model: str | None = None
    llm_component_models: dict[str, str] = field(default_factory=dict)
    embedding_backend: str = "none"
    embedding_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    openrouter_api_key: str | None = None


def test_parse_model_spec_strips_public_provider_prefix() -> None:
    parsed = parse_model_spec("google/gemini-3.1-flash-lite-preview,medium")

    assert parsed.provider_slug == "google"
    assert parsed.provider_name == "gemini"
    assert parsed.request_model == "gemini-3.1-flash-lite-preview"
    assert parsed.canonical_spec == "google/gemini-3.1-flash-lite-preview,medium"
    assert parsed.thinking_level == "medium"


def test_parse_model_spec_preserves_openrouter_vendor_segment() -> None:
    parsed = parse_model_spec("openrouter/deepseek/deepseek-v4-flash,high")

    assert parsed.provider_name == "openrouter"
    assert parsed.request_model == "deepseek/deepseek-v4-flash"
    assert parsed.thinking_level == "high"


def test_parse_embedding_model_rejects_thinking_level() -> None:
    with pytest.raises(ModelResolutionError, match="Thinking levels"):
        parse_embedding_model_spec("openai/text-embedding-3-small,high")


def test_resolution_precedence_forced_component_category_default() -> None:
    settings = ResolutionSettings(
        llm_ingest_model="anthropic/claude-haiku-4-5",
        llm_retrieval_model="google/gemini-3-flash-preview",
        llm_component_models={"extractor": "openai/gpt-5-mini"},
    )

    assert resolve_component_model(settings, "extractor") == "openai/gpt-5-mini"
    assert resolve_component_model(settings, "belief_reviser") == "anthropic/claude-haiku-4-5"
    assert resolve_component_model(settings, "need_detector") == "google/gemini-3-flash-preview"
    assert resolve_component_model(settings, "chat") == "anthropic/claude-sonnet-4-6"

    forced = ResolutionSettings(
        llm_forced_global_model="openrouter/deepseek/deepseek-v4-flash"
    )
    assert (
        resolve_component_model(forced, "extractor")
        == "openrouter/deepseek/deepseek-v4-flash"
    )


def test_required_provider_keys_include_mixed_defaults_and_embeddings() -> None:
    settings = ResolutionSettings(
        embedding_backend="sqlite_vec",
        embedding_model="openai/text-embedding-3-small",
    )

    assert required_provider_slugs(settings) == {"anthropic", "openrouter", "openai"}

    with pytest.raises(ModelResolutionError, match="ATAGIA_ANTHROPIC_API_KEY"):
        validate_required_provider_keys(settings)


def test_resolve_component_uses_openrouter_flashlite_for_extractor() -> None:
    resolved = resolve_component(ResolutionSettings(), "extractor")

    assert resolved.parsed.canonical_model == OPENROUTER_FLASH_LITE_MODEL
    assert resolved.parsed.provider_slug == "openrouter"
    assert resolved.parsed.request_model == "google/gemini-3.1-flash-lite-preview"
    assert resolved.provenance == "default"


def test_sensitive_components_stay_on_sonnet_by_default() -> None:
    sensitive_component_ids = {
        "summary_privacy_judge",
        "summary_privacy_refiner",
        "consent_confirmation",
        "export_anonymizer",
        "chat",
    }

    for component_id in sensitive_component_ids:
        resolved = resolve_component(ResolutionSettings(), component_id)
        assert resolved.parsed.canonical_model == "anthropic/claude-sonnet-4-6"


def test_default_flashlite_migration_scope_is_explicit() -> None:
    migrated_component_ids = {
        "extractor",
        "text_chunker",
        "compactor",
        "belief_reviser",
        "contract_projection",
        "consequence_builder",
        "consequence_detector",
        "topic_working_set",
        "intent_classifier",
        "need_detector",
        "applicability_scorer",
        "context_staleness",
        "metrics_computer",
    }

    for component_id in migrated_component_ids:
        assert COMPONENTS_BY_ID[component_id].default_model == OPENROUTER_FLASH_LITE_MODEL


def test_forced_global_model_limits_required_completion_provider() -> None:
    settings = ResolutionSettings(
        llm_forced_global_model="openai/gpt-5-mini",
        openai_api_key="openai-key",
    )

    assert required_provider_slugs(settings) == {"openai"}
    validate_required_provider_keys(settings)


def test_provider_qualified_model_maps_legacy_cli_aliases() -> None:
    assert provider_qualified_model("gemini", "gemini-3-flash-preview") == (
        "google/gemini-3-flash-preview"
    )
    assert provider_qualified_model("openrouter", "deepseek/deepseek-v4-flash") == (
        "openrouter/deepseek/deepseek-v4-flash"
    )
    assert provider_qualified_model("openai", "openai/gpt-5-mini,high") == (
        "openai/gpt-5-mini,high"
    )
