"""Tests for provider-qualified LLM model resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from atagia.services.model_resolution import (
    COMPONENTS_BY_ID,
    MINIMAX_M3_MODEL,
    ModelResolutionError,
    OPENROUTER_DEEPSEEK_V4_FLASH_MODEL,
    OPENROUTER_FLASH_LITE_MODEL,
    component_id_for_llm_purpose,
    parse_embedding_model_spec,
    parse_model_spec,
    provider_qualified_model,
    required_provider_slugs,
    resolve_component,
    resolve_component_model,
    resolve_intimacy_component,
    resolve_intimacy_fallback_models,
    validate_required_provider_keys,
)


@dataclass(slots=True)
class ResolutionSettings:
    llm_forced_global_model: str | None = None
    llm_ingest_model: str | None = None
    llm_retrieval_model: str | None = None
    llm_chat_model: str | None = None
    llm_component_models: dict[str, str] = field(default_factory=dict)
    llm_intimacy_ingest_model: str | None = None
    llm_intimacy_retrieval_model: str | None = None
    llm_intimacy_component_models: dict[str, str] = field(default_factory=dict)
    llm_structured_output_retry_attempts: int = 1
    llm_structured_output_rescue_enabled: bool = False
    llm_structured_output_rescue_model: str | None = None
    embedding_backend: str = "none"
    embedding_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    kimi_api_key: str | None = None
    minimax_api_key: str | None = None
    openrouter_api_key: str | None = None


def test_parse_model_spec_strips_public_provider_prefix() -> None:
    parsed = parse_model_spec("google/gemini-3.1-flash-lite,medium")

    assert parsed.provider_slug == "google"
    assert parsed.provider_name == "gemini"
    assert parsed.request_model == "gemini-3.1-flash-lite"
    assert parsed.canonical_spec == "google/gemini-3.1-flash-lite,medium"
    assert parsed.thinking_level == "medium"


def test_parse_model_spec_preserves_openrouter_vendor_segment() -> None:
    parsed = parse_model_spec("openrouter/deepseek/deepseek-v4-flash,high")

    assert parsed.provider_name == "openrouter"
    assert parsed.request_model == "deepseek/deepseek-v4-flash"
    assert parsed.thinking_level == "high"


def test_parse_model_spec_accepts_direct_minimax_provider() -> None:
    parsed = parse_model_spec("minimax/MiniMax-M3")

    assert parsed.provider_slug == "minimax"
    assert parsed.provider_name == "minimax"
    assert parsed.request_model == "MiniMax-M3"
    assert parsed.canonical_spec == "minimax/MiniMax-M3"


def test_parse_model_spec_accepts_direct_kimi_provider() -> None:
    parsed = parse_model_spec("kimi/kimi-k2.7-code")

    assert parsed.provider_slug == "kimi"
    assert parsed.provider_name == "kimi"
    assert parsed.request_model == "kimi-k2.7-code"
    assert parsed.canonical_spec == "kimi/kimi-k2.7-code"


def test_openrouter_provider_qualifies_google_vendor_models() -> None:
    assert (
        provider_qualified_model(
            "openrouter",
            "google/gemini-3.1-flash-lite",
        )
        == "openrouter/google/gemini-3.1-flash-lite"
    )


def test_explicit_provider_qualified_model_still_wins_without_openrouter_provider() -> None:
    assert (
        provider_qualified_model(
            "anthropic",
            "google/gemini-3.1-flash-lite",
        )
        == "google/gemini-3.1-flash-lite"
    )


@pytest.mark.parametrize("level", ["low", "xhigh", "max"])
def test_parse_model_spec_accepts_extended_thinking_levels(level: str) -> None:
    parsed = parse_model_spec(f"anthropic/claude-opus-4-7,{level}")

    assert parsed.request_model == "claude-opus-4-7"
    assert parsed.thinking_level == level


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
    assert (
        resolve_component_model(settings, "need_detector_language")
        == "google/gemini-3-flash-preview"
    )
    assert resolve_component_model(settings, "chat") == OPENROUTER_DEEPSEEK_V4_FLASH_MODEL

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

    assert required_provider_slugs(settings) == {
        "anthropic",
        "minimax",
        "openai",
        "openrouter",
    }

    with pytest.raises(ModelResolutionError, match="ATAGIA_ANTHROPIC_API_KEY"):
        validate_required_provider_keys(settings)


def test_resolve_component_uses_direct_minimax_m3_for_extractor() -> None:
    resolved = resolve_component(ResolutionSettings(), "extractor")

    assert resolved.parsed.canonical_model == MINIMAX_M3_MODEL
    assert resolved.parsed.provider_slug == "minimax"
    assert resolved.parsed.request_model == "MiniMax-M3"
    assert resolved.provenance == "default"


def test_extraction_watchdog_defaults_to_resolved_extractor_model() -> None:
    settings = ResolutionSettings(
        llm_component_models={"extractor": "openai/gpt-5-mini"},
    )

    resolved = resolve_component(settings, "extraction_watchdog")

    assert resolved.model_spec == "openai/gpt-5-mini"
    assert resolved.parsed.provider_slug == "openai"
    assert resolved.provenance == "extractor.component override"


def test_extraction_watchdog_supports_component_override() -> None:
    settings = ResolutionSettings(
        llm_component_models={
            "extractor": "openai/gpt-5-mini",
            "extraction_watchdog": "openai/gpt-5-nano",
        },
    )

    resolved = resolve_component(settings, "extraction_watchdog")

    assert resolved.model_spec == "openai/gpt-5-nano"
    assert resolved.provenance == "component override"


def test_intimacy_fallback_resolution_uses_component_then_category() -> None:
    settings = ResolutionSettings(
        llm_intimacy_ingest_model="openrouter/z-ai/glm-4.6",
        llm_intimacy_retrieval_model="openrouter/x-ai/grok-4.1-fast",
        llm_intimacy_component_models={
            "extractor": "google/gemini-3.1-flash-lite"
        },
    )

    extractor = resolve_intimacy_component(settings, "extractor")
    compactor = resolve_intimacy_component(settings, "compactor")
    scorer = resolve_intimacy_component(settings, "applicability_scorer")
    chat = resolve_intimacy_component(settings, "chat")

    assert extractor is not None
    assert extractor.model_spec == "google/gemini-3.1-flash-lite"
    assert extractor.provenance == "intimacy component override"
    assert compactor is not None
    assert compactor.model_spec == "openrouter/z-ai/glm-4.6"
    assert scorer is not None
    assert scorer.model_spec == "openrouter/x-ai/grok-4.1-fast"
    assert chat is None


def test_intimacy_fallback_models_include_only_configured_components() -> None:
    settings = ResolutionSettings(
        llm_intimacy_ingest_model="openrouter/z-ai/glm-4.6",
    )

    fallbacks = resolve_intimacy_fallback_models(settings)

    assert fallbacks["extractor"] == "openrouter/z-ai/glm-4.6"
    assert fallbacks["compactor"] == "openrouter/z-ai/glm-4.6"
    assert "need_detector" not in fallbacks
    assert "chat" not in fallbacks


def test_sensitive_components_stay_on_sonnet_by_default() -> None:
    sensitive_component_ids = {
        "summary_privacy_judge",
        "summary_privacy_refiner",
        "consent_confirmation",
        "export_anonymizer",
    }

    for component_id in sensitive_component_ids:
        resolved = resolve_component(ResolutionSettings(), component_id)
        assert resolved.parsed.canonical_model == "anthropic/claude-sonnet-4-6"


def test_chat_defaults_to_deepseek_v4_flash_for_dev_cost() -> None:
    resolved = resolve_component(ResolutionSettings(), "chat")

    assert resolved.parsed.canonical_model == OPENROUTER_DEEPSEEK_V4_FLASH_MODEL


def test_default_model_scope_is_explicit() -> None:
    minimax_ingest_component_ids = {
        "extractor",
        "text_chunker",
        "compactor",
        "belief_reviser",
        "contract_projection",
        "graph_projection",
        "consequence_builder",
        "consequence_detector",
        "topic_working_set",
        "intent_classifier",
        "extraction_watchdog",
        "initial_context_package_curation",
    }
    flashlite_retrieval_component_ids = {
        "need_detector_needs",
        "need_detector_language",
        "need_detector_memory",
        "need_detector_exact",
        "need_detector_shape",
        "need_detector_facets",
        "need_detector_callback",
        "need_detector_search_words",
        "need_detector_search_words_other_language",
        "coverage_expander",
        "applicability_scorer",
        "context_staleness",
        "metrics_computer",
    }

    for component_id in minimax_ingest_component_ids:
        assert COMPONENTS_BY_ID[component_id].default_model == MINIMAX_M3_MODEL

    for component_id in flashlite_retrieval_component_ids:
        assert COMPONENTS_BY_ID[component_id].default_model == OPENROUTER_FLASH_LITE_MODEL


def test_answer_postcondition_defaults_to_chat_model_family() -> None:
    resolved = resolve_component(ResolutionSettings(), "answer_postcondition")

    assert resolved.category == "chat"
    assert resolved.model_spec == OPENROUTER_DEEPSEEK_V4_FLASH_MODEL


def test_coverage_expansion_purpose_maps_to_component() -> None:
    assert component_id_for_llm_purpose("coverage_expansion") == "coverage_expander"


def test_coverage_members_card_purpose_is_wired_like_other_extraction_cards() -> None:
    """Guard against the silent-misroute class (cf. commit 227c2e3).

    A new extraction card mints a new LLM purpose that must be registered in
    every purpose map, exactly like the other extraction cards: component
    resolution (extractor), semantic temperature, and the extraction-grade
    partial-stream retry set.
    """

    from atagia.services.llm_client import _PARTIAL_STREAM_RETRY_PURPOSES
    from atagia.services.llm_temperature import PURPOSE_TEMPERATURES, purpose_temperature

    purpose = "memory_extraction_coverage_members_card"
    belief_purpose = "memory_extraction_belief_card"

    # Component resolution: maps to the extractor component like its siblings.
    assert component_id_for_llm_purpose(purpose) == "extractor"
    assert component_id_for_llm_purpose(purpose) == component_id_for_llm_purpose(
        belief_purpose
    )

    # Temperature: the same semantic temperature the other extraction cards use.
    assert purpose_temperature(purpose) == PURPOSE_TEMPERATURES[belief_purpose]

    # Partial-stream retry: extraction-grade retry coverage.
    assert purpose in _PARTIAL_STREAM_RETRY_PURPOSES


def test_need_detection_card_purposes_map_to_card_components() -> None:
    assert component_id_for_llm_purpose("need_detection_needs_card") == (
        "need_detector_needs"
    )
    assert component_id_for_llm_purpose("need_detection_language_card") == (
        "need_detector_language"
    )
    assert component_id_for_llm_purpose("need_detection_memory_card") == (
        "need_detector_memory"
    )
    assert component_id_for_llm_purpose("need_detection_exact_card") == (
        "need_detector_exact"
    )
    assert component_id_for_llm_purpose("need_detection_shape_card") == (
        "need_detector_shape"
    )
    assert component_id_for_llm_purpose("need_detection_facets_card") == (
        "need_detector_facets"
    )
    assert component_id_for_llm_purpose("need_detection_callback_card") == (
        "need_detector_callback"
    )
    assert component_id_for_llm_purpose("need_detection_search_words_card") == (
        "need_detector_search_words"
    )
    assert component_id_for_llm_purpose(
        "need_detection_search_words_other_language_card"
    ) == ("need_detector_search_words_other_language")


def test_answer_postcondition_purpose_maps_to_component() -> None:
    assert component_id_for_llm_purpose("answer_postcondition_verification") == (
        "answer_postcondition"
    )
    assert component_id_for_llm_purpose("answer_abstention_legitimacy_verification") == (
        "answer_postcondition"
    )


def test_forced_global_model_limits_required_completion_provider() -> None:
    settings = ResolutionSettings(
        llm_forced_global_model="openai/gpt-5-mini",
        openai_api_key="openai-key",
    )

    assert required_provider_slugs(settings) == {"openai"}
    validate_required_provider_keys(settings)


def test_intimacy_fallback_provider_keys_are_required() -> None:
    settings = ResolutionSettings(
        llm_forced_global_model="openai/gpt-5-mini",
        llm_intimacy_ingest_model="openrouter/z-ai/glm-4.6",
        openai_api_key="openai-key",
    )

    assert required_provider_slugs(settings) == {"openai", "openrouter"}
    with pytest.raises(ModelResolutionError, match="ATAGIA_OPENROUTER_API_KEY"):
        validate_required_provider_keys(settings)


def test_structured_output_rescue_provider_key_is_required() -> None:
    settings = ResolutionSettings(
        llm_forced_global_model="openrouter/deepseek/deepseek-v4-flash",
        llm_structured_output_rescue_enabled=True,
        llm_structured_output_rescue_model="anthropic/claude-opus-4-7",
        openrouter_api_key="openrouter-key",
    )

    assert required_provider_slugs(settings) == {"anthropic", "openrouter"}
    with pytest.raises(ModelResolutionError, match="ATAGIA_ANTHROPIC_API_KEY"):
        validate_required_provider_keys(settings)


def test_direct_minimax_provider_key_is_required_when_used() -> None:
    settings = ResolutionSettings(llm_forced_global_model="minimax/MiniMax-M3")

    assert required_provider_slugs(settings) == {"minimax"}
    with pytest.raises(ModelResolutionError, match="ATAGIA_MINIMAX_API_KEY"):
        validate_required_provider_keys(settings)

    validate_required_provider_keys(
        ResolutionSettings(
            llm_forced_global_model="minimax/MiniMax-M3",
            minimax_api_key="minimax-key",
        )
    )


def test_direct_kimi_provider_key_is_required_when_used() -> None:
    settings = ResolutionSettings(llm_forced_global_model="kimi/kimi-k2.7-code")

    assert required_provider_slugs(settings) == {"kimi"}
    with pytest.raises(ModelResolutionError, match="ATAGIA_KIMI_API_KEY"):
        validate_required_provider_keys(settings)

    validate_required_provider_keys(
        ResolutionSettings(
            llm_forced_global_model="kimi/kimi-k2.7-code",
            kimi_api_key="kimi-key",
        )
    )


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
    assert provider_qualified_model("minimax", "MiniMax-M3") == "minimax/MiniMax-M3"
    assert provider_qualified_model("kimi", "kimi-k2.7-code") == "kimi/kimi-k2.7-code"
