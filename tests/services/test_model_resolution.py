"""Tests for provider-qualified LLM model resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from atagia.services.model_resolution import (
    COMPONENTS_BY_ID,
    DEFAULT_FINITE_DECISION_MODEL,
    EXPLICIT_FINITE_DECISION_COMPONENT_IDS,
    FINITE_DECISION_COMPONENT_IDS,
    OPENROUTER_LUNA_MODEL,
    ModelResolutionError,
    OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL,
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
    llm_finite_decisions_enabled: bool = False
    llm_finite_decision_model: str | None = None
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
    typesafe_api_key: str | None = None


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
    assert resolve_component_model(settings, "chat") == OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL

    forced = ResolutionSettings(
        llm_forced_global_model="openrouter/deepseek/deepseek-v4-flash"
    )
    assert (
        resolve_component_model(forced, "extractor")
        == "openrouter/deepseek/deepseek-v4-flash"
    )


def test_finite_decisions_remain_on_generic_routes_by_default() -> None:
    settings = ResolutionSettings(
        llm_component_models={"consequence_detector": "openai/gpt-5-mini"},
    )

    assert resolve_component_model(settings, "applicability_relevance") == (
        OPENROUTER_FLASH_LITE_MODEL
    )
    assert resolve_component_model(settings, "context_staleness") == (
        OPENROUTER_FLASH_LITE_MODEL
    )
    for component_id in (
        "consequence_gate",
        "consequence_sentiment",
        "consequence_link",
    ):
        resolved = resolve_component(settings, component_id)
        assert resolved.model_spec == "openai/gpt-5-mini"
        assert resolved.provenance == "consequence_detector.component override"


def test_finite_decision_switch_routes_supported_components_to_jev() -> None:
    settings = ResolutionSettings(llm_finite_decisions_enabled=True)

    for component_id in FINITE_DECISION_COMPONENT_IDS:
        resolved = resolve_component(settings, component_id)
        assert resolved.model_spec == DEFAULT_FINITE_DECISION_MODEL
        assert resolved.provenance == "finite_decision.typesafe_default"

    assert resolve_component_model(settings, "consequence_detector") == (
        OPENROUTER_LUNA_MODEL
    )
    assert resolve_component_model(settings, "applicability_scorer") == (
        OPENROUTER_FLASH_LITE_MODEL
    )


def test_ordinary_finite_decision_model_does_not_require_typesafe() -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="local/gpu_a/decision-model",
    )

    for component_id in FINITE_DECISION_COMPONENT_IDS:
        assert resolve_component_model(settings, component_id) == (
            "local/gpu_a/decision-model"
        )
    assert required_provider_slugs(settings) == {"anthropic", "local", "openrouter"}


def test_context_staleness_component_override_can_keep_conventional_model() -> None:
    native = ResolutionSettings(llm_finite_decisions_enabled=True)
    conventional = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_component_models={"context_staleness": "openai/decision-model"},
    )

    assert "context_staleness" in FINITE_DECISION_COMPONENT_IDS
    assert resolve_component(native, "context_staleness").provenance == (
        "finite_decision.typesafe_default"
    )
    assert resolve_component_model(native, "context_staleness") == DEFAULT_FINITE_DECISION_MODEL
    assert resolve_component(conventional, "context_staleness").provenance == (
        "component override"
    )
    assert resolve_component_model(conventional, "context_staleness") == "openai/decision-model"


def test_context_staleness_native_override_requires_typesafe_key() -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="local/gpu_a/decision-model",
        llm_component_models={"context_staleness": DEFAULT_FINITE_DECISION_MODEL},
        anthropic_api_key="test-key",
        openrouter_api_key="test-key",
    )

    with pytest.raises(ModelResolutionError, match="ATAGIA_TYPESAFE_API_KEY"):
        validate_required_provider_keys(settings)

    settings.typesafe_api_key = "test-key"
    validate_required_provider_keys(settings)


def test_finite_decision_precedence_is_forced_component_decision_category() -> None:
    settings = ResolutionSettings(
        llm_retrieval_model="google/category-model",
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="openai/decision-model",
        llm_component_models={
            "need_detector_exact": "anthropic/component-model",
        },
    )

    assert resolve_component_model(settings, "need_detector_exact") == (
        "anthropic/component-model"
    )
    assert resolve_component_model(settings, "need_detector_memory") == (
        "openai/decision-model"
    )
    assert resolve_component_model(settings, "context_staleness") == (
        "openai/decision-model"
    )
    assert resolve_component_model(settings, "coverage_expander") == (
        "google/category-model"
    )

    forced = ResolutionSettings(
        llm_forced_global_model="openrouter/example/forced-model",
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="openai/decision-model",
    )
    assert resolve_component_model(forced, "need_detector_memory") == (
        "openrouter/example/forced-model"
    )
    assert resolve_component_model(forced, "context_staleness") == (
        "openrouter/example/forced-model"
    )


def test_disabled_finite_decisions_reject_typesafe_component_override() -> None:
    settings = ResolutionSettings(
        llm_component_models={
            "need_detector_memory": DEFAULT_FINITE_DECISION_MODEL,
        },
    )

    with pytest.raises(ModelResolutionError, match="FINITE_DECISIONS_ENABLED=true"):
        resolve_component(settings, "need_detector_memory")


def test_disabled_finite_decisions_reject_decision_model() -> None:
    settings = ResolutionSettings(
        llm_finite_decision_model="openai/decision-model",
    )

    with pytest.raises(ModelResolutionError, match="FINITE_DECISIONS_ENABLED=true"):
        resolve_component(settings, "need_detector_memory")


def test_finite_decision_model_rejects_typesafe_alias() -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model=DEFAULT_FINITE_DECISION_MODEL,
    )

    with pytest.raises(ModelResolutionError, match="ordinary LLM only"):
        resolve_component(settings, "need_detector_memory")


@pytest.mark.parametrize(
    ("field_name", "error_fragment"),
    [
        ("llm_forced_global_model", "FORCED_GLOBAL"),
        ("llm_ingest_model", "LLM_INGEST_MODEL"),
        ("llm_retrieval_model", "LLM_RETRIEVAL_MODEL"),
    ],
)
def test_typesafe_cannot_be_selected_globally_or_by_category(
    field_name: str,
    error_fragment: str,
) -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        **{field_name: DEFAULT_FINITE_DECISION_MODEL},
    )

    with pytest.raises(ModelResolutionError, match=error_fragment):
        resolve_component(settings, "need_detector_memory")


def test_required_provider_keys_include_mixed_defaults_and_embeddings() -> None:
    settings = ResolutionSettings(
        embedding_backend="sqlite_vec",
        embedding_model="openai/text-embedding-3-small",
    )

    assert required_provider_slugs(settings) == {
        "anthropic",
        "openai",
        "openrouter",
    }

    with pytest.raises(ModelResolutionError, match="ATAGIA_ANTHROPIC_API_KEY"):
        validate_required_provider_keys(settings)


def test_resolve_component_uses_openrouter_luna6_for_extractor() -> None:
    resolved = resolve_component(ResolutionSettings(), "extractor")

    assert resolved.parsed.canonical_model == "openrouter/openai/gpt-6-luna"
    assert resolved.parsed.provider_slug == "openrouter"
    assert resolved.parsed.request_model == "openai/gpt-6-luna"
    assert resolved.provenance == "default"


def test_extraction_watchdog_defaults_to_resolved_extractor_model() -> None:
    settings = ResolutionSettings(
        llm_component_models={"extractor": "openai/gpt-5-mini"},
    )

    resolved = resolve_component(settings, "extraction_watchdog")

    assert resolved.model_spec == "openai/gpt-5-mini"
    assert resolved.parsed.provider_slug == "openai"
    assert resolved.provenance == "extractor.component override"


def test_extraction_evidence_defaults_to_luna6_without_joining_finite_defaults() -> None:
    settings = ResolutionSettings(
        llm_component_models={"extractor": "openai/gpt-5-mini"},
        llm_finite_decisions_enabled=True,
    )
    resolved = resolve_component(settings, "extraction_evidence")
    assert resolved.model_spec == "openrouter/openai/gpt-6-luna"
    assert resolved.provenance == "default"
    assert resolve_component_model(settings, "extractor") == "openai/gpt-5-mini"
    assert "extraction_evidence" not in FINITE_DECISION_COMPONENT_IDS

    overridden = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_component_models={
            "extractor": "openai/gpt-5-mini",
            "extraction_evidence": "typesafe/jev-latest",
        },
    )
    assert resolve_component_model(overridden, "extraction_evidence") == (
        "typesafe/jev-latest"
    )
    assert component_id_for_llm_purpose(
        "memory_extraction_evidence_support_card"
    ) == "extraction_evidence_support"
    assert component_id_for_llm_purpose(
        "memory_extraction_preserve_verbatim_card"
    ) == "extraction_preserve_verbatim"
    assert component_id_for_llm_purpose(
        "memory_extraction_candidate_language_card"
    ) == "extractor"
    assert component_id_for_llm_purpose(
        "memory_extraction_source_reference_card"
    ) == "extraction_evidence"
    assert component_id_for_llm_purpose(
        "memory_extraction_source_reference_selector"
    ) == "extraction_evidence"

    startup = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_component_models={"extraction_evidence": "typesafe/jev-latest"},
        anthropic_api_key="test-key",
        openrouter_api_key="test-key",
        typesafe_api_key="test-key",
    )
    validate_required_provider_keys(startup)


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


def test_chat_defaults_to_deepseek_v4_flash_0731_for_dev_cost() -> None:
    resolved = resolve_component(ResolutionSettings(), "chat")

    assert resolved.parsed.canonical_model == OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL


def test_default_model_scope_is_explicit() -> None:
    luna_ingest_component_ids = {
        "text_chunker",
        "compactor",
        "belief_reviser",
        "contract_projection",
        "graph_projection",
        "consequence_builder",
        "consequence_detector",
        "consequence_gate",
        "consequence_link",
        "consequence_sentiment",
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

    for component_id in luna_ingest_component_ids:
        assert COMPONENTS_BY_ID[component_id].default_model == OPENROUTER_LUNA_MODEL

    for component_id in ("extractor", "extraction_evidence", "topic_working_set"):
        assert COMPONENTS_BY_ID[component_id].default_model == (
            "openrouter/openai/gpt-6-luna"
        )

    for component_id in flashlite_retrieval_component_ids:
        assert COMPONENTS_BY_ID[component_id].default_model == OPENROUTER_FLASH_LITE_MODEL


def test_answer_postcondition_defaults_to_chat_model_family() -> None:
    resolved = resolve_component(ResolutionSettings(), "answer_postcondition")

    assert resolved.category == "chat"
    assert resolved.model_spec == OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL


def test_coverage_expansion_purpose_maps_to_component() -> None:
    assert component_id_for_llm_purpose("coverage_expansion") == "coverage_expander"


def test_claim_key_batch_purpose_keeps_the_intent_classifier_contract() -> None:
    from atagia.services.llm_temperature import purpose_temperature

    single = "intent_classifier_claim_key_equivalence"
    batch = "intent_classifier_claim_key_equivalence_batch"
    assert component_id_for_llm_purpose(batch) == component_id_for_llm_purpose(single)
    assert component_id_for_llm_purpose(batch) == "intent_classifier"
    assert purpose_temperature(batch) == purpose_temperature(single)


def test_consequence_finite_purposes_have_independent_components() -> None:
    assert component_id_for_llm_purpose("consequence_gate_card") == "consequence_gate"
    assert component_id_for_llm_purpose("consequence_sentiment_card") == (
        "consequence_sentiment"
    )
    assert component_id_for_llm_purpose("consequence_link_card") == "consequence_link"
    for generative_purpose in (
        "consequence_action_card",
        "consequence_outcome_card",
        "consequence_language_card",
    ):
        assert component_id_for_llm_purpose(generative_purpose) == (
            "consequence_detector"
        )


def test_coverage_members_card_purpose_is_wired_like_other_extraction_cards() -> None:
    """Guard against the silent-misroute class (cf. commit 227c2e3).

    A new extraction card mints a new LLM purpose that must be registered in
    every purpose map, exactly like the other extraction cards: component
    resolution (extractor), semantic temperature, and the extraction-grade
    partial-stream retry set.
    """

    from atagia.services.llm_client import _PARTIAL_STREAM_RETRY_PURPOSES
    from atagia.services.llm_temperature import PURPOSE_TEMPERATURES, purpose_temperature

    for purpose, component in (
        ("memory_extraction_coverage_members_card", "extractor"),
        ("memory_extraction_coverage_member_identity_card", "extraction_member_identity"),
        ("memory_extraction_belief_key_card", "extractor"),
        ("memory_extraction_belief_value_card", "extractor"),
    ):
        assert component_id_for_llm_purpose(purpose) == component
        assert purpose_temperature(purpose) == PURPOSE_TEMPERATURES["memory_extraction_candidate_card"]
        assert purpose in _PARTIAL_STREAM_RETRY_PURPOSES


def test_need_detection_card_purposes_map_to_card_components() -> None:
    assert component_id_for_llm_purpose("need_detection_needs_card") == (
        "need_detector_needs"
    )
    for purpose, component in (
        ("need_detection_query_language_card", "need_detector_query_language"),
        ("need_detection_answer_language_card", "need_detector_answer_language"),
    ):
        assert component_id_for_llm_purpose(purpose) == component
    assert component_id_for_llm_purpose("need_detection_memory_card") == (
        "need_detector_memory"
    )


def test_unpromoted_subcards_keep_their_parent_models() -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_component_models={
            "extractor": "openai/extraction-parent",
            "topic_working_set": "openai/topic-parent",
            "need_detector_language": "openai/language-parent",
        },
    )
    for component in EXPLICIT_FINITE_DECISION_COMPONENT_IDS:
        resolved = resolve_component(settings, component)
        assert resolved.parsed.provider_slug == "openai"
        assert component not in FINITE_DECISION_COMPONENT_IDS
    assert resolve_component_model(settings, "extraction_kind") == "openai/extraction-parent"
    assert resolve_component_model(settings, "topic_title_decision") == DEFAULT_FINITE_DECISION_MODEL
    assert resolve_component_model(settings, "need_detector_query_language") == "openai/language-parent"

    defaults = ResolutionSettings(llm_finite_decisions_enabled=True)
    assert resolve_component_model(defaults, "extractor") == "openrouter/openai/gpt-6-luna"
    assert resolve_component_model(defaults, "extraction_kind") == "openrouter/openai/gpt-6-luna"
    assert resolve_component_model(defaults, "topic_title_decision") == DEFAULT_FINITE_DECISION_MODEL
    # Both language cards already follow the established finite language route.
    assert resolve_component_model(defaults, "need_detector_query_language") == DEFAULT_FINITE_DECISION_MODEL


def test_new_subcard_overrides_are_independent_and_explicitly_gated() -> None:
    settings = ResolutionSettings(
        llm_finite_decisions_enabled=True,
        llm_component_models={
            "extraction_kind": "typesafe/jev-1.13.0",
            "extraction_confidence": "typesafe/jev-1.13.0",
            "topic_summary_decision": "typesafe/jev-1.13.0",
        },
    )
    assert resolve_component_model(settings, "extraction_kind") == "typesafe/jev-1.13.0"
    assert resolve_component_model(settings, "extraction_confidence") == "typesafe/jev-1.13.0"
    assert resolve_component_model(settings, "extraction_scope") == "openrouter/openai/gpt-6-luna"
    assert resolve_component_model(settings, "extractor") == "openrouter/openai/gpt-6-luna"
    assert resolve_component_model(settings, "topic_summary_decision") == "typesafe/jev-1.13.0"
    assert resolve_component_model(settings, "topic_title_decision") == DEFAULT_FINITE_DECISION_MODEL

    settings.llm_finite_decisions_enabled = False
    with pytest.raises(ModelResolutionError, match="requires ATAGIA_LLM_FINITE_DECISIONS_ENABLED"):
        resolve_component(settings, "extraction_kind")


def test_new_routes_preserve_forced_and_intimacy_resolution() -> None:
    settings = ResolutionSettings(
        llm_forced_global_model="openai/forced",
        llm_component_models={"extraction_kind": "openai/explicit"},
        llm_intimacy_component_models={
            "extractor": "openai/intimate-extractor",
            "topic_working_set": "openai/intimate-topic",
        },
    )
    assert resolve_component_model(settings, "extraction_kind") == "openai/forced"
    assert resolve_intimacy_component(settings, "extraction_kind").model_spec == "openai/intimate-extractor"
    assert resolve_intimacy_component(settings, "topic_title_decision").model_spec == "openai/intimate-topic"


@pytest.mark.parametrize("enabled", [False, True])
def test_approved_family_routes_keep_generation_on_llms(enabled: bool) -> None:
    settings = ResolutionSettings(llm_finite_decisions_enabled=enabled)
    expected = {
        "memory_extraction_temporal_type_card": "openrouter/openai/gpt-6-luna",
        "intent_classifier_claim_key_equivalence_batch": OPENROUTER_LUNA_MODEL,
        "need_detection_query_language_card": OPENROUTER_FLASH_LITE_MODEL,
        "need_detection_answer_language_card": OPENROUTER_FLASH_LITE_MODEL,
        **{
            f"topic_working_set_{field}_decision_card": "openrouter/openai/gpt-6-luna"
            for field in ("title", "summary", "goal", "questions", "decisions")
        },
    }
    for purpose, ordinary_model in expected.items():
        component = component_id_for_llm_purpose(purpose)
        assert resolve_component_model(settings, component) == (
            DEFAULT_FINITE_DECISION_MODEL if enabled else ordinary_model
        )
    for purpose in (
        "memory_extraction_kind_card", "memory_extraction_scope_card",
        "memory_extraction_confidence_card", "memory_extraction_evidence_support_card",
        "memory_extraction_source_reference_card", "memory_extraction_belief_key_card",
        "memory_extraction_belief_value_card", "memory_extraction_temporal_interval_card",
        "memory_extraction_coverage_members_card", "memory_extraction_coverage_member_identity_card",
    ):
        assert resolve_component_model(settings, component_id_for_llm_purpose(purpose)) == (
            "openrouter/openai/gpt-6-luna"
        )
    assert resolve_component_model(settings, "topic_working_set") == "openrouter/openai/gpt-6-luna"
    settings.llm_component_models["extraction_temporal_type"] = "openai/alternative"
    settings.llm_component_models["topic_summary_decision"] = "openai/alternative"
    assert resolve_component_model(settings, "extraction_temporal_type") == "openai/alternative"
    assert resolve_component_model(settings, "topic_summary_decision") == "openai/alternative"


def test_new_purposes_have_component_and_temperature_contracts() -> None:
    from atagia.services.llm_temperature import purpose_temperature

    expected = {
        "memory_extraction_kind_card": "extraction_kind",
        "memory_extraction_scope_card": "extraction_scope",
        "memory_extraction_confidence_card": "extraction_confidence",
        "memory_extraction_evidence_support_card": "extraction_evidence_support",
        "memory_extraction_preserve_verbatim_card": "extraction_preserve_verbatim",
        "memory_extraction_temporal_type_card": "extraction_temporal_type",
        "memory_extraction_coverage_member_identity_card": "extraction_member_identity",
        "topic_working_set_title_decision_card": "topic_title_decision",
        "topic_working_set_summary_decision_card": "topic_summary_decision",
        "topic_working_set_goal_decision_card": "topic_goal_decision",
        "topic_working_set_questions_decision_card": "topic_questions_decision",
        "topic_working_set_decisions_decision_card": "topic_decisions_decision",
        "need_detection_query_language_card": "need_detector_query_language",
        "need_detection_answer_language_card": "need_detector_answer_language",
    }
    for purpose, component in expected.items():
        assert component_id_for_llm_purpose(purpose) == component
        assert purpose_temperature(purpose) is not None


def test_score_component_override_loads_from_the_existing_env_contract() -> None:
    from atagia.core.config import Settings

    settings = Settings.from_env({
        "ATAGIA_LLM_FINITE_DECISIONS_ENABLED": "true",
        "ATAGIA_LLM_MODEL__EXTRACTION_CONFIDENCE": "typesafe/jev-1.13.0",
    })
    assert settings.llm_component_models == {
        "extraction_confidence": "typesafe/jev-1.13.0"
    }
    assert resolve_component_model(settings, "extraction_confidence") == "typesafe/jev-1.13.0"
    assert resolve_component_model(settings, "extractor") == "openrouter/openai/gpt-6-luna"
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


@pytest.mark.parametrize("finite_decisions", [False, True])
def test_date_resolution_uses_dedicated_luna_low_component(finite_decisions):
    settings = ResolutionSettings(llm_finite_decisions_enabled=finite_decisions)
    assert component_id_for_llm_purpose("memory_date_resolution") == "date_resolution"
    assert resolve_component_model(settings, "date_resolution") == (
        "openrouter/openai/gpt-6-luna,low"
    )
    assert resolve_component_model(settings, "extractor") == "openrouter/openai/gpt-6-luna"
    assert resolve_component_model(settings, "applicability_scorer") == OPENROUTER_FLASH_LITE_MODEL
    settings.llm_ingest_model = "openai/ingest-model"
    assert resolve_component_model(settings, "date_resolution") == "openai/ingest-model"
    settings.llm_component_models["date_resolution"] = "openai/date-model,low"
    assert resolve_component_model(settings, "date_resolution") == "openai/date-model,low"
    settings.llm_forced_global_model = "openai/forced-model"
    assert resolve_component_model(settings, "date_resolution") == "openai/forced-model"
    settings.llm_intimacy_ingest_model = "openai/intimacy-model"
    assert resolve_intimacy_component(settings, "date_resolution").model_spec == "openai/intimacy-model"
    settings.llm_intimacy_component_models["date_resolution"] = "openai/intimacy-date-model"
    assert resolve_intimacy_component(settings, "date_resolution").model_spec == (
        "openai/intimacy-date-model"
    )
