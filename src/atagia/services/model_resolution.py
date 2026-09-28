"""LLM model spec parsing and component-level resolution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from atagia.core.env import env_bool_optional


logger = logging.getLogger(__name__)

ALLOWED_THINKING_LEVELS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
PROVIDER_SLUG_TO_NAME = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "kimi": "kimi",
    "minimax": "minimax",
    "openrouter": "openrouter",
    "typesafe": "typesafe",
    "local": "local",
}
PROVIDER_NAME_TO_SLUG = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google",
    "google": "google",
    "kimi": "kimi",
    "minimax": "minimax",
    "openrouter": "openrouter",
    "typesafe": "typesafe",
    "local": "local",
}
SUPPORTED_PROVIDER_SLUGS = tuple(PROVIDER_SLUG_TO_NAME)

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_FLASH_LITE_MODEL = "openrouter/google/gemini-3.1-flash-lite"
# The dedicated -0731 slug guarantees DeepSeek's 2026-07-31 re-post-trained
# revision; the bare deepseek-v4-flash slug routes across OpenRouter providers
# that may still serve the April preview weights.
OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
OPENROUTER_LUNA_MODEL = "openrouter/openai/gpt-5.6-luna"
# Kept as catalog handles for tests/challengers, still fully wired (provider
# adapter + MODEL_PROFILES entries): MiniMax was the ingest default until
# 2026-07-31; the bare V4 Flash slug was the chat default until 2026-08-01.
MINIMAX_M3_MODEL = "minimax/MiniMax-M3"
OPENROUTER_MINIMAX_M3_MODEL = "openrouter/minimax/minimax-m3"
OPENROUTER_DEEPSEEK_V4_FLASH_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL = "anthropic/claude-opus-4-7"
DEFAULT_FINITE_DECISION_MODEL = "typesafe/jev-latest"


class ModelResolutionError(ValueError):
    """Raised when LLM model configuration cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ParsedModelSpec:
    """A parsed provider/model spec."""

    raw_spec: str
    canonical_spec: str
    canonical_model: str
    provider_slug: str
    provider_name: str
    request_model: str
    thinking_level: str | None = None


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Canonical LLM-backed component declaration."""

    component_id: str
    category: str
    default_model: str

    @property
    def env_var(self) -> str:
        return f"ATAGIA_LLM_MODEL__{self.component_id.upper()}"

    @property
    def intimacy_env_var(self) -> str:
        return f"ATAGIA_LLM_INTIMACY_MODEL__{self.component_id.upper()}"

    @property
    def examples_env_var(self) -> str:
        return f"ATAGIA_LLM_EXAMPLES__{self.component_id.upper()}"


@dataclass(frozen=True, slots=True)
class ResolvedComponentModel:
    """Resolved component model with provenance."""

    component_id: str
    category: str
    model_spec: str
    provenance: str
    parsed: ParsedModelSpec


@dataclass(frozen=True, slots=True)
class ResolutionSnapshot:
    """Resolved component and embedding configuration for logging/validation."""

    forced_global_model: str | None
    finite_decisions_enabled: bool
    finite_decision_model: str | None
    category_models: dict[str, str | None]
    intimacy_category_models: dict[str, str | None]
    structured_output_rescue_model: ParsedModelSpec | None
    components: dict[str, ResolvedComponentModel]
    intimacy_components: dict[str, ResolvedComponentModel]
    embedding: ParsedModelSpec


CATEGORY_ENV_VARS = {
    "ingest": "ATAGIA_LLM_INGEST_MODEL",
    "retrieval": "ATAGIA_LLM_RETRIEVAL_MODEL",
    "chat": "ATAGIA_LLM_CHAT_MODEL",
}

INTIMACY_CATEGORY_ENV_VARS = {
    "ingest": "ATAGIA_LLM_INTIMACY_INGEST_MODEL",
    "retrieval": "ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL",
}

COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("extractor", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_evidence", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_kind", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_scope", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_confidence", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_evidence_support", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_preserve_verbatim", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("extraction_temporal_type", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("date_resolution", "ingest", "openrouter/openai/gpt-6-luna,low"),
    ComponentSpec("extraction_member_identity", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("text_chunker", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("compactor", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("summary_privacy_judge", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("summary_privacy_refiner", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("belief_reviser", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("contract_projection", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("graph_projection", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("consequence_builder", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("consequence_detector", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("consequence_gate", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("consequence_sentiment", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("consequence_link", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("topic_working_set", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("topic_title_decision", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("topic_summary_decision", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("topic_goal_decision", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("topic_questions_decision", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("topic_decisions_decision", "ingest", "openrouter/openai/gpt-6-luna"),
    ComponentSpec("consent_confirmation", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("intent_classifier", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("extraction_watchdog", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("initial_context_package_curation", "ingest", OPENROUTER_LUNA_MODEL),
    ComponentSpec("export_anonymizer", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("need_detector_needs", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_language", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_query_language", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_answer_language", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_memory", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_exact", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_shape", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_facets", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_callback", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("need_detector_search_words", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec(
        "need_detector_search_words_other_language",
        "retrieval",
        OPENROUTER_FLASH_LITE_MODEL,
    ),
    ComponentSpec("coverage_expander", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("applicability_scorer", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("applicability_relevance", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("context_staleness", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("metrics_computer", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("answer_postcondition", "chat", OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL),
    ComponentSpec("chat", "chat", OPENROUTER_DEEPSEEK_V4_FLASH_0731_MODEL),
)
COMPONENTS_BY_ID = {spec.component_id: spec for spec in COMPONENT_SPECS}

FINITE_DECISION_COMPONENT_IDS = frozenset(
    {
        "applicability_relevance",
        "context_staleness",
        "consequence_gate",
        "consequence_link",
        "consequence_sentiment",
        "extraction_temporal_type",
        "intent_classifier",
        "need_detector_callback",
        "need_detector_exact",
        "need_detector_facets",
        "need_detector_language",
        "need_detector_memory",
        "need_detector_needs",
        "need_detector_shape",
        "topic_title_decision",
        "topic_summary_decision",
        "topic_goal_decision",
        "topic_questions_decision",
        "topic_decisions_decision",
    }
)

# These cards do not join the global finite-decision selection set. Language
# subcards inherit their existing language route, including its current switch.
EXPLICIT_FINITE_DECISION_COMPONENT_IDS = frozenset(
    {
        "extraction_kind",
        "extraction_scope",
        "extraction_confidence",
        "extraction_evidence_support",
        "extraction_preserve_verbatim",
        "extraction_member_identity",
        "need_detector_query_language",
        "need_detector_answer_language",
    }
)
TYPED_DECISION_COMPONENT_IDS = (
    FINITE_DECISION_COMPONENT_IDS | EXPLICIT_FINITE_DECISION_COMPONENT_IDS | {"extraction_evidence"}
)

INHERITED_COMPONENT_IDS = {
    "extraction_kind": "extractor",
    "extraction_scope": "extractor",
    "extraction_confidence": "extractor",
    "extraction_evidence_support": "extractor",
    "extraction_preserve_verbatim": "extractor",
    "extraction_temporal_type": "extractor",
    "extraction_member_identity": "extractor",
    "topic_title_decision": "topic_working_set",
    "topic_summary_decision": "topic_working_set",
    "topic_goal_decision": "topic_working_set",
    "topic_questions_decision": "topic_working_set",
    "topic_decisions_decision": "topic_working_set",
    "need_detector_query_language": "need_detector_language",
    "need_detector_answer_language": "need_detector_language",
    "applicability_relevance": "applicability_scorer",
    "consequence_gate": "consequence_detector",
    "consequence_link": "consequence_detector",
    "consequence_sentiment": "consequence_detector",
    "extraction_watchdog": "extractor",
}

PURPOSE_TO_COMPONENT_ID = {
    "applicability_relevance_card": "applicability_relevance",
    "applicability_scoring": "applicability_scorer",
    "answer_abstention_legitimacy_verification": "answer_postcondition",
    "answer_evidence_use_verification": "answer_postcondition",
    "answer_postcondition_verification": "answer_postcondition",
    "belief_revision": "belief_reviser",
    "chat_reply": "chat",
    "consequence_detection": "consequence_detector",
    "consequence_action_card": "consequence_detector",
    "consequence_gate_card": "consequence_gate",
    "consequence_language_card": "consequence_detector",
    "consequence_link_card": "consequence_link",
    "consequence_outcome_card": "consequence_detector",
    "consequence_sentiment_card": "consequence_sentiment",
    "consequence_tendency_inference": "consequence_builder",
    "consent_confirmation_intent": "consent_confirmation",
    "context_cache_signal_detection": "context_staleness",
    "episode_synthesis": "compactor",
    "evaluation_contract_compliance": "metrics_computer",
    "export_anonymization_rewrite": "export_anonymizer",
    "export_anonymization_verify": "export_anonymizer",
    "extraction_watchdog": "extraction_watchdog",
    "graph_projection": "graph_projection",
    "initial_context_package_curation": "initial_context_package_curation",
    "intent_classifier_claim_key_equivalence": "intent_classifier",
    "intent_classifier_claim_key_equivalence_batch": "intent_classifier",
    "intent_classifier_explicit": "intent_classifier",
    "memory_extraction": "extractor",
    "memory_extraction_candidate_card": "extractor",
    "memory_extraction_kind_card": "extraction_kind",
    "memory_extraction_scope_card": "extraction_scope",
    "memory_extraction_confidence_card": "extraction_confidence",
    "memory_extraction_evidence_support_card": "extraction_evidence_support",
    "memory_extraction_preserve_verbatim_card": "extraction_preserve_verbatim",
    "memory_extraction_candidate_language_card": "extractor",
    "memory_extraction_source_reference_card": "extraction_evidence",
    "memory_extraction_source_reference_selector": "extraction_evidence",
    "memory_extraction_index_card": "extractor",
    "memory_extraction_belief_key_card": "extractor",
    "memory_extraction_belief_value_card": "extractor",
    "memory_extraction_temporal_type_card": "extraction_temporal_type",
    "memory_date_resolution": "date_resolution",
    "memory_extraction_temporal_interval_card": "extractor",
    "memory_extraction_coverage_members_card": "extractor",
    "memory_extraction_coverage_member_identity_card": "extraction_member_identity",
    "need_detection_needs_card": "need_detector_needs",
    "need_detection_query_language_card": "need_detector_query_language",
    "need_detection_answer_language_card": "need_detector_answer_language",
    "need_detection_memory_card": "need_detector_memory",
    "need_detection_exact_card": "need_detector_exact",
    "need_detection_shape_card": "need_detector_shape",
    "need_detection_facets_card": "need_detector_facets",
    "need_detection_callback_card": "need_detector_callback",
    "need_detection_search_words_card": "need_detector_search_words",
    "need_detection_search_words_other_language_card": "need_detector_search_words_other_language",
    "retrieval_surface_generation_dry_run": "coverage_expander",
    "coverage_expansion": "coverage_expander",
    "summary_chunk_segmentation_ranges_card": "compactor",
    "summary_chunk_segmentation_summaries_card": "compactor",
    "summary_privacy_gate_judge": "summary_privacy_judge",
    "summary_privacy_gate_refine": "summary_privacy_refiner",
    "text_chunking_level1": "text_chunker",
    "thematic_profile_synthesis": "compactor",
    "topic_working_set_update": "topic_working_set",
    "topic_working_set_route_card": "topic_working_set",
    "topic_working_set_title_card": "topic_working_set",
    "topic_working_set_summary_card": "topic_working_set",
    "topic_working_set_goal_card": "topic_working_set",
    "topic_working_set_questions_card": "topic_working_set",
    "topic_working_set_decisions_card": "topic_working_set",
    "topic_working_set_title_decision_card": "topic_title_decision",
    "topic_working_set_summary_decision_card": "topic_summary_decision",
    "topic_working_set_goal_decision_card": "topic_goal_decision",
    "topic_working_set_questions_decision_card": "topic_questions_decision",
    "topic_working_set_decisions_decision_card": "topic_decisions_decision",
    "topic_working_set_boundary_card": "topic_working_set",
    "user_language_profile_observed_card": "extractor",
    "user_language_profile_preference_card": "extractor",
    "user_language_profile_ability_card": "extractor",
    "user_language_profile_norm_card": "extractor",
    "workspace_rollup_synthesis": "compactor",
}


def normalized_model_value(value: str | None) -> str | None:
    """Return a usable model value or None when the layer is unset."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def parse_model_spec(
    value: str,
    *,
    env_name: str | None = None,
    allow_thinking: bool = True,
) -> ParsedModelSpec:
    """Parse a provider/model spec without guessing provider from model id."""
    raw = value.strip()
    if not raw:
        raise ModelResolutionError(_invalid_model_spec_message(value, env_name=env_name))

    model_part, thinking_level = _split_thinking(raw, env_name=env_name)
    if thinking_level is not None and not allow_thinking:
        source = f" in {env_name}" if env_name else ""
        raise ModelResolutionError(
            f"Invalid LLM model spec{source}: {raw!r}. Thinking levels are not supported here."
        )

    if model_part.lower().startswith("local/"):
        local_segments = model_part.split("/", 2)
        if (
            len(local_segments) != 3
            or local_segments[0].lower() != "local"
            or not local_segments[1]
            or not local_segments[2]
            or local_segments[1] != local_segments[1].strip()
            or local_segments[2] != local_segments[2].strip()
        ):
            raise ModelResolutionError(
                _invalid_model_spec_message(raw, env_name=env_name)
            )
        endpoint_id = local_segments[1]
        request_model = f"{endpoint_id}/{local_segments[2]}"
        canonical_model = f"local/{request_model}"
        canonical_spec = (
            f"{canonical_model},{thinking_level}"
            if thinking_level is not None
            else canonical_model
        )
        return ParsedModelSpec(
            raw_spec=raw,
            canonical_spec=canonical_spec,
            canonical_model=canonical_model,
            provider_slug="local",
            provider_name="local",
            request_model=request_model,
            thinking_level=thinking_level,
        )

    segments = [segment.strip() for segment in model_part.split("/") if segment.strip()]
    if not segments:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
    provider_slug = segments[0].lower()
    if provider_slug not in PROVIDER_SLUG_TO_NAME:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
    if provider_slug == "typesafe" and thinking_level is not None:
        raise ModelResolutionError("TypeSafe typed decisions do not support thinking levels")
    if provider_slug == "openrouter":
        if len(segments) != 3:
            raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
    elif len(segments) != 2:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))

    request_model = "/".join(segments[1:])
    canonical_model = f"{provider_slug}/{request_model}"
    canonical_spec = (
        f"{canonical_model},{thinking_level}" if thinking_level is not None else canonical_model
    )
    return ParsedModelSpec(
        raw_spec=raw,
        canonical_spec=canonical_spec,
        canonical_model=canonical_model,
        provider_slug=provider_slug,
        provider_name=PROVIDER_SLUG_TO_NAME[provider_slug],
        request_model=request_model,
        thinking_level=thinking_level,
    )


def parse_embedding_model_spec(value: str | None) -> ParsedModelSpec:
    """Parse the embedding model spec, using the built-in coherent default."""
    model = normalized_model_value(value) or DEFAULT_EMBEDDING_MODEL
    return parse_model_spec(
        model,
        env_name="ATAGIA_EMBEDDING_MODEL",
        allow_thinking=False,
    )


def provider_qualified_model(provider: str | None, model: str | None) -> str | None:
    """Return a provider-qualified model spec from legacy provider/model inputs."""
    normalized_model = normalized_model_value(model)
    if normalized_model is None:
        return None
    model_part = normalized_model.split(",", 1)[0]
    first_segment = model_part.split("/", 1)[0].strip().lower()
    normalized_provider = (provider or "").strip().lower()
    provider_slug = PROVIDER_NAME_TO_SLUG.get(normalized_provider)
    if provider_slug == "openrouter" and first_segment == "google":
        return parse_model_spec(f"openrouter/{normalized_model}").canonical_spec
    if first_segment in PROVIDER_SLUG_TO_NAME:
        return parse_model_spec(normalized_model).canonical_spec
    if provider_slug is None:
        raise ModelResolutionError(
            "A provider-qualified model spec is required when no valid provider "
            f"alias is available: {normalized_model!r}"
        )
    return parse_model_spec(f"{provider_slug}/{normalized_model}").canonical_spec


def component_env_models_from_env(env: dict[str, str]) -> dict[str, str]:
    """Return component override models from an env mapping."""
    values: dict[str, str] = {}
    for spec in COMPONENT_SPECS:
        value = normalized_model_value(env.get(spec.env_var))
        if value is not None:
            values[spec.component_id] = value
    return values


def intimacy_component_env_models_from_env(env: dict[str, str]) -> dict[str, str]:
    """Return intimacy fallback component models from an env mapping."""
    values: dict[str, str] = {}
    for spec in COMPONENT_SPECS:
        value = normalized_model_value(env.get(spec.intimacy_env_var))
        if value is not None:
            values[spec.component_id] = value
    return values


def component_env_examples_from_env(env: dict[str, str]) -> dict[str, bool]:
    """Return per-component prompt-examples overrides from an env mapping.

    A component is present only when its override is set, so an unset component
    falls back to the global ``card_examples_enabled`` default.
    """
    values: dict[str, bool] = {}
    for spec in COMPONENT_SPECS:
        override = env_bool_optional(spec.examples_env_var, env)
        if override is not None:
            values[spec.component_id] = override
    return values


def examples_enabled_for_component(settings: Any, component_id: str) -> bool:
    """Resolve whether a card component should include its few-shot examples.

    Per-component override wins; otherwise the global ``card_examples_enabled``
    default applies. Few-shot demonstrations reliably help small/local models
    but can hurt larger or reasoning models, so deployments can drop them per
    component without maintaining a second prompt set.
    """
    if component_id not in COMPONENTS_BY_ID:
        raise ModelResolutionError(f"Unknown LLM component id: {component_id}")
    overrides = getattr(settings, "llm_component_examples", {}) or {}
    override = overrides.get(component_id)
    if override is not None:
        return bool(override)
    return bool(getattr(settings, "card_examples_enabled", True))


def validate_finite_decision_configuration(settings: Any) -> None:
    """Validate the opt-in finite-decision routing contract.

    TypeSafe is deliberately gated by one explicit switch. Ordinary model
    overrides remain valid while the switch is off, but a dormant TypeSafe
    override is rejected so disabling the switch cannot leave native external
    calls active by surprise.
    """
    enabled = bool(getattr(settings, "llm_finite_decisions_enabled", False))
    decision_model = normalized_model_value(
        getattr(settings, "llm_finite_decision_model", None)
    )
    if decision_model is not None:
        if not enabled:
            raise ModelResolutionError(
                "ATAGIA_LLM_FINITE_DECISION_MODEL requires "
                "ATAGIA_LLM_FINITE_DECISIONS_ENABLED=true"
            )
        parsed = parse_model_spec(
            decision_model,
            env_name="ATAGIA_LLM_FINITE_DECISION_MODEL",
        )
        if parsed.provider_slug == "typesafe":
            raise ModelResolutionError(
                "ATAGIA_LLM_FINITE_DECISION_MODEL accepts an ordinary LLM only; "
                "leave it unset to use TypeSafe Jev"
            )

    forced = normalized_model_value(
        getattr(settings, "llm_forced_global_model", None)
    )
    if _selects_typesafe(forced):
        parse_model_spec(forced or "", env_name="ATAGIA_LLM_FORCED_GLOBAL_MODEL")
        raise ModelResolutionError(
            "TypeSafe cannot be selected through ATAGIA_LLM_FORCED_GLOBAL_MODEL; "
            "it supports only finite-choice components"
        )

    for category, env_var in CATEGORY_ENV_VARS.items():
        category_model = normalized_model_value(_category_model(settings, category))
        if _selects_typesafe(category_model):
            parse_model_spec(category_model or "", env_name=env_var)
            raise ModelResolutionError(
                f"TypeSafe cannot be selected through {env_var}; it supports only "
                "finite-choice components"
            )

    component_overrides = getattr(settings, "llm_component_models", {}) or {}
    for component_id, model in component_overrides.items():
        normalized = normalized_model_value(model)
        if normalized is None:
            continue
        if not _selects_typesafe(normalized):
            continue
        parse_model_spec(
            normalized,
            env_name=f"ATAGIA_LLM_MODEL__{component_id.upper()}",
        )
        if not enabled:
            raise ModelResolutionError(
                f"TypeSafe override for {component_id} requires "
                "ATAGIA_LLM_FINITE_DECISIONS_ENABLED=true"
            )
        if component_id not in TYPED_DECISION_COMPONENT_IDS:
            raise ModelResolutionError(
                "TypeSafe supports only finite-choice components, not "
                f"{component_id}; keep generative components on their existing models"
            )


def _selects_typesafe(value: str | None) -> bool:
    """Return whether a configured model explicitly names TypeSafe."""
    normalized = normalized_model_value(value)
    if normalized is None:
        return False
    model_part = normalized.split(",", 1)[0].strip()
    provider_slug = model_part.split("/", 1)[0].strip().lower()
    return provider_slug == "typesafe"


def _finite_decision_model(settings: Any) -> ParsedModelSpec | None:
    validate_finite_decision_configuration(settings)
    if not bool(getattr(settings, "llm_finite_decisions_enabled", False)):
        return None
    configured = normalized_model_value(
        getattr(settings, "llm_finite_decision_model", None)
    )
    return parse_model_spec(
        configured or DEFAULT_FINITE_DECISION_MODEL,
        env_name=(
            "ATAGIA_LLM_FINITE_DECISION_MODEL"
            if configured is not None
            else "default:finite_decisions"
        ),
    )


def resolve_component(settings: Any, component_id: str) -> ResolvedComponentModel:
    """Resolve a component model from forced/component/decision/category layers."""
    component = COMPONENTS_BY_ID.get(component_id)
    if component is None:
        raise ModelResolutionError(f"Unknown LLM component id: {component_id}")

    decision_model = _finite_decision_model(settings)

    forced = normalized_model_value(getattr(settings, "llm_forced_global_model", None))
    if forced is not None:
        parsed = parse_model_spec(forced, env_name="ATAGIA_LLM_FORCED_GLOBAL_MODEL")
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parsed.canonical_spec,
            provenance="forced_global",
            parsed=parsed,
        )

    component_overrides = getattr(settings, "llm_component_models", {}) or {}
    component_value = normalized_model_value(component_overrides.get(component.component_id))
    if component_value is not None:
        parsed = parse_model_spec(component_value, env_name=component.env_var)
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parsed.canonical_spec,
            provenance="component override",
            parsed=parsed,
        )

    if component.component_id in FINITE_DECISION_COMPONENT_IDS and decision_model is not None:
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=decision_model.canonical_spec,
            provenance=(
                "finite_decision_model"
                if normalized_model_value(
                    getattr(settings, "llm_finite_decision_model", None)
                )
                is not None
                else "finite_decision.typesafe_default"
            ),
            parsed=decision_model,
        )

    inherited_component = INHERITED_COMPONENT_IDS.get(component.component_id)
    if inherited_component is not None:
        parent = resolve_component(settings, inherited_component)
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parent.model_spec,
            provenance=f"{inherited_component}.{parent.provenance}",
            parsed=parent.parsed,
        )

    category_value = normalized_model_value(_category_model(settings, component.category))
    if category_value is not None:
        parsed = parse_model_spec(category_value, env_name=CATEGORY_ENV_VARS[component.category])
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parsed.canonical_spec,
            provenance=f"category.{component.category}",
            parsed=parsed,
        )

    parsed = parse_model_spec(component.default_model, env_name=f"default:{component.component_id}")
    return ResolvedComponentModel(
        component_id=component.component_id,
        category=component.category,
        model_spec=parsed.canonical_spec,
        provenance="default",
        parsed=parsed,
    )


def resolve_intimacy_component(
    settings: Any,
    component_id: str,
) -> ResolvedComponentModel | None:
    """Resolve an optional intimacy-specific fallback model for a component."""
    component = COMPONENTS_BY_ID.get(component_id)
    if component is None:
        raise ModelResolutionError(f"Unknown LLM component id: {component_id}")

    component_overrides = getattr(settings, "llm_intimacy_component_models", {}) or {}
    unknown_components = set(component_overrides).difference(COMPONENTS_BY_ID)
    if unknown_components:
        raise ModelResolutionError(
            "Unknown intimacy LLM component id(s): "
            f"{', '.join(sorted(unknown_components))}"
        )

    component_value = normalized_model_value(component_overrides.get(component.component_id))
    if component_value is not None:
        parsed = parse_model_spec(component_value, env_name=component.intimacy_env_var)
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parsed.canonical_spec,
            provenance="intimacy component override",
            parsed=parsed,
        )

    inherited_component = INHERITED_COMPONENT_IDS.get(component_id)
    if inherited_component in {
        "applicability_scorer",
        "consequence_detector",
        "extractor",
        "need_detector_language",
        "topic_working_set",
    }:
        parent = resolve_intimacy_component(settings, inherited_component)
        if parent is not None:
            return ResolvedComponentModel(
                component_id=component_id,
                category=component.category,
                model_spec=parent.model_spec,
                provenance=f"{inherited_component}.{parent.provenance}",
                parsed=parent.parsed,
            )

    category_value = normalized_model_value(
        _intimacy_category_model(settings, component.category)
    )
    if category_value is not None:
        env_var = INTIMACY_CATEGORY_ENV_VARS[component.category]
        parsed = parse_model_spec(category_value, env_name=env_var)
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=parsed.canonical_spec,
            provenance=f"intimacy category.{component.category}",
            parsed=parsed,
        )

    return None


def resolve_component_model(settings: Any, component_id: str) -> str:
    """Return the canonical model spec for one LLM-backed component."""
    return resolve_component(settings, component_id).model_spec


def resolve_intimacy_component_model(settings: Any, component_id: str) -> str | None:
    """Return the optional intimacy fallback model for one component."""
    resolved = resolve_intimacy_component(settings, component_id)
    return resolved.model_spec if resolved is not None else None


def resolve_all_components(settings: Any) -> dict[str, ResolvedComponentModel]:
    """Resolve every known LLM-backed component."""
    return {
        spec.component_id: resolve_component(settings, spec.component_id)
        for spec in COMPONENT_SPECS
    }


def resolve_all_intimacy_components(settings: Any) -> dict[str, ResolvedComponentModel]:
    """Resolve all configured intimacy fallback component models."""
    resolved: dict[str, ResolvedComponentModel] = {}
    for spec in COMPONENT_SPECS:
        component = resolve_intimacy_component(settings, spec.component_id)
        if component is not None:
            resolved[spec.component_id] = component
    return resolved


def resolve_intimacy_fallback_models(settings: Any) -> dict[str, str]:
    """Return component id to intimacy fallback model mappings."""
    return {
        component_id: resolved.model_spec
        for component_id, resolved in resolve_all_intimacy_components(settings).items()
    }


def resolve_structured_output_rescue_model(settings: Any) -> ParsedModelSpec | None:
    """Return the optional structured-output rescue model."""
    if not bool(getattr(settings, "llm_structured_output_rescue_enabled", False)):
        return None
    value = normalized_model_value(
        getattr(settings, "llm_structured_output_rescue_model", None)
    )
    if value is None:
        raise ModelResolutionError(
            "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL is required when "
            "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED is true."
        )
    return parse_model_spec(
        value,
        env_name="ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL",
    )


def component_id_for_llm_purpose(purpose: str | None) -> str | None:
    """Return the component id associated with a stable LLM request purpose."""
    if purpose is None:
        return None
    return PURPOSE_TO_COMPONENT_ID.get(purpose)


def build_resolution_snapshot(settings: Any) -> ResolutionSnapshot:
    """Build the full resolution snapshot for boot validation/logging."""
    finite_decision_model = _finite_decision_model(settings)
    return ResolutionSnapshot(
        forced_global_model=normalized_model_value(
            getattr(settings, "llm_forced_global_model", None)
        ),
        finite_decisions_enabled=bool(
            getattr(settings, "llm_finite_decisions_enabled", False)
        ),
        finite_decision_model=(
            finite_decision_model.canonical_spec
            if finite_decision_model is not None
            else None
        ),
        category_models={
            "ingest": normalized_model_value(getattr(settings, "llm_ingest_model", None)),
            "retrieval": normalized_model_value(getattr(settings, "llm_retrieval_model", None)),
            "chat": normalized_model_value(getattr(settings, "llm_chat_model", None)),
        },
        intimacy_category_models={
            "ingest": normalized_model_value(
                getattr(settings, "llm_intimacy_ingest_model", None)
            ),
            "retrieval": normalized_model_value(
                getattr(settings, "llm_intimacy_retrieval_model", None)
            ),
        },
        structured_output_rescue_model=resolve_structured_output_rescue_model(settings),
        components=resolve_all_components(settings),
        intimacy_components=resolve_all_intimacy_components(settings),
        embedding=parse_embedding_model_spec(getattr(settings, "embedding_model", None)),
    )


def required_completion_provider_slugs(settings: Any) -> set[str]:
    """Return provider slugs required by resolved completion components."""
    providers: set[str] = set()
    forced = normalized_model_value(getattr(settings, "llm_forced_global_model", None))
    if forced is not None:
        providers.add(
            parse_model_spec(
                forced,
                env_name="ATAGIA_LLM_FORCED_GLOBAL_MODEL",
            ).provider_slug
        )
    else:
        providers.update(
            resolved.parsed.provider_slug
            for resolved in resolve_all_components(settings).values()
        )
    providers.update(
        resolved.parsed.provider_slug
        for resolved in resolve_all_intimacy_components(settings).values()
    )
    rescue_model = resolve_structured_output_rescue_model(settings)
    if rescue_model is not None:
        providers.add(rescue_model.provider_slug)
    return providers


def required_provider_slugs(settings: Any) -> set[str]:
    """Return all provider slugs required at bootstrap."""
    providers = required_completion_provider_slugs(settings)
    if getattr(settings, "embedding_backend", "none") != "none":
        providers.add(parse_embedding_model_spec(getattr(settings, "embedding_model", None)).provider_slug)
    return providers


def validate_required_provider_keys(settings: Any) -> None:
    """Fail fast when a resolved provider has no configured key."""
    validate_finite_decision_configuration(settings)
    for spec in COMPONENT_SPECS:
        resolved = resolve_component(settings, spec.component_id)
        if (
            resolved.parsed.provider_slug == "typesafe"
            and spec.component_id not in TYPED_DECISION_COMPONENT_IDS
        ):
            raise ModelResolutionError(
                "TypeSafe supports only finite-choice components, not "
                f"{spec.component_id}; keep generative components on their existing models"
            )
    fallback_models = list(resolve_all_intimacy_components(settings).values())
    rescue = resolve_structured_output_rescue_model(settings)
    if any(model.parsed.provider_slug == "typesafe" for model in fallback_models) or (
        rescue is not None and rescue.provider_slug == "typesafe"
    ):
        raise ModelResolutionError("TypeSafe cannot be used as a generative fallback or rescue model")
    provider_keys = {
        "anthropic": getattr(settings, "anthropic_api_key", None),
        "openai": getattr(settings, "openai_api_key", None),
        "google": getattr(settings, "google_api_key", None),
        "kimi": getattr(settings, "kimi_api_key", None),
        "minimax": getattr(settings, "minimax_api_key", None),
        "openrouter": getattr(settings, "openrouter_api_key", None),
        "typesafe": getattr(settings, "typesafe_api_key", None),
    }
    missing = [
        provider
        for provider in sorted(required_provider_slugs(settings).difference({"local"}))
        if not normalized_model_value(provider_keys.get(provider))
    ]
    if missing:
        env_names = {
            "anthropic": "ATAGIA_ANTHROPIC_API_KEY",
            "openai": "ATAGIA_OPENAI_API_KEY",
            "google": "ATAGIA_GOOGLE_API_KEY",
            "kimi": "ATAGIA_KIMI_API_KEY",
            "minimax": "ATAGIA_MINIMAX_API_KEY",
            "openrouter": "ATAGIA_OPENROUTER_API_KEY",
            "typesafe": "ATAGIA_TYPESAFE_API_KEY",
        }
        hints = ", ".join(env_names[provider] for provider in missing)
        raise ModelResolutionError(
            "Missing API key(s) for resolved LLM provider(s): "
            f"{', '.join(missing)}. Set {hints}, or use "
            "ATAGIA_LLM_FORCED_GLOBAL_MODEL to run all completion components on one provider."
        )


def format_resolution_log(settings: Any) -> str:
    """Format a stable LLM resolution summary block."""
    snapshot = build_resolution_snapshot(settings)
    lines = ["Atagia LLM resolution:"]
    forced = snapshot.forced_global_model or "<none>"
    lines.append(
        f"  forced_global_model    : {forced:<45} (env: ATAGIA_LLM_FORCED_GLOBAL_MODEL)"
    )
    lines.append(
        "  finite_decisions       : "
        f"{str(snapshot.finite_decisions_enabled).lower():<45} "
        "(env: ATAGIA_LLM_FINITE_DECISIONS_ENABLED)"
    )
    finite_model = snapshot.finite_decision_model or "<disabled>"
    lines.append(
        f"  finite_decision_model  : {finite_model:<45} "
        "(env: ATAGIA_LLM_FINITE_DECISION_MODEL)"
    )
    for category in ("ingest", "retrieval", "chat"):
        value = snapshot.category_models[category] or "<unset>"
        lines.append(
            f"  category.{category:<9} : {value:<45} (env: {CATEGORY_ENV_VARS[category]})"
        )
    for category in ("ingest", "retrieval"):
        value = snapshot.intimacy_category_models[category] or "<unset>"
        lines.append(
            f"  intimacy.{category:<8} : {value:<45} (env: {INTIMACY_CATEGORY_ENV_VARS[category]})"
        )
    lines.append(
        f"  embedding              : {snapshot.embedding.canonical_model:<45} (env: ATAGIA_EMBEDDING_MODEL)"
    )
    configured_rescue_model = normalized_model_value(
        getattr(settings, "llm_structured_output_rescue_model", None)
    )
    rescue_model = (
        snapshot.structured_output_rescue_model.canonical_spec
        if snapshot.structured_output_rescue_model is not None
        else f"<disabled; model={configured_rescue_model or '<unset>'}>"
    )
    retry_attempts = getattr(settings, "llm_structured_output_retry_attempts", 1)
    lines.append(
        f"  structured_retry       : {retry_attempts!s:<45} (env: ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS)"
    )
    lines.append(
        f"  structured_rescue      : {rescue_model:<45} (env: ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL)"
    )
    lines.append("  ----")
    for component_id in COMPONENTS_BY_ID:
        resolved = snapshot.components[component_id]
        lines.append(
            f"  {component_id:<23}: {resolved.model_spec:<45} [from: {resolved.provenance}]"
        )
    if snapshot.intimacy_components:
        lines.append("  ---- intimacy fallbacks")
        for component_id in COMPONENTS_BY_ID:
            resolved = snapshot.intimacy_components.get(component_id)
            if resolved is None:
                continue
            lines.append(
                f"  {component_id:<23}: {resolved.model_spec:<45} [from: {resolved.provenance}]"
            )
    else:
        lines.append("  intimacy_fallbacks    : <none configured>")
    if snapshot.forced_global_model is not None:
        lines.append(f"*** FORCED GLOBAL MODEL ACTIVE: {snapshot.forced_global_model} ***")
        lines.append("*** All completion component resolutions above are overridden. ***")
    return "\n".join(lines)


def log_resolution(settings: Any) -> None:
    """Log the resolved LLM configuration once at boot."""
    logger.info("%s", format_resolution_log(settings))


def _split_thinking(raw: str, *, env_name: str | None) -> tuple[str, str | None]:
    if raw.count(",") > 1:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
    if "," not in raw:
        return raw, None
    model_part, raw_level = raw.split(",", 1)
    thinking_level = raw_level.strip().lower()
    if thinking_level not in ALLOWED_THINKING_LEVELS:
        source = f" in {env_name}" if env_name else ""
        raise ModelResolutionError(
            f"Invalid thinking level {raw_level.strip()!r}{source}. Atagia accepts "
            "none, minimal, low, medium, high, xhigh, or max."
        )
    return model_part.strip(), thinking_level


def _invalid_model_spec_message(value: str, *, env_name: str | None) -> str:
    source = f" in {env_name}" if env_name else ""
    return (
        f"Invalid LLM model spec{source}: {value!r}. Expected provider/model where "
        "provider is one of {anthropic, openai, google, kimi, minimax, openrouter, local}. "
        "For OpenRouter use openrouter/vendor/model (e.g. "
        "openrouter/deepseek/deepseek-v4-flash). For a configured local endpoint "
        "use local/endpoint_id/served_model_id."
    )


def _category_model(settings: Any, category: str) -> str | None:
    if category == "ingest":
        return getattr(settings, "llm_ingest_model", None)
    if category == "retrieval":
        return getattr(settings, "llm_retrieval_model", None)
    if category == "chat":
        return getattr(settings, "llm_chat_model", None)
    raise ModelResolutionError(f"Unknown LLM category: {category}")


def _intimacy_category_model(settings: Any, category: str) -> str | None:
    if category == "ingest":
        return getattr(settings, "llm_intimacy_ingest_model", None)
    if category == "retrieval":
        return getattr(settings, "llm_intimacy_retrieval_model", None)
    return None
