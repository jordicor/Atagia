"""LLM model spec parsing and component-level resolution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any


logger = logging.getLogger(__name__)

ALLOWED_THINKING_LEVELS = frozenset({"none", "minimal", "medium", "high"})
PROVIDER_SLUG_TO_NAME = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "openrouter": "openrouter",
}
PROVIDER_NAME_TO_SLUG = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google",
    "google": "google",
    "openrouter": "openrouter",
}
SUPPORTED_PROVIDER_SLUGS = tuple(PROVIDER_SLUG_TO_NAME)

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_FLASH_LITE_MODEL = "openrouter/google/gemini-3.1-flash-lite-preview"


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
    category_models: dict[str, str | None]
    intimacy_category_models: dict[str, str | None]
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
    ComponentSpec("extractor", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("text_chunker", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("compactor", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("summary_privacy_judge", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("summary_privacy_refiner", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("belief_reviser", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("contract_projection", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("consequence_builder", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("consequence_detector", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("topic_working_set", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("consent_confirmation", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("intent_classifier", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("extraction_watchdog", "ingest", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("export_anonymizer", "ingest", "anthropic/claude-sonnet-4-6"),
    ComponentSpec("need_detector", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("applicability_scorer", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("context_staleness", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("metrics_computer", "retrieval", OPENROUTER_FLASH_LITE_MODEL),
    ComponentSpec("chat", "chat", "anthropic/claude-sonnet-4-6"),
)
COMPONENTS_BY_ID = {spec.component_id: spec for spec in COMPONENT_SPECS}

PURPOSE_TO_COMPONENT_ID = {
    "applicability_scoring": "applicability_scorer",
    "belief_revision": "belief_reviser",
    "chat_reply": "chat",
    "consequence_detection": "consequence_detector",
    "consequence_tendency_inference": "consequence_builder",
    "consent_confirmation_intent": "consent_confirmation",
    "context_cache_signal_detection": "context_staleness",
    "episode_synthesis": "compactor",
    "evaluation_contract_compliance": "metrics_computer",
    "export_anonymization_rewrite": "export_anonymizer",
    "export_anonymization_verify": "export_anonymizer",
    "extraction_watchdog": "extraction_watchdog",
    "intent_classifier_claim_key_equivalence": "intent_classifier",
    "intent_classifier_explicit": "intent_classifier",
    "memory_extraction": "extractor",
    "need_detection": "need_detector",
    "summary_chunk_segmentation": "compactor",
    "summary_privacy_gate_judge": "summary_privacy_judge",
    "summary_privacy_gate_refine": "summary_privacy_refiner",
    "text_chunking_level1": "text_chunker",
    "thematic_profile_synthesis": "compactor",
    "topic_working_set_update": "topic_working_set",
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

    segments = [segment.strip() for segment in model_part.split("/") if segment.strip()]
    if not segments:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
    provider_slug = segments[0].lower()
    if provider_slug not in PROVIDER_SLUG_TO_NAME:
        raise ModelResolutionError(_invalid_model_spec_message(raw, env_name=env_name))
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
    if first_segment in PROVIDER_SLUG_TO_NAME:
        return parse_model_spec(normalized_model).canonical_spec
    normalized_provider = (provider or "").strip().lower()
    provider_slug = PROVIDER_NAME_TO_SLUG.get(normalized_provider)
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


def resolve_component(settings: Any, component_id: str) -> ResolvedComponentModel:
    """Resolve a component model from forced/component/category/default layers."""
    component = COMPONENTS_BY_ID.get(component_id)
    if component is None:
        raise ModelResolutionError(f"Unknown LLM component id: {component_id}")

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

    if component.component_id == "extraction_watchdog":
        extractor = resolve_component(settings, "extractor")
        return ResolvedComponentModel(
            component_id=component.component_id,
            category=component.category,
            model_spec=extractor.model_spec,
            provenance=f"extractor.{extractor.provenance}",
            parsed=extractor.parsed,
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


def component_id_for_llm_purpose(purpose: str | None) -> str | None:
    """Return the component id associated with a stable LLM request purpose."""
    if purpose is None:
        return None
    return PURPOSE_TO_COMPONENT_ID.get(purpose)


def build_resolution_snapshot(settings: Any) -> ResolutionSnapshot:
    """Build the full resolution snapshot for boot validation/logging."""
    return ResolutionSnapshot(
        forced_global_model=normalized_model_value(
            getattr(settings, "llm_forced_global_model", None)
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
    return providers


def required_provider_slugs(settings: Any) -> set[str]:
    """Return all provider slugs required at bootstrap."""
    providers = required_completion_provider_slugs(settings)
    if getattr(settings, "embedding_backend", "none") != "none":
        providers.add(parse_embedding_model_spec(getattr(settings, "embedding_model", None)).provider_slug)
    return providers


def validate_required_provider_keys(settings: Any) -> None:
    """Fail fast when a resolved provider has no configured key."""
    provider_keys = {
        "anthropic": getattr(settings, "anthropic_api_key", None),
        "openai": getattr(settings, "openai_api_key", None),
        "google": getattr(settings, "google_api_key", None),
        "openrouter": getattr(settings, "openrouter_api_key", None),
    }
    missing = [
        provider for provider in sorted(required_provider_slugs(settings))
        if not normalized_model_value(provider_keys.get(provider))
    ]
    if missing:
        env_names = {
            "anthropic": "ATAGIA_ANTHROPIC_API_KEY",
            "openai": "ATAGIA_OPENAI_API_KEY",
            "google": "ATAGIA_GOOGLE_API_KEY",
            "openrouter": "ATAGIA_OPENROUTER_API_KEY",
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
            "none, minimal, medium, or high."
        )
    return model_part.strip(), thinking_level


def _invalid_model_spec_message(value: str, *, env_name: str | None) -> str:
    source = f" in {env_name}" if env_name else ""
    return (
        f"Invalid LLM model spec{source}: {value!r}. Expected provider/model where "
        "provider is one of {anthropic, openai, google, openrouter}. For OpenRouter "
        "use openrouter/vendor/model (e.g. openrouter/deepseek/deepseek-v4-flash)."
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
