"""Inference-access authorization and startup auditing helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from atagia.services.inference_routes import (
    BaseUrlClass,
    InferenceAccessMode,
    InferenceCostClass,
    InferenceOperation,
    InferenceRouteError,
    ResolvedInferenceRoute,
    resolve_local_inference_route,
)
from atagia.services.local_endpoint_catalog import (
    LocalEndpointCatalog,
    validate_local_base_url,
)
from atagia.services.model_resolution import (
    COMPONENT_SPECS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FINITE_DECISION_MODEL,
    ModelResolutionError,
    normalized_model_value,
    parse_model_spec,
    resolve_component_model,
    resolve_intimacy_component_model,
)
from atagia.services.openrouter_zero_cost import (
    OpenRouterZeroCostAttestation,
    is_exact_zero_cost_openrouter_model,
    is_exact_zero_cost_openrouter_spec,
)


@dataclass(frozen=True, slots=True)
class InferenceAccessPolicy:
    """Internal route policy reserved for inference safety-mode activation."""

    mode: InferenceAccessMode = InferenceAccessMode.UNRESTRICTED
    openrouter_zero_cost_attestation: OpenRouterZeroCostAttestation | None = None

    def denial_reason(
        self,
        route: ResolvedInferenceRoute,
        *,
        local_catalog: LocalEndpointCatalog | None,
    ) -> str | None:
        """Return a fail-closed denial reason without performing network I/O."""
        if self.mode is InferenceAccessMode.UNRESTRICTED:
            return None
        if self.mode is InferenceAccessMode.ZERO_COST:
            return self._zero_cost_denial_reason(route, local_catalog=local_catalog)
        if self.mode is not InferenceAccessMode.LOCAL_ONLY:
            return f"Unsupported inference access mode {self.mode.value!r}."
        if route.provider_slug != "local":
            return "local_only permits only catalog-backed local routes."
        return self._local_route_denial_reason(route, local_catalog=local_catalog)

    def _zero_cost_denial_reason(
        self,
        route: ResolvedInferenceRoute,
        *,
        local_catalog: LocalEndpointCatalog | None,
    ) -> str | None:
        if route.provider_slug == "local":
            return self._local_route_denial_reason(route, local_catalog=local_catalog)
        if route.operation is InferenceOperation.EMBEDDING:
            return "zero_cost permits embeddings only from catalog-backed local routes."
        if route.provider_slug != "openrouter":
            return "zero_cost does not support this external provider."
        if (
            route.endpoint_id is not None
            or route.base_url_class is not BaseUrlClass.EXTERNAL
        ):
            return "zero_cost OpenRouter route identity is malformed."
        try:
            parsed = parse_model_spec(route.canonical_model_spec, allow_thinking=True)
        except ModelResolutionError:
            return "zero_cost denied a malformed OpenRouter route."
        if (
            parsed.provider_slug != "openrouter"
            or parsed.canonical_spec != route.canonical_model_spec
        ):
            return "zero_cost OpenRouter route identity is malformed."
        if not is_exact_zero_cost_openrouter_model(parsed.request_model):
            if parsed.request_model.endswith(":free"):
                return "zero_cost denied a malformed OpenRouter free route."
            return "zero_cost denies ordinary or paid OpenRouter models."
        if route.cost_class is not InferenceCostClass.ZERO_COST_OPENROUTER_CANDIDATE:
            return "zero_cost denied a malformed or unknown-cost OpenRouter free route."
        if self.openrouter_zero_cost_attestation is None:
            return (
                "zero_cost OpenRouter requires a verified dedicated free-tier "
                "no-BYOK attestation."
            )
        return None

    @staticmethod
    def _local_route_denial_reason(
        route: ResolvedInferenceRoute,
        *,
        local_catalog: LocalEndpointCatalog | None,
    ) -> str | None:
        if (
            route.base_url_class is not BaseUrlClass.LOCAL
            or route.endpoint_id is None
            or local_catalog is None
        ):
            return "restricted modes permit only catalog-backed local routes."
        try:
            catalog_route = resolve_local_inference_route(
                local_catalog,
                route.canonical_model_spec,
                route.operation,
            )
        except InferenceRouteError:
            return (
                "restricted modes require a registered endpoint and model capability."
            )
        if catalog_route != route:
            return "restricted route identity did not match the local catalog."
        return None

    @property
    def restricted(self) -> bool:
        """Whether this internal policy requires restricted transports."""
        return self.mode is not InferenceAccessMode.UNRESTRICTED


@dataclass(frozen=True, slots=True)
class InferenceStartupRoute:
    """One configured inference route examined before restricted startup."""

    source: str
    model_spec: str
    route: ResolvedInferenceRoute


def local_provider_registry_key(endpoint_id: str) -> str:
    """Return the private provider registry key for one local endpoint."""
    return f"local:{endpoint_id}"


def resolve_internal_inference_route(
    model_spec: str,
    operation: InferenceOperation,
    *,
    local_catalog: LocalEndpointCatalog | None,
) -> ResolvedInferenceRoute:
    """Resolve a local or ordinary model spec for internal policy enforcement.

    This remains separate from public provider-model parsing because catalog
    membership and operation capability are runtime policy concerns.
    """
    normalized = normalized_model_value(model_spec)
    if normalized is None:
        raise InferenceRouteError("Inference model specification must not be empty.")
    if normalized.lower().startswith("local/"):
        if local_catalog is None:
            raise InferenceRouteError(
                "Local model specifications require an injected local endpoint catalog."
            )
        return resolve_local_inference_route(local_catalog, normalized, operation)
    try:
        parsed = parse_model_spec(
            normalized,
            allow_thinking=operation is InferenceOperation.COMPLETION,
        )
    except ModelResolutionError as exc:
        raise InferenceRouteError(str(exc)) from exc
    cost_class = InferenceCostClass.METERED
    if parsed.provider_slug == "openrouter" and is_exact_zero_cost_openrouter_spec(
        model_spec,
        canonical_spec=parsed.canonical_spec,
        request_model=parsed.request_model,
    ):
        # A free-looking suffix is only a candidate. The immutable policy
        # attestation remains necessary before the route can be admitted.
        cost_class = InferenceCostClass.ZERO_COST_OPENROUTER_CANDIDATE
    return ResolvedInferenceRoute(
        operation=operation,
        canonical_model_spec=parsed.canonical_spec,
        provider_slug=parsed.provider_slug,
        endpoint_id=None,
        base_url_class=BaseUrlClass.EXTERNAL,
        cost_class=cost_class,
    )


def audit_inference_startup_routes(
    settings: Any,
    policy: InferenceAccessPolicy,
    *,
    local_catalog: LocalEndpointCatalog | None,
) -> tuple[InferenceStartupRoute, ...]:
    """Resolve and authorize every configured internal LLM route at startup."""
    routes = resolve_inference_startup_routes(
        settings,
        local_catalog=local_catalog,
    )
    authorize_inference_startup_routes(
        routes,
        policy,
        local_catalog=local_catalog,
    )
    audit_privacy_filter_startup_urls(settings, policy)
    return routes


def resolve_inference_startup_routes(
    settings: Any,
    *,
    local_catalog: LocalEndpointCatalog | None,
) -> tuple[InferenceStartupRoute, ...]:
    """Resolve configured inference routes without authorizing or opening I/O."""
    return tuple(
        InferenceStartupRoute(
            source=source,
            model_spec=model_spec,
            route=resolve_internal_inference_route(
                model_spec,
                operation,
                local_catalog=local_catalog,
            ),
        )
        for source, model_spec, operation in _configured_model_routes(settings)
    )


def authorize_inference_startup_routes(
    routes: tuple[InferenceStartupRoute, ...],
    policy: InferenceAccessPolicy,
    *,
    local_catalog: LocalEndpointCatalog | None,
) -> None:
    """Authorize already resolved startup routes against one immutable policy."""
    for candidate in routes:
        denial = policy.denial_reason(candidate.route, local_catalog=local_catalog)
        if denial is not None:
            raise InferenceRouteError(
                f"Inference access policy denied configured route {candidate.source!r} "
                f"({candidate.model_spec!r}): {denial}"
            )


def audit_privacy_filter_startup_urls(
    settings: Any,
    policy: InferenceAccessPolicy,
) -> None:
    """Validate enabled OPF sidecar destinations for a restricted policy."""
    if not policy.restricted or not bool(
        getattr(settings, "opf_privacy_filter_enabled", False)
    ):
        return
    for source, value in (
        ("opf_primary_url", getattr(settings, "opf_primary_url", None)),
        ("opf_fallback_url", getattr(settings, "opf_fallback_url", None)),
    ):
        if not isinstance(value, str):
            raise InferenceRouteError(
                f"Inference access policy denied {source}: a local URL is required."
            )
        try:
            validate_local_base_url(value, context=source)
        except InferenceRouteError as exc:
            raise InferenceRouteError(
                f"Inference access policy denied {source}: {exc}"
            ) from exc


def _configured_model_routes(
    settings: Any,
) -> Iterator[tuple[str, str, InferenceOperation]]:
    forced = normalized_model_value(getattr(settings, "llm_forced_global_model", None))
    if forced is not None:
        yield "forced_global", forced, InferenceOperation.COMPLETION

    for category in ("ingest", "retrieval", "chat"):
        model = normalized_model_value(getattr(settings, f"llm_{category}_model", None))
        if model is not None:
            yield f"category.{category}", model, InferenceOperation.COMPLETION
    if bool(getattr(settings, "llm_finite_decisions_enabled", False)):
        decision_model = normalized_model_value(
            getattr(settings, "llm_finite_decision_model", None)
        ) or DEFAULT_FINITE_DECISION_MODEL
        yield "finite_decision", decision_model, InferenceOperation.COMPLETION
    for component_id, model in (
        getattr(settings, "llm_component_models", {}) or {}
    ).items():
        normalized = normalized_model_value(model)
        if normalized is not None:
            yield f"component.{component_id}", normalized, InferenceOperation.COMPLETION

    for component in COMPONENT_SPECS:
        model = _effective_component_model(settings, component.component_id)
        yield (
            f"normal.{component.component_id}",
            model,
            InferenceOperation.COMPLETION,
        )

    for category in ("ingest", "retrieval"):
        model = normalized_model_value(
            getattr(settings, f"llm_intimacy_{category}_model", None)
        )
        if model is not None:
            yield f"intimacy_category.{category}", model, InferenceOperation.COMPLETION
    for component_id, model in (
        getattr(settings, "llm_intimacy_component_models", {}) or {}
    ).items():
        normalized = normalized_model_value(model)
        if normalized is not None:
            yield (
                f"intimacy_component.{component_id}",
                normalized,
                InferenceOperation.COMPLETION,
            )

    for component in COMPONENT_SPECS:
        model = _effective_intimacy_model(settings, component.component_id)
        if model is not None:
            yield (
                f"intimacy.{component.component_id}",
                model,
                InferenceOperation.COMPLETION,
            )

    if bool(getattr(settings, "llm_structured_output_rescue_enabled", False)):
        rescue_model = normalized_model_value(
            getattr(settings, "llm_structured_output_rescue_model", None)
        )
        if rescue_model is None:
            raise InferenceRouteError(
                "Structured-output rescue is enabled without a rescue model."
            )
        yield "structured_output_rescue", rescue_model, InferenceOperation.COMPLETION

    if getattr(settings, "embedding_backend", "none") != "none":
        embedding_model = (
            normalized_model_value(getattr(settings, "embedding_model", None))
            or DEFAULT_EMBEDDING_MODEL
        )
        yield "embedding", embedding_model, InferenceOperation.EMBEDDING

    proxy_upstream = normalized_model_value(
        getattr(settings, "openai_proxy_upstream_model", None)
    )
    if proxy_upstream is not None:
        yield "proxy_upstream", proxy_upstream, InferenceOperation.COMPLETION
    yield (
        "proxy_chat_fallback",
        _effective_component_model(settings, "chat"),
        InferenceOperation.COMPLETION,
    )


def _effective_component_model(settings: Any, component_id: str) -> str:
    try:
        return resolve_component_model(settings, component_id)
    except ModelResolutionError as exc:
        raise InferenceRouteError(str(exc)) from exc


def _effective_intimacy_model(settings: Any, component_id: str) -> str | None:
    try:
        return resolve_intimacy_component_model(settings, component_id)
    except ModelResolutionError as exc:
        raise InferenceRouteError(str(exc)) from exc
