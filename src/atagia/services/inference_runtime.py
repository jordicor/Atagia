"""Public inference-access activation and startup preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from atagia.core.config import Settings
from atagia.services.inference_policy import (
    InferenceAccessPolicy,
    InferenceStartupRoute,
    audit_privacy_filter_startup_urls,
    authorize_inference_startup_routes,
    resolve_inference_startup_routes,
    resolve_internal_inference_route,
)
from atagia.services.inference_routes import (
    InferenceAccessMode,
    InferenceCostClass,
    InferenceOperation,
    InferenceRouteError,
)
from atagia.services.local_endpoint_catalog import (
    LocalEndpointCatalog,
    load_local_endpoint_catalog,
)
from atagia.services.model_resolution import (
    ModelResolutionError,
    parse_model_spec,
    required_provider_slugs,
)
from atagia.services.openrouter_zero_cost import attest_openrouter_zero_cost


@dataclass(frozen=True, slots=True)
class PreparedInferenceAccess:
    """Catalog, policy, and audited routes ready for provider construction."""

    policy: InferenceAccessPolicy
    local_catalog: LocalEndpointCatalog | None
    startup_routes: tuple[InferenceStartupRoute, ...]


def configured_inference_access_mode(settings: Settings) -> InferenceAccessMode:
    """Return the validated public mode as its internal enum."""
    try:
        return InferenceAccessMode(settings.inference_access_mode)
    except ValueError as exc:
        raise InferenceRouteError(
            f"Unsupported inference access mode {settings.inference_access_mode!r}."
        ) from exc


async def prepare_inference_access(
    settings: Settings,
    *,
    http_client_factory: Callable[..., Any] | None = None,
    additional_completion_models: Mapping[str, str] | None = None,
) -> PreparedInferenceAccess:
    """Load, attest, and authorize all configured routes before provider setup."""
    mode = configured_inference_access_mode(settings)
    local_catalog = (
        load_local_endpoint_catalog(settings.local_llm_endpoints_file)
        if settings.local_llm_endpoints_file is not None
        else None
    )
    if mode is InferenceAccessMode.LOCAL_ONLY and local_catalog is None:
        raise InferenceRouteError(
            "local_only requires ATAGIA_LOCAL_LLM_ENDPOINTS_FILE."
        )
    if mode is InferenceAccessMode.UNRESTRICTED and local_catalog is None:
        try:
            local_route_selected = "local" in required_provider_slugs(settings)
            extra_models = list((additional_completion_models or {}).values())
            if (
                settings.openai_proxy_upstream_model is not None
                and settings.openai_proxy_upstream_model.strip()
            ):
                extra_models.append(settings.openai_proxy_upstream_model)
            local_route_selected = local_route_selected or any(
                parse_model_spec(model).provider_slug == "local"
                for model in extra_models
            )
        except ModelResolutionError as exc:
            raise InferenceRouteError(str(exc)) from exc
        if local_route_selected:
            raise InferenceRouteError(
                "Local model specifications require ATAGIA_LOCAL_LLM_ENDPOINTS_FILE."
            )
        # Preserve the existing default startup path exactly. The normal model
        # resolver and provider factory retain their established validation;
        # restricted preflight must not make dormant/shadowed overrides newly
        # fatal when no inference-access feature was selected.
        return PreparedInferenceAccess(
            policy=InferenceAccessPolicy(mode=mode),
            local_catalog=None,
            startup_routes=(),
        )

    routes = resolve_inference_startup_routes(
        settings,
        local_catalog=local_catalog,
    )
    routes += tuple(
        InferenceStartupRoute(
            source=source,
            model_spec=model_spec,
            route=resolve_internal_inference_route(
                model_spec,
                InferenceOperation.COMPLETION,
                local_catalog=local_catalog,
            ),
        )
        for source, model_spec in (additional_completion_models or {}).items()
    )
    policy = InferenceAccessPolicy(mode=mode)
    zero_cost_candidates = tuple(
        candidate
        for candidate in routes
        if candidate.route.cost_class
        is InferenceCostClass.ZERO_COST_OPENROUTER_CANDIDATE
        and candidate.route.operation is InferenceOperation.COMPLETION
    )
    if mode is InferenceAccessMode.ZERO_COST and zero_cost_candidates:
        # Do not open even the free control-plane check for a configuration
        # that is already inadmissible for another reason.
        authorize_inference_startup_routes(
            tuple(
                candidate
                for candidate in routes
                if not (
                    candidate.route.cost_class
                    is InferenceCostClass.ZERO_COST_OPENROUTER_CANDIDATE
                    and candidate.route.operation is InferenceOperation.COMPLETION
                )
            ),
            policy,
            local_catalog=local_catalog,
        )
        audit_privacy_filter_startup_urls(settings, policy)
        attestation = await attest_openrouter_zero_cost(
            api_key=settings.openrouter_api_key,
            profile=settings.zero_cost_openrouter_profile,
            base_url=settings.openrouter_base_url,
            http_client_factory=http_client_factory,
        )
        policy = replace(
            policy,
            openrouter_zero_cost_attestation=attestation,
        )

    authorize_inference_startup_routes(
        routes,
        policy,
        local_catalog=local_catalog,
    )
    audit_privacy_filter_startup_urls(settings, policy)
    return PreparedInferenceAccess(
        policy=policy,
        local_catalog=local_catalog,
        startup_routes=routes,
    )


def require_unrestricted_inference(settings: Settings, *, entry_point: str) -> None:
    """Refuse direct or externally governed tools in restricted modes."""
    mode = configured_inference_access_mode(settings)
    if mode is InferenceAccessMode.UNRESTRICTED:
        return
    raise InferenceRouteError(
        f"{entry_point} cannot run with inference_access_mode={mode.value!r}: "
        "this entry point constructs AI transports outside Atagia's governed client."
    )


def format_inference_access_diagnostics(
    prepared: PreparedInferenceAccess,
) -> str:
    """Format a secret-free summary of the admitted startup configuration."""
    endpoint_ids = (
        [endpoint.endpoint_id for endpoint in prepared.local_catalog.endpoints]
        if prepared.local_catalog is not None
        else []
    )
    admitted_specs = sorted(
        {candidate.route.canonical_model_spec for candidate in prepared.startup_routes}
    )
    return "\n".join(
        (
            "Atagia inference access:",
            f"  mode                 : {prepared.policy.mode.value}",
            "  local_endpoint_ids   : "
            + (", ".join(endpoint_ids) if endpoint_ids else "<none>"),
            "  admitted_model_specs : "
            + (", ".join(admitted_specs) if admitted_specs else "<none>"),
            "  openrouter_free      : "
            + (
                "attested"
                if prepared.policy.openrouter_zero_cost_attestation is not None
                else "not enabled"
            ),
        )
    )
