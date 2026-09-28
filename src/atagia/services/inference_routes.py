"""Route primitives and local endpoint catalog validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import TYPE_CHECKING

from atagia.services.model_resolution import ALLOWED_THINKING_LEVELS


if TYPE_CHECKING:
    from atagia.services.local_endpoint_catalog import LocalEndpointCatalog


_LOCAL_MODEL_ENDPOINT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class InferenceRouteError(ValueError):
    """Raised when an internal inference route cannot be resolved safely."""


class InferenceAccessMode(str, Enum):
    """Supported inference-access modes."""

    UNRESTRICTED = "unrestricted"
    LOCAL_ONLY = "local_only"
    ZERO_COST = "zero_cost"


class InferenceOperation(str, Enum):
    """The inference operation a route is authorized to perform."""

    COMPLETION = "completion"
    EMBEDDING = "embedding"


class BaseUrlClass(str, Enum):
    """The network locality class of an inference destination."""

    LOCAL = "local"
    EXTERNAL = "external"


class InferenceCostClass(str, Enum):
    """The cost classification attached to a resolved inference route."""

    LOCAL = "local"
    ZERO_COST_OPENROUTER_CANDIDATE = "zero_cost_openrouter_candidate"
    PROVIDER_VERIFIED_FREE = "provider_verified_free"
    METERED = "metered"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResolvedInferenceRoute:
    """Immutable route metadata produced by internal model resolution."""

    operation: InferenceOperation
    canonical_model_spec: str
    provider_slug: str
    endpoint_id: str | None
    base_url_class: BaseUrlClass
    cost_class: InferenceCostClass


@dataclass(frozen=True, slots=True)
class ParsedLocalModelSpec:
    """Parsed form of a catalog-backed ``local/...`` model specification."""

    raw_spec: str
    canonical_spec: str
    endpoint_id: str
    served_model_id: str
    thinking_level: str | None = None


def parse_local_model_spec(
    value: str,
    *,
    allow_thinking: bool = True,
) -> ParsedLocalModelSpec:
    """Parse a local model spec without changing public provider resolution."""
    if not isinstance(value, str):
        raise InferenceRouteError("Local model spec must be a string.")
    raw = value.strip()
    if not raw:
        raise InferenceRouteError("Local model spec must not be empty.")

    model_part, thinking_level = _split_local_thinking(raw)
    if thinking_level is not None and not allow_thinking:
        raise InferenceRouteError(
            "Thinking levels are not supported for this local model spec."
        )

    segments = model_part.split("/", 2)
    if len(segments) != 3 or segments[0].lower() != "local":
        raise InferenceRouteError(
            "Invalid local model spec. Expected local/<endpoint_id>/<served_model_id>."
        )
    endpoint_id = segments[1]
    served_model_id = segments[2]
    if not _LOCAL_MODEL_ENDPOINT_ID_PATTERN.fullmatch(endpoint_id):
        raise InferenceRouteError(
            "Local model spec endpoint ID must use lowercase ASCII letters, digits, underscores, or hyphens."
        )
    if not served_model_id or served_model_id != served_model_id.strip():
        raise InferenceRouteError("Local model spec served model ID must be non-empty.")

    canonical_model = f"local/{endpoint_id}/{served_model_id}"
    canonical_spec = (
        f"{canonical_model},{thinking_level}"
        if thinking_level is not None
        else canonical_model
    )
    return ParsedLocalModelSpec(
        raw_spec=raw,
        canonical_spec=canonical_spec,
        endpoint_id=endpoint_id,
        served_model_id=served_model_id,
        thinking_level=thinking_level,
    )


def resolve_local_inference_route(
    catalog: LocalEndpointCatalog,
    model_spec: str,
    operation: InferenceOperation,
) -> ResolvedInferenceRoute:
    """Resolve one catalog-backed local route without any network activity."""
    parsed = parse_local_model_spec(
        model_spec,
        allow_thinking=operation is InferenceOperation.COMPLETION,
    )
    endpoint = catalog.endpoint_for_id(parsed.endpoint_id)
    if endpoint is None:
        raise InferenceRouteError(
            f"Local model spec references unknown endpoint ID {parsed.endpoint_id!r}."
        )
    if parsed.served_model_id not in endpoint.models_for(operation):
        raise InferenceRouteError(
            f"Local endpoint {parsed.endpoint_id!r} does not serve model "
            f"{parsed.served_model_id!r} for {operation.value}."
        )
    return ResolvedInferenceRoute(
        operation=operation,
        canonical_model_spec=parsed.canonical_spec,
        provider_slug="local",
        endpoint_id=endpoint.endpoint_id,
        base_url_class=BaseUrlClass.LOCAL,
        cost_class=InferenceCostClass.LOCAL,
    )


def _split_local_thinking(raw: str) -> tuple[str, str | None]:
    if raw.count(",") > 1:
        raise InferenceRouteError("Invalid local model spec thinking level.")
    if "," not in raw:
        return raw, None
    model_part, raw_level = raw.split(",", 1)
    thinking_level = raw_level.strip().lower()
    if thinking_level not in ALLOWED_THINKING_LEVELS:
        raise InferenceRouteError(
            f"Invalid local model spec thinking level {raw_level.strip()!r}."
        )
    return model_part.strip(), thinking_level
