"""Validated immutable catalog of explicitly configured local LLM endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from atagia.services.inference_routes import InferenceOperation, InferenceRouteError


LOCAL_ENDPOINT_API_KEY_PLACEHOLDER = "atagia-local-no-api-key"
_ENDPOINT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_ENVIRONMENT_VARIABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PRIVATE_IPV4_NETWORKS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)
_IPV6_ULA_NETWORK = ipaddress.IPv6Network("fc00::/7")


class LocalEndpointAdapter(str, Enum):
    """The only local endpoint protocol supported by catalog version one."""

    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True, slots=True)
class LocalEndpoint:
    """One explicit local OpenAI-compatible inference endpoint."""

    endpoint_id: str
    adapter: LocalEndpointAdapter
    base_url: str
    api_key_env: str | None
    chat_models: tuple[str, ...]
    embedding_models: tuple[str, ...]

    def models_for(self, operation: InferenceOperation) -> tuple[str, ...]:
        """Return the models this endpoint serves for one operation."""
        if operation is InferenceOperation.COMPLETION:
            return self.chat_models
        if operation is InferenceOperation.EMBEDDING:
            return self.embedding_models
        raise InferenceRouteError(f"Unsupported inference operation: {operation!r}")


@dataclass(frozen=True, slots=True)
class LocalEndpointCatalog:
    """Immutable version-one catalog of configured local endpoints."""

    endpoints: tuple[LocalEndpoint, ...]

    def endpoint_for_id(self, endpoint_id: str) -> LocalEndpoint | None:
        """Return the endpoint with an exact configured identity, if present."""
        for endpoint in self.endpoints:
            if endpoint.endpoint_id == endpoint_id:
                return endpoint
        return None


def load_local_endpoint_catalog(path: str | Path) -> LocalEndpointCatalog:
    """Load and validate a version-one local endpoint catalog JSON file."""
    catalog_path = Path(path)
    if not catalog_path.is_absolute():
        raise InferenceRouteError("Local endpoint catalog path must be absolute.")
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InferenceRouteError(
            f"Unable to read local endpoint catalog at {catalog_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise InferenceRouteError(
            f"Local endpoint catalog at {catalog_path} is not valid JSON."
        ) from exc
    return parse_local_endpoint_catalog(document)


def parse_local_endpoint_catalog(document: object) -> LocalEndpointCatalog:
    """Validate a decoded version-one local endpoint catalog document."""
    catalog = _require_mapping(document, "Local endpoint catalog")
    _require_keys(
        catalog, required={"version", "endpoints"}, optional=set(), context="catalog"
    )

    version = catalog["version"]
    if type(version) is not int or version != 1:
        raise InferenceRouteError("Local endpoint catalog version must be integer 1.")

    raw_endpoints = catalog["endpoints"]
    if not isinstance(raw_endpoints, list) or not raw_endpoints:
        raise InferenceRouteError(
            "Local endpoint catalog endpoints must be a non-empty list."
        )

    endpoints = tuple(
        _parse_local_endpoint(raw_endpoint, index)
        for index, raw_endpoint in enumerate(raw_endpoints)
    )
    endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
    if len(set(endpoint_ids)) != len(endpoint_ids):
        raise InferenceRouteError(
            "Local endpoint catalog contains duplicate endpoint IDs."
        )
    return LocalEndpointCatalog(endpoints=endpoints)


def local_endpoint_api_key(
    endpoint: LocalEndpoint,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return an endpoint credential or the fixed non-secret SDK placeholder."""
    if endpoint.api_key_env is None:
        return LOCAL_ENDPOINT_API_KEY_PLACEHOLDER
    environment = os.environ if environ is None else environ
    value = environment.get(endpoint.api_key_env)
    if value is None or not value.strip():
        raise InferenceRouteError(
            f"Local endpoint {endpoint.endpoint_id!r} requires environment variable "
            f"{endpoint.api_key_env!r}, but it is unset or empty."
        )
    return value


def _parse_local_endpoint(document: object, index: int) -> LocalEndpoint:
    context = f"endpoint at index {index}"
    endpoint = _require_mapping(document, f"Local {context}")
    _require_keys(
        endpoint,
        required={"id", "adapter", "base_url", "chat_models", "embedding_models"},
        optional={"api_key_env"},
        context=context,
    )

    endpoint_id = _require_string(endpoint["id"], f"Local {context} ID")
    _validate_endpoint_id(endpoint_id, context=f"Local {context} ID")

    adapter_value = _require_string(endpoint["adapter"], f"Local {context} adapter")
    try:
        adapter = LocalEndpointAdapter(adapter_value)
    except ValueError as exc:
        raise InferenceRouteError(
            f"Local {context} adapter must be 'openai_compatible'."
        ) from exc

    base_url = _require_string(endpoint["base_url"], f"Local {context} base URL")
    validate_local_base_url(base_url, context=f"Local {context} base URL")

    api_key_env: str | None = None
    if "api_key_env" in endpoint:
        api_key_env = _require_string(
            endpoint["api_key_env"], f"Local {context} api_key_env"
        )
        if not _ENVIRONMENT_VARIABLE_NAME_PATTERN.fullmatch(api_key_env):
            raise InferenceRouteError(
                f"Local {context} api_key_env must be a valid environment variable name."
            )

    chat_models = _parse_model_ids(endpoint["chat_models"], context, "chat_models")
    embedding_models = _parse_model_ids(
        endpoint["embedding_models"], context, "embedding_models"
    )
    if not chat_models and not embedding_models:
        raise InferenceRouteError(
            f"Local {context} must declare at least one chat or embedding model."
        )
    return LocalEndpoint(
        endpoint_id=endpoint_id,
        adapter=adapter,
        base_url=base_url,
        api_key_env=api_key_env,
        chat_models=chat_models,
        embedding_models=embedding_models,
    )


def _parse_model_ids(
    value: object, endpoint_context: str, field_name: str
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InferenceRouteError(
            f"Local {endpoint_context} {field_name} must be a list."
        )
    models: list[str] = []
    for index, model_id in enumerate(value):
        model = _require_string(
            model_id,
            f"Local {endpoint_context} {field_name}[{index}]",
        )
        if "," in model:
            raise InferenceRouteError(
                f"Local {endpoint_context} {field_name}[{index}] must not contain a comma."
            )
        models.append(model)
    if len(set(models)) != len(models):
        raise InferenceRouteError(
            f"Local {endpoint_context} {field_name} contains duplicate model IDs."
        )
    return tuple(models)


def _validate_endpoint_id(value: str, *, context: str) -> None:
    if not _ENDPOINT_ID_PATTERN.fullmatch(value):
        raise InferenceRouteError(
            f"{context} must use lowercase ASCII letters, digits, underscores, or hyphens."
        )


def validate_local_base_url(value: str, *, context: str) -> None:
    if value != value.strip():
        raise InferenceRouteError(
            f"{context} must not contain leading or trailing whitespace."
        )
    if any(character in value for character in "\t\r\n"):
        raise InferenceRouteError(f"{context} must not contain control characters.")
    if "?" in value or "#" in value:
        raise InferenceRouteError(f"{context} must not contain a query or fragment.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise InferenceRouteError(f"{context} is malformed.") from exc
    if parsed.scheme not in {"http", "https"}:
        raise InferenceRouteError(f"{context} must use http or https.")
    if not parsed.netloc or parsed.hostname is None or port is None:
        raise InferenceRouteError(f"{context} must include a literal host and port.")
    if parsed.username is not None or parsed.password is not None:
        raise InferenceRouteError(f"{context} must not include user info.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise InferenceRouteError(
            f"{context} host must be a literal loopback, RFC1918 IPv4, or IPv6 ULA address."
        ) from exc
    if not _is_allowed_local_address(address):
        raise InferenceRouteError(
            f"{context} host must be a literal loopback, RFC1918 IPv4, or IPv6 ULA address."
        )


def _is_allowed_local_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return any(address in network for network in _PRIVATE_IPV4_NETWORKS)
    return address in _IPV6_ULA_NETWORK


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceRouteError(f"{context} must be a JSON object.")
    if not all(isinstance(key, str) for key in value):
        raise InferenceRouteError(f"{context} keys must be strings.")
    return value


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required.difference(value)
    unexpected = set(value).difference(required | optional)
    if missing:
        raise InferenceRouteError(
            f"Local {context} is missing required field(s): {', '.join(sorted(missing))}."
        )
    if unexpected:
        raise InferenceRouteError(
            f"Local {context} contains unsupported field(s): {', '.join(sorted(unexpected))}."
        )


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InferenceRouteError(
            f"{context} must be a non-empty string without surrounding whitespace."
        )
    return value
