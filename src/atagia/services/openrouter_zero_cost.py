"""Narrow OpenRouter controls for the zero-cost policy.

The external OpenRouter path is conditional on an administrative deployment
invariant.  This module verifies only the documented provider contract needed
by the internal policy; it does not attempt to inspect BYOK inventory or model
pricing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import inspect
import re
from typing import Any

import httpx


OPENROUTER_CANONICAL_BASE_URL = "https://openrouter.ai/api/v1"
ZERO_COST_OPENROUTER_PROFILE = "dedicated_free_tier_no_byok"
_OPENROUTER_SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_OPENROUTER_FREE_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*:free")


class OpenRouterZeroCostControlPlaneError(ValueError):
    """Raised when OpenRouter cannot be attested for zero-cost use."""


@dataclass(frozen=True, slots=True)
class OpenRouterZeroCostAttestation:
    """Immutable result of the required OpenRouter free-tier key check."""

    profile: str
    base_url: str
    inference_key_fingerprint: str
    is_free_tier: bool

    def __post_init__(self) -> None:
        if self.profile != ZERO_COST_OPENROUTER_PROFILE:
            raise OpenRouterZeroCostControlPlaneError(
                "Zero-cost OpenRouter requires the dedicated_free_tier_no_byok profile."
            )
        if self.base_url != OPENROUTER_CANONICAL_BASE_URL:
            raise OpenRouterZeroCostControlPlaneError(
                "Zero-cost OpenRouter requires the official canonical OpenRouter base URL."
            )
        if self.is_free_tier is not True:
            raise OpenRouterZeroCostControlPlaneError(
                "Zero-cost OpenRouter requires a key whose data.is_free_tier is exactly true."
            )

    def matches(self, *, api_key: str | None, base_url: str | None) -> bool:
        """Return whether this attestation belongs to this exact key and origin."""
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        try:
            canonical_base_url = canonical_openrouter_base_url(base_url)
        except OpenRouterZeroCostControlPlaneError:
            return False
        return (
            canonical_base_url == self.base_url
            and _inference_key_fingerprint(api_key) == self.inference_key_fingerprint
        )


def canonical_openrouter_base_url(base_url: str | None) -> str:
    """Return the sole OpenRouter origin permitted by zero-cost mode."""
    if base_url is None:
        return OPENROUTER_CANONICAL_BASE_URL
    if base_url == OPENROUTER_CANONICAL_BASE_URL:
        return OPENROUTER_CANONICAL_BASE_URL
    raise OpenRouterZeroCostControlPlaneError(
        "Zero-cost OpenRouter rejects custom base URLs; use the official "
        "https://openrouter.ai/api/v1 origin."
    )


def is_exact_zero_cost_openrouter_model(request_model: str) -> bool:
    """Return whether an OpenRouter request model is an admitted free form."""
    if request_model == "openrouter/free":
        return True
    author_and_model = request_model.split("/")
    if len(author_and_model) != 2:
        return False
    author, model = author_and_model
    return bool(
        _OPENROUTER_SLUG_PATTERN.fullmatch(author)
        and _OPENROUTER_FREE_MODEL_PATTERN.fullmatch(model)
    )


def is_exact_zero_cost_openrouter_spec(
    raw_spec: str,
    *,
    canonical_spec: str,
    request_model: str,
) -> bool:
    """Return whether raw input is an exact canonical free OpenRouter spec."""
    if raw_spec != canonical_spec:
        return False
    model_part = raw_spec.split(",", 1)[0]
    segments = model_part.split("/")
    return (
        len(segments) == 3
        and segments[0] == "openrouter"
        and is_exact_zero_cost_openrouter_model(request_model)
    )


async def attest_openrouter_zero_cost(
    *,
    api_key: str | None,
    profile: str | None,
    base_url: str | None,
    http_client_factory: Callable[..., Any] | None = None,
) -> OpenRouterZeroCostAttestation:
    """Verify the documented free-tier key state without following redirects."""
    if profile != ZERO_COST_OPENROUTER_PROFILE:
        raise OpenRouterZeroCostControlPlaneError(
            "Zero-cost OpenRouter requires the dedicated_free_tier_no_byok profile."
        )
    if not isinstance(api_key, str) or not api_key.strip():
        raise OpenRouterZeroCostControlPlaneError(
            "Zero-cost OpenRouter requires an inference API key for the free-tier check."
        )
    canonical_base_url = canonical_openrouter_base_url(base_url)
    client_factory = http_client_factory or httpx.AsyncClient
    client = client_factory(trust_env=False, follow_redirects=False)
    try:
        response = await client.get(
            f"{canonical_base_url}/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if getattr(response, "status_code", None) != 200:
            raise OpenRouterZeroCostControlPlaneError(
                "Zero-cost OpenRouter free-tier check did not return HTTP 200."
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise OpenRouterZeroCostControlPlaneError(
                "Zero-cost OpenRouter free-tier check returned malformed JSON."
            ) from exc
    except OpenRouterZeroCostControlPlaneError:
        raise
    except Exception as exc:
        raise OpenRouterZeroCostControlPlaneError(
            "Zero-cost OpenRouter free-tier check was unavailable."
        ) from exc
    finally:
        aclose = getattr(client, "aclose", None)
        close = aclose if callable(aclose) else getattr(client, "close", None)
        if callable(close):
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Mapping) or data.get("is_free_tier") is not True:
        raise OpenRouterZeroCostControlPlaneError(
            "Zero-cost OpenRouter requires data.is_free_tier to be exactly true."
        )
    return OpenRouterZeroCostAttestation(
        profile=profile,
        base_url=canonical_base_url,
        inference_key_fingerprint=_inference_key_fingerprint(api_key),
        is_free_tier=True,
    )


def _inference_key_fingerprint(api_key: str) -> str:
    """Return a stable, non-logged binding for one inference credential."""
    return sha256(api_key.encode("utf-8")).hexdigest()
