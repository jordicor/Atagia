"""OpenRouter provider adapter."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from atagia.services.llm_client import InferenceAccessDeniedError, LLMCompletionRequest
from atagia.services.llm_schema import (
    has_json_schema_nullability,
    strip_json_schema_nullability,
)
from atagia.services.providers.gemini import _sanitize_schema_for_gemini
from atagia.services.providers.openai import OpenAICompatibleProvider, _response_format
from atagia.services.openrouter_zero_cost import canonical_openrouter_base_url


_NATIVE_STRUCTURED_OUTPUT_VENDOR_PREFIXES = frozenset(
    {
        "anthropic",
        "google",
        "openai",
        "x-ai",
    }
)


def _openrouter_vendor_prefix(model: str) -> str | None:
    segments = [
        segment.strip().lower() for segment in model.split("/") if segment.strip()
    ]
    if len(segments) >= 2 and segments[0] == "openrouter":
        return segments[1]
    if len(segments) >= 2:
        return segments[0]
    return None


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter uses the OpenAI-compatible SDK surface."""

    name = "openrouter"
    supports_native_structured_output = False

    def __init__(
        self,
        api_key: str,
        *,
        site_url: str,
        app_name: str,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        client: AsyncOpenAI | None = None,
        zero_cost: bool = False,
    ) -> None:
        self._zero_cost = zero_cost
        resolved_base_url = (
            canonical_openrouter_base_url(base_url)
            if zero_cost
            else base_url or "https://openrouter.ai/api/v1"
        )
        super().__init__(
            api_key=api_key,
            base_url=resolved_base_url,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
            request_timeout_seconds=request_timeout_seconds,
            client=client,
            restricted_transport=zero_cost,
        )

    def supports_native_structured_output_for(
        self, request: LLMCompletionRequest
    ) -> bool:
        if request.metadata.get("openrouter_native_structured_output") is True:
            return True
        vendor = _openrouter_vendor_prefix(request.model)
        return vendor in _NATIVE_STRUCTURED_OUTPUT_VENDOR_PREFIXES

    def _completion_response_format(
        self,
        request: LLMCompletionRequest,
    ) -> dict[str, Any] | None:
        if not self.supports_native_structured_output_for(request):
            return None
        vendor = _openrouter_vendor_prefix(request.model)
        if vendor == "openai":
            return _response_format(request.response_schema, preserve_nullability=True)
        if vendor == "google" and request.response_schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": False,
                    "schema": _sanitize_schema_for_gemini(request.response_schema),
                },
            }
        if request.response_schema is not None and has_json_schema_nullability(
            request.response_schema
        ):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": False,
                    "schema": strip_json_schema_nullability(request.response_schema),
                },
            }
        return super()._completion_response_format(request)

    def _completion_kwargs(
        self, request: LLMCompletionRequest, *, stream: bool
    ) -> dict[str, Any]:
        kwargs = super()._completion_kwargs(request, stream=stream)
        if self._zero_cost:
            if "provider_extra_body" in request.metadata:
                raise InferenceAccessDeniedError(
                    "Zero-cost OpenRouter does not permit caller-supplied provider_extra_body."
                )
            if request.tools:
                raise InferenceAccessDeniedError(
                    "Zero-cost OpenRouter does not permit tools."
                )
            provider_preferences: dict[str, Any] = {
                "allow_fallbacks": False,
                "max_price": {
                    "prompt": 0,
                    "completion": 0,
                    "request": 0,
                    "image": 0,
                },
            }
            if "response_format" in kwargs:
                provider_preferences["require_parameters"] = True
            kwargs["extra_body"] = {"provider": provider_preferences}
            return kwargs
        date_resolution = request.metadata.get("purpose") == "memory_date_resolution"
        if "response_format" not in kwargs and not date_resolution:
            return kwargs

        extra_body = dict(kwargs.get("extra_body") or {})
        raw_provider_preferences = extra_body.get("provider")
        provider_preferences = (
            dict(raw_provider_preferences)
            if isinstance(raw_provider_preferences, dict)
            else {}
        )
        if "response_format" in kwargs:
            provider_preferences["require_parameters"] = True
        if date_resolution:
            provider_preferences["allow_fallbacks"] = False
        extra_body["provider"] = provider_preferences
        kwargs["extra_body"] = extra_body
        return kwargs
