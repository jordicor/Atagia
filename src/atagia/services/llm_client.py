"""Provider-agnostic LLM client abstractions."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from atagia.core.llm_output_limits import apply_min_output_threshold
from atagia.services.structured_json import (
    StructuredJSONDecodeError,
    decode_structured_json_payload,
)
from atagia.services.model_profiles import MODEL_PROFILES
from atagia.services.model_resolution import (
    ModelResolutionError,
    ParsedModelSpec,
    parse_model_spec,
)

T = TypeVar("T")

_STRICT_JSON_FALLBACK_INSTRUCTION = (
    "Return exactly one raw JSON object or array. Start with { or [. "
    "Do not include markdown fences, explanations, preambles, tags, or any text "
    "outside the JSON value. Anything outside the first JSON value will be ignored. "
    "Every item you want Atagia to consider must be represented inside the JSON fields."
)


class LLMError(RuntimeError):
    """Base LLM client error."""


class ConfigurationError(LLMError):
    """Raised when the client is configured with an unsupported provider."""


class StructuredOutputError(LLMError):
    """Raised when structured output validation fails."""

    def __init__(self, message: str, *, details: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.details = details


class TransientLLMError(LLMError):
    """Raised for retryable provider failures."""


class OutputLimitExceededError(LLMError):
    """Raised when the model truncates output (finish_reason=length / stop_reason=max_tokens)."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry settings for transient provider errors."""

    attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0


class LLMMessage(BaseModel):
    """Message sent to or returned from a provider."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    name: str | None = None


class LLMToolSpec(BaseModel):
    """Portable tool declaration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class LLMCompletionRequest(BaseModel):
    """Normalized completion request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: list[LLMToolSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    include_thinking: bool = False


class LLMCompletionResponse(BaseModel):
    """Normalized completion response."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    output_text: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMEmbeddingRequest(BaseModel):
    """Normalized embedding request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input_texts: list[str]
    dimensions: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMEmbeddingVector(BaseModel):
    """Single embedding vector."""

    model_config = ConfigDict(extra="forbid")

    index: int
    values: list[float]


class LLMEmbeddingResponse(BaseModel):
    """Normalized embedding response."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    vectors: list[LLMEmbeddingVector]
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMStreamEvent(BaseModel):
    """Streaming event emitted by a provider."""

    model_config = ConfigDict(extra="forbid")

    type: str
    content: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class LLMProvider:
    """Provider adapter interface."""

    name: str
    supports_embeddings: bool = True
    supports_embedding_dimensions: bool = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise NotImplementedError

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise NotImplementedError

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.complete(request)
        if response.thinking:
            yield LLMStreamEvent(type="thinking", content=response.thinking)
        if response.output_text:
            yield LLMStreamEvent(type="text", content=response.output_text)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                yield LLMStreamEvent(type="tool_call", payload=tool_call)
        yield LLMStreamEvent(type="done", payload={"usage": response.usage})


class LLMClient(Generic[T]):
    """Registry-based LLM client with retry helpers."""

    def __init__(
        self,
        provider_name: str | None = None,
        providers: list[LLMProvider] | None = None,
        embedding_provider_name: str | None = None,
        retry_policy: RetryPolicy | None = None,
        allow_unqualified_single_provider_models: bool = False,
    ) -> None:
        self._provider_name = provider_name.strip().lower() if provider_name is not None else None
        self._embedding_provider_name = (
            embedding_provider_name.strip().lower()
            if embedding_provider_name is not None
            else None
        )
        self._providers = {
            provider.name.strip().lower(): provider for provider in (providers or [])
        }
        self._retry_policy = retry_policy or RetryPolicy()
        self._allow_unqualified_single_provider_models = allow_unqualified_single_provider_models

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name.strip().lower()] = provider

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    @property
    def embedding_provider_name(self) -> str | None:
        return self._embedding_provider_name or self._provider_name

    @property
    def provider(self) -> LLMProvider:
        return self._provider()

    @property
    def embedding_provider(self) -> LLMProvider:
        return self._provider(self.embedding_provider_name)

    def _provider(self, provider_name: str | None = None) -> LLMProvider:
        resolved_name = provider_name or self._provider_name
        if resolved_name is None:
            if len(self._providers) == 1:
                return next(iter(self._providers.values()))
            raise ConfigurationError("No LLM provider was selected for this request")
        provider = self._providers.get(resolved_name)
        if provider is None:
            raise ConfigurationError(f"Unsupported LLM provider: {resolved_name}")
        return provider

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        normalized_request = request.model_copy(
            update={"max_output_tokens": apply_min_output_threshold(request.max_output_tokens)}
        )
        provider, provider_request = self._completion_provider_request(normalized_request)
        return await self._with_retries(lambda: provider.complete(provider_request))

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        provider, provider_request = self._embedding_provider_request(request)
        return await self._with_retries(lambda: provider.embed(provider_request))

    def supports_embedding_dimensions(self, model_spec: str) -> bool:
        """Return whether the provider for an embedding model supports dimensions."""
        if self._provider_name is not None:
            return self.embedding_provider.supports_embedding_dimensions
        parsed = self._parse_model_spec(model_spec, allow_thinking=False)
        return self._provider(parsed.provider_name).supports_embedding_dimensions

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        normalized_request = request.model_copy(
            update={"max_output_tokens": apply_min_output_threshold(request.max_output_tokens)}
        )
        provider, provider_request = self._completion_provider_request(normalized_request)
        delay = self._retry_policy.base_delay_seconds
        last_error: Exception | None = None
        for attempt in range(1, self._retry_policy.attempts + 1):
            emitted_any = False
            try:
                async for event in provider.stream(provider_request):
                    emitted_any = True
                    yield event
                return
            except TransientLLMError as exc:
                if emitted_any:
                    raise
                last_error = exc
                if attempt == self._retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_policy.max_delay_seconds)
        if last_error is None:
            raise LLMError("LLM stream failed without a captured error")
        raise last_error

    def _completion_provider_request(
        self,
        request: LLMCompletionRequest,
    ) -> tuple[LLMProvider, LLMCompletionRequest]:
        if self._provider_name is not None:
            return self.provider, request
        parsed = self._parse_model_spec(request.model, allow_thinking=True)
        provider = self._provider(parsed.provider_name)
        return provider, self._completion_request_for_provider(request, parsed)

    def _embedding_provider_request(
        self,
        request: LLMEmbeddingRequest,
    ) -> tuple[LLMProvider, LLMEmbeddingRequest]:
        if self._provider_name is not None:
            return self._provider(self._provider_name), request
        parsed = self._parse_model_spec(request.model, allow_thinking=False)
        provider = self._provider(parsed.provider_name)
        metadata = dict(request.metadata)
        metadata.setdefault("atagia_model_spec", parsed.canonical_model)
        provider_request = request.model_copy(
            update={
                "model": parsed.request_model,
                "metadata": metadata,
            }
        )
        return provider, provider_request

    def _parse_model_spec(self, model: str, *, allow_thinking: bool) -> ParsedModelSpec:
        try:
            return parse_model_spec(model, allow_thinking=allow_thinking)
        except ModelResolutionError as exc:
            if self._allow_unqualified_single_provider_models and len(self._providers) == 1:
                provider = next(iter(self._providers))
                return ParsedModelSpec(
                    raw_spec=model,
                    canonical_spec=model,
                    canonical_model=model,
                    provider_slug=provider,
                    provider_name=provider,
                    request_model=model,
                    thinking_level=None,
                )
            raise ConfigurationError(str(exc)) from exc

    def _completion_request_for_provider(
        self,
        request: LLMCompletionRequest,
        parsed: ParsedModelSpec,
    ) -> LLMCompletionRequest:
        metadata = copy.deepcopy(request.metadata)
        metadata.setdefault("atagia_model_spec", parsed.canonical_spec)
        metadata.setdefault("atagia_canonical_model", parsed.canonical_model)
        metadata.setdefault("atagia_provider_slug", parsed.provider_slug)
        metadata = self._apply_model_profile(parsed, metadata)
        return request.model_copy(
            update={
                "model": parsed.request_model,
                "metadata": metadata,
            }
        )

    def _apply_model_profile(
        self,
        parsed: ParsedModelSpec,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        profile = MODEL_PROFILES.get(parsed.canonical_model)
        if profile is None:
            return metadata

        if profile.extra_kwargs:
            metadata.update(copy.deepcopy(profile.extra_kwargs))

        provider_value: str | int | None = None
        level = parsed.thinking_level or profile.default_thinking_level
        if level is not None and profile.thinking_level_map:
            if level in profile.thinking_level_map:
                provider_value = profile.thinking_level_map[level]
            elif profile.default_thinking_level in profile.thinking_level_map:
                provider_value = profile.thinking_level_map[profile.default_thinking_level]

        if parsed.provider_slug == "openai":
            if provider_value is not None:
                metadata["reasoning_effort"] = provider_value
            return metadata

        if parsed.provider_slug == "anthropic":
            if provider_value == -1:
                metadata["anthropic_thinking_adaptive"] = True
                metadata.pop("thinking_budget_tokens", None)
            elif isinstance(provider_value, int) and provider_value > 0:
                metadata["thinking_budget_tokens"] = provider_value
                metadata.pop("anthropic_thinking_adaptive", None)
            return metadata

        if parsed.provider_slug == "google":
            if provider_value is not None:
                metadata["gemini_thinking_level"] = provider_value
            return metadata

        if parsed.provider_slug == "openrouter":
            profile_body = copy.deepcopy(profile.extra_body or {})
            request_body = copy.deepcopy(metadata.get("provider_extra_body") or {})
            body = self._deep_merge_dicts(profile_body, request_body)
            if provider_value is not None:
                reasoning = body.get("reasoning")
                if not isinstance(reasoning, dict):
                    reasoning = {}
                reasoning["effort"] = provider_value
                body["reasoning"] = reasoning
            if body:
                metadata["provider_extra_body"] = body
            return metadata

        return metadata

    @classmethod
    def _deep_merge_dicts(cls, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = cls._deep_merge_dicts(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    async def complete_structured(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
    ) -> T:
        used_schema_fallback = False
        try:
            response = await self.complete(request)
        except LLMError as exc:
            if not self._should_retry_without_schema(exc, request):
                raise
            used_schema_fallback = True
            response = await self.complete(
                request.model_copy(
                    update={
                        "response_schema": None,
                        "messages": [
                            *request.messages,
                            LLMMessage(
                                role="user",
                                content=_STRICT_JSON_FALLBACK_INSTRUCTION,
                            ),
                        ],
                    }
                )
            )
        try:
            payload = self._decode_json_payload(response.output_text)
        except StructuredJSONDecodeError as exc:
            if used_schema_fallback:
                raise StructuredOutputError(
                    "Provider returned non-JSON structured output after schema fallback",
                    details=exc.details,
                ) from exc
            raise StructuredOutputError(
                "Provider returned non-JSON structured output",
                details=exc.details,
            ) from exc

        adapter = TypeAdapter(schema)
        try:
            return adapter.validate_python(payload)
        except Exception as exc:
            raise StructuredOutputError(
                "Provider returned invalid structured output",
                details=self._structured_error_details(exc),
            ) from exc

    async def _with_retries(self, operation: Any) -> Any:
        delay = self._retry_policy.base_delay_seconds
        last_error: Exception | None = None
        for attempt in range(1, self._retry_policy.attempts + 1):
            try:
                return await operation()
            except TransientLLMError as exc:
                last_error = exc
                if attempt == self._retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_policy.max_delay_seconds)
        if last_error is None:
            raise LLMError("LLM operation failed without a captured error")
        raise last_error

    @staticmethod
    def _should_retry_without_schema(exc: LLMError, request: LLMCompletionRequest) -> bool:
        if request.response_schema is None:
            return False
        message = str(exc).lower()
        return (
            "compiled grammar is too large" in message
            or "additionalproperties" in message and "must be explicitly set to false" in message
            or "maxitems" in message and "not supported" in message
        )

    @staticmethod
    def _decode_json_payload(output_text: str) -> Any:
        return decode_structured_json_payload(output_text).data

    @classmethod
    def _structured_error_details(cls, exc: Exception) -> tuple[str, ...]:
        if isinstance(exc, json.JSONDecodeError):
            return ("$: Response was not valid JSON.",)
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                normalized_errors = errors(include_url=False)
            except TypeError:
                normalized_errors = errors()
            details = [
                f"{cls._format_error_path(tuple(error.get('loc', ())))}: {error.get('msg', 'Invalid structured output')}"
                for error in normalized_errors
            ]
            if details:
                return tuple(details)
        return ("$: Structured output validation failed.",)

    @staticmethod
    def _format_error_path(location: tuple[Any, ...]) -> str:
        path = "$"
        for segment in location:
            if isinstance(segment, int):
                path += f"[{segment}]"
            else:
                path += f".{segment}"
        return path
