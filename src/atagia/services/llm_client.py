"""Provider-agnostic LLM client abstractions."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
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
    component_id_for_llm_purpose,
    parse_model_spec,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

_STRICT_JSON_FALLBACK_INSTRUCTION = (
    "Return exactly one raw JSON object or array. Start with { or [. "
    "Do not include markdown fences, explanations, preambles, tags, or any text "
    "outside the JSON value. Anything outside the first JSON value will be ignored. "
    "Every item you want Atagia to consider must be represented inside the JSON fields."
)


def known_intimacy_context_metadata(
    *,
    reason: str,
    boundary: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Return sanitized metadata for already-known intimate analytical context."""
    metadata: dict[str, Any] = {
        "atagia_intimacy_context": True,
        "atagia_intimacy_context_reason": reason,
    }
    if boundary is not None:
        metadata["source_intimacy_boundary"] = boundary
    if confidence is not None:
        metadata["source_intimacy_boundary_confidence"] = float(confidence)
    return metadata


class LLMError(RuntimeError):
    """Base LLM client error."""


class LLMPolicyBlockedError(LLMError):
    """Raised when a provider refuses or blocks a request for policy reasons."""


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


class _PreOutputStreamError(RuntimeError):
    """Internal wrapper for stream errors raised before any output was emitted."""

    def __init__(self, original: LLMError) -> None:
        super().__init__(str(original))
        self.original = original


class LLMClient(Generic[T]):
    """Registry-based LLM client with retry helpers."""

    def __init__(
        self,
        provider_name: str | None = None,
        providers: list[LLMProvider] | None = None,
        retry_policy: RetryPolicy | None = None,
        allow_unqualified_single_provider_models: bool = False,
        intimacy_fallback_models: dict[str, str] | None = None,
        intimacy_proactive_routing_enabled: bool = False,
    ) -> None:
        self._provider_name = provider_name.strip().lower() if provider_name is not None else None
        self._providers = {
            provider.name.strip().lower(): provider for provider in (providers or [])
        }
        self._retry_policy = retry_policy or RetryPolicy()
        self._allow_unqualified_single_provider_models = allow_unqualified_single_provider_models
        self._intimacy_fallback_models = dict(intimacy_fallback_models or {})
        self._intimacy_proactive_routing_enabled = intimacy_proactive_routing_enabled

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name.strip().lower()] = provider

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    @property
    def provider(self) -> LLMProvider:
        return self._provider()

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
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            return await self._complete_once(proactive_request)
        try:
            return await self._complete_once(normalized_request)
        except LLMError as exc:
            fallback_request = self._intimacy_fallback_request(normalized_request, exc)
            if fallback_request is None:
                raise
            return await self._complete_once(fallback_request)

    async def _complete_once(
        self,
        request: LLMCompletionRequest,
    ) -> LLMCompletionResponse:
        provider, provider_request = self._completion_provider_request(request)
        return await self._with_retries(lambda: provider.complete(provider_request))

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        provider, provider_request = self._embedding_provider_request(request)
        return await self._with_retries(lambda: provider.embed(provider_request))

    def supports_embedding_dimensions(self, model_spec: str) -> bool:
        """Return whether the provider for an embedding model supports dimensions."""
        if self._provider_name is not None:
            return self.provider.supports_embedding_dimensions
        parsed = self._parse_model_spec(model_spec, allow_thinking=False)
        return self._provider(parsed.provider_name).supports_embedding_dimensions

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        normalized_request = request.model_copy(
            update={"max_output_tokens": apply_min_output_threshold(request.max_output_tokens)}
        )
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            async for event in self._stream_once(proactive_request):
                yield event
            return
        try:
            async for event in self._stream_once(normalized_request):
                yield event
            return
        except _PreOutputStreamError as exc:
            fallback_request = self._intimacy_fallback_request(normalized_request, exc.original)
            if fallback_request is None:
                raise exc.original from exc
            async for event in self._stream_once(fallback_request):
                yield event

    async def _stream_once(
        self,
        request: LLMCompletionRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        provider, provider_request = self._completion_provider_request(request)
        delay = self._retry_policy.base_delay_seconds
        last_error: LLMError | None = None
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
            except LLMError as exc:
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
        if last_error is None:
            raise LLMError("LLM stream failed without a captured error")
        raise _PreOutputStreamError(last_error)

    async def complete_streamed(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
    ) -> LLMCompletionResponse:
        normalized_request = request.model_copy(
            update={"max_output_tokens": apply_min_output_threshold(request.max_output_tokens)}
        )
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            return await self._complete_streamed_once(proactive_request, observer=observer)
        try:
            return await self._complete_streamed_once(normalized_request, observer=observer)
        except _PreOutputStreamError as exc:
            fallback_request = self._intimacy_fallback_request(normalized_request, exc.original)
            if fallback_request is None:
                raise exc.original from exc
            return await self._complete_streamed_once(fallback_request, observer=observer)

    async def _complete_streamed_once(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
    ) -> LLMCompletionResponse:
        provider, provider_request = self._completion_provider_request(request)
        delay = self._retry_policy.base_delay_seconds
        last_error: LLMError | None = None
        for attempt in range(1, self._retry_policy.attempts + 1):
            emitted_any = False
            output_text = ""
            thinking = ""
            tool_calls: list[dict[str, Any]] = []
            usage: dict[str, Any] = {}
            stream_iterator = provider.stream(provider_request)
            try:
                async for event in stream_iterator:
                    emitted_any = True
                    if event.type == "text" and event.content:
                        output_text += event.content
                        if observer is not None:
                            await observer.on_text(event.content, output_text, provider_request)
                    elif event.type == "thinking" and event.content:
                        thinking += event.content
                    elif event.type == "tool_call":
                        tool_calls.append(dict(event.payload))
                    elif event.type == "done":
                        event_usage = event.payload.get("usage")
                        if isinstance(event_usage, dict):
                            usage = event_usage
                return LLMCompletionResponse(
                    provider=provider.name,
                    model=str(
                        provider_request.metadata.get("atagia_model_spec")
                        or provider_request.model
                    ),
                    output_text=output_text,
                    thinking=thinking or None,
                    tool_calls=tool_calls,
                    usage=usage,
                )
            except TransientLLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                if emitted_any:
                    raise
                last_error = exc
                if attempt == self._retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_policy.max_delay_seconds)
            except LLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
            except BaseException:
                await self._close_stream_iterator(stream_iterator)
                raise
        if last_error is None:
            raise LLMError("LLM streamed completion failed without a captured error")
        raise _PreOutputStreamError(last_error)

    def _intimacy_fallback_request(
        self,
        request: LLMCompletionRequest,
        exc: LLMError,
    ) -> LLMCompletionRequest | None:
        if not self._is_policy_blocked_error(exc):
            return None
        if bool(request.metadata.get("atagia_intimacy_fallback_used")):
            return None
        if bool(request.metadata.get("atagia_intimacy_proactive_route")):
            return None

        fallback_model = self._resolve_intimacy_fallback_model(request)
        if fallback_model is None:
            return None
        if fallback_model == request.model:
            return None

        component_id = self._component_id_for_request(request)
        logger.warning(
            "Retrying LLM request with intimacy fallback model "
            "component_id=%s purpose=%s primary_model=%s fallback_model=%s error_class=%s",
            component_id or "<unknown>",
            request.metadata.get("purpose") or "<unset>",
            request.model,
            fallback_model,
            exc.__class__.__name__,
        )
        metadata = copy.deepcopy(request.metadata)
        metadata.update(
            {
                "atagia_intimacy_fallback_used": True,
                "atagia_intimacy_primary_model": request.model,
                "atagia_intimacy_primary_error_class": exc.__class__.__name__,
                "atagia_intimacy_primary_error_reason": self._safe_error_label(exc),
            }
        )
        if component_id is not None:
            metadata.setdefault("atagia_component_id", component_id)
        return request.model_copy(
            update={
                "model": fallback_model,
                "metadata": metadata,
            }
        )

    def _proactive_intimacy_request(
        self,
        request: LLMCompletionRequest,
    ) -> LLMCompletionRequest | None:
        if not self._intimacy_proactive_routing_enabled:
            return None
        if bool(request.metadata.get("atagia_intimacy_fallback_used")):
            return None
        if bool(request.metadata.get("atagia_intimacy_proactive_route")):
            return None
        if not self._metadata_indicates_known_intimacy(request.metadata):
            return None

        fallback_model = self._resolve_intimacy_fallback_model(request)
        if fallback_model is None or fallback_model == request.model:
            return None

        component_id = self._component_id_for_request(request)
        logger.info(
            "Routing LLM request directly to intimacy model "
            "component_id=%s purpose=%s primary_model=%s intimacy_model=%s",
            component_id or "<unknown>",
            request.metadata.get("purpose") or "<unset>",
            request.model,
            fallback_model,
        )
        metadata = copy.deepcopy(request.metadata)
        metadata.update(
            {
                "atagia_intimacy_proactive_route": True,
                "atagia_intimacy_primary_model": request.model,
            }
        )
        if component_id is not None:
            metadata.setdefault("atagia_component_id", component_id)
        return request.model_copy(
            update={
                "model": fallback_model,
                "metadata": metadata,
            }
        )

    def _resolve_intimacy_fallback_model(
        self,
        request: LLMCompletionRequest,
    ) -> str | None:
        explicit = request.metadata.get("atagia_intimacy_fallback_model")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        component_id = self._component_id_for_request(request)
        if component_id is None:
            return None
        return self._intimacy_fallback_models.get(component_id)

    @staticmethod
    def _component_id_for_request(request: LLMCompletionRequest) -> str | None:
        component_id = request.metadata.get("atagia_component_id")
        if isinstance(component_id, str) and component_id.strip():
            return component_id.strip()
        purpose = request.metadata.get("purpose")
        return component_id_for_llm_purpose(purpose if isinstance(purpose, str) else None)

    @classmethod
    def _metadata_indicates_known_intimacy(cls, metadata: dict[str, Any]) -> bool:
        for key in (
            "atagia_intimacy_context",
            "atagia_known_intimacy_context",
        ):
            if cls._truthy(metadata.get(key)):
                return True
        for key in (
            "intimacy_boundary",
            "source_intimacy_boundary",
            "candidate_intimacy_boundary",
        ):
            if cls._nonordinary_intimacy_boundary(metadata.get(key)):
                return True
        boundaries = metadata.get("intimacy_boundaries")
        if isinstance(boundaries, list):
            return any(cls._nonordinary_intimacy_boundary(value) for value in boundaries)
        return False

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _nonordinary_intimacy_boundary(value: Any) -> bool:
        if value is None:
            return False
        raw_value = getattr(value, "value", value)
        normalized = str(raw_value).strip().lower()
        return normalized not in {"", "ordinary", "none", "null"}

    @staticmethod
    def _is_policy_blocked_error(exc: LLMError) -> bool:
        if isinstance(exc, LLMPolicyBlockedError):
            return True
        if isinstance(exc, (OutputLimitExceededError, StructuredOutputError, TransientLLMError)):
            return False
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "blocked the response",
                "blocked the prompt",
                "content_filter",
                "finish_reason=safety",
                "finish_reason=refusal",
                "prompt:prohibited_content",
                "response:safety",
                "response:refusal",
                "stop_reason=refusal",
                "policy_refusal",
                "safety block",
            )
        )

    @staticmethod
    def _safe_error_label(exc: LLMError) -> str:
        message = str(exc).lower()
        for marker in (
            "content_filter",
            "refusal",
            "prohibited_content",
            "safety",
            "blocked",
        ):
            if marker in message:
                return marker
        return "policy_blocked"

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
        return self._validate_structured_response(response, schema, used_schema_fallback)

    async def complete_structured_streamed(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
        *,
        observer: Any | None = None,
    ) -> T:
        used_schema_fallback = False
        try:
            response = await self.complete_streamed(request, observer=observer)
        except LLMError as exc:
            if not self._should_retry_without_schema(exc, request):
                raise
            used_schema_fallback = True
            response = await self.complete_streamed(
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
                ),
                observer=observer,
            )
        return self._validate_structured_response(response, schema, used_schema_fallback)

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

    @staticmethod
    async def _close_stream_iterator(stream_iterator: Any) -> None:
        aclose = getattr(stream_iterator, "aclose", None)
        if not callable(aclose):
            return
        result = aclose()
        if hasattr(result, "__await__"):
            await result

    def _validate_structured_response(
        self,
        response: LLMCompletionResponse,
        schema: type[T],
        used_schema_fallback: bool,
    ) -> T:
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
