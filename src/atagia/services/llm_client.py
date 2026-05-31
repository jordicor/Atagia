"""Provider-agnostic LLM client abstractions."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, AsyncIterator, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from atagia.core.llm_output_limits import apply_min_output_threshold
from atagia.services.structured_json import (
    StructuredJSONDecodeError,
    decode_structured_json_payload,
    render_compact_schema_spec,
)
from atagia.services.model_profiles import MODEL_PROFILES
from atagia.services.model_resolution import (
    ModelResolutionError,
    ParsedModelSpec,
    component_id_for_llm_purpose,
    parse_model_spec,
)
from atagia.services.llm_run_guard import (
    LLMRunGuard,
    LLMRunGuardConfig,
    LLMRunGuardDecision,
)
from atagia.services.llm_reliability import (
    LLMRunawayAbort,
    LLMTechnicalRecoveryConfig,
    TechnicalRunawayObserver,
    compose_stream_observers,
)
from atagia.services.llm_temperature import (
    MIN_COMPLETION_TEMPERATURE,
    purpose_temperature,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

_STRICT_JSON_FALLBACK_INSTRUCTION = (
    "Return exactly one raw JSON object or array. Start with { or [. "
    "Do not include markdown fences, explanations, preambles, tags, or any text "
    "outside the JSON value. Anything outside the first JSON value will be ignored. "
    "Every item you want Atagia to consider must be represented inside the JSON fields."
)
_STRUCTURED_OUTPUT_REPAIR_MAX_DETAILS = 8
_STRUCTURED_OUTPUT_REPAIR_MAX_OUTPUT_CHARS = 4000
_TECHNICAL_OUTPUT_LIMIT_RETRY_INSTRUCTION = (
    "Your previous generation for this same request was stopped by a provider "
    "output limit or technical runaway-output watchdog. Regenerate the complete "
    "answer for the original task, but keep it concise enough to finish within "
    "the requested output budget. Do not mention this retry or the technical "
    "failure. If the task requires JSON or structured output, return exactly one "
    "complete valid JSON value and no extra prose."
)
_TECHNICAL_RECOVERY_EXCERPT_CHARS = 1200


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

    def __init__(
        self,
        message: str,
        *,
        details: tuple[str, ...] = (),
        output_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.output_text = output_text


class TransientLLMError(LLMError):
    """Raised for retryable provider failures."""


class LLMRequestError(LLMError):
    """Raised for non-transient client-request-class provider errors (HTTP 4xx)."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class OutputLimitExceededError(LLMError):
    """Raised when the model truncates output (finish_reason=length / stop_reason=max_tokens)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        finish_reason: str | None = None,
        max_output_tokens: int | None = None,
        partial_output_chars: int | None = None,
        partial_output_excerpt: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.finish_reason = finish_reason
        self.max_output_tokens = max_output_tokens
        self.partial_output_chars = partial_output_chars
        self.partial_output_excerpt = partial_output_excerpt
        self.diagnostics = dict(diagnostics or {})


class LLMRunGuardError(LLMError):
    """Raised when runtime LLM budget or health thresholds are exceeded."""

    def __init__(self, decision: LLMRunGuardDecision) -> None:
        self.decision = decision
        message = "; ".join(decision.violations) or "LLM run guard blocked the request"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry settings for transient provider errors."""

    attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0


# Interactive retrieval gates degrade gracefully (e.g. need detection falls back
# to base search), so they should not pay long backoff inside a live turn. These
# are the exact `request.metadata["purpose"]` strings used by those stages.
INTERACTIVE_RETRIEVAL_PURPOSES: frozenset[str] = frozenset(
    {
        "need_detection",
        "need_detection_anchor_review",
        "need_detection_multi_facet_exact_review",
        "need_detection_degraded_exact_contract_review",
        "need_detection_unknown_only_contract_review",
        "applicability_scoring",
        "context_cache_signal_detection",
        "coverage_expansion",
    }
)

_DEFAULT_INTERACTIVE_RETRY_POLICY = RetryPolicy(attempts=2, max_delay_seconds=1.5)


@dataclass(frozen=True, slots=True)
class _ResolvedTemperature:
    """Temperature chosen for one provider-bound completion request."""

    value: float | None
    source: str
    requested: float | None = None
    reason: str | None = None


class LLMMessage(BaseModel):
    """Message sent to or returned from a provider."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


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


@dataclass(frozen=True, slots=True)
class StructuredCompletionResult(Generic[T]):
    """Structured completion payload plus the raw provider response."""

    value: T
    response: LLMCompletionResponse
    used_schema_fallback: bool = False
    used_structured_output_retry: bool = False
    used_structured_output_rescue: bool = False


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
    supports_native_structured_output: bool = True

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

    def supports_native_structured_output_for(self, request: LLMCompletionRequest) -> bool:
        """Return whether this provider can enforce the schema for this request."""
        return self.supports_native_structured_output


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
        interactive_retry_policy: RetryPolicy | None = None,
        allow_unqualified_single_provider_models: bool = False,
        intimacy_fallback_models: dict[str, str] | None = None,
        intimacy_proactive_routing_enabled: bool = False,
        structured_output_retry_attempts: int = 1,
        structured_output_rescue_enabled: bool = False,
        structured_output_rescue_model: str | None = None,
        technical_recovery_config: LLMTechnicalRecoveryConfig | None = None,
        llm_run_guard: LLMRunGuard | None = None,
    ) -> None:
        self._provider_name = provider_name.strip().lower() if provider_name is not None else None
        self._providers = {provider.name.strip().lower(): provider for provider in (providers or [])}
        self._retry_policy = retry_policy or RetryPolicy()
        self._interactive_retry_policy = (
            interactive_retry_policy or _DEFAULT_INTERACTIVE_RETRY_POLICY
        )
        self._allow_unqualified_single_provider_models = allow_unqualified_single_provider_models
        self._intimacy_fallback_models = dict(intimacy_fallback_models or {})
        self._intimacy_proactive_routing_enabled = intimacy_proactive_routing_enabled
        if structured_output_retry_attempts < 0:
            raise ValueError("structured_output_retry_attempts must be non-negative")
        self._structured_output_retry_attempts = structured_output_retry_attempts
        self._structured_output_rescue_enabled = structured_output_rescue_enabled
        rescue_model = structured_output_rescue_model.strip() if structured_output_rescue_model else None
        self._structured_output_rescue_model = rescue_model or None
        self._technical_recovery_config = (
            technical_recovery_config or LLMTechnicalRecoveryConfig.default_enabled()
        )
        self._llm_run_guard = llm_run_guard

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name.strip().lower()] = provider

    @property
    def llm_run_guard(self) -> LLMRunGuard | None:
        """Return the optional runtime LLM guard used by this client."""
        return self._llm_run_guard

    def llm_run_guard_snapshot(self) -> dict[str, Any] | None:
        """Return a JSON-safe runtime LLM guard snapshot for admin surfaces."""
        if self._llm_run_guard is None:
            return None
        return self._llm_run_guard.runtime_snapshot()

    def reset_llm_run_guard(self) -> dict[str, Any] | None:
        """Reset process-wide LLM guard counters after operator intervention."""
        if self._llm_run_guard is None:
            return None
        return self._llm_run_guard.reset_runtime()

    def llm_run_guard_scope(
        self,
        *,
        run_id: str,
        kind: str,
        config: LLMRunGuardConfig | None = None,
    ) -> Any:
        """Return a context manager applying a scoped LLM budget to this task."""
        if self._llm_run_guard is None:
            from contextlib import nullcontext

            return nullcontext(None)
        return self._llm_run_guard.maybe_scoped_run(
            run_id=run_id,
            kind=kind,
            config=config,
        )

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
        return await self._with_output_limit_recovery(
            normalized_request,
            self._complete_with_intimacy_routing,
            operation_name="completion",
        )

    async def _complete_with_intimacy_routing(
        self,
        normalized_request: LLMCompletionRequest,
    ) -> LLMCompletionResponse:
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
        return await self._with_retries(
            lambda: provider.complete(provider_request),
            request=provider_request,
            call_type="completion",
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        provider, provider_request = self._embedding_provider_request(request)
        return await self._with_retries(
            lambda: provider.embed(provider_request),
            request=provider_request,
            call_type="embedding",
        )

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
        *,
        observer: Any | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        provider, provider_request = self._completion_provider_request(request)
        observer = self._stream_observer(observer)
        retry_policy = self._retry_policy_for(provider_request)
        delay = retry_policy.base_delay_seconds
        last_error: LLMError | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            emitted_any = False
            output_text = ""
            usage: dict[str, Any] = {}
            stream_iterator = provider.stream(provider_request)
            started_at = perf_counter()
            self._guard_before_call(provider_request, call_type="stream")
            try:
                async for event in stream_iterator:
                    emitted_any = True
                    if event.type == "text" and event.content:
                        output_text += event.content
                        if observer is not None:
                            await observer.on_text(event.content, output_text, provider_request)
                    if event.type == "done" and isinstance(event.payload.get("usage"), dict):
                        usage = dict(event.payload["usage"])
                    yield event
                self._guard_record_success(
                    provider_request,
                    call_type="stream",
                    provider=provider.name,
                    response_model=str(
                        provider_request.metadata.get("atagia_model_spec")
                        or provider_request.model
                    ),
                    usage=usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                return
            except LLMRunawayAbort as exc:
                await self._close_stream_iterator(stream_iterator)
                output_limit_error = self._runaway_abort_error(exc, provider_request)
                self._guard_record_failure(
                    provider_request,
                    call_type="stream",
                    exc=output_limit_error,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise output_limit_error from exc
                raise _PreOutputStreamError(output_limit_error) from exc
            except TransientLLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._guard_record_failure(
                    provider_request,
                    call_type="stream",
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, retry_policy.max_delay_seconds)
            except LLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._guard_record_failure(
                    provider_request,
                    call_type="stream",
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
            except BaseException:
                await self._close_stream_iterator(stream_iterator)
                raise
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
        return await self._with_output_limit_recovery(
            normalized_request,
            lambda retry_request: self._complete_streamed_with_intimacy_routing(
                retry_request,
                observer=observer,
            ),
            operation_name="streamed_completion",
            retry_observer=observer,
        )

    async def _complete_streamed_with_intimacy_routing(
        self,
        normalized_request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
    ) -> LLMCompletionResponse:
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
        observer = self._stream_observer(observer)
        retry_policy = self._retry_policy_for(provider_request)
        delay = retry_policy.base_delay_seconds
        last_error: LLMError | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            emitted_any = False
            output_text = ""
            thinking = ""
            tool_calls: list[dict[str, Any]] = []
            usage: dict[str, Any] = {}
            stream_iterator = provider.stream(provider_request)
            started_at = perf_counter()
            self._guard_before_call(provider_request, call_type="streamed_completion")
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
                response = LLMCompletionResponse(
                    provider=provider.name,
                    model=str(provider_request.metadata.get("atagia_model_spec") or provider_request.model),
                    output_text=output_text,
                    thinking=thinking or None,
                    tool_calls=tool_calls,
                    usage=usage,
                )
                self._guard_record_success(
                    provider_request,
                    call_type="streamed_completion",
                    provider=response.provider,
                    response_model=response.model,
                    usage=response.usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                return response
            except LLMRunawayAbort as exc:
                await self._close_stream_iterator(stream_iterator)
                output_limit_error = self._runaway_abort_error(exc, provider_request)
                self._guard_record_failure(
                    provider_request,
                    call_type="streamed_completion",
                    exc=output_limit_error,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise output_limit_error from exc
                raise _PreOutputStreamError(output_limit_error) from exc
            except TransientLLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._guard_record_failure(
                    provider_request,
                    call_type="streamed_completion",
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, retry_policy.max_delay_seconds)
            except LLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._guard_record_failure(
                    provider_request,
                    call_type="streamed_completion",
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
            except BaseException:
                await self._close_stream_iterator(stream_iterator)
                raise
        if last_error is None:
            raise LLMError("LLM streamed completion failed without a captured error")
        raise _PreOutputStreamError(last_error)

    async def _with_output_limit_recovery(
        self,
        request: LLMCompletionRequest,
        operation: Any,
        *,
        operation_name: str,
        allow_recovery: bool = True,
        retry_observer: Any | None = None,
    ) -> LLMCompletionResponse:
        attempts = (
            self._technical_recovery_config.output_limit_retry_attempts
            if self._should_retry_output_limit(request, allow_recovery=allow_recovery)
            else 0
        )
        current_request = request
        last_error: OutputLimitExceededError | None = None
        for recovery_attempt in range(0, attempts + 1):
            try:
                response = await operation(current_request)
            except OutputLimitExceededError as exc:
                last_error = exc
                if recovery_attempt >= attempts:
                    logger.error(
                        "LLM technical recovery exhausted operation=%s purpose=%s "
                        "model=%s attempts=%s finish_reason=%s partial_output_chars=%s "
                        "diagnostics=%s",
                        operation_name,
                        request.metadata.get("purpose") or "<unset>",
                        request.model,
                        attempts,
                        exc.finish_reason,
                        exc.partial_output_chars,
                        exc.diagnostics,
                    )
                    raise
                await self._reset_stream_observer_for_retry(retry_observer, exc)
                current_request = self._technical_recovery_request(
                    request,
                    exc,
                    retry_attempt=recovery_attempt + 1,
                    operation_name=operation_name,
                )
                logger.warning(
                    "Retrying LLM request after technical output-limit failure "
                    "operation=%s purpose=%s model=%s retry_attempt=%s "
                    "finish_reason=%s partial_output_chars=%s",
                    operation_name,
                    request.metadata.get("purpose") or "<unset>",
                    request.model,
                    recovery_attempt + 1,
                    exc.finish_reason,
                    exc.partial_output_chars,
                )
                continue
            if recovery_attempt == 0:
                return response
            return self._technical_recovery_response(
                response,
                primary_request=request,
                retry_request=current_request,
                retry_attempt=recovery_attempt,
                error=last_error,
                operation_name=operation_name,
            )
        if last_error is None:
            raise LLMError("LLM technical recovery failed without a captured error")
        raise last_error

    def _should_retry_output_limit(
        self,
        request: LLMCompletionRequest,
        *,
        allow_recovery: bool,
    ) -> bool:
        if not allow_recovery:
            return False
        if not self._technical_recovery_config.output_limit_retries_enabled():
            return False
        strategy = request.metadata.get("atagia_technical_recovery_output_limit_strategy")
        if isinstance(strategy, str) and strategy.strip().lower() == "caller":
            return False
        enabled = request.metadata.get("atagia_technical_recovery_output_limit_retry")
        if isinstance(enabled, bool) and not enabled:
            return False
        return True

    def _technical_recovery_request(
        self,
        request: LLMCompletionRequest,
        error: OutputLimitExceededError,
        *,
        retry_attempt: int,
        operation_name: str,
    ) -> LLMCompletionRequest:
        metadata = copy.deepcopy(request.metadata)
        metadata.update(
            {
                "atagia_technical_recovery_retry": True,
                "atagia_technical_recovery_retry_attempt": retry_attempt,
                "atagia_technical_recovery_operation": operation_name,
                "atagia_technical_recovery_primary_model": request.model,
                "atagia_technical_recovery_failure_class": error.__class__.__name__,
                "atagia_technical_recovery_finish_reason": error.finish_reason,
                "atagia_technical_recovery_partial_output_chars": (
                    error.partial_output_chars
                ),
            }
        )
        return request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    LLMMessage(
                        role="user",
                        content=_TECHNICAL_OUTPUT_LIMIT_RETRY_INSTRUCTION,
                    ),
                ],
                "metadata": metadata,
            }
        )

    @staticmethod
    def _technical_recovery_response(
        response: LLMCompletionResponse,
        *,
        primary_request: LLMCompletionRequest,
        retry_request: LLMCompletionRequest,
        retry_attempt: int,
        error: OutputLimitExceededError | None,
        operation_name: str,
    ) -> LLMCompletionResponse:
        raw_response = copy.deepcopy(response.raw_response)
        raw_response["atagia_technical_recovery"] = {
            "operation": operation_name,
            "primary_model": primary_request.model,
            "retry_model": retry_request.model,
            "retry_attempt": retry_attempt,
            "failure_class": error.__class__.__name__ if error is not None else None,
            "finish_reason": error.finish_reason if error is not None else None,
            "partial_output_chars": (
                error.partial_output_chars if error is not None else None
            ),
            "diagnostics": error.diagnostics if error is not None else {},
        }
        return response.model_copy(update={"raw_response": raw_response})

    def _stream_observer(self, observer: Any | None) -> Any | None:
        if not self._technical_recovery_config.runaway_detection_enabled():
            return observer
        return compose_stream_observers(
            observer,
            TechnicalRunawayObserver(self._technical_recovery_config),
        )

    @staticmethod
    async def _reset_stream_observer_for_retry(
        observer: Any | None,
        error: OutputLimitExceededError,
    ) -> None:
        if observer is None:
            return
        reset = getattr(observer, "reset_for_retry", None)
        if not callable(reset):
            return
        result = reset(error)
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def _runaway_abort_error(
        exc: LLMRunawayAbort,
        request: LLMCompletionRequest,
    ) -> OutputLimitExceededError:
        partial_text = exc.accumulated_text[-_TECHNICAL_RECOVERY_EXCERPT_CHARS:]
        return OutputLimitExceededError(
            "Technical watchdog detected runaway LLM output",
            provider=str(request.metadata.get("atagia_provider_slug") or "atagia"),
            finish_reason="technical_runaway_watchdog",
            max_output_tokens=request.max_output_tokens,
            partial_output_chars=len(exc.accumulated_text),
            partial_output_excerpt=partial_text,
            diagnostics=exc.signals.to_diagnostics(),
        )

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
            "Retrying LLM request with intimacy fallback model component_id=%s purpose=%s primary_model=%s fallback_model=%s error_class=%s",
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
            "Routing LLM request directly to intimacy model component_id=%s purpose=%s primary_model=%s intimacy_model=%s",
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
        temperature = self._resolve_completion_temperature(
            parsed,
            request.temperature,
            metadata,
        )
        self._record_temperature_metadata(metadata, temperature)
        return request.model_copy(
            update={
                "model": parsed.request_model,
                "metadata": metadata,
                "temperature": temperature.value,
            }
        )

    def _resolve_completion_temperature(
        self,
        parsed: ParsedModelSpec,
        temperature: float | None,
        metadata: dict[str, Any],
    ) -> _ResolvedTemperature:
        profile = MODEL_PROFILES.get(parsed.canonical_model)
        requested = temperature
        reason: str | None = None
        source = "unset"
        if temperature is not None:
            value = float(temperature)
            source = "request"
        elif profile is not None and profile.temperature_default is not None:
            value = float(profile.temperature_default)
            source = "model_profile_default"
            reason = parsed.canonical_model
        else:
            policy = purpose_temperature(metadata.get("purpose"))
            if policy is None:
                return _ResolvedTemperature(
                    value=None,
                    source=source,
                    requested=requested,
                )
            value = float(policy.value)
            source = "purpose_default"
            reason = policy.reason

        if value < MIN_COMPLETION_TEMPERATURE:
            value = MIN_COMPLETION_TEMPERATURE
            source = f"{source}+minimum_floor"
        if profile is not None and profile.temperature_floor is not None:
            floor = float(profile.temperature_floor)
            if value < floor:
                value = floor
                source = f"{source}+model_floor"
                reason = parsed.canonical_model
        return _ResolvedTemperature(
            value=value,
            source=source,
            requested=requested,
            reason=reason,
        )

    @staticmethod
    def _record_temperature_metadata(
        metadata: dict[str, Any],
        resolved: _ResolvedTemperature,
    ) -> None:
        metadata["atagia_temperature_source"] = resolved.source
        if resolved.requested is not None:
            metadata["atagia_requested_temperature"] = resolved.requested
        if resolved.value is not None:
            metadata["atagia_effective_temperature"] = resolved.value
        if resolved.reason is not None:
            metadata["atagia_temperature_reason"] = resolved.reason

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
            metadata = self._apply_profile_extra_body(profile, metadata)
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
                metadata.pop("anthropic_output_effort", None)
            elif isinstance(provider_value, str):
                metadata["anthropic_thinking_adaptive"] = True
                metadata["anthropic_output_effort"] = provider_value
                metadata.pop("thinking_budget_tokens", None)
            return metadata

        if parsed.provider_slug == "google":
            if provider_value is not None:
                metadata["gemini_thinking_level"] = provider_value
            return metadata

        if parsed.provider_slug == "openrouter":
            metadata = self._apply_profile_extra_body(profile, metadata)
            body = copy.deepcopy(metadata.get("provider_extra_body") or {})
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

    def _apply_profile_extra_body(
        self,
        profile: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        profile_body = copy.deepcopy(profile.extra_body or {})
        request_body = copy.deepcopy(metadata.get("provider_extra_body") or {})
        body = self._deep_merge_dicts(profile_body, request_body)
        if body:
            metadata["provider_extra_body"] = body
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
        return (await self.complete_structured_with_response(request, schema)).value

    async def complete_structured_with_response(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
    ) -> StructuredCompletionResult[T]:
        try:
            return await self._complete_structured_once(request, schema)
        except StructuredOutputError as exc:
            initial_error = exc
            last_error = exc

        retry_used = False

        for retry_attempt in range(1, self._structured_output_retry_attempts + 1):
            retry_used = True
            retry_request = self._structured_output_repair_request(
                request,
                schema,
                last_error,
                retry_attempt=retry_attempt,
            )
            try:
                result = await self._complete_structured_once(retry_request, schema)
            except StructuredOutputError as exc:
                last_error = exc
                continue
            return StructuredCompletionResult(
                value=result.value,
                response=self._structured_output_repair_response(
                    result.response,
                    repair_kind="retry",
                    primary_model=request.model,
                    repair_model=retry_request.model,
                    retry_attempts=retry_attempt,
                ),
                used_schema_fallback=result.used_schema_fallback,
                used_structured_output_retry=True,
            )

        rescue_request = self._structured_output_rescue_request(request, schema, last_error)
        if rescue_request is None:
            raise last_error from initial_error

        try:
            result = await self._complete_structured_once(rescue_request, schema)
        except StructuredOutputError as exc:
            raise exc from last_error
        return StructuredCompletionResult(
            value=result.value,
            response=self._structured_output_repair_response(
                result.response,
                repair_kind="rescue",
                primary_model=request.model,
                repair_model=rescue_request.model,
                retry_attempts=self._structured_output_retry_attempts,
            ),
            used_schema_fallback=result.used_schema_fallback,
            used_structured_output_retry=retry_used,
            used_structured_output_rescue=True,
        )

    async def _complete_structured_once(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
    ) -> StructuredCompletionResult[T]:
        completion_request = request
        used_schema_fallback = False
        if self._should_prompt_for_structured_json(request):
            completion_request = self._schema_prompt_fallback_request(request)
            used_schema_fallback = True
        try:
            response = await self.complete(completion_request)
        except LLMError as exc:
            if not self._should_retry_without_schema(exc, completion_request):
                raise
            used_schema_fallback = True
            response = await self.complete(self._schema_drop_fallback_request(request, exc))
        value = self._validate_structured_response(response, schema, used_schema_fallback)
        return StructuredCompletionResult(
            value=value,
            response=response,
            used_schema_fallback=used_schema_fallback,
        )

    async def complete_structured_streamed(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
        *,
        observer: Any | None = None,
    ) -> T:
        try:
            return await self._complete_structured_streamed_once(
                request,
                schema,
                observer=observer,
            )
        except StructuredOutputError as exc:
            initial_error = exc
            last_error = exc

        for retry_attempt in range(1, self._structured_output_retry_attempts + 1):
            retry_request = self._structured_output_repair_request(
                request,
                schema,
                last_error,
                retry_attempt=retry_attempt,
            )
            try:
                return await self._complete_structured_streamed_once(
                    retry_request,
                    schema,
                    observer=observer,
                )
            except StructuredOutputError as exc:
                last_error = exc

        rescue_request = self._structured_output_rescue_request(request, schema, last_error)
        if rescue_request is None:
            raise last_error from initial_error
        try:
            return (await self._complete_structured_once(rescue_request, schema)).value
        except StructuredOutputError as exc:
            raise exc from last_error

    async def _complete_structured_streamed_once(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
        *,
        observer: Any | None = None,
    ) -> T:
        completion_request = request
        used_schema_fallback = False
        if self._should_prompt_for_structured_json(request):
            completion_request = self._schema_prompt_fallback_request(request)
            used_schema_fallback = True
        try:
            response = await self.complete_streamed(completion_request, observer=observer)
        except LLMError as exc:
            if not self._should_retry_without_schema(exc, completion_request):
                raise
            used_schema_fallback = True
            response = await self.complete_streamed(
                self._schema_drop_fallback_request(request, exc),
                observer=observer,
            )
        return self._validate_structured_response(response, schema, used_schema_fallback)

    def _structured_output_repair_request(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
        error: StructuredOutputError,
        *,
        retry_attempt: int,
    ) -> LLMCompletionRequest:
        metadata = copy.deepcopy(request.metadata)
        metadata.update(
            {
                "atagia_structured_output_retry": True,
                "atagia_structured_output_retry_attempt": retry_attempt,
                "atagia_structured_output_retry_primary_model": request.model,
                "atagia_structured_output_failure_class": error.__class__.__name__,
            }
        )
        resolved_schema = request.response_schema or self._json_schema_for(schema)
        return request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    LLMMessage(
                        role="user",
                        content=self._structured_output_repair_instruction(
                            error,
                            phase="retry",
                            schema=resolved_schema,
                        ),
                    ),
                ],
                "metadata": metadata,
                "response_schema": resolved_schema,
            }
        )

    def _structured_output_rescue_request(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
        error: StructuredOutputError,
    ) -> LLMCompletionRequest | None:
        if not self._structured_output_rescue_enabled:
            return None
        if self._structured_output_rescue_model is None:
            return None

        metadata = copy.deepcopy(request.metadata)
        metadata.update(
            {
                "atagia_structured_output_rescue": True,
                "atagia_structured_output_rescue_model": self._structured_output_rescue_model,
                "atagia_structured_output_rescue_original_model": request.model,
                "atagia_structured_output_rescue_retry_attempts": self._structured_output_retry_attempts,
                "atagia_structured_output_failure_class": error.__class__.__name__,
            }
        )
        logger.warning(
            "Escalating structured-output repair to rescue model purpose=%s primary_model=%s rescue_model=%s retry_attempts=%s",
            request.metadata.get("purpose") or "<unset>",
            request.model,
            self._structured_output_rescue_model,
            self._structured_output_retry_attempts,
        )
        resolved_schema = request.response_schema or self._json_schema_for(schema)
        return request.model_copy(
            update={
                "model": self._structured_output_rescue_model,
                "messages": [
                    *request.messages,
                    LLMMessage(
                        role="user",
                        content=self._structured_output_repair_instruction(
                            error,
                            phase="rescue",
                            schema=resolved_schema,
                        ),
                    ),
                ],
                "metadata": metadata,
                "response_schema": resolved_schema,
            }
        )

    @staticmethod
    def _json_schema_for(schema: type[Any]) -> dict[str, Any]:
        return TypeAdapter(schema).json_schema()

    @classmethod
    def _structured_output_repair_instruction(
        cls,
        error: StructuredOutputError,
        *,
        phase: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        if phase == "rescue":
            opening = (
                "The primary model and its corrective retry failed this structured-output task. "
                "You are the configured rescue model for the same original task."
            )
        else:
            opening = (
                "Your previous response for this structured-output task did not satisfy "
                "Atagia's JSON contract."
            )
        details = cls._structured_output_error_details_for_prompt(error)
        output_excerpt = cls._structured_output_excerpt_for_prompt(error.output_text)
        parts = [
            opening,
            "Regenerate the complete answer for the original task. Do not only patch the broken JSON.",
            "Validation errors:",
            details,
        ]
        if output_excerpt:
            parts.extend(
                [
                    "Previous output excerpt:",
                    output_excerpt,
                ]
            )
        parts.append(cls._schema_prompt_fallback_instruction(schema))
        return "\n\n".join(parts)

    @staticmethod
    def _structured_output_error_details_for_prompt(error: StructuredOutputError) -> str:
        if not error.details:
            return "- Structured output validation failed."
        lines = [
            f"- {detail}"
            for detail in error.details[:_STRUCTURED_OUTPUT_REPAIR_MAX_DETAILS]
        ]
        remaining = len(error.details) - len(lines)
        if remaining > 0:
            lines.append(f"- ... {remaining} additional validation issue(s) omitted.")
        return "\n".join(lines)

    @staticmethod
    def _structured_output_excerpt_for_prompt(output_text: str | None) -> str:
        if not output_text:
            return ""
        text = output_text.strip()
        if not text:
            return ""
        if len(text) > _STRUCTURED_OUTPUT_REPAIR_MAX_OUTPUT_CHARS:
            text = text[:_STRUCTURED_OUTPUT_REPAIR_MAX_OUTPUT_CHARS] + "\n...[truncated]"
        return text

    @staticmethod
    def _structured_output_repair_response(
        response: LLMCompletionResponse,
        *,
        repair_kind: str,
        primary_model: str,
        repair_model: str,
        retry_attempts: int,
    ) -> LLMCompletionResponse:
        raw_response = copy.deepcopy(response.raw_response)
        raw_response["atagia_structured_output_repair"] = {
            "kind": repair_kind,
            "primary_model": primary_model,
            "repair_model": repair_model,
            "retry_attempts": retry_attempts,
        }
        return response.model_copy(update={"raw_response": raw_response})

    def _should_prompt_for_structured_json(self, request: LLMCompletionRequest) -> bool:
        if request.response_schema is None:
            return False
        provider, provider_request = self._completion_provider_request(request)
        return not provider.supports_native_structured_output_for(provider_request)

    @classmethod
    def _schema_prompt_fallback_request(cls, request: LLMCompletionRequest) -> LLMCompletionRequest:
        return request.model_copy(
            update={
                "response_schema": None,
                "messages": [
                    *request.messages,
                    LLMMessage(
                        role="user",
                        content=cls._schema_prompt_fallback_instruction(request.response_schema),
                    ),
                ],
            }
        )

    @staticmethod
    def _schema_prompt_fallback_instruction(schema: dict[str, Any] | None) -> str:
        if not schema:
            return _STRICT_JSON_FALLBACK_INSTRUCTION
        spec = render_compact_schema_spec(schema)
        if not spec:
            return _STRICT_JSON_FALLBACK_INSTRUCTION
        return (
            f"{_STRICT_JSON_FALLBACK_INSTRUCTION}\n\n"
            f"The JSON object must follow this structure "
            f"(field name, type, enum values, required/optional):\n{spec}"
        )

    @classmethod
    def _schema_drop_fallback_request(
        cls,
        request: LLMCompletionRequest,
        exc: LLMRequestError,
    ) -> LLMCompletionRequest:
        """Build the prompt-JSON fallback after a typed native-schema 4xx failure.

        The schema is dropped and the F0.1 compact spec is appended to the
        instruction so the model still knows the expected fields. The fallback
        reason is recorded in metadata, matching the existing retry-trace pattern.
        """
        fallback = cls._schema_prompt_fallback_request(request)
        metadata = copy.deepcopy(fallback.metadata)
        metadata.update(
            {
                "atagia_structured_output_schema_drop_fallback": True,
                "atagia_structured_output_schema_drop_reason": "client_request_error_4xx",
                "atagia_structured_output_schema_drop_status_code": exc.status_code,
                "atagia_structured_output_schema_drop_error_class": exc.__class__.__name__,
            }
        )
        logger.warning(
            "Retrying structured output via prompt-JSON fallback after native-schema "
            "client error purpose=%s model=%s status_code=%s",
            request.metadata.get("purpose") or "<unset>",
            request.model,
            exc.status_code,
        )
        return fallback.model_copy(update={"metadata": metadata})

    async def _with_retries(
        self,
        operation: Any,
        *,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        call_type: str,
    ) -> Any:
        retry_policy = self._retry_policy_for(request)
        delay = retry_policy.base_delay_seconds
        last_error: Exception | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            started_at = perf_counter()
            self._guard_before_call(request, call_type=call_type)
            try:
                response = await operation()
            except TransientLLMError as exc:
                self._guard_record_failure(
                    request,
                    call_type=call_type,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, retry_policy.max_delay_seconds)
            except Exception as exc:
                self._guard_record_failure(
                    request,
                    call_type=call_type,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                raise
            else:
                self._guard_record_success(
                    request,
                    call_type=call_type,
                    provider=getattr(response, "provider", None),
                    response_model=getattr(response, "model", None),
                    usage=getattr(response, "usage", None),
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                return response
        if last_error is None:
            raise LLMError("LLM operation failed without a captured error")
        raise last_error

    def _guard_before_call(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        *,
        call_type: str,
    ) -> None:
        if self._llm_run_guard is None:
            return
        decision = self._llm_run_guard.check_before_call(
            call_type=call_type,
            purpose=self._request_purpose(request),
            request_model=request.model,
        )
        self._raise_if_guard_blocks(decision)

    def _guard_record_success(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        *,
        call_type: str,
        provider: str | None,
        response_model: str | None,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> None:
        if self._llm_run_guard is None:
            return
        decision = self._llm_run_guard.record_success(
            call_type=call_type,
            purpose=self._request_purpose(request),
            request_model=request.model,
            response_model=response_model,
            provider=provider,
            usage=usage or {},
            latency_ms=latency_ms,
        )
        self._raise_if_guard_blocks(decision)

    def _guard_record_failure(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        *,
        call_type: str,
        exc: Exception,
        latency_ms: float,
    ) -> None:
        if self._llm_run_guard is None:
            return
        decision = self._llm_run_guard.record_failure(
            call_type=call_type,
            purpose=self._request_purpose(request),
            request_model=request.model,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )
        self._raise_if_guard_blocks(decision)

    @staticmethod
    def _raise_if_guard_blocks(decision: LLMRunGuardDecision) -> None:
        if decision.should_block:
            logger.error(
                "LLM run guard blocked further provider calls",
                extra={
                    "violations": list(decision.violations),
                    "llm_guard": decision.snapshot,
                },
            )
            raise LLMRunGuardError(decision)

    @staticmethod
    def _request_purpose(
        request: LLMCompletionRequest | LLMEmbeddingRequest,
    ) -> str | None:
        purpose = request.metadata.get("purpose")
        return purpose if isinstance(purpose, str) else None

    def _retry_policy_for(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
    ) -> RetryPolicy:
        """Resolve the retry policy for a request by its interactive purpose.

        Interactive retrieval gates use the short policy so a live turn does not
        pay long backoff; every other purpose keeps the client's base policy
        (which may have been injected via the constructor).
        """
        purpose = self._request_purpose(request)
        if purpose is not None and purpose in INTERACTIVE_RETRIEVAL_PURPOSES:
            return self._interactive_retry_policy
        return self._retry_policy

    @staticmethod
    def _should_retry_without_schema(exc: LLMError, request: LLMCompletionRequest) -> bool:
        """Decide whether to retry via prompt-JSON after a native-schema failure.

        ``request`` is the request that was actually sent. When the schema was
        already dropped for the prompt-JSON fallback, ``response_schema`` is
        ``None`` and there is nothing to retry without — so the gate also
        guarantees the failed attempt used the native structured path. The
        trigger is a typed client-request-class 4xx, not error-string matching.
        """
        if request.response_schema is None:
            return False
        return isinstance(exc, LLMRequestError) and 400 <= exc.status_code < 500

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
                    output_text=response.output_text,
                ) from exc
            raise StructuredOutputError(
                "Provider returned non-JSON structured output",
                details=exc.details,
                output_text=response.output_text,
            ) from exc

        adapter = TypeAdapter(schema)
        try:
            return adapter.validate_python(payload)
        except Exception as exc:
            raise StructuredOutputError(
                "Provider returned invalid structured output",
                details=self._structured_error_details(exc),
                output_text=response.output_text,
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
