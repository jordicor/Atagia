"""Provider-agnostic LLM client abstractions."""

from __future__ import annotations

import asyncio
import copy
from contextlib import aclosing, nullcontext
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import random
from uuid import uuid4
from dataclasses import dataclass
from time import perf_counter
from typing import Any, AsyncIterator, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from atagia.core.llm_output_limits import apply_min_output_threshold
from atagia.core.text_utils import strip_card_output_wrappers
from atagia.diagnostics.recorder import DiagnosticRecorder, bind_attempt, current_operation
from atagia.models.schemas_decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    ScoreAnswer,
    ScoreDecision,
    ScoreQuestion,
)
from atagia.services.inference_policy import (
    InferenceAccessPolicy,
    local_provider_registry_key,
    resolve_internal_inference_route,
)
from atagia.services.inference_routes import (
    BaseUrlClass,
    InferenceAccessMode,
    InferenceCostClass,
    InferenceOperation,
    InferenceRouteError,
    ParsedLocalModelSpec,
    ResolvedInferenceRoute,
    parse_local_model_spec,
)
from atagia.services.local_endpoint_catalog import LocalEndpointCatalog
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
    LLMCallMeter,
    LLMCallOutcome,
    LLMRunGuard,
    LLMRunGuardCall,
    LLMRunGuardConfig,
    LLMRunGuardDecision,
    begin_isolated_llm_call_meter,
    begin_llm_call_meter,
    bind_llm_call_meter,
    end_llm_call_meter,
    record_call_on_active_meter,
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
Q = TypeVar("Q", ChoiceQuestion, ScoreQuestion)
logger = logging.getLogger(__name__)
_TYPESAFE_SINGLE_QUESTION_BYTES = 30_000
_TYPESAFE_REQUEST_BYTES = 60_000


def _typesafe_question_batches(
    model: str,
    messages: list[LLMMessage],
    questions: dict[str, Q],
) -> list[dict[str, Q]]:
    """Pack prepared questions under conservative Jev 32k/64k token bounds.

    UTF-8 JSON byte length bounds token count without guessing Jev's tokenizer.
    Headroom covers framing. Only questions are split; the state is never cut.
    Each returned batch becomes a separate metered provider attempt.
    """
    state = [{"role": message.role, "content": message.content} for message in messages]
    request_model = parse_model_spec(model).request_model

    def size(batch: dict[str, Q]) -> int:
        payload = {
            "model": request_model,
            "state": state,
            "questions": {
                key: question.model_dump(mode="json") for key, question in batch.items()
            },
        }
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024

    batches: list[dict[str, Q]] = []
    current: dict[str, Q] = {}
    for question_id, question in questions.items():
        single = {question_id: question}
        if size(single) > _TYPESAFE_SINGLE_QUESTION_BYTES:
            raise ConfigurationError("TypeSafe state and one question exceed the safe 32k-token bound")
        proposed = {**current, question_id: question}
        if current and size(proposed) > _TYPESAFE_REQUEST_BYTES:
            batches.append(current)
            current = single
        else:
            current = proposed
    if current:
        batches.append(current)
    return batches


_DIAGNOSTIC_METADATA_KEYS = frozenset({
    "purpose", "stage", "user_id", "turn_id", "job_id", "conversation_id",
    "reasoning_effort", "verbosity", "openai_tool_choice", "service_tier",
    "anthropic_prompt_cache", "thinking_budget_tokens", "anthropic_thinking_adaptive",
    "anthropic_output_effort", "gemini_thinking_level", "gemini_google_search",
    "openrouter_native_structured_output", "provider_extra_body", "task_type", "title",
    "atagia_model_spec", "atagia_provider_slug", "atagia_component_id",
    "atagia_temperature_source", "atagia_requested_temperature", "atagia_effective_temperature",
    "atagia_temperature_reason", "atagia_partial_stream_retry",
})


def _diagnostic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in _DIAGNOSTIC_METADATA_KEYS if key in metadata}


def _diagnostic_has_credentials(value: Any) -> bool:
    if isinstance(value, dict):
        forbidden = {"api_key", "authorization", "headers", "extra_headers", "password", "secret", "credentials", "access_token"}
        return any(str(key).lower() in forbidden or _diagnostic_has_credentials(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_diagnostic_has_credentials(item) for item in value)
    return False


def _diagnostic_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
    """Base LLM client error: the caller did not get a usable model answer.

    NOT "the provider call failed". The hierarchy already contains local
    refusals in which no provider was ever touched (``ConfigurationError``,
    ``LLMRunGuardError``) and provider successes that failed downstream
    (``StructuredOutputError``), and it contains them deliberately: every
    handler in this repo that catches ``LLMError`` is answering "I did not get
    an answer, degrade" -- a 503, an abstention, a skipped background refresh --
    and that answer is correct for a local refusal too.

    What that shared base does NOT license is REROUTING. A handler that responds
    by calling the provider again (a fallback model, a retry, a second provider)
    must first exclude the refusals that no re-call can satisfy; see
    ``_is_policy_blocked_error``, which does exactly that.
    """


class LLMPolicyBlockedError(LLMError):
    """Raised when a provider refuses or blocks a request for policy reasons."""


class InferenceAccessDeniedError(LLMError):
    """Terminal refusal raised before an inference route reaches a provider."""


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
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.output_text = output_text
        self.reason = reason


class TransientLLMError(LLMError):
    """Raised for retryable provider failures."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = _normalize_retry_after_seconds(retry_after_seconds)


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
    """Raised when the run guard refuses a call: budget spent or health blown.

    A LOCAL REFUSAL -- the provider was never touched. Re-issuing the same
    request through any fallback path cannot help, because the refusal is about
    the RUN rather than about the request: every fallback's own ``begin_call``
    refuses it too. Nothing may respond to this by calling a provider again.

    It stays an ``LLMError`` on purpose, and the alternative was measured rather
    than assumed. Every reachable ``except LLMError`` in this repo responds by
    degrading -- ``chat_service`` turns it into a 503, the OpenAI proxy route
    into a structured ``llm_unavailable`` 503, ``answer_postcondition`` into an
    abstention that still delivers the already-generated answer, the ingest
    worker into a skipped refresh -- and every one of those is the right answer
    for "stop spending". Reparenting off ``LLMError`` would turn each of them
    into an uncaught ``RuntimeError`` unless it were individually re-taught this
    class: two 503s would become 500s, the proxy would lose its error envelope,
    and the postcondition path would discard an answer the operator already paid
    for, which is the exact failure the guard's own design forbids. The failure
    mode of forgetting is what settles it: under this hierarchy a handler that
    has never heard of the guard degrades correctly, and under a reparented one
    it crashes.
    """

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
    retry_delays_seconds: tuple[float, ...] = ()
    jitter_fraction: float = 0.15


class _DispatchPermit:
    """Release a provider slot once, including before retry backoff."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._semaphore.release()


# Interactive retrieval gates degrade gracefully (e.g. need detection falls back
# to base search), so they should not pay long backoff inside a live turn. These
# are the exact `request.metadata["purpose"]` strings used by those stages.
INTERACTIVE_RETRIEVAL_PURPOSES: frozenset[str] = frozenset(
    {
        "need_detection_query_language_card",
        "need_detection_answer_language_card",
        "need_detection_needs_card",
        "need_detection_memory_card",
        "need_detection_exact_card",
        "need_detection_shape_card",
        "need_detection_facets_card",
        "need_detection_callback_card",
        "need_detection_search_words_card",
        "need_detection_search_words_other_language_card",
        "applicability_scoring",
        "applicability_relevance_card",
        "context_cache_signal_detection",
        "coverage_expansion",
    }
)

_DEFAULT_INTERACTIVE_RETRY_POLICY = RetryPolicy(attempts=2, max_delay_seconds=1.5)
_DEFAULT_EXTRACTION_RETRY_POLICY = RetryPolicy(
    attempts=5,
    base_delay_seconds=1.0,
    max_delay_seconds=15.0,
    retry_delays_seconds=(1.0, 3.0, 8.0, 15.0),
)
_MEMORY_EXTRACTION_PURPOSES: frozenset[str] = frozenset(
    {
        "memory_date_resolution",
        "memory_extraction",
        "memory_extraction_candidate_card",
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
        "memory_extraction_evidence_support_card",
        "memory_extraction_preserve_verbatim_card",
        "memory_extraction_candidate_language_card",
        "memory_extraction_source_reference_card",
        "memory_extraction_index_card",
        "memory_extraction_belief_key_card",
        "memory_extraction_belief_value_card",
        "memory_extraction_temporal_type_card",
        "memory_extraction_temporal_interval_card",
        "memory_extraction_coverage_members_card",
        "memory_extraction_coverage_member_identity_card",
    }
)
_PARTIAL_STREAM_RETRY_PURPOSES: frozenset[str] = _MEMORY_EXTRACTION_PURPOSES
_CONSEQUENCE_DETECTION_PURPOSES: frozenset[str] = frozenset(
    {
        "consequence_detection",
        "consequence_gate_card",
        "consequence_action_card",
        "consequence_outcome_card",
        "consequence_sentiment_card",
        "consequence_link_card",
        "consequence_language_card",
    }
)
_BACKGROUND_DEFERABLE_RETRY_AFTER_PURPOSES: frozenset[str] = frozenset(
    {
        *_MEMORY_EXTRACTION_PURPOSES,
        *_CONSEQUENCE_DETECTION_PURPOSES,
        "consequence_tendency_inference",
    }
)


def _normalize_retry_after_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def retry_after_seconds_from_headers(headers: Any) -> float | None:
    """Parse an HTTP ``Retry-After`` value as seconds when available."""
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw_value = getter("retry-after")
    if raw_value is None:
        raw_value = getter("Retry-After")
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    numeric = _normalize_retry_after_seconds(text)
    if numeric is not None:
        return numeric
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def retry_after_seconds_from_exception(exc: BaseException) -> float | None:
    """Extract ``Retry-After`` from common SDK exception shapes."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    retry_after = retry_after_seconds_from_headers(headers)
    if retry_after is not None:
        return retry_after
    return retry_after_seconds_from_headers(getattr(exc, "headers", None))


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
    choice_questions: dict[str, ChoiceQuestion] | None = None
    score_questions: dict[str, ScoreQuestion] | None = None
    finite_choice: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    include_thinking: bool = False
    # External answer requests own a hard caller/server ceiling. They must not
    # inherit Atagia's structured-output minimum or technical truncation retry.
    external_answer: bool = False


class LLMCompletionResponse(BaseModel):
    """Normalized completion response."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    output_text: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    choice_answers: dict[str, ChoiceAnswer] = Field(default_factory=dict)
    score_answers: dict[str, ScoreAnswer] = Field(default_factory=dict)


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


def normalize_completion_finish_reason(
    value: Any,
    *,
    has_tool_calls: bool = False,
) -> str | None:
    """Normalize provider stop labels to the OpenAI completion vocabulary."""

    if value is None:
        return "tool_calls" if has_tool_calls else None
    label = str(getattr(value, "name", None) or value).rsplit(".", 1)[-1]
    normalized = label.strip().lower()
    if not normalized:
        return "tool_calls" if has_tool_calls else None
    if has_tool_calls and normalized in {
        "stop",
        "end_turn",
        "stop_sequence",
        "tool_use",
        "tool_calls",
        "function_call",
        "finish_reason_stop",
    }:
        return "tool_calls"
    if normalized in {"stop", "end_turn", "stop_sequence", "finish_reason_stop"}:
        return "stop"
    if normalized in {"tool_use", "tool_calls", "function_call"}:
        return "tool_calls"
    if normalized in {"length", "max_tokens", "model_context_window_exceeded"}:
        return "length"
    if normalized in {
        "error",
        "pause_turn",
        "malformed_function_call",
        "finish_reason_unspecified",
        "unspecified",
    }:
        return "error"
    return normalized


class LLMProvider:
    """Provider adapter interface."""

    name: str
    supports_embeddings: bool = True
    supports_embedding_dimensions: bool = False
    supports_native_structured_output: bool = True
    supports_choices: bool = False
    supports_scores: bool = False

    async def aclose(self) -> None:
        """Release resources owned by this adapter, when applicable."""

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise NotImplementedError

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise NotImplementedError

    async def stream(
        self, request: LLMCompletionRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        response = await self.complete(request)
        if response.thinking:
            yield LLMStreamEvent(type="thinking", content=response.thinking)
        if response.output_text:
            yield LLMStreamEvent(type="text", content=response.output_text)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                yield LLMStreamEvent(type="tool_call", payload=tool_call)
        done_payload: dict[str, Any] = {}
        if response.usage:
            done_payload["usage"] = response.usage
        if response.finish_reason is not None:
            done_payload["finish_reason"] = response.finish_reason
        yield LLMStreamEvent(type="done", payload=done_payload)

    def supports_native_structured_output_for(
        self, request: LLMCompletionRequest
    ) -> bool:
        """Return whether this provider can enforce the schema for this request."""
        return self.supports_native_structured_output


class _PreOutputStreamError(RuntimeError):
    """Internal wrapper for stream errors raised before any output was emitted."""

    def __init__(self, original: LLMError) -> None:
        super().__init__(str(original))
        self.original = original


@dataclass(frozen=True, slots=True)
class _ResolvedProviderRequest:
    """Provider-bound request plus immutable, separately resolved route."""

    provider_name: str
    request: LLMCompletionRequest | LLMEmbeddingRequest
    route: ResolvedInferenceRoute


class LLMClient(Generic[T]):
    """Registry-based LLM client with retry helpers."""

    def __init__(
        self,
        provider_name: str | None = None,
        providers: list[LLMProvider] | None = None,
        retry_policy: RetryPolicy | None = None,
        interactive_retry_policy: RetryPolicy | None = None,
        extraction_retry_policy: RetryPolicy | None = None,
        allow_unqualified_single_provider_models: bool = False,
        intimacy_fallback_models: dict[str, str] | None = None,
        intimacy_proactive_routing_enabled: bool = False,
        structured_output_retry_attempts: int = 1,
        structured_output_rescue_enabled: bool = False,
        structured_output_rescue_model: str | None = None,
        technical_recovery_config: LLMTechnicalRecoveryConfig | None = None,
        llm_run_guard: LLMRunGuard | None = None,
        inference_access_policy: InferenceAccessPolicy | None = None,
        local_endpoint_catalog: LocalEndpointCatalog | None = None,
        max_concurrent_requests_per_provider: int = 4,
        diagnostic_recorder: DiagnosticRecorder | None = None,
    ) -> None:
        if max_concurrent_requests_per_provider <= 0:
            raise ValueError("max_concurrent_requests_per_provider must be positive")
        self._provider_name = (
            provider_name.strip().lower() if provider_name is not None else None
        )
        self._providers = {
            provider.name.strip().lower(): provider for provider in (providers or [])
        }
        self._dispatch_semaphores = {
            name: asyncio.Semaphore(max_concurrent_requests_per_provider)
            for name in self._providers
        }
        self._max_concurrent_requests_per_provider = max_concurrent_requests_per_provider
        self._retry_policy = retry_policy or RetryPolicy()
        self._interactive_retry_policy = (
            interactive_retry_policy or _DEFAULT_INTERACTIVE_RETRY_POLICY
        )
        self._extraction_retry_policy = (
            extraction_retry_policy or retry_policy or _DEFAULT_EXTRACTION_RETRY_POLICY
        )
        self._allow_unqualified_single_provider_models = (
            allow_unqualified_single_provider_models
        )
        self._intimacy_fallback_models = dict(intimacy_fallback_models or {})
        self._intimacy_proactive_routing_enabled = intimacy_proactive_routing_enabled
        if structured_output_retry_attempts < 0:
            raise ValueError("structured_output_retry_attempts must be non-negative")
        self._structured_output_retry_attempts = structured_output_retry_attempts
        self._structured_output_rescue_enabled = structured_output_rescue_enabled
        rescue_model = (
            structured_output_rescue_model.strip()
            if structured_output_rescue_model
            else None
        )
        self._structured_output_rescue_model = rescue_model or None
        self._technical_recovery_config = (
            technical_recovery_config or LLMTechnicalRecoveryConfig.default_enabled()
        )
        self._llm_run_guard = llm_run_guard
        self._inference_access_policy = (
            inference_access_policy
            or InferenceAccessPolicy(InferenceAccessMode.UNRESTRICTED)
        )
        self._local_endpoint_catalog = local_endpoint_catalog
        self._diagnostic_recorder = diagnostic_recorder

    def register_provider(self, provider: LLMProvider) -> None:
        name = provider.name.strip().lower()
        self._providers[name] = provider
        self._dispatch_semaphores.setdefault(
            name, asyncio.Semaphore(self._max_concurrent_requests_per_provider)
        )

    async def _acquire_dispatch(self, provider_name: str) -> _DispatchPermit:
        """Limit in-flight calls per provider across one engine client's tasks.

        Separate client instances and OS processes do not share these slots.
        The run guard starts after acquisition, so queued or cancelled waiters
        cannot consume a provider attempt or a guard reservation.
        """
        self._provider(provider_name)
        semaphore = self._dispatch_semaphores[provider_name]
        await semaphore.acquire()
        return _DispatchPermit(semaphore)

    async def aclose(self) -> None:
        """Close owned provider transports after a runtime or benchmark."""
        for provider in self._providers.values():
            await provider.aclose()
        if self._diagnostic_recorder is not None:
            self._diagnostic_recorder.close()

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

    def begin_turn_call_meter(self) -> LLMCallMeter:
        """Bind a per-turn LLM call meter to the current async context.

        The returned meter SHOULD be released with ``end_turn_call_meter`` in a
        ``finally``. The meter counts every provider round-trip made in this
        context (retrieval cards, staleness, planner, scoring, chat reply) and
        is independent of the run guard.
        """
        return begin_llm_call_meter()

    def begin_isolated_call_meter(self) -> LLMCallMeter:
        """Bind a meter for background work, detached from the spawning turn.

        A task spawned from a turn inherits a copy of that turn's meter stack,
        so its calls would land on a row that has already been written. This
        replaces the inherited stack for the duration of the background task.
        """
        return begin_isolated_llm_call_meter()

    def bind_turn_call_meter(self, meter: LLMCallMeter) -> None:
        """Re-bind an existing turn meter in a second async scope of the same turn."""
        bind_llm_call_meter(meter)

    def end_turn_call_meter(self, meter: LLMCallMeter) -> None:
        """Unbind a per-turn LLM call meter, by identity, from this context."""
        end_llm_call_meter(meter)

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
        if request.choice_questions is not None or request.score_questions is not None:
            # Typed requests must stay on the selected typed provider.
            return await self._complete_once(request)
        normalized_request = request.model_copy(
            update={
                "max_output_tokens": (
                    request.max_output_tokens
                    if request.external_answer
                    else apply_min_output_threshold(request.max_output_tokens)
                )
            }
        )
        if request.finite_choice:
            # Keep the normal LLM output floor and the configured proactive
            # privacy route, but never repair or switch providers after failure.
            proactive_request = self._proactive_intimacy_request(normalized_request)
            if proactive_request is not None:
                if parse_model_spec(proactive_request.model).provider_slug == "typesafe":
                    raise ConfigurationError("A text choice cannot route to TypeSafe")
                return await self._complete_once(proactive_request)
            return await self._complete_once(normalized_request)
        route = self._completion_route_for_request(normalized_request)
        return await self._with_output_limit_recovery(
            normalized_request,
            lambda current_request: self._complete_with_intimacy_routing(
                current_request,
                route=route,
            ),
            operation_name="completion",
            allow_recovery=not normalized_request.external_answer,
        )

    async def complete_choice_questions(
        self,
        *,
        model: str,
        messages: list[LLMMessage],
        questions: dict[str, ChoiceQuestion],
        metadata: dict[str, Any],
        max_output_tokens: int = 64,
        concurrency: int = 2,
        dispatch_semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, str]:
        """Run prepared questions of one card against the selected model.

        Each question contains its complete task and target in ``instructions``;
        question IDs only connect answers to callers. The shared state is sent
        once to TypeSafe, while an ordinary LLM receives one narrow question per
        request. A failed sibling cancels and joins the others before returning.
        Pass the extraction card's shared semaphore without acquiring it in the
        caller; this method acquires it for each provider request.
        """
        if not questions or not messages:
            raise ValueError("Finite choices require questions and state messages")
        if any(not question_id.strip() for question_id in questions):
            raise ValueError("Finite choice question IDs must be nonempty")
        if not isinstance(metadata.get("purpose"), str) or not metadata["purpose"].strip():
            raise ValueError("Finite choices require a request purpose")
        if concurrency < 1:
            raise ValueError("Finite choice concurrency must be positive")

        async def complete_bounded(request: LLMCompletionRequest) -> LLMCompletionResponse:
            if dispatch_semaphore is None:
                return await self.complete(request)
            async with dispatch_semaphore:
                return await self.complete(request)

        if parse_model_spec(model).provider_slug == "typesafe":
            answers: dict[str, str] = {}
            for batch in _typesafe_question_batches(model, messages, questions):
                response = await complete_bounded(
                    LLMCompletionRequest(
                        model=model,
                        messages=messages,
                        choice_questions=batch,
                        metadata=metadata,
                    )
                )
                if response.choice_answers.keys() != batch.keys():
                    raise LLMError("Finite choice answer IDs do not match the questions")
                selected = {
                    question_id: answer.choice
                    for question_id, answer in response.choice_answers.items()
                }
                if any(
                    answer not in batch[question_id].criteria
                    for question_id, answer in selected.items()
                ):
                    raise LLMError("Finite choice response contains an unknown option")
                answers.update(selected)
            return answers

        semaphore = asyncio.Semaphore(concurrency)

        async def complete_one(question_id: str, question: ChoiceQuestion) -> str:
            options = "\n".join(
                f"- {json.dumps(option, ensure_ascii=False)}"
                + (f": {description}" if description is not None else "")
                for option, description in question.criteria.items()
            )
            instruction = (
                f"{question.instructions}\n\nChoose exactly one option key:\n{options}\n"
                "Return only the exact option key, without quotes or explanation."
            )
            async with semaphore:
                response = await complete_bounded(
                    LLMCompletionRequest(
                        model=model,
                        messages=[LLMMessage(role="system", content=instruction), *messages],
                        max_output_tokens=max_output_tokens,
                        finite_choice=True,
                        metadata={**metadata, "stage": question_id},
                    )
                )
            answer = response.output_text.strip()
            if answer not in question.criteria:
                answer = strip_card_output_wrappers(answer)
            if answer not in question.criteria:
                raise LLMError("Finite choice response contains an unknown option")
            return answer

        tasks = [
            asyncio.create_task(complete_one(question_id, question))
            for question_id, question in questions.items()
        ]
        try:
            values = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return dict(zip(questions, values, strict=True))

    async def complete_score_questions(
        self,
        *,
        model: str,
        messages: list[LLMMessage],
        questions: dict[str, ScoreQuestion],
        metadata: dict[str, Any],
        max_output_tokens: int = 64,
        concurrency: int = 2,
        dispatch_semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, ScoreDecision]:
        """Rate prepared five-level questions with a common [0, 1] result.

        TypeSafe returns its weighted 0..4 score and separate certainty. An LLM
        returns a continuous [0, 1] number guided by the same rubric. The
        shared dispatch semaphore is acquired here per provider request.
        """
        if not questions or not messages:
            raise ValueError("Finite scores require questions and state messages")
        if any(not question_id.strip() for question_id in questions):
            raise ValueError("Finite score question IDs must be nonempty")
        if not isinstance(metadata.get("purpose"), str) or not metadata["purpose"].strip():
            raise ValueError("Finite scores require a request purpose")
        if concurrency < 1:
            raise ValueError("Finite score concurrency must be positive")

        async def complete_bounded(request: LLMCompletionRequest) -> LLMCompletionResponse:
            if dispatch_semaphore is None:
                return await self.complete(request)
            async with dispatch_semaphore:
                return await self.complete(request)

        if parse_model_spec(model).provider_slug == "typesafe":
            answers: dict[str, ScoreDecision] = {}
            for batch in _typesafe_question_batches(model, messages, questions):
                response = await complete_bounded(
                    LLMCompletionRequest(
                        model=model,
                        messages=messages,
                        score_questions=batch,
                        metadata=metadata,
                    )
                )
                if response.score_answers.keys() != batch.keys():
                    raise LLMError("Finite score answer IDs do not match the questions")
                answers.update({
                    question_id: ScoreDecision(
                        normalized_score=answer.normalized_score,
                        typed_answer=answer,
                    )
                    for question_id, answer in response.score_answers.items()
                })
            return answers

        semaphore = asyncio.Semaphore(concurrency)

        async def complete_one(question_id: str, question: ScoreQuestion) -> ScoreDecision:
            rubric = "\n".join(
                f"- {index / 4:.2f}: {description}"
                for index, description in enumerate(question.criteria)
            )
            instruction = (
                f"{question.instructions}\n\nUse these anchors for a continuous score "
                f"between 0 and 1:\n{rubric}\nReturn only one number between 0 and 1. "
                "Intermediate values are allowed; do not round to an anchor."
            )
            async with semaphore:
                response = await complete_bounded(
                    LLMCompletionRequest(
                        model=model,
                        messages=[LLMMessage(role="system", content=instruction), *messages],
                        max_output_tokens=max_output_tokens,
                        finite_choice=True,
                        metadata={**metadata, "stage": question_id},
                    )
                )
            try:
                value = float(strip_card_output_wrappers(response.output_text))
            except ValueError:
                raise LLMError("Finite score response is not a single number") from None
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise LLMError("Finite score response is outside [0, 1]")
            return ScoreDecision(normalized_score=value)

        tasks = [
            asyncio.create_task(complete_one(question_id, question))
            for question_id, question in questions.items()
        ]
        try:
            values = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return dict(zip(questions, values, strict=True))

    async def _complete_with_intimacy_routing(
        self,
        normalized_request: LLMCompletionRequest,
        *,
        route: ResolvedInferenceRoute,
    ) -> LLMCompletionResponse:
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            return await self._complete_once(proactive_request)
        try:
            return await self._complete_once(normalized_request, route=route)
        except LLMError as exc:
            fallback_request = self._intimacy_fallback_request(normalized_request, exc)
            if fallback_request is None:
                raise
            return await self._complete_once(fallback_request)

    async def _complete_once(
        self,
        request: LLMCompletionRequest,
        *,
        route: ResolvedInferenceRoute | None = None,
    ) -> LLMCompletionResponse:
        if request.choice_questions is not None and request.score_questions is not None:
            raise ConfigurationError("Typed decisions require one homogeneous question kind")
        resolved = self._completion_provider_request(request, route=route)
        provider_request = resolved.request
        assert isinstance(provider_request, LLMCompletionRequest)
        if request.choice_questions is not None or request.score_questions is not None:
            provider = self._provider_for_resolved_request(resolved)
            if request.choice_questions is not None and not provider.supports_choices:
                raise ConfigurationError("Selected provider does not support typed choices")
            if request.score_questions is not None and not provider.supports_scores:
                raise ConfigurationError("Selected provider does not support typed scores")
        return await self._with_retries(
            lambda: self._provider(resolved.provider_name).complete(provider_request),
            request=provider_request,
            route=resolved.route,
            provider_name=resolved.provider_name,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        resolved = self._embedding_provider_request(request)
        provider_request = resolved.request
        assert isinstance(provider_request, LLMEmbeddingRequest)
        return await self._with_retries(
            lambda: self._provider(resolved.provider_name).embed(provider_request),
            request=provider_request,
            route=resolved.route,
            provider_name=resolved.provider_name,
        )

    def supports_embedding_dimensions(self, model_spec: str) -> bool:
        """Return whether the provider for an embedding model supports dimensions."""
        if self._provider_name is not None:
            return self.provider.supports_embedding_dimensions
        resolved = self._embedding_provider_request(
            LLMEmbeddingRequest(model=model_spec, input_texts=[])
        )
        return self._provider_for_resolved_request(
            resolved
        ).supports_embedding_dimensions

    async def stream(
        self, request: LLMCompletionRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        if request.choice_questions is not None or request.score_questions is not None or request.finite_choice:
            raise ConfigurationError("Typed choices require complete(), not text streaming")
        normalized_request = request.model_copy(
            update={
                "max_output_tokens": (
                    request.max_output_tokens
                    if request.external_answer
                    else apply_min_output_threshold(request.max_output_tokens)
                )
            }
        )
        route = self._completion_route_for_request(normalized_request)
        # `aclosing` is what makes an abandoned stream observable. A bare
        # `async for` leaves the inner generator suspended when this one is
        # closed, so `_stream_once` would only learn about the disconnect
        # whenever the garbage collector got around to finalizing it -- after the
        # turn that paid for the round-trip has already written its telemetry.
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            async with aclosing(self._stream_once(proactive_request)) as events:
                async for event in events:
                    yield event
            return
        try:
            async with aclosing(
                self._stream_once(normalized_request, route=route)
            ) as events:
                async for event in events:
                    yield event
            return
        except _PreOutputStreamError as exc:
            fallback_request = self._intimacy_fallback_request(
                normalized_request, exc.original
            )
            if fallback_request is None:
                raise exc.original from exc
            async with aclosing(self._stream_once(fallback_request)) as events:
                async for event in events:
                    yield event

    async def _stream_once(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
        route: ResolvedInferenceRoute | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        recorder = self._diagnostic_recorder
        if recorder is None:
            async with aclosing(self._stream_once_impl(request, observer=observer, route=route)) as events:
                async for event in events:
                    yield event
            return
        safe_metadata = _diagnostic_metadata(request.metadata)
        if _diagnostic_has_credentials(safe_metadata.get("provider_extra_body")):
            recorder._fail(ValueError("provider_extra_body contains credential fields"))
        request_ref = recorder.blob(request.model_copy(update={"metadata": safe_metadata}))
        prompt_ref = recorder.blob(request.messages)
        contract_ref = recorder.blob({"choice_questions": request.choice_questions, "score_questions": request.score_questions, "response_schema": request.response_schema, "tools": request.tools})
        with recorder.operation(self._request_purpose(request), component=component_id_for_llm_purpose(self._request_purpose(request)), input_data={"request": request_ref, "requested_model": request.model, "prompt_sha256": prompt_ref["sha256"] if prompt_ref else None, "contract_sha256": contract_ref["sha256"] if contract_ref else None}):
            async with aclosing(self._stream_once_impl(request, observer=observer, route=route)) as events:
                async for event in events:
                    yield event

    async def _stream_once_impl(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
        route: ResolvedInferenceRoute | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        resolved = self._completion_provider_request(request, route=route)
        provider_request = resolved.request
        assert isinstance(provider_request, LLMCompletionRequest)
        observer = self._stream_observer(observer)
        retry_policy = self._retry_policy_for(provider_request)
        last_error: LLMError | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            emitted_any = False
            output_text = ""
            usage: dict[str, Any] = {}
            self._authorize_inference_route(resolved.route)
            permit = await self._acquire_dispatch(resolved.provider_name)
            try:
                guarded_call = self._begin_guarded_call(provider_request)
            except BaseException:
                permit.release()
                raise
            try:
                provider = self._provider(resolved.provider_name)
                started_at = perf_counter()
                stream_iterator = provider.stream(provider_request)
            except BaseException:
                permit.release()
                raise
            recorder = self._diagnostic_recorder
            current = current_operation(recorder)
            attempt_id = uuid4().hex if recorder is not None and current is not None else None
            if recorder is not None and current is not None and attempt_id is not None:
                recorder.event("provider_attempt", phase="start", trace_id=current[0], operation_id=current[1], attempt_id=attempt_id, purpose=self._request_purpose(provider_request), status="started", data={"attempt_number": attempt, "resolved_provider": resolved.provider_name, "resolved_model": provider_request.model, "started_at": _diagnostic_utc_now()})
                original_iterator = stream_iterator

                async def captured_stream() -> AsyncIterator[LLMStreamEvent]:
                    with bind_attempt(recorder, attempt_id):
                        async for captured_event in original_iterator:
                            yield captured_event

                stream_iterator = captured_stream()
            try:
                async for event in stream_iterator:
                    emitted_any = True
                    if event.type == "text" and event.content:
                        output_text += event.content
                        if observer is not None:
                            await observer.on_text(
                                event.content, output_text, provider_request
                            )
                    # Read usage from WHEREVER the stream publishes it, not only
                    # from the terminal event. A cancelled stream never reaches
                    # its terminal event, so anything published earlier is the
                    # only usage that call will ever have; the terminal event
                    # still wins for a completed stream because it arrives last.
                    event_usage = event.payload.get("usage")
                    if isinstance(event_usage, dict):
                        usage = dict(event_usage)
                    yield event
            except LLMRunawayAbort as exc:
                self._diagnostic_attempt_end(attempt_id, provider_request, "partial" if emitted_any else "failure", error=exc, partial_output=output_text, partial_usage=usage)
                await self._close_stream_iterator(stream_iterator)
                output_limit_error = self._runaway_abort_error(exc, provider_request)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=output_limit_error,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise output_limit_error from exc
                raise _PreOutputStreamError(output_limit_error) from exc
            except TransientLLMError as exc:
                self._diagnostic_attempt_end(attempt_id, provider_request, "partial" if emitted_any else "failure", error=exc, partial_output=output_text, partial_usage=usage)
                await self._close_stream_iterator(stream_iterator)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                permit.release()
                await self._sleep_before_retry(
                    retry_policy,
                    attempt,
                    exc,
                    request=provider_request,
                )
            except LLMError as exc:
                self._diagnostic_attempt_end(attempt_id, provider_request, "partial" if emitted_any else "failure", error=exc, partial_output=output_text, partial_usage=usage)
                await self._close_stream_iterator(stream_iterator)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
            except Exception as exc:
                # A provider or adapter error that was not normalized into an
                # LLMError is still a failed round-trip, not an abandoned one.
                self._diagnostic_attempt_end(attempt_id, provider_request, "partial" if emitted_any else "failure", error=exc, partial_output=output_text, partial_usage=usage)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                await self._close_stream_iterator(stream_iterator)
                raise
            except BaseException:
                # Everything that is NOT an Exception -- CancelledError,
                self._diagnostic_attempt_end(attempt_id, provider_request, "cancelled", partial_output=output_text, partial_usage=usage)
                # GeneratorExit, KeyboardInterrupt, SystemExit -- means the
                # consumer or the process went away mid-stream (a proxy client
                # disconnect is the common one). Record before closing the
                # iterator: the close is awaited inside a cancellation, so it can
                # be interrupted, and the round-trip is already spent either way.
                self._record_provider_call_cancelled(
                    provider_request,
                    guarded_call,
                    usage=usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                await self._close_stream_iterator(stream_iterator)
                raise
            else:
                # OUTSIDE the try. Recording a success inside it would let any
                # LLMError raised while recording (LLMRunGuardError is one) be
                # caught by this method's own `except LLMError`, which would
                # record a SECOND, failed round-trip for the same provider call.
                self._record_provider_call_success(
                    provider_request,
                    guarded_call,
                    usage=usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                self._diagnostic_attempt_end(attempt_id, provider_request, "success", partial_output=output_text, partial_usage=usage)
                return
            finally:
                permit.release()
        if last_error is None:
            raise LLMError("LLM stream failed without a captured error")
        raise _PreOutputStreamError(last_error)

    async def complete_streamed(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
    ) -> LLMCompletionResponse:
        if request.choice_questions is not None or request.score_questions is not None or request.finite_choice:
            raise ConfigurationError("Finite choices require complete(), not text streaming")
        normalized_request = request.model_copy(
            update={
                "max_output_tokens": (
                    request.max_output_tokens
                    if request.external_answer
                    else apply_min_output_threshold(request.max_output_tokens)
                )
            }
        )
        route = self._completion_route_for_request(normalized_request)
        return await self._with_output_limit_recovery(
            normalized_request,
            lambda retry_request: self._complete_streamed_with_intimacy_routing(
                retry_request,
                observer=observer,
                route=route,
            ),
            operation_name="streamed_completion",
            retry_observer=observer,
            allow_recovery=not normalized_request.external_answer,
        )

    async def _complete_streamed_with_intimacy_routing(
        self,
        normalized_request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
        route: ResolvedInferenceRoute,
    ) -> LLMCompletionResponse:
        proactive_request = self._proactive_intimacy_request(normalized_request)
        if proactive_request is not None:
            return await self._complete_streamed_once(
                proactive_request, observer=observer
            )
        try:
            return await self._complete_streamed_once(
                normalized_request,
                observer=observer,
                route=route,
            )
        except _PreOutputStreamError as exc:
            fallback_request = self._intimacy_fallback_request(
                normalized_request, exc.original
            )
            if fallback_request is None:
                raise exc.original from exc
            return await self._complete_streamed_once(
                fallback_request, observer=observer
            )

    async def _complete_streamed_once(
        self,
        request: LLMCompletionRequest,
        *,
        observer: Any | None = None,
        route: ResolvedInferenceRoute | None = None,
    ) -> LLMCompletionResponse:
        resolved = self._completion_provider_request(request, route=route)
        provider_request = resolved.request
        assert isinstance(provider_request, LLMCompletionRequest)
        observer = self._stream_observer(observer)
        retry_policy = self._retry_policy_for(provider_request)
        last_error: LLMError | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            emitted_any = False
            output_text = ""
            thinking = ""
            tool_calls: list[dict[str, Any]] = []
            usage: dict[str, Any] = {}
            finish_reason: str | None = None
            self._authorize_inference_route(resolved.route)
            permit = await self._acquire_dispatch(resolved.provider_name)
            try:
                guarded_call = self._begin_guarded_call(provider_request)
            except BaseException:
                permit.release()
                raise
            try:
                provider = self._provider(resolved.provider_name)
                started_at = perf_counter()
                stream_iterator = provider.stream(provider_request)
            except BaseException:
                permit.release()
                raise
            try:
                async for event in stream_iterator:
                    emitted_any = True
                    # Usage is read outside the type dispatch below for the same
                    # reason as in ``_stream_once``: a cancelled stream never
                    # reaches its terminal event, so whatever was published
                    # earlier is the only usage this call will ever have.
                    event_usage = event.payload.get("usage")
                    if isinstance(event_usage, dict):
                        usage = dict(event_usage)
                    if event.type == "text" and event.content:
                        output_text += event.content
                        if observer is not None:
                            await observer.on_text(
                                event.content, output_text, provider_request
                            )
                    elif event.type == "thinking" and event.content:
                        thinking += event.content
                    elif event.type == "tool_call":
                        tool_calls.append(dict(event.payload))
                    elif event.type == "done":
                        event_finish_reason = event.payload.get("finish_reason")
                        if isinstance(event_finish_reason, str):
                            finish_reason = event_finish_reason
            except LLMRunawayAbort as exc:
                await self._close_stream_iterator(stream_iterator)
                output_limit_error = self._runaway_abort_error(exc, provider_request)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=output_limit_error,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise output_limit_error from exc
                raise _PreOutputStreamError(output_limit_error) from exc
            except TransientLLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                can_retry_partial = (
                    emitted_any
                    and self._should_retry_after_partial_stream(provider_request)
                )
                if emitted_any and not can_retry_partial:
                    raise
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                if can_retry_partial:
                    logger.warning(
                        "Retrying streamed completion after partial transient error "
                        "purpose=%s model=%s attempt=%s",
                        provider_request.metadata.get("purpose") or "<unset>",
                        provider_request.model,
                        attempt,
                    )
                    await self._reset_stream_observer_for_retry(observer, exc)
                permit.release()
                await self._sleep_before_retry(
                    retry_policy,
                    attempt,
                    exc,
                    request=provider_request,
                )
            except LLMError as exc:
                await self._close_stream_iterator(stream_iterator)
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                if emitted_any:
                    raise
                raise _PreOutputStreamError(exc) from exc
            except Exception as exc:
                # Not normalized into an LLMError, but still a failed round-trip.
                self._record_provider_call_failure(
                    provider_request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                await self._close_stream_iterator(stream_iterator)
                raise
            except BaseException:
                # Not an Exception: the caller or the process abandoned the call.
                # Record before closing the iterator -- the close is awaited
                # inside a cancellation and can itself be interrupted, and the
                # round-trip is already spent either way.
                self._record_provider_call_cancelled(
                    provider_request,
                    guarded_call,
                    usage=usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                await self._close_stream_iterator(stream_iterator)
                raise
            else:
                # OUTSIDE the try, for the same reason as `_stream_once`: a
                # success recorded inside it could be re-caught by this method's
                # own `except LLMError` and metered a second time as a failure.
                response = LLMCompletionResponse(
                    provider=provider.name,
                    model=str(
                        provider_request.metadata.get("atagia_model_spec")
                        or provider_request.model
                    ),
                    output_text=output_text,
                    thinking=thinking or None,
                    tool_calls=tool_calls,
                    usage=usage,
                    finish_reason=finish_reason,
                )
                self._record_provider_call_success(
                    provider_request,
                    guarded_call,
                    usage=response.usage,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                return response
            finally:
                permit.release()
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
            except InferenceAccessDeniedError:
                # An access denial is terminal and cannot be repaired by a
                # second request, even if the request otherwise qualifies for
                # technical output-limit recovery.
                raise
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
        strategy = request.metadata.get(
            "atagia_technical_recovery_output_limit_strategy"
        )
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
        error: LLMError,
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
        return component_id_for_llm_purpose(
            purpose if isinstance(purpose, str) else None
        )

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
            return any(
                cls._nonordinary_intimacy_boundary(value) for value in boundaries
            )
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
        """Whether ``exc`` is a provider policy refusal worth rerouting.

        This predicate gates the ONE path in the client that answers an error by
        calling a provider again (the intimacy fallback model), so it is where a
        local refusal has to be excluded: the run guard's message is free text
        assembled from violation strings, and the substring test below is one
        unlucky wording away from reading "guard blocked the request" as
        "provider blocked the response" and spending a second call to be refused
        again.
        """
        if isinstance(exc, (InferenceAccessDeniedError, LLMRunGuardError)):
            return False
        if isinstance(exc, LLMPolicyBlockedError):
            return True
        if isinstance(
            exc, (OutputLimitExceededError, StructuredOutputError, TransientLLMError)
        ):
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
        *,
        route: ResolvedInferenceRoute | None = None,
    ) -> _ResolvedProviderRequest:
        if self._provider_name is not None:
            return _ResolvedProviderRequest(
                provider_name=self._provider_name,
                request=request,
                route=route
                or self._direct_provider_route(
                    request.model,
                    InferenceOperation.COMPLETION,
                ),
            )
        resolved_route = route or self._resolve_inference_route(
            request.model, InferenceOperation.COMPLETION
        )
        if resolved_route.provider_slug == "local":
            parsed = parse_local_model_spec(resolved_route.canonical_model_spec)
            return _ResolvedProviderRequest(
                provider_name=local_provider_registry_key(parsed.endpoint_id),
                request=self._local_completion_request_for_provider(request, parsed),
                route=resolved_route,
            )
        self._reject_zero_cost_openrouter_caller_features(request, resolved_route)
        parsed = self._parse_model_spec(request.model, allow_thinking=True)
        return _ResolvedProviderRequest(
            provider_name=parsed.provider_name,
            request=self._completion_request_for_provider(request, parsed),
            route=resolved_route,
        )

    def _completion_route_for_request(
        self,
        request: LLMCompletionRequest,
    ) -> ResolvedInferenceRoute:
        if self._provider_name is not None:
            return self._direct_provider_route(
                request.model,
                InferenceOperation.COMPLETION,
            )
        return self._resolve_inference_route(
            request.model,
            InferenceOperation.COMPLETION,
        )

    def _embedding_provider_request(
        self,
        request: LLMEmbeddingRequest,
    ) -> _ResolvedProviderRequest:
        if self._provider_name is not None:
            return _ResolvedProviderRequest(
                provider_name=self._provider_name,
                request=request,
                route=self._direct_provider_route(
                    request.model,
                    InferenceOperation.EMBEDDING,
                ),
            )
        route = self._resolve_inference_route(
            request.model,
            InferenceOperation.EMBEDDING,
        )
        if route.provider_slug == "local":
            parsed = parse_local_model_spec(
                route.canonical_model_spec,
                allow_thinking=False,
            )
            return _ResolvedProviderRequest(
                provider_name=local_provider_registry_key(parsed.endpoint_id),
                request=self._local_embedding_request_for_provider(request, parsed),
                route=route,
            )
        parsed = self._parse_model_spec(request.model, allow_thinking=False)
        metadata = dict(request.metadata)
        metadata.setdefault("atagia_model_spec", parsed.canonical_model)
        return _ResolvedProviderRequest(
            provider_name=parsed.provider_name,
            request=request.model_copy(
                update={
                    "model": parsed.request_model,
                    "metadata": metadata,
                }
            ),
            route=route,
        )

    def _resolve_inference_route(
        self,
        model_spec: str,
        operation: InferenceOperation,
    ) -> ResolvedInferenceRoute:
        try:
            return resolve_internal_inference_route(
                model_spec,
                operation,
                local_catalog=self._local_endpoint_catalog,
            )
        except InferenceRouteError as exc:
            if (
                self._inference_access_policy.mode
                is not InferenceAccessMode.UNRESTRICTED
                or model_spec.strip().lower().startswith("local/")
            ):
                if (
                    self._inference_access_policy.mode is InferenceAccessMode.ZERO_COST
                    and model_spec.strip().lower().startswith("openrouter/")
                ):
                    raise InferenceAccessDeniedError(
                        "Inference access denied: zero_cost denied a malformed "
                        "OpenRouter route."
                    ) from exc
                raise ConfigurationError(str(exc)) from exc
            parsed = self._parse_model_spec(
                model_spec,
                allow_thinking=operation is InferenceOperation.COMPLETION,
            )
            return ResolvedInferenceRoute(
                operation=operation,
                canonical_model_spec=parsed.canonical_spec,
                provider_slug=parsed.provider_slug,
                endpoint_id=None,
                base_url_class=BaseUrlClass.EXTERNAL,
                cost_class=InferenceCostClass.METERED,
            )

    @staticmethod
    def _direct_provider_route(
        model_spec: str,
        operation: InferenceOperation,
    ) -> ResolvedInferenceRoute:
        """Classify the legacy direct-provider shortcut as non-local.

        The shortcut has no catalog-backed endpoint identity, so a restricted
        policy must reject it even if request metadata claims otherwise.
        """
        return ResolvedInferenceRoute(
            operation=operation,
            canonical_model_spec=model_spec,
            provider_slug="direct",
            endpoint_id=None,
            base_url_class=BaseUrlClass.EXTERNAL,
            cost_class=InferenceCostClass.UNKNOWN,
        )

    def _provider_for_resolved_request(
        self,
        resolved: _ResolvedProviderRequest,
    ) -> LLMProvider:
        self._authorize_inference_route(resolved.route)
        return self._provider(resolved.provider_name)

    def _authorize_inference_route(self, route: ResolvedInferenceRoute) -> None:
        denial = self._inference_access_policy.denial_reason(
            route,
            local_catalog=self._local_endpoint_catalog,
        )
        if denial is not None:
            raise InferenceAccessDeniedError(
                f"Inference access denied for route {route.canonical_model_spec!r}: {denial}"
            )

    def _local_completion_request_for_provider(
        self,
        request: LLMCompletionRequest,
        parsed: ParsedLocalModelSpec,
    ) -> LLMCompletionRequest:
        self._reject_restricted_local_provider_extra_body(request.metadata)
        metadata = copy.deepcopy(request.metadata)
        metadata["atagia_model_spec"] = parsed.canonical_spec
        metadata["atagia_canonical_model"] = parsed.canonical_spec
        metadata["atagia_provider_slug"] = "local"
        metadata["atagia_local_endpoint_id"] = parsed.endpoint_id
        if parsed.thinking_level is None:
            metadata.pop("reasoning_effort", None)
        else:
            metadata["reasoning_effort"] = parsed.thinking_level
        return request.model_copy(
            update={
                "model": parsed.served_model_id,
                "metadata": metadata,
            }
        )

    def _local_embedding_request_for_provider(
        self,
        request: LLMEmbeddingRequest,
        parsed: ParsedLocalModelSpec,
    ) -> LLMEmbeddingRequest:
        self._reject_restricted_local_provider_extra_body(request.metadata)
        metadata = copy.deepcopy(request.metadata)
        metadata["atagia_model_spec"] = parsed.canonical_spec
        metadata["atagia_canonical_model"] = parsed.canonical_spec
        metadata["atagia_provider_slug"] = "local"
        metadata["atagia_local_endpoint_id"] = parsed.endpoint_id
        return request.model_copy(
            update={
                "model": parsed.served_model_id,
                "metadata": metadata,
            }
        )

    def _reject_restricted_local_provider_extra_body(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """Keep a restricted local route's wire model owned by its catalog."""
        if (
            self._inference_access_policy.mode is InferenceAccessMode.ZERO_COST
            and "provider_extra_body" in metadata
        ):
            raise InferenceAccessDeniedError(
                "Inference access denied: zero_cost local routes do not permit "
                "provider_extra_body."
            )
        if self._inference_access_policy.restricted and metadata.get(
            "provider_extra_body"
        ):
            raise InferenceAccessDeniedError(
                "Inference access denied: restricted local routes do not permit "
                "provider_extra_body."
            )

    def _reject_zero_cost_openrouter_caller_features(
        self,
        request: LLMCompletionRequest,
        route: ResolvedInferenceRoute,
    ) -> None:
        """Reject caller-controlled OpenRouter routing and tool features."""
        if (
            self._inference_access_policy.mode is not InferenceAccessMode.ZERO_COST
            or route.provider_slug != "openrouter"
        ):
            return
        if "provider_extra_body" in request.metadata:
            raise InferenceAccessDeniedError(
                "Inference access denied: zero_cost OpenRouter does not permit "
                "caller-supplied provider_extra_body."
            )
        if request.tools:
            raise InferenceAccessDeniedError(
                "Inference access denied: zero_cost OpenRouter does not permit tools."
            )

    def _parse_model_spec(self, model: str, *, allow_thinking: bool) -> ParsedModelSpec:
        try:
            return parse_model_spec(model, allow_thinking=allow_thinking)
        except ModelResolutionError as exc:
            if (
                self._allow_unqualified_single_provider_models
                and len(self._providers) == 1
            ):
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
        if (
            self._inference_access_policy.mode is InferenceAccessMode.ZERO_COST
            and parsed.provider_slug == "openrouter"
        ):
            # The adapter owns the complete OpenRouter routing body. Profiles
            # may set ordinary-provider preferences, but none may reach this
            # restricted request.
            metadata.pop("provider_extra_body", None)
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
        if profile is not None and profile.omit_temperature:
            return _ResolvedTemperature(
                value=None,
                source="model_profile_omitted",
                requested=requested,
                reason=parsed.canonical_model,
            )
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
                provider_value = profile.thinking_level_map[
                    profile.default_thinking_level
                ]

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

        if parsed.provider_slug in {"kimi", "minimax"}:
            metadata = self._apply_profile_extra_body(profile, metadata)
            body = copy.deepcopy(metadata.get("provider_extra_body") or {})
            if parsed.provider_slug == "minimax" and isinstance(provider_value, str):
                thinking = body.get("thinking")
                if not isinstance(thinking, dict):
                    thinking = {}
                thinking["type"] = provider_value
                body["thinking"] = thinking
            if body:
                metadata["provider_extra_body"] = body
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
    def _deep_merge_dicts(
        cls, base: dict[str, Any], overlay: dict[str, Any]
    ) -> dict[str, Any]:
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
        recorder = self._diagnostic_recorder
        if recorder is None:
            return await self._complete_structured_with_response_impl(request, schema)
        with recorder.operation("structured_completion", component=component_id_for_llm_purpose(self._request_purpose(request)), card=self._request_purpose(request), user_id=str(request.metadata.get("user_id")) if request.metadata.get("user_id") is not None else None, input_data={"requested_model": request.model, "schema": recorder.blob(schema.model_json_schema() if hasattr(schema, "model_json_schema") else str(schema))}):
            result = await self._complete_structured_with_response_impl(request, schema)
            recorder.no_call("structured_validation", component="llm_client", data={"status": "success", "parsed": recorder.blob(result.value), "used_schema_fallback": result.used_schema_fallback, "used_structured_output_retry": result.used_structured_output_retry, "used_structured_output_rescue": result.used_structured_output_rescue})
            return result

    async def _complete_structured_with_response_impl(
        self,
        request: LLMCompletionRequest,
        schema: type[T],
    ) -> StructuredCompletionResult[T]:
        if request.choice_questions is not None or request.score_questions is not None or request.finite_choice:
            raise ConfigurationError("Typed choices require complete(), not JSON generation")
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

        rescue_request = self._structured_output_rescue_request(
            request, schema, last_error
        )
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
            response = await self.complete(
                self._schema_drop_fallback_request(request, exc)
            )
        try:
            value = self._validate_structured_response(
                response, schema, used_schema_fallback
            )
        except StructuredOutputError as exc:
            recorder = self._diagnostic_recorder
            if recorder is not None:
                recorder.no_call("structured_validation", component="llm_client", data={"status": "failure", "error_type": type(exc).__name__, "reason": exc.reason, "details": exc.details})
            raise
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
        if request.choice_questions is not None or request.score_questions is not None or request.finite_choice:
            raise ConfigurationError("Typed choices require complete(), not JSON generation")
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

        rescue_request = self._structured_output_rescue_request(
            request, schema, last_error
        )
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
            response = await self.complete_streamed(
                completion_request, observer=observer
            )
        except LLMError as exc:
            if not self._should_retry_without_schema(exc, completion_request):
                raise
            used_schema_fallback = True
            response = await self.complete_streamed(
                self._schema_drop_fallback_request(request, exc),
                observer=observer,
            )
        return self._validate_structured_response(
            response, schema, used_schema_fallback
        )

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
    def _structured_output_error_details_for_prompt(
        error: StructuredOutputError,
    ) -> str:
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
            text = (
                text[:_STRUCTURED_OUTPUT_REPAIR_MAX_OUTPUT_CHARS] + "\n...[truncated]"
            )
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
        resolved = self._completion_provider_request(request)
        provider = self._provider_for_resolved_request(resolved)
        provider_request = resolved.request
        assert isinstance(provider_request, LLMCompletionRequest)
        return not provider.supports_native_structured_output_for(provider_request)

    @classmethod
    def _schema_prompt_fallback_request(
        cls, request: LLMCompletionRequest
    ) -> LLMCompletionRequest:
        return request.model_copy(
            update={
                "response_schema": None,
                "messages": [
                    *request.messages,
                    LLMMessage(
                        role="user",
                        content=cls._schema_prompt_fallback_instruction(
                            request.response_schema
                        ),
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
        route: ResolvedInferenceRoute,
        provider_name: str,
    ) -> Any:
        recorder = self._diagnostic_recorder
        if recorder is None:
            return await self._with_retries_impl(operation, request=request, route=route, provider_name=provider_name)
        metadata = request.metadata
        safe_metadata = _diagnostic_metadata(metadata)
        if _diagnostic_has_credentials(safe_metadata.get("provider_extra_body")):
            recorder._fail(ValueError("provider_extra_body contains credential fields"))
        request_ref = recorder.blob(request.model_copy(update={"metadata": safe_metadata}))
        prompt_ref = recorder.blob(request.messages) if isinstance(request, LLMCompletionRequest) else recorder.blob(request.input_texts)
        contract_ref = recorder.blob({"choice_questions": request.choice_questions, "score_questions": request.score_questions, "response_schema": request.response_schema, "tools": request.tools}) if isinstance(request, LLMCompletionRequest) else None
        with recorder.operation(
            self._request_purpose(request),
            component=component_id_for_llm_purpose(self._request_purpose(request)),
            card=str(metadata.get("stage")) if metadata.get("stage") else None,
            user_id=str(metadata.get("user_id")) if metadata.get("user_id") is not None else None,
            turn_id=str(metadata.get("turn_id")) if metadata.get("turn_id") is not None else None,
            job_id=str(metadata.get("job_id")) if metadata.get("job_id") is not None else None,
            input_data={"request": request_ref, "requested_model": request.model, "resolved_provider": provider_name, "prompt_sha256": prompt_ref["sha256"] if prompt_ref else None, "contract_sha256": contract_ref["sha256"] if contract_ref else None},
        ):
            return await self._with_retries_impl(operation, request=request, route=route, provider_name=provider_name)

    async def _with_retries_impl(
        self,
        operation: Any,
        *,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        route: ResolvedInferenceRoute,
        provider_name: str,
    ) -> Any:
        retry_policy = self._retry_policy_for(request)
        last_error: Exception | None = None
        for attempt in range(1, retry_policy.attempts + 1):
            self._authorize_inference_route(route)
            permit = await self._acquire_dispatch(provider_name)
            try:
                guarded_call = self._begin_guarded_call(request)
            except BaseException:
                permit.release()
                raise
            started_at = perf_counter()
            recorder = self._diagnostic_recorder
            current = current_operation(recorder)
            attempt_id = uuid4().hex if recorder is not None and current is not None else None
            if recorder is not None and current is not None and attempt_id is not None:
                recorder.event("provider_attempt", phase="start", trace_id=current[0], operation_id=current[1], attempt_id=attempt_id, purpose=self._request_purpose(request), status="started", data={"attempt_number": attempt, "request_kind": "embedding" if isinstance(request, LLMEmbeddingRequest) else "completion", "resolved_provider": provider_name, "resolved_model": request.model, "started_at": _diagnostic_utc_now()})
            try:
                with bind_attempt(recorder, attempt_id) if recorder is not None and attempt_id is not None else nullcontext():
                    response = await operation()
            except TransientLLMError as exc:
                self._diagnostic_attempt_end(attempt_id, request, "failure", error=exc)
                self._record_provider_call_failure(
                    request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                last_error = exc
                if attempt == retry_policy.attempts:
                    break
                permit.release()
                await self._sleep_before_retry(
                    retry_policy,
                    attempt,
                    exc,
                    request=request,
                )
            except Exception as exc:
                self._diagnostic_attempt_end(attempt_id, request, "failure", error=exc)
                self._record_provider_call_failure(
                    request,
                    guarded_call,
                    exc=exc,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                raise
            except BaseException:
                self._diagnostic_attempt_end(attempt_id, request, "cancelled")
                # CancelledError and GeneratorExit do not derive from Exception,
                # so without this clause an abandoned call spends provider work
                # that no counter ever sees. A non-streamed call has no partial
                # response to read usage from: the provider either returned or
                # it did not.
                self._record_provider_call_cancelled(
                    request,
                    guarded_call,
                    usage=None,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                raise
            else:
                self._record_provider_call_success(
                    request,
                    guarded_call,
                    usage=getattr(response, "usage", None),
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                )
                self._diagnostic_attempt_end(attempt_id, request, "success", response=response)
                return response
            finally:
                permit.release()
        if last_error is None:
            raise LLMError("LLM operation failed without a captured error")
        raise last_error

    def _diagnostic_attempt_end(
        self,
        attempt_id: str | None,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        status: str,
        *,
        response: Any = None,
        error: BaseException | None = None,
        partial_output: str | None = None,
        partial_usage: dict[str, Any] | None = None,
    ) -> None:
        recorder = self._diagnostic_recorder
        current = current_operation(recorder)
        if recorder is None or current is None or attempt_id is None:
            return
        raw = getattr(response, "raw_response", None)
        if isinstance(response, LLMEmbeddingResponse):
            response_content: Any = {
                "provider": response.provider,
                "model": response.model,
                "vector_count": len(response.vectors),
                "dimensions": [len(vector.values) for vector in response.vectors],
            }
            raw = None
        else:
            response_content = response
        usage = getattr(response, "usage", None) if response is not None else partial_usage
        reported_cost = None
        if isinstance(usage, dict):
            if "cost" in usage:
                reported_cost = usage["cost"]
            elif isinstance(usage.get("cost_details"), dict):
                reported_cost = usage["cost_details"].get("upstream_inference_cost")
        recorder.event("provider_attempt", phase="end", trace_id=current[0], operation_id=current[1], attempt_id=attempt_id, purpose=self._request_purpose(request), status=status, data={"response": recorder.blob(response_content) if response is not None else None, "raw_response": recorder.blob(raw) if raw is not None else None, "partial_output": recorder.blob(partial_output) if partial_output is not None else None, "usage": usage if isinstance(usage, dict) else None, "usage_provenance": "provider" if usage else "unknown", "cost_usd": reported_cost, "cost_provenance": "provider" if reported_cost is not None else "unknown", "error_type": type(error).__name__ if error is not None else None, "finished_at": _diagnostic_utc_now()})

    def _begin_guarded_call(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
    ) -> LLMRunGuardCall | None:
        """Gate one provider round-trip and return the ticket to record it on.

        THE PRE-CALL CHECK IS THE ONLY GATE. Every outcome recording below is
        pure bookkeeping that latches a verdict for the NEXT call; none of them
        may turn a round-trip that already reached the provider into an error to
        its caller. Returns ``None`` when no guard is configured, which is the
        normal state for bare clients in tests -- the per-turn meter still runs.
        """
        if self._llm_run_guard is None:
            return None
        guarded_call = self._llm_run_guard.begin_call(
            purpose=self._request_purpose(request),
            request_model=request.model,
        )
        decision = guarded_call.decision
        if decision.should_block:
            logger.error(
                "LLM run guard blocked further provider calls",
                extra={
                    "violations": list(decision.violations),
                    "llm_guard": decision.snapshot,
                },
            )
            raise LLMRunGuardError(decision)
        return guarded_call

    def _record_provider_call_success(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        guarded_call: LLMRunGuardCall | None,
        *,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> None:
        # Per-turn trace meter first: it must reflect every provider round-trip
        # regardless of whether a run guard is configured (the guard is optional;
        # bare clients in tests have none). This is the single success choke point
        # for completion/streamed_completion/stream, so it cannot miss a call.
        record_call_on_active_meter(
            purpose=self._request_purpose(request),
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.SUCCESS,
        )
        if guarded_call is None or self._llm_run_guard is None:
            return
        self._log_if_guard_tripped(
            self._llm_run_guard.record_success(
                guarded_call,
                usage=usage or {},
                latency_ms=latency_ms,
            )
        )

    def _record_provider_call_failure(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        guarded_call: LLMRunGuardCall | None,
        *,
        exc: BaseException,
        latency_ms: float,
    ) -> None:
        # A failed attempt is still a real provider round-trip: count it in the
        # per-turn meter (guard-independent) before the guard's failure logic.
        record_call_on_active_meter(
            purpose=self._request_purpose(request),
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.FAILURE,
        )
        if guarded_call is None or self._llm_run_guard is None:
            return
        self._log_if_guard_tripped(
            self._llm_run_guard.record_failure(
                guarded_call,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
        )

    def _record_provider_call_cancelled(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        guarded_call: LLMRunGuardCall | None,
        *,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> None:
        """Record a round-trip whose caller went away (disconnect, cancellation).

        The provider did the work and the operator pays for it, so it must be
        counted; it is not a provider failure, so it must not move the health
        signals. Recorded BEFORE the iterator is closed and the exception is
        re-raised, so telemetry survives a cleanup that itself gets cancelled.

        ``usage`` is whatever the stream had already published when the caller
        left, which is normally nothing: a cancelled stream never delivers the
        terminal event that carries the provider's usage totals. See
        ``LLMRunGuard.record_cancellation`` for what that costs.
        """
        record_call_on_active_meter(
            purpose=self._request_purpose(request),
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.CANCELLED,
        )
        if guarded_call is None or self._llm_run_guard is None:
            return
        self._log_if_guard_tripped(
            self._llm_run_guard.record_cancellation(
                guarded_call,
                usage=usage or {},
                latency_ms=latency_ms,
            )
        )

    @staticmethod
    def _log_if_guard_tripped(decision: LLMRunGuardDecision) -> None:
        """Report the outcome on which the run crossed into violation.

        Keyed on ``tripped``, NOT on ``should_block``: audit mode returns
        ``should_block=False`` by construction, so a log keyed on blocking never
        records the moment the guard would have fired -- which is the only event
        audit mode exists to produce. Keyed on ``violations`` being non-empty it
        would instead fire on every call for as long as the run stays degraded.
        ``tripped`` is true exactly once per trip, in both modes.

        Deliberately not an exception: the round-trip this decision came from is
        already spent, and its result belongs to the caller that paid for it.
        """
        if not decision.tripped:
            return
        logger.error(
            "LLM run guard tripped"
            + (
                "; the next provider call will be blocked"
                if decision.should_block
                else " (audit mode: calls continue, nothing is blocked)"
            ),
            extra={
                "violations": list(decision.violations),
                "llm_guard": decision.snapshot,
            },
        )

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
        if purpose is not None and purpose in _PARTIAL_STREAM_RETRY_PURPOSES:
            return self._extraction_retry_policy
        return self._retry_policy

    async def _sleep_before_retry(
        self,
        retry_policy: RetryPolicy,
        attempt: int,
        exc: TransientLLMError,
        *,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
    ) -> None:
        if self._should_defer_long_retry_after(request, retry_policy, exc):
            raise exc
        await asyncio.sleep(self._retry_delay_seconds(retry_policy, attempt, exc))

    def _should_defer_long_retry_after(
        self,
        request: LLMCompletionRequest | LLMEmbeddingRequest,
        retry_policy: RetryPolicy,
        exc: TransientLLMError,
    ) -> bool:
        retry_after = exc.retry_after_seconds
        if retry_after is None or retry_after <= retry_policy.max_delay_seconds:
            return False
        return (
            self._request_purpose(request) in _BACKGROUND_DEFERABLE_RETRY_AFTER_PURPOSES
        )

    @staticmethod
    def _retry_delay_seconds(
        retry_policy: RetryPolicy,
        attempt: int,
        exc: TransientLLMError,
    ) -> float:
        retry_after = exc.retry_after_seconds
        if retry_after is not None:
            return min(retry_after, retry_policy.max_delay_seconds)
        if retry_policy.retry_delays_seconds:
            index = min(max(0, attempt - 1), len(retry_policy.retry_delays_seconds) - 1)
            delay = float(retry_policy.retry_delays_seconds[index])
        else:
            delay = retry_policy.base_delay_seconds * (2 ** max(0, attempt - 1))
        delay = min(max(0.0, delay), retry_policy.max_delay_seconds)
        jitter_fraction = max(0.0, retry_policy.jitter_fraction)
        if delay <= 0.0 or jitter_fraction <= 0.0:
            return delay
        jittered = delay * random.uniform(1.0 - jitter_fraction, 1.0 + jitter_fraction)
        return min(max(0.0, jittered), retry_policy.max_delay_seconds)

    @staticmethod
    def _should_retry_after_partial_stream(request: LLMCompletionRequest) -> bool:
        purpose = request.metadata.get("purpose")
        if (
            not isinstance(purpose, str)
            or purpose not in _PARTIAL_STREAM_RETRY_PURPOSES
        ):
            return False
        mode = request.metadata.get("atagia_partial_stream_retry")
        if isinstance(mode, str) and mode.strip().lower() == "discard_and_retry":
            return True
        return False

    @staticmethod
    def _should_retry_without_schema(
        exc: LLMError, request: LLMCompletionRequest
    ) -> bool:
        """Decide whether to retry via prompt-JSON after a native-schema failure.

        ``request`` is the request that was actually sent. When the schema was
        already dropped for the prompt-JSON fallback, ``response_schema`` is
        ``None`` and there is nothing to retry without — so the gate also
        guarantees the failed attempt used the native structured path. The
        trigger is a typed client-request-class 4xx, not error-string matching.
        """
        if isinstance(exc, InferenceAccessDeniedError):
            return False
        if request.response_schema is None:
            return False
        return isinstance(exc, LLMRequestError) and 400 <= exc.status_code < 500

    @staticmethod
    def _decode_json_payload(output_text: str) -> Any:
        return decode_structured_json_payload(output_text).data

    @staticmethod
    async def _close_stream_iterator(stream_iterator: Any) -> None:
        """Close a provider iterator without letting its close become the outcome.

        EVERY CALLER REACHES HERE FROM AN EXCEPT HANDLER, holding an outcome it
        is about to record and re-raise. A provider whose iterator raises on
        close must not displace that outcome, and it would in three separate
        ways: at the call sites that close BEFORE recording, the round-trip's
        failure is never metered at all; at the ones that close after, an
        unrelated teardown error replaces the LLMError the caller diagnosed; and
        in the proxy's abandon path it escapes ``_ClosingStreamingResponse``'s
        ``finally`` into the ASGI task error log, having skipped the claim
        resolution queued behind it.

        So the close failure is LOGGED with its traceback and the original
        outcome continues. That is a diagnosis, not a fallback: nothing here is
        being recovered or retried, because there is nothing left to recover --
        the round-trip is already spent, already recorded, and the only thing
        lost is a socket the provider's own transport still owns.

        ``BaseException`` is deliberately NOT caught. A close that is cancelled
        is the enclosing cancellation being re-delivered at this suspension
        point, and swallowing that would strand the cancel.
        """
        aclose = getattr(stream_iterator, "aclose", None)
        if not callable(aclose):
            return
        try:
            result = aclose()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.warning(
                "Provider stream iterator raised while closing", exc_info=True
            )

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
                    reason="schema_fallback_non_json",
                ) from exc
            raise StructuredOutputError(
                "Provider returned non-JSON structured output",
                details=exc.details,
                output_text=response.output_text,
                reason="non_json",
            ) from exc

        adapter = TypeAdapter(schema)
        try:
            return adapter.validate_python(payload)
        except Exception as exc:
            raise StructuredOutputError(
                "Provider returned invalid structured output",
                details=self._structured_error_details(exc),
                output_text=response.output_text,
                reason="schema_validation",
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
