"""Provider-agnostic LLM client abstractions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
import re
from typing import Any, AsyncIterator, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

T = TypeVar("T")
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


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
        provider_name: str,
        providers: list[LLMProvider] | None = None,
        embedding_provider_name: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._provider_name = provider_name.strip().lower()
        self._providers = {
            provider.name.strip().lower(): provider for provider in (providers or [])
        }
        self._embedding_provider_name = (
            embedding_provider_name.strip().lower()
            if embedding_provider_name is not None
            else self._provider_name
        )
        self._retry_policy = retry_policy or RetryPolicy()

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name.strip().lower()] = provider

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def embedding_provider_name(self) -> str:
        return self._embedding_provider_name

    @property
    def provider(self) -> LLMProvider:
        return self._provider()

    @property
    def embedding_provider(self) -> LLMProvider:
        return self._provider(self._embedding_provider_name)

    def _provider(self, provider_name: str | None = None) -> LLMProvider:
        resolved_name = provider_name or self._provider_name
        provider = self._providers.get(resolved_name)
        if provider is None:
            raise ConfigurationError(f"Unsupported LLM provider: {resolved_name}")
        return provider

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return await self._with_retries(lambda: self._provider().complete(request))

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return await self._with_retries(
            lambda: self._provider(self._embedding_provider_name).embed(request)
        )

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        delay = self._retry_policy.base_delay_seconds
        last_error: Exception | None = None
        for attempt in range(1, self._retry_policy.attempts + 1):
            emitted_any = False
            try:
                provider = self._provider()
                async for event in provider.stream(request):
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
                                content=(
                                    "Return raw JSON only. Do not include markdown fences, "
                                    "explanations, or any text outside the JSON object."
                                ),
                            ),
                        ],
                    }
                )
            )
        try:
            payload = self._decode_json_payload(response.output_text)
        except json.JSONDecodeError as exc:
            if used_schema_fallback:
                raise StructuredOutputError(
                    "Provider returned non-JSON structured output after schema fallback",
                    details=self._structured_error_details(exc),
                ) from exc
            raise StructuredOutputError(
                "Provider returned non-JSON structured output",
                details=self._structured_error_details(exc),
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
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

        stripped = output_text.strip()
        fence_match = _JSON_FENCE_PATTERN.search(stripped)
        if fence_match is not None:
            fenced_payload = fence_match.group(1).strip()
            return json.loads(fenced_payload)

        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped[index:])
                return payload
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("Expecting value", output_text, 0)

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
