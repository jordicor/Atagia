"""Anthropic provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)

from atagia.core.llm_output_limits import ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMPolicyBlockedError,
    LLMProvider,
    LLMRequestError,
    LLMStreamEvent,
    OutputLimitExceededError,
    TransientLLMError,
    retry_after_seconds_from_exception,
)
from atagia.services.llm_schema import strip_json_schema_nullability


_ANTHROPIC_TRANSIENT_STATUS_CODES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504, 529}
)


def _anthropic_status_error(exc: APIStatusError) -> LLMError:
    """Convert an Anthropic ``APIStatusError`` (incl. ``BadRequestError``) to an LLM error.

    ``BadRequestError`` is a subclass of ``APIStatusError`` in the SDK, so its
    400 status flows through here and becomes a typed ``LLMRequestError``.
    """
    status_code = int(getattr(exc, "status_code", 0) or 0)
    if status_code in _ANTHROPIC_TRANSIENT_STATUS_CODES:
        return TransientLLMError(
            str(exc),
            retry_after_seconds=retry_after_seconds_from_exception(exc),
        )
    if 400 <= status_code < 500:
        return LLMRequestError(str(exc), status_code=status_code)
    return LLMError(str(exc))


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _usage_to_dict(usage: Any) -> dict[str, int | float]:
    """Extract only numeric usage values (tokens, costs) from the API response."""
    if usage is None:
        return {}
    raw: dict[str, Any] = {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump(exclude_none=True)
    elif isinstance(usage, dict):
        raw = {k: v for k, v in usage.items() if v is not None}
    return {k: v for k, v in raw.items() if isinstance(v, (int, float))}


_TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
_BLOCKED_STOP_REASONS = frozenset({"refusal"})
_TRANSIENT_STOP_REASONS = frozenset({"pause_turn"})
_SUCCESS_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})
_ANTHROPIC_NO_TEMPERATURE_MODELS = frozenset(
    {"claude-opus-4-7", "claude-opus-4-8"}
)
_ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS = 12_000


def _canonical_model_name(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("anthropic/"):
        return normalized.split("/", 1)[1]
    return normalized


def _model_accepts_temperature(model: str) -> bool:
    return _canonical_model_name(model) not in _ANTHROPIC_NO_TEMPERATURE_MODELS


def _stop_reason_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    return label or None


def _stop_reason_error(stop_reason: str | None) -> LLMError | None:
    if stop_reason is None or stop_reason in _SUCCESS_STOP_REASONS:
        return None
    if stop_reason == "max_tokens":
        return OutputLimitExceededError(
            "Anthropic stopped because it reached max output tokens "
            f"(stop_reason={stop_reason})"
        )
    if stop_reason == "model_context_window_exceeded":
        return LLMError(
            "Anthropic stopped because it reached the model context window "
            f"(stop_reason={stop_reason})"
        )
    if stop_reason in _BLOCKED_STOP_REASONS:
        return LLMPolicyBlockedError(
            f"Anthropic blocked the response (stop_reason={stop_reason})"
        )
    if stop_reason in _TRANSIENT_STOP_REASONS:
        return TransientLLMError(
            f"Anthropic returned retryable stop reason: {stop_reason}"
        )
    return LLMError(f"Anthropic returned unsupported stop reason: {stop_reason}")


def _empty_content_error(stop_reason: str | None) -> TransientLLMError:
    label = stop_reason or "unknown"
    return TransientLLMError(
        f"Anthropic returned no output content (stop_reason={label})"
    )


def _transient_error(exc: BaseException) -> TransientLLMError:
    return TransientLLMError(
        str(exc),
        retry_after_seconds=retry_after_seconds_from_exception(exc),
    )


def _split_messages(
    request: LLMCompletionRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    use_prompt_cache = bool(request.metadata.get("anthropic_prompt_cache", False))
    system_blocks: list[dict[str, Any]] = []
    conversation_messages: list[dict[str, Any]] = []

    for message in request.messages:
        if message.role == "system":
            block: dict[str, Any] = {"type": "text", "text": message.content}
            if use_prompt_cache:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
            continue

        if message.role == "tool":
            conversation_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.name or "tool",
                            "content": message.content,
                        }
                    ],
                }
            )
            continue

        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or "tool"),
                        "name": str(tool_call.get("name") or "tool"),
                        "input": _tool_call_input(tool_call),
                    }
                )
            conversation_messages.append({"role": "assistant", "content": content})
            continue

        conversation_messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    return system_blocks, conversation_messages


def _tool_call_input(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = tool_call.get("input", tool_call.get("arguments", {}))
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"arguments": raw_arguments}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def _stream_tool_payload(
    content_block: Any,
    buffer: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if getattr(content_block, "type", None) == "tool_use":
        return {
            "id": getattr(content_block, "id", None),
            "type": "tool_use",
            "name": getattr(content_block, "name", None),
            "input": getattr(content_block, "input", {}),
        }
    if buffer is None:
        return None
    partial_json = str(buffer.pop("_partial_json", "") or "")
    if partial_json:
        try:
            parsed = json.loads(partial_json)
        except json.JSONDecodeError:
            parsed = {"arguments": partial_json}
        buffer["input"] = parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return buffer


def _thinking_config(request: LLMCompletionRequest, max_tokens: int) -> dict[str, Any] | None:
    if request.metadata.get("anthropic_thinking_adaptive"):
        return {"type": "adaptive"}
    budget = request.metadata.get("thinking_budget_tokens")
    if budget == -1:
        return {"type": "adaptive"}
    if isinstance(budget, int) and budget > 0:
        capped_budget = min(budget, max_tokens - 1)
        if capped_budget < 1:
            return None
        return {"type": "enabled", "budget_tokens": capped_budget}
    return None


_UNSUPPORTED_NUMBER_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
_UNSUPPORTED_ARRAY_KEYS = {"minItems", "maxItems"}


def _sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema features that the Anthropic API does not accept.

    Anthropic also requires every object node with `properties` to declare
    `additionalProperties: false` explicitly, even when the input schema omits
    the field entirely (Pydantic models with `extra="ignore"` do not emit it).
    """
    if not isinstance(node, dict):
        if isinstance(node, list):
            return [_sanitize_schema(item) for item in node]
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            cleaned[key] = False
            continue
        if key in _UNSUPPORTED_NUMBER_KEYS or key in _UNSUPPORTED_ARRAY_KEYS:
            continue
        cleaned[key] = _sanitize_schema(value)
    if cleaned.get("type") == "object" and isinstance(cleaned.get("properties"), dict):
        cleaned.setdefault("additionalProperties", False)
    return cleaned


def _anthropic_output_effort(request: LLMCompletionRequest) -> str | None:
    effort = request.metadata.get("anthropic_output_effort")
    if isinstance(effort, str) and effort in {"low", "medium", "high", "xhigh", "max"}:
        return effort
    return None


def _native_schema_payload(request: LLMCompletionRequest) -> dict[str, Any] | None:
    """Return the exact schema payload Anthropic will receive for this request.

    The native-schema gate and the senders must agree on the same object so the
    size cap decides on what is actually transmitted.
    """
    if request.response_schema is None:
        return None
    return _sanitize_schema(strip_json_schema_nullability(request.response_schema))


def _output_format_block(request: LLMCompletionRequest) -> dict[str, Any] | None:
    schema = _native_schema_payload(request)
    if schema is None:
        return None
    return {"type": "json_schema", "schema": schema}


def _output_config(request: LLMCompletionRequest) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    output_format = _output_format_block(request)
    if output_format is not None:
        config["format"] = output_format
    effort = _anthropic_output_effort(request)
    if effort is not None:
        config["effort"] = effort
    return config or None


class AnthropicProvider(LLMProvider):
    """Anthropic-backed implementation of the LLMProvider interface."""

    name = "anthropic"
    supports_embeddings = False

    def supports_native_structured_output_for(self, request: LLMCompletionRequest) -> bool:
        payload = _native_schema_payload(request)
        if payload is None:
            return True
        schema_text = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        return len(schema_text) <= _ANTHROPIC_NATIVE_SCHEMA_MAX_CHARS

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=request_timeout_seconds,
        )

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        system_blocks, messages = _split_messages(request)
        max_tokens = max(1, request.max_output_tokens or ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if request.temperature is not None and _model_accepts_temperature(request.model):
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        output_config = _output_config(request)
        if output_config is not None:
            kwargs["output_config"] = output_config
        thinking = _thinking_config(request, max_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking
        try:
            response = await self._client.messages.create(**kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise _transient_error(exc) from exc
        except APIStatusError as exc:
            raise _anthropic_status_error(exc) from exc
        except APIError as exc:
            raise LLMError(str(exc)) from exc

        output_text = ""
        thinking_text = ""
        tool_calls: list[dict[str, Any]] = []
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                output_text += getattr(block, "text", "")
            elif block_type == "thinking":
                thinking_text += getattr(block, "thinking", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", None),
                        "type": "tool_use",
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", {}),
                    }
                )

        stop_reason = _stop_reason_label(getattr(response, "stop_reason", None))
        stop_error = _stop_reason_error(stop_reason)
        if stop_error is not None:
            raise stop_error
        if not output_text and not tool_calls:
            raise _empty_content_error(stop_reason)

        return LLMCompletionResponse(
            provider=self.name,
            model=str(getattr(response, "model", None) or request.model),
            output_text=output_text,
            thinking=thinking_text or None,
            tool_calls=tool_calls,
            usage=_usage_to_dict(getattr(response, "usage", None)),
            raw_response=_model_dump(response),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise LLMError("Anthropic does not currently expose embeddings through this adapter")

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        system_blocks, messages = _split_messages(request)
        max_tokens = max(1, request.max_output_tokens or ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if request.temperature is not None and _model_accepts_temperature(request.model):
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        output_config = _output_config(request)
        if output_config is not None:
            kwargs["output_config"] = output_config
        thinking = _thinking_config(request, max_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking

        emitted_output_or_tool = False
        pending_error: Exception | None = None
        tool_buffers: dict[int, dict[str, Any]] = {}
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text":
                        emitted_output_or_tool = True
                        yield LLMStreamEvent(type="text", content=event.text)
                    elif event.type == "thinking":
                        yield LLMStreamEvent(type="thinking", content=event.thinking)
                    elif event.type == "content_block_start":
                        content_block = getattr(event, "content_block", None)
                        if getattr(content_block, "type", None) == "tool_use":
                            tool_buffers[int(getattr(event, "index", 0) or 0)] = {
                                "id": getattr(content_block, "id", None),
                                "type": "tool_use",
                                "name": getattr(content_block, "name", None),
                                "input": getattr(content_block, "input", {}) or {},
                                "_partial_json": "",
                            }
                    elif event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        buffer = tool_buffers.get(int(getattr(event, "index", 0) or 0))
                        if buffer is not None and getattr(delta, "type", None) == "input_json_delta":
                            buffer["_partial_json"] += str(
                                getattr(delta, "partial_json", "") or ""
                            )
                    elif event.type == "content_block_stop":
                        index = int(getattr(event, "index", 0) or 0)
                        payload = _stream_tool_payload(
                            getattr(event, "content_block", None),
                            tool_buffers.pop(index, None),
                        )
                        if payload is not None:
                            emitted_output_or_tool = True
                            yield LLMStreamEvent(
                                type="tool_call",
                                payload=payload,
                            )
                final_message = await stream.get_final_message()
                stop_reason = _stop_reason_label(
                    getattr(final_message, "stop_reason", None)
                )
                stop_error = _stop_reason_error(stop_reason)
                if stop_error is not None:
                    pending_error = stop_error
                elif stop_reason is not None and not emitted_output_or_tool:
                    pending_error = _empty_content_error(stop_reason)
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise _transient_error(exc) from exc
        except APIStatusError as exc:
            raise _anthropic_status_error(exc) from exc
        except APIError as exc:
            raise LLMError(str(exc)) from exc

        yield LLMStreamEvent(
            type="done",
            payload={"usage": _usage_to_dict(getattr(final_message, "usage", None))},
        )
        if pending_error is not None:
            raise pending_error
