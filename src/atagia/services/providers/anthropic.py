"""Anthropic provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
    LLMStreamEvent,
    TransientLLMError,
)


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

        conversation_messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    return system_blocks, conversation_messages


def _thinking_config(request: LLMCompletionRequest, max_tokens: int) -> dict[str, Any] | None:
    if not request.include_thinking:
        return None

    budget = request.metadata.get("thinking_budget_tokens")
    model_lower = request.model.lower()
    is_adaptive_model = any(name in model_lower for name in ("opus-4-6", "sonnet-4-6"))

    if budget == -1 and is_adaptive_model:
        return {"type": "adaptive"}

    if isinstance(budget, int) and budget > 0:
        return {"type": "enabled", "budget_tokens": min(budget, max_tokens - 1)}

    if is_adaptive_model:
        return {"type": "adaptive"}

    default_budget = min(1024, max_tokens - 1)
    if default_budget < 1:
        return None
    return {"type": "enabled", "budget_tokens": default_budget}


_UNSUPPORTED_NUMBER_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
_UNSUPPORTED_ARRAY_KEYS = {"minItems", "maxItems"}


def _sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema features that the Anthropic API does not accept."""
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
    return cleaned


def _completion_output_config(request: LLMCompletionRequest) -> dict[str, Any] | None:
    if request.response_schema is None:
        return None
    return {
        "format": {
            "type": "json_schema",
            "schema": _sanitize_schema(request.response_schema),
        }
    }


def _stream_output_format(request: LLMCompletionRequest) -> dict[str, Any] | None:
    if request.response_schema is None:
        return None
    return {
        "type": "json_schema",
        "schema": _sanitize_schema(request.response_schema),
    }


class AnthropicProvider(LLMProvider):
    """Anthropic-backed implementation of the LLMProvider interface."""

    name = "anthropic"
    supports_embeddings = False

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        system_blocks, messages = _split_messages(request)
        max_tokens = max(1, request.max_output_tokens or 8192)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if request.temperature is not None:
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
        output_config = _completion_output_config(request)
        if output_config is not None:
            kwargs["output_config"] = output_config
        thinking = _thinking_config(request, max_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking
        try:
            response = await self._client.messages.create(**kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except APIStatusError as exc:
            if int(getattr(exc, "status_code", 0) or 0) in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except (BadRequestError, APIError) as exc:
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

        return LLMCompletionResponse(
            provider=self.name,
            model=getattr(response, "model", request.model),
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
        max_tokens = max(1, request.max_output_tokens or 8192)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if request.temperature is not None:
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
        output_format = _stream_output_format(request)
        if output_format is not None:
            kwargs["output_format"] = output_format
        thinking = _thinking_config(request, max_tokens)
        if thinking is not None:
            kwargs["thinking"] = thinking

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield LLMStreamEvent(type="text", content=event.text)
                    elif event.type == "thinking":
                        yield LLMStreamEvent(type="thinking", content=event.thinking)
                    elif event.type == "content_block_stop":
                        content_block = getattr(event, "content_block", None)
                        if getattr(content_block, "type", None) == "tool_use":
                            yield LLMStreamEvent(
                                type="tool_call",
                                payload={
                                    "id": getattr(content_block, "id", None),
                                    "type": "tool_use",
                                    "name": getattr(content_block, "name", None),
                                    "input": getattr(content_block, "input", {}),
                                },
                            )
                final_message = await stream.get_final_message()
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except APIStatusError as exc:
            if int(getattr(exc, "status_code", 0) or 0) in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except (BadRequestError, APIError) as exc:
            raise LLMError(str(exc)) from exc

        yield LLMStreamEvent(
            type="done",
            payload={"usage": _usage_to_dict(getattr(final_message, "usage", None))},
        )
