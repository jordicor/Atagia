"""OpenAI-compatible provider adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
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


def _getattr_or_key(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return {key: value for key, value in usage.items() if value is not None}
    return {}


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            for item in content
        )
    return str(content)


def _uses_max_completion_tokens(model: str) -> bool:
    model_lower = model.lower()
    return model_lower.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_messages(request: LLMCompletionRequest) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in request.messages:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            if message.role == "tool":
                payload["tool_call_id"] = message.name
            else:
                payload["name"] = message.name
        converted.append(payload)
    return converted


def _openai_tools(request: LLMCompletionRequest) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in request.tools
    ]


def _has_free_form_objects(schema: dict[str, Any]) -> bool:
    """Return True if the schema contains bare object nodes without properties.

    These nodes come from ``dict[str, Any]`` fields in Pydantic and are
    incompatible with OpenAI strict mode, which requires
    ``additionalProperties: false`` on every object -- effectively forcing
    property-less objects to always be ``{}``.
    """
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object" and "properties" not in schema:
        return True
    return any(
        _has_free_form_objects(v)
        for v in schema.values()
        if isinstance(v, (dict, list))
    ) or any(
        _has_free_form_objects(item)
        for v in schema.values()
        if isinstance(v, list)
        for item in v
        if isinstance(item, dict)
    )


def _add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Add ``additionalProperties: false`` to objects that have ``properties``.

    Only touches structured objects (those with explicit property definitions).
    Bare ``{"type": "object"}`` nodes (from ``dict[str, Any]``) are left
    untouched so they remain valid free-form maps.
    """
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    if (
        result.get("type") == "object"
        and "properties" in result
        and "additionalProperties" not in result
    ):
        result["additionalProperties"] = False
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = _add_additional_properties_false(value)
        elif isinstance(value, list):
            result[key] = [
                _add_additional_properties_false(item) if isinstance(item, dict) else item
                for item in value
            ]
    return result


def _response_format(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    strict = not _has_free_form_objects(schema)
    sanitized = _add_additional_properties_false(schema) if strict else schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "strict": strict,
            "schema": sanitized,
        },
    }


class OpenAICompatibleProvider(LLMProvider):
    """Shared implementation for OpenAI-style chat completion APIs."""

    name = "openai-compatible"
    supports_embedding_dimensions = True

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            max_retries=0,
        )

    def _completion_kwargs(self, request: LLMCompletionRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": _openai_messages(request),
            "stream": stream,
        }
        if request.tools:
            kwargs["tools"] = _openai_tools(request)
            kwargs["tool_choice"] = "auto"
        response_format = _response_format(request.response_schema)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if request.metadata.get("user_id"):
            kwargs["user"] = str(request.metadata["user_id"])
        if request.metadata.get("reasoning_effort"):
            kwargs["reasoning_effort"] = request.metadata["reasoning_effort"]
        if request.metadata.get("verbosity"):
            kwargs["verbosity"] = request.metadata["verbosity"]
        if request.temperature is not None and not _uses_max_completion_tokens(request.model):
            kwargs["temperature"] = request.temperature
        max_tokens = request.max_output_tokens
        if max_tokens is not None:
            token_field = "max_completion_tokens" if _uses_max_completion_tokens(request.model) else "max_tokens"
            kwargs[token_field] = max_tokens
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _map_completion_response(
        self,
        request: LLMCompletionRequest,
        response: Any,
    ) -> LLMCompletionResponse:
        choices = _getattr_or_key(response, "choices", [])
        first_choice = choices[0] if choices else None
        message = _getattr_or_key(first_choice, "message")
        tool_calls = []
        for tool_call in _getattr_or_key(message, "tool_calls", []) or []:
            function = _getattr_or_key(tool_call, "function", {})
            tool_calls.append(
                {
                    "id": _getattr_or_key(tool_call, "id"),
                    "type": _getattr_or_key(tool_call, "type", "function"),
                    "name": _getattr_or_key(function, "name"),
                    "arguments": _getattr_or_key(function, "arguments", ""),
                }
            )

        return LLMCompletionResponse(
            provider=self.name,
            model=_getattr_or_key(response, "model", request.model),
            output_text=_stringify_content(_getattr_or_key(message, "content")),
            thinking=_getattr_or_key(message, "reasoning", None),
            tool_calls=tool_calls,
            usage=_usage_to_dict(_getattr_or_key(response, "usage")),
            raw_response=_model_dump(response),
        )

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        try:
            response = await self._client.chat.completions.create(**self._completion_kwargs(request, stream=False))
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except (BadRequestError, APIStatusError, APIError) as exc:
            raise LLMError(str(exc)) from exc
        return self._map_completion_response(request, response)

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": request.input_texts,
        }
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        if request.metadata.get("user_id"):
            kwargs["user"] = str(request.metadata["user_id"])
        try:
            response = await self._client.embeddings.create(**kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except (BadRequestError, APIStatusError, APIError) as exc:
            raise LLMError(str(exc)) from exc

        vectors = [
            LLMEmbeddingVector(index=_getattr_or_key(item, "index", 0), values=list(_getattr_or_key(item, "embedding", [])))
            for item in _getattr_or_key(response, "data", [])
        ]
        return LLMEmbeddingResponse(
            provider=self.name,
            model=_getattr_or_key(response, "model", request.model),
            vectors=vectors,
            raw_response=_model_dump(response),
        )

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        try:
            stream = await self._client.chat.completions.create(**self._completion_kwargs(request, stream=True))
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except (BadRequestError, APIStatusError, APIError) as exc:
            raise LLMError(str(exc)) from exc

        usage: dict[str, Any] = {}
        tool_buffers: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in stream:
                chunk_usage = _usage_to_dict(_getattr_or_key(chunk, "usage"))
                if chunk_usage:
                    usage = chunk_usage

                for choice in _getattr_or_key(chunk, "choices", []):
                    delta = _getattr_or_key(choice, "delta", {})
                    thinking = _getattr_or_key(delta, "reasoning", None) or _getattr_or_key(
                        delta, "reasoning_content", None
                    )
                    if thinking:
                        yield LLMStreamEvent(type="thinking", content=thinking)

                    content = _getattr_or_key(delta, "content")
                    if content:
                        yield LLMStreamEvent(type="text", content=content)

                    for tool_call in _getattr_or_key(delta, "tool_calls", []) or []:
                        index = _getattr_or_key(tool_call, "index", 0)
                        function = _getattr_or_key(tool_call, "function", {})
                        buffer = tool_buffers.setdefault(
                            index,
                            {
                                "id": None,
                                "type": _getattr_or_key(tool_call, "type", "function"),
                                "name": "",
                                "arguments": "",
                            },
                        )
                        if _getattr_or_key(tool_call, "id"):
                            buffer["id"] = _getattr_or_key(tool_call, "id")
                        if _getattr_or_key(function, "name"):
                            buffer["name"] = _getattr_or_key(function, "name")
                        arguments = _getattr_or_key(function, "arguments", "")
                        if arguments:
                            buffer["arguments"] += arguments

                    if _getattr_or_key(choice, "finish_reason") == "tool_calls":
                        for index in sorted(tool_buffers):
                            yield LLMStreamEvent(type="tool_call", payload=tool_buffers[index])
                        tool_buffers.clear()
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except (BadRequestError, APIStatusError, APIError) as exc:
            raise LLMError(str(exc)) from exc

        for index in sorted(tool_buffers):
            yield LLMStreamEvent(type="tool_call", payload=tool_buffers[index])
        yield LLMStreamEvent(type="done", payload={"usage": usage})


class OpenAIProvider(OpenAICompatibleProvider):
    """Concrete OpenAI provider."""

    name = "openai"
