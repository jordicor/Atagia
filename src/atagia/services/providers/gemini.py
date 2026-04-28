"""Native Gemini provider adapter using the modern Google Gen AI SDK.

Current Gemini model families documented by Google AI for Developers include
Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`), Gemini 3 Flash Preview
(`gemini-3-flash-preview`), Gemini 3.1 Flash-Lite Preview
(`gemini-3.1-flash-lite-preview`), and Gemini Embedding
(`gemini-embedding-2`). The adapter accepts any model string supplied by the
caller and does not gate requests to a hardcoded model list.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import copy
import json
from typing import Any
from uuid import uuid4

import httpx
from google import genai as google_genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMError,
    LLMProvider,
    LLMStreamEvent,
    OutputLimitExceededError,
    TransientLLMError,
)


_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BLOCKING_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    }
)
_TRUNCATION_FINISH_REASONS = frozenset({"MAX_TOKENS"})
_TRANSIENT_FINISH_REASONS = frozenset(
    {
        "OTHER",
        "MALFORMED_FUNCTION_CALL",
        "FINISH_REASON_UNSPECIFIED",
        "UNSPECIFIED",
    }
)
_SUCCESS_FINISH_REASONS = frozenset({"STOP", "FINISH_REASON_STOP"})
_UNSUPPORTED_SCHEMA_KEYS = {
    "additionalProperties",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxItems",
    "minimum",
    "maximum",
    "minItems",
    "multipleOf",
}


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _getattr_or_key(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _usage_to_dict(usage: Any) -> dict[str, int | float]:
    """Normalize Gemini usage metadata to common numeric token fields."""
    if usage is None:
        return {}
    raw = _model_dump(usage)
    aliases = {
        "prompt_token_count": "input_tokens",
        "cached_content_token_count": "cache_tokens",
        "response_token_count": "output_tokens",
        "candidates_token_count": "output_tokens",
        "tool_use_prompt_token_count": "tool_use_prompt_tokens",
        "thoughts_token_count": "thinking_tokens",
        "total_token_count": "total_tokens",
    }
    normalized: dict[str, int | float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        normalized[aliases.get(key, key)] = value
    return normalized


def _json_pointer_get(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise LLMError(f"Gemini schema only supports local JSON refs, got {ref!r}")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise LLMError(f"Gemini schema ref could not be resolved: {ref!r}")
        current = current[part]
    return current


def _is_null_schema(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "null"


def _sanitize_schema_node(
    node: Any,
    *,
    root: dict[str, Any],
    ref_stack: tuple[str, ...] = (),
) -> Any:
    if isinstance(node, list):
        return [
            _sanitize_schema_node(item, root=root, ref_stack=ref_stack)
            for item in node
        ]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = str(node["$ref"])
        if ref in ref_stack:
            raise LLMError(f"Gemini schema contains a cyclic ref: {ref!r}")
        resolved = copy.deepcopy(_json_pointer_get(root, ref))
        for key, value in node.items():
            if key != "$ref" and not key.startswith("$"):
                resolved[key] = value
        return _sanitize_schema_node(
            resolved,
            root=root,
            ref_stack=(*ref_stack, ref),
        )

    cleaned: dict[str, Any] = {}
    nullable = False
    for key, value in node.items():
        if key in {"$defs", "definitions"} or key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key.startswith("$"):
            continue
        if key == "type" and isinstance(value, list):
            non_null_types = [item for item in value if item != "null"]
            nullable = nullable or len(non_null_types) != len(value)
            if len(non_null_types) == 1:
                cleaned["type"] = non_null_types[0]
            elif non_null_types:
                cleaned["anyOf"] = [{"type": item} for item in non_null_types]
            continue
        if key in {"anyOf", "oneOf"} and isinstance(value, list):
            variants = []
            for variant in value:
                if _is_null_schema(variant):
                    nullable = True
                    continue
                variants.append(
                    _sanitize_schema_node(variant, root=root, ref_stack=ref_stack)
                )
            if len(variants) == 1:
                for child_key, child_value in variants[0].items():
                    cleaned.setdefault(child_key, child_value)
            elif variants:
                cleaned["anyOf"] = variants
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                property_name: _sanitize_schema_node(
                    property_schema,
                    root=root,
                    ref_stack=ref_stack,
                )
                for property_name, property_schema in value.items()
            }
            continue
        if key == "items":
            cleaned["items"] = _sanitize_schema_node(
                value,
                root=root,
                ref_stack=ref_stack,
            )
            continue
        cleaned[key] = _sanitize_schema_node(value, root=root, ref_stack=ref_stack)

    if cleaned.get("type") == "null":
        cleaned.pop("type")
        nullable = True
    if nullable:
        cleaned["nullable"] = True
    return cleaned


def _sanitize_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip or inline JSON Schema features rejected by Gemini response schemas."""
    root = copy.deepcopy(schema)
    sanitized = _sanitize_schema_node(root, root=root)
    if not isinstance(sanitized, dict):
        raise LLMError("Gemini response schemas must sanitize to a JSON object")
    return sanitized


def _split_messages(
    request: LLMCompletionRequest,
) -> tuple[str | None, list[genai_types.Content]]:
    system_messages: list[str] = []
    contents: list[genai_types.Content] = []

    for message in request.messages:
        if message.role == "system":
            system_messages.append(message.content)
            continue
        if message.role == "assistant":
            contents.append(
                genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=message.content)],
                )
            )
            continue
        if message.role == "tool":
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_function_response(
                            name=message.name or "tool",
                            response={"content": message.content},
                        )
                    ],
                )
            )
            continue
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=message.content)],
            )
        )

    return "\n\n".join(system_messages) or None, contents


def _tools_for_gemini(tools: list[Any]) -> list[genai_types.Tool]:
    declarations = []
    for tool in tools:
        parameters = tool.input_schema or {"type": "object", "properties": {}}
        declarations.append(
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or None,
                parameters_json_schema=_sanitize_schema_for_gemini(parameters),
            )
        )
    if not declarations:
        return []
    return [genai_types.Tool(function_declarations=declarations)]


def _thinking_config(request: LLMCompletionRequest) -> genai_types.ThinkingConfig | None:
    """Build a ThinkingConfig for Gemini 3.x using `thinking_level`.

    Gemini 3 defaults to `thinking_level=HIGH` for supported models, which adds
    latency for callers that did not opt in. Atagia's model profiles set an
    explicit `metadata["gemini_thinking_level"]` when a supported model should
    override that provider default. Visible thought text still depends only on
    `include_thinking`.

    Note: per Google docs, `MINIMAL` does not guarantee that thinking is fully
    off — it is the closest equivalent available in Gemini 3. The legacy
    `thinking_budget` field still exists in the SDK but `thinking_level` is the
    official API for Gemini 3 and the two cannot be combined in one request.
    """
    level_override = request.metadata.get("gemini_thinking_level")
    if level_override:
        try:
            level = genai_types.ThinkingLevel[str(level_override).upper()]
        except KeyError as exc:
            raise LLMError(
                f"Unknown gemini_thinking_level value: {level_override!r}"
            ) from exc
        return genai_types.ThinkingConfig(
            include_thoughts=request.include_thinking,
            thinking_level=level,
        )
    if not request.include_thinking:
        return None
    return genai_types.ThinkingConfig(
        include_thoughts=True,
        thinking_level=genai_types.ThinkingLevel.HIGH,
    )


def _generation_config(request: LLMCompletionRequest) -> genai_types.GenerateContentConfig:
    system_instruction, _contents = _split_messages(request)
    kwargs: dict[str, Any] = {}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = _sanitize_schema_for_gemini(request.response_schema)
    tools = _tools_for_gemini(request.tools)
    if request.metadata.get("gemini_google_search"):
        tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
    if tools:
        kwargs["tools"] = tools
        kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(
            disable=True,
        )
    thinking = _thinking_config(request)
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    return genai_types.GenerateContentConfig(**kwargs)


def _completion_kwargs(request: LLMCompletionRequest) -> dict[str, Any]:
    _system_instruction, contents = _split_messages(request)
    return {
        "model": request.model,
        "contents": contents,
        "config": _generation_config(request),
    }


def _map_exception(exc: Exception) -> LLMError:
    if isinstance(exc, json.JSONDecodeError):
        return TransientLLMError("Provider returned a non-JSON HTTP response")
    if isinstance(exc, genai_errors.UnknownApiResponseError):
        return TransientLLMError(str(exc))
    if isinstance(exc, genai_errors.ServerError):
        return TransientLLMError(str(exc))
    if isinstance(exc, genai_errors.APIError):
        status_code = int(getattr(exc, "code", 0) or 0)
        if status_code in _TRANSIENT_STATUS_CODES or status_code >= 500:
            return TransientLLMError(str(exc))
        return LLMError(str(exc))
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return TransientLLMError(str(exc))
    if isinstance(exc, LLMError):
        return exc
    return LLMError(str(exc))


def _iter_parts(response: Any) -> list[Any]:
    parts: list[Any] = []
    candidates = _getattr_or_key(response, "candidates", []) or []
    for candidate in candidates:
        content = _getattr_or_key(candidate, "content")
        parts.extend(_getattr_or_key(content, "parts", []) or [])
    parts.extend(_getattr_or_key(response, "parts", []) or [])
    return parts


def _safe_response_text(response: Any) -> str:
    try:
        text = _getattr_or_key(response, "text", "")
    except ValueError:
        return ""
    return text or ""


def _finish_reason_label(value: Any) -> str | None:
    if value is None:
        return None
    label = getattr(value, "name", None) or str(value)
    if not label:
        return None
    return label.rsplit(".", 1)[-1].upper()


def _candidate_finish_reason(response: Any) -> str | None:
    """Return the first candidate finish reason as a normalized label."""
    candidates = _getattr_or_key(response, "candidates", []) or []
    if not candidates:
        return None
    return _finish_reason_label(_getattr_or_key(candidates[0], "finish_reason"))


def _finish_reason_error(finish_reason: str | None) -> LLMError | None:
    if finish_reason is None or finish_reason in _SUCCESS_FINISH_REASONS:
        return None
    if finish_reason in _TRUNCATION_FINISH_REASONS:
        return OutputLimitExceededError(
            "Gemini stopped because it reached max output tokens "
            f"(finish_reason={finish_reason})"
        )
    if finish_reason in _TRANSIENT_FINISH_REASONS:
        return TransientLLMError(
            f"Gemini returned retryable finish reason: {finish_reason}"
        )
    return None


def _empty_content_error(finish_reason: str | None) -> TransientLLMError:
    label = finish_reason or "unknown"
    return TransientLLMError(
        f"Gemini returned no output content (finish_reason={label})"
    )


def _blocked_reason(response: Any) -> str | None:
    """Return a normalized block reason if Gemini refused the prompt or response."""
    prompt_feedback = _getattr_or_key(response, "prompt_feedback")
    if prompt_feedback is not None:
        label = _finish_reason_label(_getattr_or_key(prompt_feedback, "block_reason"))
        if label and label not in {"BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED"}:
            return f"prompt:{label}"
    candidates = _getattr_or_key(response, "candidates", []) or []
    for candidate in candidates:
        label = _finish_reason_label(_getattr_or_key(candidate, "finish_reason"))
        if label in _BLOCKING_FINISH_REASONS:
            return f"response:{label}"
    return None


def _tool_call_from_function_call(function_call: Any) -> dict[str, Any]:
    call_id = _getattr_or_key(function_call, "id") or f"call_{uuid4().hex[:8]}"
    return {
        "id": call_id,
        "type": "function",
        "name": _getattr_or_key(function_call, "name"),
        "input": _as_dict(_getattr_or_key(function_call, "args", {})),
    }


def _collect_response_content(response: Any) -> tuple[str, str | None, list[dict[str, Any]]]:
    output_text = ""
    thinking_text = ""
    tool_calls: list[dict[str, Any]] = []
    saw_text_part = False

    for part in _iter_parts(response):
        function_call = _getattr_or_key(part, "function_call")
        if function_call is not None:
            tool_calls.append(_tool_call_from_function_call(function_call))
            continue
        text = _getattr_or_key(part, "text")
        if not text:
            continue
        saw_text_part = True
        if _getattr_or_key(part, "thought", False):
            thinking_text += text
        else:
            output_text += text

    if not saw_text_part:
        output_text = _safe_response_text(response)

    return output_text, thinking_text or None, tool_calls


class GeminiProvider(LLMProvider):
    """Gemini-backed implementation of the LLMProvider interface."""

    name = "gemini"
    supports_embeddings = True
    supports_embedding_dimensions = True

    def __init__(
        self,
        api_key: str,
        *,
        client: google_genai.Client | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        if vertex_project is not None or vertex_location is not None:
            raise NotImplementedError("Vertex AI Gemini clients are not implemented yet")
        self._client = google_genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        try:
            response = await self._client.aio.models.generate_content(
                **_completion_kwargs(request)
            )
        except Exception as exc:
            raise _map_exception(exc) from exc

        block_reason = _blocked_reason(response)
        if block_reason is not None:
            raise LLMError(f"Gemini blocked the response ({block_reason})")

        output_text, thinking, tool_calls = _collect_response_content(response)
        finish_reason = _candidate_finish_reason(response)
        finish_error = _finish_reason_error(finish_reason)
        if finish_error is not None:
            raise finish_error
        if not output_text and not tool_calls:
            raise _empty_content_error(finish_reason)

        return LLMCompletionResponse(
            provider=self.name,
            model=str(_getattr_or_key(response, "model_version", None) or request.model),
            output_text=output_text,
            thinking=thinking,
            tool_calls=tool_calls,
            usage=_usage_to_dict(_getattr_or_key(response, "usage_metadata")),
            raw_response=_model_dump(response),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        config_kwargs: dict[str, Any] = {}
        if request.dimensions is not None:
            config_kwargs["output_dimensionality"] = request.dimensions
        if request.metadata.get("task_type"):
            config_kwargs["task_type"] = request.metadata["task_type"]
        if request.metadata.get("title"):
            config_kwargs["title"] = request.metadata["title"]
        config = genai_types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        kwargs: dict[str, Any] = {
            "model": request.model,
            "contents": request.input_texts,
        }
        if config is not None:
            kwargs["config"] = config

        try:
            response = await self._client.aio.models.embed_content(**kwargs)
        except Exception as exc:
            raise _map_exception(exc) from exc

        vectors = [
            LLMEmbeddingVector(
                index=index,
                values=list(_getattr_or_key(embedding, "values", []) or []),
            )
            for index, embedding in enumerate(_getattr_or_key(response, "embeddings", []) or [])
        ]
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=vectors,
            raw_response=_model_dump(response),
        )

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        emitted_any = False
        emitted_output_or_tool = False
        usage: dict[str, int | float] = {}
        pending_error: Exception | None = None
        last_finish_reason: str | None = None
        try:
            stream = await self._client.aio.models.generate_content_stream(
                **_completion_kwargs(request)
            )
            async for chunk in stream:
                chunk_usage = _usage_to_dict(_getattr_or_key(chunk, "usage_metadata"))
                if chunk_usage:
                    usage = chunk_usage
                output_text, thinking, tool_calls = _collect_response_content(chunk)
                if thinking:
                    emitted_any = True
                    yield LLMStreamEvent(type="thinking", content=thinking)
                if output_text:
                    emitted_any = True
                    emitted_output_or_tool = True
                    yield LLMStreamEvent(type="text", content=output_text)
                for tool_call in tool_calls:
                    emitted_any = True
                    emitted_output_or_tool = True
                    yield LLMStreamEvent(type="tool_call", payload=tool_call)
                block_reason = _blocked_reason(chunk)
                if block_reason is not None:
                    pending_error = LLMError(
                        f"Gemini blocked the response ({block_reason})"
                    )
                    break
                finish_reason = _candidate_finish_reason(chunk)
                if finish_reason is not None:
                    last_finish_reason = finish_reason
                finish_error = _finish_reason_error(finish_reason)
                if finish_error is not None:
                    pending_error = finish_error
                    break
        except Exception as exc:
            if not emitted_any:
                raise _map_exception(exc) from exc
            pending_error = exc

        if pending_error is None and not emitted_output_or_tool:
            pending_error = _empty_content_error(last_finish_reason)

        yield LLMStreamEvent(type="done", payload={"usage": usage})
        if pending_error is not None:
            raise pending_error
