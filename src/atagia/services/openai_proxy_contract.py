"""Stable transport and replay primitives for the OpenAI-compatible proxy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Mapping

from atagia.services.llm_client import (
    LLMCompletionResponse,
    normalize_completion_finish_reason,
)


@dataclass(slots=True)
class OpenAIProxyProtocolError(Exception):
    """An OpenAI-shaped error whose HTTP mapping is part of the proxy API."""

    status_code: int
    message: str
    error_type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None
    retry_after_seconds: float | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class OpenAIProxyTurnClaims:
    """Canonical message-ID and sequence claims from headers and metadata."""

    request_message_id: str | None
    request_source_seq: int | None
    response_message_id: str | None
    response_source_seq: int | None


@dataclass(frozen=True, slots=True)
class OpenAIProxyCompletionEnvelope:
    """Provider-derived fields required for a lossless completion replay."""

    content: str
    tool_calls: tuple[dict[str, Any], ...]
    finish_reason: str | None
    usage: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": self.content,
            "tool_calls": [deepcopy(item) for item in self.tool_calls],
            "finish_reason": self.finish_reason,
        }
        if self.usage is not None:
            payload["usage"] = deepcopy(self.usage)
        return payload


def resolve_proxy_text_claim(
    field_name: str,
    claims: Mapping[str, Any],
    *,
    required: bool = False,
) -> str | None:
    """Resolve matching non-null string claims without precedence guessing."""

    supplied: list[tuple[str, str]] = []
    for source, value in claims.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise OpenAIProxyProtocolError(
                400,
                f"OpenAI proxy {field_name} claims must be non-empty strings",
                param=field_name,
                code=f"invalid_{field_name}",
            )
        supplied.append((source, value))
    if required and not supplied:
        raise OpenAIProxyProtocolError(
            400,
            f"OpenAI proxy requests require {field_name}",
            param=field_name,
            code=f"missing_{field_name}",
        )
    if not supplied:
        return None
    canonical = supplied[0][1]
    if any(value != canonical for _, value in supplied[1:]):
        raise OpenAIProxyProtocolError(
            400,
            f"Conflicting OpenAI proxy {field_name} claims",
            param=field_name,
            code=f"conflicting_{field_name}",
        )
    return canonical


def resolve_proxy_positive_int_claim(
    field_name: str,
    claims: Mapping[str, Any],
) -> int | None:
    """Resolve matching positive integer claims from headers and metadata."""

    supplied: list[tuple[str, int]] = []
    for source, value in claims.items():
        if value is None:
            continue
        if isinstance(value, bool):
            parsed = None
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = int(value)
            except ValueError:
                parsed = None
        else:
            parsed = None
        if parsed is None or parsed <= 0:
            raise OpenAIProxyProtocolError(
                400,
                f"OpenAI proxy {field_name} claims must be positive integers",
                param=field_name,
                code=f"invalid_{field_name}",
            )
        supplied.append((source, parsed))
    if not supplied:
        return None
    canonical = supplied[0][1]
    if any(value != canonical for _, value in supplied[1:]):
        raise OpenAIProxyProtocolError(
            400,
            f"Conflicting OpenAI proxy {field_name} claims",
            param=field_name,
            code=f"conflicting_{field_name}",
        )
    return canonical


def resolve_proxy_turn_claims(
    metadata: Mapping[str, Any],
    *,
    request_message_id_header: str | None,
    request_source_seq_header: str | None,
    response_message_id_header: str | None,
    response_source_seq_header: str | None,
) -> OpenAIProxyTurnClaims:
    """Collect every accepted turn-ID claim and reject contradictions."""

    return OpenAIProxyTurnClaims(
        request_message_id=resolve_proxy_text_claim(
            "message_id",
            {
                "header": request_message_id_header,
                "metadata.atagia_message_id": metadata.get("atagia_message_id"),
                "metadata.message_id": metadata.get("message_id"),
            },
        ),
        request_source_seq=resolve_proxy_positive_int_claim(
            "source_seq",
            {
                "header": request_source_seq_header,
                "metadata.atagia_source_seq": metadata.get("atagia_source_seq"),
                "metadata.source_seq": metadata.get("source_seq"),
            },
        ),
        response_message_id=resolve_proxy_text_claim(
            "response_message_id",
            {
                "header": response_message_id_header,
                "metadata.atagia_response_message_id": metadata.get(
                    "atagia_response_message_id"
                ),
                "metadata.response_message_id": metadata.get("response_message_id"),
            },
        ),
        response_source_seq=resolve_proxy_positive_int_claim(
            "response_source_seq",
            {
                "header": response_source_seq_header,
                "metadata.atagia_response_source_seq": metadata.get(
                    "atagia_response_source_seq"
                ),
                "metadata.response_source_seq": metadata.get("response_source_seq"),
            },
        ),
    )


def validate_proxy_turn_claim_relationships(
    claims: OpenAIProxyTurnClaims,
    *,
    require_id_pair: bool,
) -> None:
    """Validate sequence ownership and, when requested, all-or-none ID pairs."""

    if claims.request_source_seq is not None and claims.request_message_id is None:
        raise OpenAIProxyProtocolError(
            400,
            "source_seq requires message_id",
            param="source_seq",
            code="source_seq_without_message_id",
        )
    if claims.response_source_seq is not None and claims.response_message_id is None:
        raise OpenAIProxyProtocolError(
            400,
            "response_source_seq requires response_message_id",
            param="response_source_seq",
            code="response_source_seq_without_message_id",
        )
    if require_id_pair and (
        (claims.request_message_id is None) != (claims.response_message_id is None)
    ):
        raise OpenAIProxyProtocolError(
            400,
            "message_id and response_message_id must be supplied together",
            code="incomplete_message_id_pair",
        )


def completion_envelope_from_response(
    response: LLMCompletionResponse,
) -> OpenAIProxyCompletionEnvelope:
    """Build the canonical replay envelope without fabricating usage or reason."""

    return OpenAIProxyCompletionEnvelope(
        content=response.output_text,
        tool_calls=tuple(normalize_openai_tool_calls(response.tool_calls)),
        finish_reason=normalize_completion_finish_reason(
            response.finish_reason,
            has_tool_calls=bool(response.tool_calls),
        ),
        usage=deepcopy(response.usage) if response.usage else None,
    )


def normalize_openai_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize provider tool calls for response and replay persistence."""

    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = (
            tool_call.get("function")
            if isinstance(tool_call.get("function"), dict)
            else {}
        )
        name = str(function.get("name") or tool_call.get("name") or "tool")
        raw_arguments = (
            function.get("arguments")
            if "arguments" in function
            else tool_call.get("arguments", tool_call.get("input", {}))
        )
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, str)
            else json.dumps(raw_arguments, ensure_ascii=False)
        )
        normalized.append(
            {
                "id": str(tool_call.get("id") or f"call_atagia_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized
