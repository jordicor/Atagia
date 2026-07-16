"""Mechanical request and attachment resource budgets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from string import ascii_letters, digits
from typing import Any, Iterable, Iterator, Mapping

from atagia.integrations.message_projection import (
    MAX_MESSAGE_CONTENT_NESTING,
    MESSAGE_ATTACHMENT_BLOCK_TYPES,
    iter_message_content_blocks,
    iter_message_text_chunks,
    iter_tool_message_text_chunks,
)


_BASE64_ALPHABET = frozenset(ascii_letters + digits + "+/")
_ASCII_C0_AND_SPACE = "".join(chr(codepoint) for codepoint in range(0x21))
_ASCII_C0_AND_SPACE_SET = frozenset(_ASCII_C0_AND_SPACE)
DEFAULT_REQUEST_MAX_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_REQUEST_MAX_MESSAGE_TEXT_BYTES = 256 * 1024
DEFAULT_REQUEST_MAX_ATTACHMENTS = 16
DEFAULT_REQUEST_MAX_ATTACHMENT_DECODED_BYTES = 10 * 1024 * 1024
DEFAULT_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES = 20 * 1024 * 1024
DEFAULT_REQUEST_MAX_METADATA_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class RequestBudgetLimits:
    """Configured external request limits in bytes/counts."""

    body_bytes: int
    message_text_bytes: int
    attachments: int
    attachment_decoded_bytes: int
    attachments_decoded_bytes: int
    metadata_bytes: int

    @classmethod
    def from_settings(cls, settings: Any) -> "RequestBudgetLimits":
        return cls(
            body_bytes=int(
                getattr(
                    settings, "request_max_body_bytes", DEFAULT_REQUEST_MAX_BODY_BYTES
                )
            ),
            message_text_bytes=int(
                getattr(
                    settings,
                    "request_max_message_text_bytes",
                    DEFAULT_REQUEST_MAX_MESSAGE_TEXT_BYTES,
                )
            ),
            attachments=int(
                getattr(
                    settings, "request_max_attachments", DEFAULT_REQUEST_MAX_ATTACHMENTS
                )
            ),
            attachment_decoded_bytes=int(
                getattr(
                    settings,
                    "request_max_attachment_decoded_bytes",
                    DEFAULT_REQUEST_MAX_ATTACHMENT_DECODED_BYTES,
                )
            ),
            attachments_decoded_bytes=int(
                getattr(
                    settings,
                    "request_max_attachments_decoded_bytes",
                    DEFAULT_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES,
                )
            ),
            metadata_bytes=int(
                getattr(
                    settings,
                    "request_max_metadata_bytes",
                    DEFAULT_REQUEST_MAX_METADATA_BYTES,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RequestBudgetExceededError(Exception):
    """A payload is structurally valid but exceeds a configured resource cap."""

    field: str
    limit: int

    def __str__(self) -> str:
        return f"{self.field} exceeds the configured limit of {self.limit}"


@dataclass(frozen=True, slots=True)
class RequestPayloadStructureError(Exception):
    """A bounded payload contains invalid mechanical encoding."""

    field: str
    message: str

    def __str__(self) -> str:
        return self.message


def validate_direct_message_request_budget(
    *,
    message_text: str,
    attachments: Iterable[Any],
    metadata: Mapping[str, Any] | None,
    limits: RequestBudgetLimits,
) -> None:
    """Validate direct chat/sidecar fields before attachment decoding."""

    _validate_text_bytes("message_text", message_text, limits.message_text_bytes)
    attachment_list = list(attachments)
    if len(attachment_list) > limits.attachments:
        raise RequestBudgetExceededError("attachments", limits.attachments)
    if metadata is not None:
        _validate_metadata("metadata", metadata, limits.metadata_bytes)

    aggregate_decoded_bytes = 0
    for index, attachment in enumerate(attachment_list):
        value = _as_mapping(attachment)
        attachment_metadata = value.get("metadata")
        if isinstance(attachment_metadata, Mapping):
            _validate_metadata(
                f"attachments.{index}.metadata",
                attachment_metadata,
                limits.metadata_bytes,
            )
        decoded_size = 0
        content_text = value.get("content_text")
        if isinstance(content_text, str):
            decoded_size = _add_utf8_size(
                f"attachments.{index}.decoded_bytes",
                content_text,
                decoded_size,
                limits.attachment_decoded_bytes,
            )
        content_base64 = value.get("content_base64")
        if isinstance(content_base64, str):
            decoded_size += decoded_base64_size(
                content_base64,
                field=f"attachments.{index}.content_base64",
            )
        if decoded_size > limits.attachment_decoded_bytes:
            raise RequestBudgetExceededError(
                f"attachments.{index}.decoded_bytes",
                limits.attachment_decoded_bytes,
            )
        aggregate_decoded_bytes += decoded_size
        if aggregate_decoded_bytes > limits.attachments_decoded_bytes:
            raise RequestBudgetExceededError(
                "attachments.decoded_bytes",
                limits.attachments_decoded_bytes,
            )


def validate_openai_proxy_request_budget(
    request: Any,
    *,
    limits: RequestBudgetLimits,
) -> None:
    """Validate OpenAI text, metadata, and typed multimodal attachment blocks."""

    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, Mapping):
        _validate_metadata("metadata", metadata, limits.metadata_bytes)

    attachment_count = 0
    aggregate_decoded_bytes = 0
    for message_index, message in enumerate(getattr(request, "messages", ())):
        content = getattr(message, "content", None)
        message_field = f"messages.{message_index}.content"
        try:
            for block_path, block in iter_message_content_blocks(content):
                block_field = _message_block_field(message_field, block_path)
                block_type = str(block.get("type") or "").strip().lower()
                decoded_size = _openai_block_decoded_size(
                    block,
                    field=block_field,
                )
                if (
                    decoded_size is None
                    and block_type not in MESSAGE_ATTACHMENT_BLOCK_TYPES
                ):
                    continue
                attachment_count += 1
                if attachment_count > limits.attachments:
                    raise RequestBudgetExceededError("attachments", limits.attachments)
                attachment_metadata = block.get("metadata")
                if isinstance(attachment_metadata, Mapping):
                    _validate_metadata(
                        f"{block_field}.metadata",
                        attachment_metadata,
                        limits.metadata_bytes,
                    )
                resolved_size = decoded_size or 0
                if resolved_size > limits.attachment_decoded_bytes:
                    raise RequestBudgetExceededError(
                        f"{block_field}.decoded_bytes",
                        limits.attachment_decoded_bytes,
                    )
                aggregate_decoded_bytes += resolved_size
                if aggregate_decoded_bytes > limits.attachments_decoded_bytes:
                    raise RequestBudgetExceededError(
                        "attachments.decoded_bytes",
                        limits.attachments_decoded_bytes,
                    )
        except (RecursionError, ValueError) as exc:
            raise RequestPayloadStructureError(
                message_field,
                f"{message_field} has an unsupported structure",
            ) from exc
        try:
            is_tool_message = (
                str(getattr(message, "role", "")).strip().lower() == "tool"
            )
            if is_tool_message:
                _validate_json_string_lower_bound(
                    message_field,
                    content,
                    limits.message_text_bytes,
                )
                projected_chunks = iter_tool_message_text_chunks(content)
            else:
                projected_chunks = iter_message_text_chunks(
                    content,
                    include_attachment_placeholders=False,
                )
            _validate_text_chunks(
                message_field,
                projected_chunks,
                limits.message_text_bytes,
            )
        except (RecursionError, TypeError, ValueError) as exc:
            raise RequestPayloadStructureError(
                message_field,
                f"{message_field} has an unsupported structure",
            ) from exc


def _message_block_field(base: str, path: tuple[str | int, ...]) -> str:
    return "".join((base, *(f".{component}" for component in path)))


def decoded_base64_size(value: str, *, field: str) -> int:
    """Validate standard base64 mechanically and return decoded byte length."""

    return _decoded_base64_size_range(value, 0, len(value), field=field)


def _decoded_base64_size_range(
    value: str,
    start: int,
    end: int,
    *,
    field: str,
) -> int:

    encoded_chars = 0
    padding = 0
    saw_padding = False
    for index in range(start, end):
        character = value[index]
        if character.isspace():
            continue
        encoded_chars += 1
        if character == "=":
            saw_padding = True
            padding += 1
            if padding > 2:
                raise RequestPayloadStructureError(field, f"{field} is invalid base64")
            continue
        if saw_padding or character not in _BASE64_ALPHABET:
            raise RequestPayloadStructureError(field, f"{field} is invalid base64")
    if encoded_chars == 0:
        return 0
    if encoded_chars % 4 != 0:
        raise RequestPayloadStructureError(field, f"{field} is invalid base64")
    return (encoded_chars // 4) * 3 - padding


def _openai_block_decoded_size(
    block: Mapping[str, Any],
    *,
    field: str,
) -> int | None:
    for key in ("data", "content_base64", "base64"):
        value = block.get(key)
        if isinstance(value, str):
            return decoded_base64_size(value, field=f"{field}.{key}")
    for key in ("image_url", "file", "input_audio", "source"):
        nested = block.get(key)
        if isinstance(nested, Mapping):
            nested_size = _openai_block_decoded_size(nested, field=f"{field}.{key}")
            if nested_size is not None:
                return nested_size
        elif isinstance(nested, str):
            size = _data_url_decoded_size(nested, field=f"{field}.{key}")
            if size is not None:
                return size
    url = block.get("url")
    if isinstance(url, str):
        return _data_url_decoded_size(url, field=f"{field}.url")
    return None


def _data_url_decoded_size(value: str, *, field: str) -> int | None:
    start = 0
    end = len(value)
    while start < end and value[start] in _ASCII_C0_AND_SPACE_SET:
        start += 1
    while end > start and value[end - 1] in _ASCII_C0_AND_SPACE_SET:
        end -= 1
    if not _ascii_case_insensitive_equal_at(value, start, "data:"):
        return None
    marker = ";base64,"
    comma_index = value.find(",", start + 5, end)
    marker_index = comma_index - (len(marker) - 1)
    if marker_index < start + 5 or not _ascii_case_insensitive_equal_at(
        value,
        marker_index,
        marker,
    ):
        raise RequestPayloadStructureError(field, f"{field} is not a base64 data URL")
    return _decoded_base64_size_range(
        value,
        comma_index + 1,
        end,
        field=field,
    )


def _ascii_case_insensitive_equal_at(value: str, index: int, expected: str) -> bool:
    if index < 0 or index + len(expected) > len(value):
        return False
    for offset, expected_character in enumerate(expected):
        actual_character = value[index + offset]
        if actual_character == expected_character:
            continue
        if (
            "A" <= actual_character <= "Z"
            and chr(ord(actual_character) + 32) == expected_character
        ):
            continue
        return False
    return True


def _validate_text_bytes(field: str, value: str, limit: int) -> None:
    _add_utf8_size(field, value, 0, limit)


def _validate_text_chunks(field: str, chunks: Iterable[str], limit: int) -> None:
    size = 0
    for chunk in chunks:
        size = _add_utf8_size(field, chunk, size, limit)


def _validate_json_string_lower_bound(
    field: str,
    value: Any,
    limit: int,
) -> None:
    """Reject large JSON strings before JSON encoding duplicates them."""

    size = 0
    stack: list[Iterator[Any]] = [iter((value,))]
    while stack:
        iterator = stack[-1]
        try:
            current = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        if isinstance(current, str):
            size = _add_utf8_size(field, current, size, limit)
            continue
        if isinstance(current, list):
            if len(stack) > MAX_MESSAGE_CONTENT_NESTING:
                raise ValueError("message content nesting exceeds the supported limit")
            stack.append(iter(current))
            continue
        if isinstance(current, Mapping):
            if len(stack) > MAX_MESSAGE_CONTENT_NESTING:
                raise ValueError("message content nesting exceeds the supported limit")
            stack.append(_iter_mapping_keys_and_values(current))


def _iter_mapping_keys_and_values(value: Mapping[Any, Any]) -> Iterator[Any]:
    for key, item in value.items():
        yield key
        yield item


def _add_utf8_size(field: str, value: str, size: int, limit: int) -> int:
    for offset in range(0, len(value), 4_096):
        size += len(value[offset : offset + 4_096].encode("utf-8"))
        if size > limit:
            raise RequestBudgetExceededError(field, limit)
    return size


def _validate_metadata(field: str, value: Mapping[str, Any], limit: int) -> None:
    size = 0
    try:
        _validate_json_string_lower_bound(field, value, limit)
        chunks = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
        ).iterencode(value)
        for chunk in chunks:
            size = _add_utf8_size(field, chunk, size, limit)
    except RequestBudgetExceededError:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise RequestPayloadStructureError(
            field,
            f"{field} must be JSON serializable",
        ) from exc


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    return {}
