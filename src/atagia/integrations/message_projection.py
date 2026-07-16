"""Message-shape projection helpers for sidecar integrations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from typing import Any

TextFileLoader = Callable[[dict[str, Any]], str]
MessageBlockPath = tuple[str | int, ...]
MAX_MESSAGE_CONTENT_NESTING = 128

MESSAGE_TEXT_BLOCK_TYPES = frozenset({"input_text", "output_text", "text"})
MESSAGE_ATTACHMENT_BLOCK_TYPES = frozenset(
    {
        "audio",
        "document",
        "document_bytes",
        "document_url",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "text_file",
    }
)


def tool_message_to_text(value: Any) -> str:
    """Project tool content exactly as the OpenAI proxy sends it to the model."""

    return "".join(iter_tool_message_text_chunks(value))


def iter_tool_message_text_chunks(value: Any) -> Iterator[str]:
    """Yield the exact tool-message projection without building a second copy."""

    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (dict, list)):
        yield from json.JSONEncoder(ensure_ascii=False).iterencode(value)
        return
    yield str(value)


def message_to_text(
    value: Any,
    *,
    text_file_loader: TextFileLoader | None = None,
    include_attachment_placeholders: bool = True,
    _depth: int = 0,
) -> str:
    """Convert common host/provider message shapes into safe Atagia text.

    Binary or base64-heavy blocks are represented as placeholders unless the host
    supplies a text-file loader. Hosts that have first-class attachment metadata
    should pass attachments separately to Atagia rather than embedding raw bytes
    in the projected text.
    """
    return "".join(
        iter_message_text_chunks(
            value,
            text_file_loader=text_file_loader,
            include_attachment_placeholders=include_attachment_placeholders,
            _depth=_depth,
        )
    )


def iter_message_text_chunks(
    value: Any,
    *,
    text_file_loader: TextFileLoader | None = None,
    include_attachment_placeholders: bool = True,
    _depth: int = 0,
) -> Iterator[str]:
    """Yield the ``message_to_text`` projection incrementally."""

    if _depth > MAX_MESSAGE_CONTENT_NESTING:
        raise ValueError("message content nesting exceeds the supported limit")

    if value is None:
        return

    if isinstance(value, bytes):
        try:
            yield value.decode("utf-8", errors="ignore")
        except Exception:
            return
        return

    if isinstance(value, str):
        # A string is already typed text. JSON-looking text must remain
        # byte-for-byte intact; only values that arrived as dict/list content
        # take the structured projection path below.
        yield value
        return

    if isinstance(value, list):
        emitted_part = False
        for item in value:
            item_started = False
            for chunk in iter_message_text_chunks(
                item,
                text_file_loader=text_file_loader,
                include_attachment_placeholders=include_attachment_placeholders,
                _depth=_depth + 1,
            ):
                if not chunk:
                    continue
                if not item_started:
                    if emitted_part:
                        yield "\n"
                    item_started = True
                    emitted_part = True
                yield chunk
        return

    if isinstance(value, dict):
        if value.get("multi_ai") and isinstance(value.get("responses"), list):
            emitted_response = False
            for response in value["responses"]:
                if not isinstance(response, dict):
                    continue
                label = response.get("model") or response.get("machine") or "model"
                response_started = False
                for chunk in iter_message_text_chunks(
                    response.get("content"),
                    text_file_loader=text_file_loader,
                    include_attachment_placeholders=include_attachment_placeholders,
                    _depth=_depth + 1,
                ):
                    if not chunk:
                        continue
                    if not response_started:
                        if emitted_response:
                            yield "\n\n"
                        yield f"[{label}]\n"
                        response_started = True
                        emitted_response = True
                    yield chunk
            return

        block_type = value.get("type")
        if block_type in MESSAGE_TEXT_BLOCK_TYPES:
            yield str(value.get("text") or "")
            return
        if block_type == "text_file":
            if text_file_loader is not None:
                try:
                    yield text_file_loader(value)
                    return
                except Exception:
                    pass
            if not include_attachment_placeholders:
                return
            filename = _filename_from_block(value, default="attached text file")
            yield f"[Text file attached: {filename}]"
            return
        if block_type in {"image", "image_url", "input_image"}:
            if not include_attachment_placeholders:
                return
            yield "[Image attached]"
            return
        if block_type in {
            "document",
            "document_bytes",
            "document_url",
            "file",
            "input_file",
        }:
            if not include_attachment_placeholders:
                return
            filename = _filename_from_block(value, default="document")
            yield f"[Document attached: {filename}]"
            return
        if block_type in {"audio", "input_audio"}:
            if not include_attachment_placeholders:
                return
            filename = _filename_from_block(value, default="audio")
            yield f"[Audio attached: {filename}]"
            return

        if "message" in value:
            yield from iter_message_text_chunks(
                value.get("message"),
                text_file_loader=text_file_loader,
                include_attachment_placeholders=include_attachment_placeholders,
                _depth=_depth + 1,
            )
            return
        if "content" in value:
            yield from iter_message_text_chunks(
                value.get("content"),
                text_file_loader=text_file_loader,
                include_attachment_placeholders=include_attachment_placeholders,
                _depth=_depth + 1,
            )
            return

    yield str(value)


def iter_message_content_blocks(
    value: Any,
) -> Iterator[tuple[MessageBlockPath, dict[str, Any]]]:
    """Yield structured blocks through the same wrappers as ``message_to_text``.

    Paths are relative to the message ``content`` field. Lists preserve their
    source indices; ``message``, ``content``, and ``multi_ai`` response wrappers
    preserve their source keys. The traversal is lazy and depth-bounded, so
    callers can reject an excessive attachment count without copying the list.
    """

    yield from _iter_message_content_blocks(value, (), 0)


def _iter_message_content_blocks(
    value: Any,
    path: MessageBlockPath,
    depth: int,
) -> Iterator[tuple[MessageBlockPath, dict[str, Any]]]:
    if depth > MAX_MESSAGE_CONTENT_NESTING:
        raise ValueError("message content nesting exceeds the supported limit")
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_message_content_blocks(
                item,
                (*path, index),
                depth + 1,
            )
        return
    if not isinstance(value, dict):
        return
    if value.get("multi_ai") and isinstance(value.get("responses"), list):
        for index, response in enumerate(value["responses"]):
            if not isinstance(response, dict):
                continue
            yield from _iter_message_content_blocks(
                response.get("content"),
                (*path, "responses", index, "content"),
                depth + 1,
            )
        return

    block_type = value.get("type")
    if (
        block_type in MESSAGE_TEXT_BLOCK_TYPES
        or block_type in MESSAGE_ATTACHMENT_BLOCK_TYPES
    ):
        yield path, value
        return
    if "message" in value:
        yield from _iter_message_content_blocks(
            value.get("message"),
            (*path, "message"),
            depth + 1,
        )
        return
    if "content" in value:
        yield from _iter_message_content_blocks(
            value.get("content"),
            (*path, "content"),
            depth + 1,
        )
        return
    yield path, value


def _filename_from_block(value: dict[str, Any], *, default: str) -> str:
    for key in ("filename", "name"):
        candidate = value.get(key)
        if candidate:
            return str(candidate)

    for nested_key in ("document_url", "file", "text_file"):
        nested = value.get(nested_key)
        if isinstance(nested, dict):
            for key in ("filename", "name"):
                candidate = nested.get(key)
                if candidate:
                    return str(candidate)

    return default
