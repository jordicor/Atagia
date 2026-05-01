"""Message-shape projection helpers for sidecar integrations."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

TextFileLoader = Callable[[dict[str, Any]], str]


def message_to_text(
    value: Any,
    *,
    text_file_loader: TextFileLoader | None = None,
) -> str:
    """Convert common host/provider message shapes into safe Atagia text.

    Binary or base64-heavy blocks are represented as placeholders unless the host
    supplies a text-file loader. Hosts that have first-class attachment metadata
    should pass attachments separately to Atagia rather than embedding raw bytes
    in the projected text.
    """
    if value is None:
        return ""

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            parsed_text = message_to_text(
                parsed,
                text_file_loader=text_file_loader,
            )
            return parsed_text or value
        return value

    if isinstance(value, list):
        parts = [
            message_to_text(item, text_file_loader=text_file_loader)
            for item in value
        ]
        return "\n".join(part for part in parts if part)

    if isinstance(value, dict):
        if value.get("multi_ai") and isinstance(value.get("responses"), list):
            response_parts = []
            for response in value["responses"]:
                if not isinstance(response, dict):
                    continue
                label = response.get("model") or response.get("machine") or "model"
                text = message_to_text(
                    response.get("content"),
                    text_file_loader=text_file_loader,
                )
                if text:
                    response_parts.append(f"[{label}]\n{text}")
            return "\n\n".join(response_parts)

        block_type = value.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            return str(value.get("text") or "")
        if block_type == "text_file":
            if text_file_loader is not None:
                try:
                    return text_file_loader(value)
                except Exception:
                    pass
            filename = _filename_from_block(value, default="attached text file")
            return f"[Text file attached: {filename}]"
        if block_type in {"image", "image_url", "input_image"}:
            return "[Image attached]"
        if block_type in {
            "document",
            "document_bytes",
            "document_url",
            "file",
            "input_file",
        }:
            filename = _filename_from_block(value, default="document")
            return f"[Document attached: {filename}]"
        if block_type in {"audio", "input_audio"}:
            filename = _filename_from_block(value, default="audio")
            return f"[Audio attached: {filename}]"

        if "message" in value:
            return message_to_text(
                value.get("message"),
                text_file_loader=text_file_loader,
            )
        if "content" in value:
            return message_to_text(
                value.get("content"),
                text_file_loader=text_file_loader,
            )

    return str(value)


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
