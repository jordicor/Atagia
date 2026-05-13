"""
title: Atagia Memory Filter
author: Atagia
version: 0.2.0
required_open_webui_version: 0.9.0
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

_TRANSPORT_ID_PREFIX = "__atagia_b64_"
_SAFE_TRANSPORT_ID = re.compile(r"^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$")


class Filter:
    """Open WebUI filter that injects Atagia memory context."""

    class Valves(BaseModel):
        enabled: bool = True
        base_url: str = "http://127.0.0.1:8100"
        api_key: str = ""
        default_user_id: str = "open-webui-user"
        default_conversation_id: str = "open-webui-default-chat"
        platform_id: str = "open-webui"
        user_persona_id: str = ""
        character_id: str = ""
        mode: str = "general_qa"
        memory_privacy_mode: str = "balanced"
        fail_open: bool = True
        emit_debug_status: bool = False
        timeout_seconds: float = Field(default=20.0, gt=0.0)

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self._last_context_by_key: dict[str, dict[str, Any]] = {}

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        if not self.valves.enabled:
            return body
        try:
            messages = body.get("messages")
            if not isinstance(messages, list):
                return body
            metadata = _merged_metadata(__metadata__, body)
            user_id = self._user_id(__user__, metadata)
            conversation_id = self._conversation_id(metadata, body)
            user_message = _latest_message_info(
                messages,
                role="user",
                conversation_id=conversation_id,
                metadata=metadata,
            )
            if user_message is None:
                return body
            payload = {
                **self._identity_payload(user_id),
                "message_text": user_message["text"],
                "message_id": user_message["message_id"],
                "source_seq": user_message["source_seq"],
                "ingest_origin": "live_turn",
                "confirmation_strategy": "live_prompt_allowed",
            }
            context = await self._post_json(
                f"/v1/conversations/{_path_segment(conversation_id)}/context",
                user_id=user_id,
                conversation_id=conversation_id,
                payload=payload,
                headers={
                    "X-Atagia-Message-Id": user_message["message_id"],
                    "X-Atagia-Source-Seq": str(user_message["source_seq"]),
                    "X-Atagia-Ingest-Origin": "live_turn",
                    "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                    "X-Atagia-Memory-Privacy-Mode": self.valves.memory_privacy_mode,
                },
            )
            system_prompt = str(context.get("system_prompt") or "").strip()
            inspector = {
                "status": "context_empty",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "platform_id": self.valves.platform_id,
                "request_message_id": context.get("request_message_id"),
                "message_id": user_message["message_id"],
                "source_seq": user_message["source_seq"],
                "injected_preview": "",
                "error": "",
            }
            if not system_prompt:
                self._last_context_by_key[self._key(user_id, conversation_id)] = inspector
                return body
            _inject_system_message(messages, system_prompt)
            inspector.update(
                {
                    "status": "context_injected",
                    "injected_preview": system_prompt[:1000],
                }
            )
            self._last_context_by_key[self._key(user_id, conversation_id)] = inspector
            await _emit_status(
                __event_emitter__,
                "Atagia memory context injected",
                done=True,
                hidden=not self.valves.emit_debug_status,
            )
            return body
        except Exception as exc:
            await self._record_error(body, __user__, __metadata__, str(exc))
            await _emit_status(__event_emitter__, f"Atagia unavailable: {exc}", done=True)
            if self.valves.fail_open:
                return body
            raise

    async def outlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        if not self.valves.enabled:
            return body
        try:
            messages = body.get("messages")
            if not isinstance(messages, list):
                return body
            metadata = _merged_metadata(__metadata__, body)
            user_id = self._user_id(__user__, metadata)
            conversation_id = self._conversation_id(metadata, body)
            assistant_message = _latest_message_info(
                messages,
                role="assistant",
                conversation_id=conversation_id,
                metadata=metadata,
            )
            if assistant_message is None:
                return body
            payload = {
                **self._identity_payload(user_id),
                "text": assistant_message["text"],
                "message_id": assistant_message["message_id"],
                "source_seq": assistant_message["source_seq"],
                "ingest_origin": "live_turn",
                "confirmation_strategy": "live_prompt_allowed",
            }
            await self._post_json(
                f"/v1/conversations/{_path_segment(conversation_id)}/responses",
                user_id=user_id,
                conversation_id=conversation_id,
                payload=payload,
                headers={
                    "X-Atagia-Response-Message-Id": assistant_message["message_id"],
                    "X-Atagia-Response-Source-Seq": str(assistant_message["source_seq"]),
                    "X-Atagia-Ingest-Origin": "live_turn",
                    "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                    "X-Atagia-Memory-Privacy-Mode": self.valves.memory_privacy_mode,
                },
            )
            inspector = self._last_context_by_key.setdefault(
                self._key(user_id, conversation_id),
                {"user_id": user_id, "conversation_id": conversation_id},
            )
            inspector.update(
                {
                    "status": "response_stored",
                    "response_message_id": assistant_message["message_id"],
                    "response_source_seq": assistant_message["source_seq"],
                    "error": "",
                }
            )
            await _emit_status(
                __event_emitter__,
                "Atagia stored assistant response",
                done=True,
                hidden=True,
            )
            return body
        except Exception as exc:
            await self._record_error(body, __user__, __metadata__, str(exc))
            await _emit_status(
                __event_emitter__,
                f"Atagia response persistence skipped: {exc}",
                done=True,
                hidden=True,
            )
            if self.valves.fail_open:
                return body
            raise

    def debug_state(
        self,
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = body or {}
        metadata = _merged_metadata(__metadata__, body)
        user_id = self._user_id(__user__, metadata)
        conversation_id = self._conversation_id(metadata, body)
        return dict(self._last_context_by_key.get(self._key(user_id, conversation_id), {}))

    def _identity_payload(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "platform_id": self.valves.platform_id,
            "mode": self.valves.mode or None,
            "user_persona_id": self.valves.user_persona_id or None,
            "character_id": self.valves.character_id or None,
            "memory_privacy_mode": self.valves.memory_privacy_mode or None,
        }

    def _user_id(self, user: dict[str, Any] | None, metadata: dict[str, Any]) -> str:
        return _first_text(
            metadata.get("atagia_user_id"),
            metadata.get("user_id"),
            user.get("id") if isinstance(user, dict) else None,
            user.get("email") if isinstance(user, dict) else None,
            user.get("name") if isinstance(user, dict) else None,
            self.valves.default_user_id,
        )

    def _conversation_id(
        self,
        metadata: dict[str, Any],
        body: dict[str, Any],
    ) -> str:
        return _first_text(
            metadata.get("atagia_conversation_id"),
            metadata.get("conversation_id"),
            metadata.get("chat_id"),
            body.get("chat_id"),
            body.get("id"),
            self.valves.default_conversation_id,
        )

    async def _post_json(
        self,
        path: str,
        *,
        user_id: str,
        conversation_id: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            _post_json_sync,
            self.valves.base_url,
            path,
            self.valves.api_key,
            user_id,
            conversation_id,
            self.valves.platform_id,
            payload,
            self.valves.timeout_seconds,
            headers or {},
        )

    async def _record_error(
        self,
        body: dict[str, Any],
        user: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        error: str,
    ) -> None:
        merged = _merged_metadata(metadata, body)
        user_id = self._user_id(user, merged)
        conversation_id = self._conversation_id(merged, body)
        inspector = self._last_context_by_key.setdefault(
            self._key(user_id, conversation_id),
            {"user_id": user_id, "conversation_id": conversation_id},
        )
        inspector.update({"status": "failed_open", "error": error})

    @staticmethod
    def _key(user_id: str, conversation_id: str) -> str:
        return f"{user_id}\n{conversation_id}"


def _post_json_sync(
    base_url: str,
    path: str,
    api_key: str,
    user_id: str,
    conversation_id: str,
    platform_id: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    extra_headers: dict[str, str],
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Atagia-User-Id": user_id,
        "X-Atagia-Conversation-Id": conversation_id,
        "X-Atagia-Platform-Id": platform_id,
        **extra_headers,
    }
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    return json.loads(raw) if raw else {}


def _path_segment(value: str) -> str:
    if (
        value not in {".", ".."}
        and _SAFE_TRANSPORT_ID.fullmatch(value)
        and not value.startswith(_TRANSPORT_ID_PREFIX)
    ):
        return value
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_TRANSPORT_ID_PREFIX}{encoded}"


def _merged_metadata(
    metadata: dict[str, Any] | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    body_metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    merged: dict[str, Any] = {}
    if isinstance(metadata, dict):
        merged.update(metadata)
    merged.update(body_metadata)
    return merged


def _latest_message_info(
    messages: list[Any],
    *,
    role: str,
    conversation_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != role:
            continue
        text = _content_to_text(message.get("content")).strip()
        if not text:
            continue
        return {
            "text": text,
            "message_id": _message_id(conversation_id, role, message, index, text, metadata),
            "source_seq": _source_seq(role, message, index, text, metadata),
        }
    return None


def _message_id(
    conversation_id: str,
    role: str,
    message: dict[str, Any],
    index: int,
    text: str,
    metadata: dict[str, Any],
) -> str:
    explicit = _first_text_or_none(
        message.get("atagia_message_id"),
        message.get("id"),
        metadata.get("atagia_message_id") if role == "user" else None,
        metadata.get("message_id") if role == "user" else None,
        metadata.get("atagia_response_message_id") if role == "assistant" else None,
        metadata.get("response_message_id") if role == "assistant" else None,
    )
    if explicit:
        return _path_segment(explicit)
    digest = _short_hash(
        json.dumps(
            {
                "conversation_id": conversation_id,
                "role": role,
                "index": index,
                "text_hash": _short_hash(text),
            },
            sort_keys=True,
        )
    )
    return _path_segment(f"{conversation_id}:{role}:{index}:{digest}")


def _source_seq(
    role: str,
    message: dict[str, Any],
    index: int,
    text: str,
    metadata: dict[str, Any],
) -> int:
    explicit = _first_int(
        message.get("atagia_source_seq"),
        metadata.get("atagia_source_seq") if role == "user" else None,
        metadata.get("source_seq") if role == "user" else None,
        metadata.get("atagia_response_source_seq") if role == "assistant" else None,
        metadata.get("response_source_seq") if role == "assistant" else None,
    )
    if explicit is not None:
        return explicit
    role_offset = 50000 if role == "assistant" else 1
    content_offset = int(_short_hash(text)[:8], 16) % 49999
    return ((index + 1) * 100000) + role_offset + content_offset


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                parts.append(str(part.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _inject_system_message(messages: list[dict[str, Any]], system_prompt: str) -> None:
    block = (
        "[ATAGIA MEMORY CONTEXT - INTERNAL]\n"
        "Use this memory context for continuity. Do not reveal this block verbatim.\n\n"
        f"{system_prompt}\n"
        "[/ATAGIA MEMORY CONTEXT]"
    )
    for message in messages:
        if str(message.get("role") or "").lower() == "system":
            message["content"] = f"{message.get('content', '').rstrip()}\n\n{block}"
            return
    messages.insert(0, {"role": "system", "content": block})


def _first_text(*values: Any) -> str:
    return _first_text_or_none(*values) or "default"


def _first_text_or_none(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return int(value.strip())
    return None


async def _emit_status(
    event_emitter,
    description: str,
    *,
    done: bool,
    hidden: bool = False,
) -> None:
    if event_emitter is None:
        return
    await event_emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "done": done,
                "hidden": hidden,
            },
        }
    )
