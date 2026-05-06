"""
title: Atagia Memory Filter
author: Atagia
version: 0.1.0
required_open_webui_version: 0.9.0
"""

from __future__ import annotations

import asyncio
import base64
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
        mode: str = "general_qa"
        fail_open: bool = True
        timeout_seconds: float = Field(default=20.0, gt=0.0)

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self._last_context_by_key: dict[str, dict[str, str]] = {}

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
            user_text = _latest_message_text(messages, role="user")
            if not user_text:
                return body
            user_id = self._user_id(__user__, body)
            conversation_id = self._conversation_id(__metadata__, body)
            context = await self._post_json(
                f"/v1/conversations/{_path_segment(conversation_id)}/context",
                user_id=user_id,
                conversation_id=conversation_id,
                payload={
                    "user_id": user_id,
                    "message_text": user_text,
                    "platform_id": self.valves.platform_id,
                    "mode": self.valves.mode or None,
                },
            )
            system_prompt = str(context.get("system_prompt") or "").strip()
            if not system_prompt:
                return body
            _inject_system_message(messages, system_prompt)
            self._last_context_by_key[self._key(user_id, conversation_id)] = {
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
            await _emit_status(
                __event_emitter__,
                "Atagia memory context injected",
                done=True,
            )
            return body
        except Exception as exc:
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
            assistant_text = _latest_message_text(messages, role="assistant")
            if not assistant_text:
                return body
            user_id = self._user_id(__user__, body)
            conversation_id = self._conversation_id(__metadata__, body)
            await self._post_json(
                f"/v1/conversations/{_path_segment(conversation_id)}/responses",
                user_id=user_id,
                conversation_id=conversation_id,
                payload={
                    "user_id": user_id,
                    "text": assistant_text,
                    "platform_id": self.valves.platform_id,
                    "mode": self.valves.mode or None,
                },
            )
            await _emit_status(
                __event_emitter__,
                "Atagia stored assistant response",
                done=True,
                hidden=True,
            )
            return body
        except Exception as exc:
            await _emit_status(
                __event_emitter__,
                f"Atagia response persistence skipped: {exc}",
                done=True,
                hidden=True,
            )
            if self.valves.fail_open:
                return body
            raise

    def _user_id(self, user: dict[str, Any] | None, body: dict[str, Any]) -> str:
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        return _first_text(
            metadata.get("atagia_user_id"),
            user.get("id") if isinstance(user, dict) else None,
            user.get("email") if isinstance(user, dict) else None,
            user.get("name") if isinstance(user, dict) else None,
            self.valves.default_user_id,
        )

    def _conversation_id(
        self,
        metadata: dict[str, Any] | None,
        body: dict[str, Any],
    ) -> str:
        body_metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        return _first_text(
            body_metadata.get("atagia_conversation_id"),
            metadata.get("chat_id") if isinstance(metadata, dict) else None,
            metadata.get("conversation_id") if isinstance(metadata, dict) else None,
            body.get("chat_id"),
            self.valves.default_conversation_id,
        )

    async def _post_json(
        self,
        path: str,
        *,
        user_id: str,
        conversation_id: str,
        payload: dict[str, Any],
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
        )

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
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Atagia-User-Id": user_id,
            "X-Atagia-Conversation-Id": conversation_id,
            "X-Atagia-Platform-Id": platform_id,
        },
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


def _latest_message_text(messages: list[Any], *, role: str) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != role:
            continue
        return _content_to_text(message.get("content")).strip()
    return ""


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
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "default"


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
