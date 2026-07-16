"""
title: Atagia Memory Filter
author: Atagia
version: 0.3.0
required_open_webui_version: 0.9.0
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import re
import secrets
import time
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

_TRANSPORT_ID_PREFIX = "__atagia_b64_"
_SAFE_TRANSPORT_ID = re.compile(r"^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$")
_IDENTITY_SCHEMA = "atagia.external-message.v1"
_CORRELATION_KEY = "atagia_filter_correlation_v1"
_BLOCK_BEGIN = "<!-- ATAGIA:FILTER:MEMORY_CONTEXT:v1 -->"
_BLOCK_END = "<!-- /ATAGIA:FILTER:MEMORY_CONTEXT:v1 -->"


@dataclass(frozen=True, slots=True)
class _Scope:
    atagia_user_id: str
    atagia_conversation_id: str
    host_account_id: str
    host_conversation_id: str


@dataclass(frozen=True, slots=True)
class _DiagnosticEntry:
    updated_at: float
    state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CorrelationEntry:
    updated_at: float
    scope: _Scope


class Filter:
    """Open WebUI 0.9.0 filter that injects and persists Atagia context."""

    class Valves(BaseModel):
        enabled: bool = True
        base_url: str = "http://127.0.0.1:8100"
        api_key: str = ""
        installation_id: str = ""
        default_host_account_id: str = ""
        default_user_id: str = ""
        default_conversation_id: str = ""
        platform_id: str = "open-webui"
        user_persona_id: str = ""
        character_id: str = ""
        mode: str = "general_qa"
        memory_privacy_mode: str = "balanced"
        fail_open: bool = True
        emit_debug_status: bool = False
        timeout_seconds: float = Field(default=20.0, gt=0.0)
        diagnostic_cache_max_entries: int = Field(default=512, ge=1, le=10000)
        diagnostic_cache_ttl_seconds: float = Field(default=900.0, gt=0.0)

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self._clock = clock or time.monotonic
        self._diagnostic_state: OrderedDict[tuple[str, str], _DiagnosticEntry] = (
            OrderedDict()
        )
        self._correlations: OrderedDict[str, _CorrelationEntry] = OrderedDict()

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __event_emitter__=None,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body

        # An old block can never survive reentry, disablement, an empty result,
        # or a fail-open request.
        _replace_owned_context(messages, None)
        if not self.valves.enabled:
            return body

        metadata = _metadata_view(__metadata__, body)
        carrier = _metadata_carrier(__metadata__, body)
        scope: _Scope | None = None
        try:
            scope = self._resolve_scope(__user__, metadata, prefer_correlation=False)
            user_message = _latest_message_info(
                messages,
                role="user",
                scope=scope,
                metadata=metadata,
                installation_id=self.valves.installation_id,
            )
            if user_message is None:
                return body
            payload = {
                **self._identity_payload(scope.atagia_user_id),
                "message_text": user_message["text"],
                "message_id": user_message["message_id"],
                "ingest_origin": "live_turn",
                "confirmation_strategy": "live_prompt_allowed",
            }
            headers = {
                "X-Atagia-Message-Id": user_message["message_id"],
                "X-Atagia-Ingest-Origin": "live_turn",
                "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                "X-Atagia-Memory-Privacy-Mode": self.valves.memory_privacy_mode,
            }
            if user_message["source_seq"] is not None:
                payload["source_seq"] = user_message["source_seq"]
                headers["X-Atagia-Source-Seq"] = str(user_message["source_seq"])

            context = await self._post_json(
                f"/v1/conversations/{_path_segment(scope.atagia_conversation_id)}/context",
                user_id=scope.atagia_user_id,
                conversation_id=scope.atagia_conversation_id,
                payload=payload,
                headers=headers,
            )
            system_prompt = str(context.get("system_prompt") or "").strip()
            request_message_id = _first_text_or_none(
                context.get("request_message_id"), user_message["message_id"]
            )
            if carrier is not None:
                carrier[_CORRELATION_KEY] = self._correlation_set(scope)
            state = {
                "status": "context_injected" if system_prompt else "context_empty",
                "has_context": bool(system_prompt),
                "atagia_user_id": scope.atagia_user_id,
                "atagia_conversation_id": scope.atagia_conversation_id,
                "request_message_id": request_message_id,
                "request_source_seq": user_message["source_seq"],
                "response_message_id": None,
                "response_source_seq": None,
                "error_code": None,
            }
            self._state_set(scope, state)
            if system_prompt:
                _replace_owned_context(messages, system_prompt)
                await _emit_status(
                    __event_emitter__,
                    "Atagia memory context injected",
                    done=True,
                    hidden=not self.valves.emit_debug_status,
                )
            return body
        except Exception:
            if scope is None:
                scope = self._scope_for_error(__user__, metadata)
            self._record_error(scope, "upstream_context_unavailable")
            await _emit_status(
                __event_emitter__, "Atagia context unavailable", done=True, hidden=True
            )
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
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body
        metadata = _metadata_view(__metadata__, body)
        carrier = _metadata_carrier(__metadata__, body)
        scope: _Scope | None = None
        correlation_token: str | None = None
        try:
            scope = self._resolve_scope(__user__, metadata, prefer_correlation=True)
            correlation_token = self._correlation_token(metadata)
            assistant_message = _latest_message_info(
                messages,
                role="assistant",
                scope=scope,
                metadata=metadata,
                installation_id=self.valves.installation_id,
            )
            if assistant_message is None:
                return body
            payload = {
                **self._identity_payload(scope.atagia_user_id),
                "text": assistant_message["text"],
                "message_id": assistant_message["message_id"],
                "ingest_origin": "live_turn",
                "confirmation_strategy": "live_prompt_allowed",
            }
            headers = {
                "X-Atagia-Response-Message-Id": assistant_message["message_id"],
                "X-Atagia-Ingest-Origin": "live_turn",
                "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                "X-Atagia-Memory-Privacy-Mode": self.valves.memory_privacy_mode,
            }
            if assistant_message["source_seq"] is not None:
                payload["source_seq"] = assistant_message["source_seq"]
                headers["X-Atagia-Response-Source-Seq"] = str(
                    assistant_message["source_seq"]
                )
            await self._post_json(
                f"/v1/conversations/{_path_segment(scope.atagia_conversation_id)}/responses",
                user_id=scope.atagia_user_id,
                conversation_id=scope.atagia_conversation_id,
                payload=payload,
                headers=headers,
            )
            prior = self._state_get(scope) or self._base_state(scope)
            prior.update(
                {
                    "status": "response_stored",
                    "response_message_id": assistant_message["message_id"],
                    "response_source_seq": assistant_message["source_seq"],
                    "error_code": None,
                }
            )
            self._state_set(scope, prior)
            await _emit_status(
                __event_emitter__,
                "Atagia stored assistant response",
                done=True,
                hidden=True,
            )
            if carrier is not None:
                carrier.pop(_CORRELATION_KEY, None)
            if correlation_token is not None:
                self._correlation_delete(correlation_token)
            return body
        except Exception:
            if scope is None:
                scope = self._scope_for_error(__user__, metadata)
            self._record_error(scope, "upstream_response_unavailable")
            if carrier is not None:
                carrier.pop(_CORRELATION_KEY, None)
            if correlation_token is not None:
                self._correlation_delete(correlation_token)
            await _emit_status(
                __event_emitter__,
                "Atagia response persistence unavailable",
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
        metadata = _metadata_view(__metadata__, body)
        scope = self._resolve_scope(__user__, metadata, prefer_correlation=True)
        return self._state_get(scope) or {}

    def delete_state(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> int:
        """Purge diagnostic metadata for a host user/chat deletion event."""
        self._purge_expired()
        matching = [
            key
            for key in self._diagnostic_state
            if (user_id is None or key[0] == user_id)
            and (conversation_id is None or key[1] == conversation_id)
        ]
        for key in matching:
            self._diagnostic_state.pop(key, None)
        correlation_tokens = [
            token
            for token, entry in self._correlations.items()
            if (user_id is None or entry.scope.atagia_user_id == user_id)
            and (
                conversation_id is None
                or entry.scope.atagia_conversation_id == conversation_id
            )
        ]
        for token in correlation_tokens:
            self._correlations.pop(token, None)
        return len(matching)

    def _identity_payload(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "platform_id": self.valves.platform_id,
            "mode": self.valves.mode or None,
            "user_persona_id": self.valves.user_persona_id or None,
            "character_id": self.valves.character_id or None,
            "memory_privacy_mode": self.valves.memory_privacy_mode or None,
        }

    def _resolve_scope(
        self,
        user: dict[str, Any] | None,
        metadata: dict[str, Any],
        *,
        prefer_correlation: bool,
    ) -> _Scope:
        host_account_id, atagia_user_id = self._trusted_user_mapping(user)
        self._reject_identity_claims(
            metadata,
            host_account_id=host_account_id,
            atagia_user_id=atagia_user_id,
        )
        host_conversation_id = _first_text_or_none(
            metadata.get("open_webui_conversation_id"),
            metadata.get("chat_id"),
            metadata.get("conversation_id"),
        )
        atagia_conversation_id = _first_text_or_none(
            metadata.get("atagia_conversation_id"),
            metadata.get("conversation_id"),
            metadata.get("chat_id"),
            self.valves.default_conversation_id,
        )
        if prefer_correlation:
            correlation_token = self._correlation_token(metadata)
            if correlation_token is not None:
                correlation = self._correlation_get(correlation_token)
                if correlation is None:
                    raise ValueError("Atagia correlation token is invalid or expired")
                if (
                    correlation.scope.host_account_id != host_account_id
                    or correlation.scope.atagia_user_id != atagia_user_id
                ):
                    raise ValueError("Atagia correlation does not belong to this user")
                claimed_host_conversation_id = (
                    host_conversation_id or atagia_conversation_id
                )
                if (
                    claimed_host_conversation_id
                    != correlation.scope.host_conversation_id
                    or (
                        atagia_conversation_id is not None
                        and atagia_conversation_id
                        != correlation.scope.atagia_conversation_id
                    )
                ):
                    raise ValueError(
                        "Atagia correlation does not belong to this conversation"
                    )
                return correlation.scope
        return _Scope(
            atagia_user_id=_required_text(atagia_user_id, "Atagia user"),
            atagia_conversation_id=_required_text(
                atagia_conversation_id, "Atagia conversation"
            ),
            host_account_id=_required_text(host_account_id, "Open WebUI account"),
            host_conversation_id=_required_text(
                host_conversation_id or atagia_conversation_id,
                "Open WebUI conversation",
            ),
        )

    def _trusted_user_mapping(
        self,
        user: dict[str, Any] | None,
    ) -> tuple[str, str]:
        host_account_id = _first_text_or_none(
            user.get("id") if isinstance(user, dict) else None,
            user.get("email") if isinstance(user, dict) else None,
            self.valves.default_host_account_id,
        )
        host_account_id = _required_text(host_account_id, "Open WebUI account")
        if self.valves.default_user_id:
            configured_host = _required_text(
                self.valves.default_host_account_id,
                "default_host_account_id when default_user_id is configured",
            )
            if host_account_id != configured_host:
                raise ValueError(
                    "Configured Atagia user mapping does not match __user__"
                )
            return host_account_id, self.valves.default_user_id
        return host_account_id, host_account_id

    @staticmethod
    def _reject_identity_claims(
        metadata: dict[str, Any],
        *,
        host_account_id: str,
        atagia_user_id: str,
    ) -> None:
        expected = {
            "open_webui_account_id": host_account_id,
            "atagia_user_id": atagia_user_id,
            "user_id": atagia_user_id,
        }
        for field, expected_value in expected.items():
            claimed = _first_text_or_none(metadata.get(field))
            if claimed is not None and claimed != expected_value:
                raise ValueError(
                    f"Untrusted {field} identity claim conflicts with __user__"
                )

    @staticmethod
    def _correlation_token(metadata: dict[str, Any]) -> str | None:
        value = metadata.get(_CORRELATION_KEY)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Atagia correlation token must be an opaque string")
        return value.strip()

    def _correlation_set(self, scope: _Scope) -> str:
        now = self._clock()
        self._purge_expired(now)
        token = secrets.token_urlsafe(32)
        self._correlations[token] = _CorrelationEntry(now, scope)
        while len(self._correlations) > self.valves.diagnostic_cache_max_entries:
            self._correlations.popitem(last=False)
        return token

    def _correlation_get(self, token: str) -> _CorrelationEntry | None:
        self._purge_expired()
        entry = self._correlations.get(token)
        if entry is not None:
            self._correlations.move_to_end(token)
        return entry

    def _correlation_delete(self, token: str) -> None:
        self._correlations.pop(token, None)

    def _scope_for_error(
        self, user: dict[str, Any] | None, metadata: dict[str, Any]
    ) -> _Scope:
        try:
            return self._resolve_scope(user, metadata, prefer_correlation=True)
        except Exception:
            # Diagnostic-only fallback. It is never used for an Atagia request
            # or external identity.
            return _Scope("unresolved", "unresolved", "unresolved", "unresolved")

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

    def _base_state(self, scope: _Scope) -> dict[str, Any]:
        return {
            "status": "idle",
            "has_context": False,
            "atagia_user_id": scope.atagia_user_id,
            "atagia_conversation_id": scope.atagia_conversation_id,
            "request_message_id": None,
            "request_source_seq": None,
            "response_message_id": None,
            "response_source_seq": None,
            "error_code": None,
        }

    def _record_error(self, scope: _Scope, error_code: str) -> None:
        state = self._state_get(scope) or self._base_state(scope)
        state.update({"status": "failed_open", "error_code": error_code})
        self._state_set(scope, state)

    def _state_set(self, scope: _Scope, state: dict[str, Any]) -> None:
        now = self._clock()
        self._purge_expired(now)
        key = (scope.atagia_user_id, scope.atagia_conversation_id)
        self._diagnostic_state.pop(key, None)
        self._diagnostic_state[key] = _DiagnosticEntry(now, dict(state))
        while len(self._diagnostic_state) > self.valves.diagnostic_cache_max_entries:
            self._diagnostic_state.popitem(last=False)

    def _state_get(self, scope: _Scope) -> dict[str, Any] | None:
        self._purge_expired()
        key = (scope.atagia_user_id, scope.atagia_conversation_id)
        entry = self._diagnostic_state.get(key)
        if entry is None:
            return None
        self._diagnostic_state.move_to_end(key)
        return dict(entry.state)

    def _purge_expired(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        ttl = self.valves.diagnostic_cache_ttl_seconds
        expired = [
            key
            for key, entry in self._diagnostic_state.items()
            if current - entry.updated_at >= ttl
        ]
        for key in expired:
            self._diagnostic_state.pop(key, None)
        expired_correlations = [
            token
            for token, entry in self._correlations.items()
            if current - entry.updated_at >= ttl
        ]
        for token in expired_correlations:
            self._correlations.pop(token, None)


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
    encoded = (
        base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")
    )
    return f"{_TRANSPORT_ID_PREFIX}{encoded}"


def _metadata_view(
    metadata: dict[str, Any] | None, body: dict[str, Any]
) -> dict[str, Any]:
    body_metadata = (
        body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    )
    merged = dict(body_metadata)
    if "chat_id" not in merged and body.get("chat_id") is not None:
        merged["chat_id"] = body["chat_id"]
    if "conversation_id" not in merged and body.get("id") is not None:
        merged["conversation_id"] = body["id"]
    if isinstance(metadata, dict):
        merged.update(metadata)
    return merged


def _metadata_carrier(
    metadata: dict[str, Any] | None, body: dict[str, Any]
) -> dict[str, Any] | None:
    if isinstance(metadata, dict):
        return metadata
    body_metadata = body.get("metadata")
    return body_metadata if isinstance(body_metadata, dict) else None


def _latest_message_info(
    messages: list[Any],
    *,
    role: Literal["user", "assistant"],
    scope: _Scope,
    metadata: dict[str, Any],
    installation_id: str,
) -> dict[str, Any] | None:
    host_messages = [message for message in messages if not _is_owned_message(message)]
    for position in range(len(host_messages) - 1, -1, -1):
        message = host_messages[position]
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != role:
            continue
        text = _content_to_text(message.get("content")).strip()
        if not text:
            continue
        ordinal = position + 1
        host_message_id = _first_scalar(
            message.get("id"),
            message.get("message_id"),
            metadata.get("message_id") if role == "user" else None,
            metadata.get("response_message_id") if role == "assistant" else None,
        )
        source_namespace = "host_message" if host_message_id else "live_event"
        host_message_id = host_message_id or f"ordinal:{ordinal}"
        generation_id = (
            _first_scalar(
                message.get("generation_id"),
                message.get("generationId"),
                metadata.get("generation_id") if role == "assistant" else None,
                metadata.get("response_generation_id") if role == "assistant" else None,
            )
            or "default"
        )
        message_id = _canonical_message_id(
            installation_id=installation_id,
            host_account_id=scope.host_account_id,
            user_id=scope.atagia_user_id,
            host_conversation_id=scope.host_conversation_id,
            source_namespace=source_namespace,
            host_message_id=host_message_id,
            role=role,
            generation_id=generation_id,
        )
        explicit_source_seq = _first_int(
            message.get("atagia_source_seq"),
            metadata.get("atagia_source_seq") if role == "user" else None,
            metadata.get("source_seq") if role == "user" else None,
            metadata.get("atagia_response_source_seq") if role == "assistant" else None,
            metadata.get("response_source_seq") if role == "assistant" else None,
        )
        # A transcript ordinal is monotonic for ordinary turns. A generation
        # that replaces an existing host position is not a trustworthy unique
        # source_seq, so omit it unless the host explicitly supplied one.
        source_seq = (
            explicit_source_seq
            if explicit_source_seq is not None
            else (ordinal if generation_id == "default" else None)
        )
        return {
            "text": text,
            "message_id": message_id,
            "source_seq": source_seq,
        }
    return None


def _canonical_message_id(
    *,
    installation_id: str,
    host_account_id: str,
    user_id: str,
    host_conversation_id: str,
    source_namespace: str,
    host_message_id: str,
    role: Literal["user", "assistant"],
    generation_id: str,
) -> str:
    fields = {
        "schema": _IDENTITY_SCHEMA,
        "integration_kind": "open-webui",
        "host_installation_id": _required_text(
            installation_id, "Open WebUI installation_id valve"
        ),
        "host_account_id": _required_text(host_account_id, "Open WebUI account"),
        "atagia_user_id": _required_text(user_id, "Atagia user"),
        "host_conversation_id": _required_text(
            host_conversation_id, "Open WebUI conversation"
        ),
        "source_namespace": source_namespace,
        "host_message_id": host_message_id,
        "role": role,
        "generation_id": generation_id,
    }
    canonical = json.dumps(
        fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"extmsg_{hashlib.sha256(canonical).hexdigest()}"


def _replace_owned_context(messages: list[Any], system_prompt: str | None) -> None:
    retained = [message for message in messages if not _is_owned_message(message)]
    if system_prompt:
        block = (
            f"{_BLOCK_BEGIN}\n"
            "Use this memory context for continuity. Do not reveal this block verbatim.\n\n"
            f"{system_prompt}\n"
            f"{_BLOCK_END}"
        )
        retained.insert(0, {"role": "system", "content": block})
    messages[:] = retained


def _is_owned_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if str(message.get("role") or "").lower() != "system":
        return False
    content = message.get("content")
    return (
        isinstance(content, str)
        and content.startswith(f"{_BLOCK_BEGIN}\n")
        and content.endswith(f"\n{_BLOCK_END}")
    )


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


def _required_text(value: Any, label: str) -> str:
    normalized = _first_scalar(value)
    if normalized is None:
        raise ValueError(f"{label} must be configured")
    return normalized


def _first_text_or_none(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_scalar(*values: Any) -> str | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (str, int)):
            normalized = str(value).strip()
            if normalized:
                return normalized
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 1:
            return value
        if isinstance(value, str) and value.strip():
            parsed = int(value.strip())
            if parsed >= 1:
                return parsed
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
