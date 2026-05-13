"""Copyable Hermes MemoryProvider implementation for Atagia."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from queue import Queue
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # pragma: no cover - exercised in real Hermes installs.
    from agent.memory_provider import MemoryProvider
except Exception:  # pragma: no cover - local tests use the fallback.
    class MemoryProvider:  # type: ignore[no-redef]
        """Fallback base class used when Hermes is not installed."""


_STOP = object()


@dataclass(slots=True)
class AtagiaConfig:
    """Runtime configuration for the Atagia Hermes provider."""

    enabled: bool = True
    base_url: str = field(default_factory=lambda: os.getenv("ATAGIA_BASE_URL", "http://127.0.0.1:8100"))
    api_key: str = field(default_factory=lambda: os.getenv("ATAGIA_SERVICE_API_KEY", ""))
    user_id: str = "hermes-user"
    platform_id: str = "hermes"
    conversation_id: str = "hermes-default-session"
    character_id: str | None = None
    user_persona_id: str | None = None
    mode: str = "general_qa"
    memory_privacy_mode: str = "balanced"
    fail_open: bool = True
    timeout_seconds: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AtagiaConfig":
        values = dict(data or {})
        return cls(
            enabled=bool(values.get("enabled", True)),
            base_url=str(values.get("base_url") or os.getenv("ATAGIA_BASE_URL", "http://127.0.0.1:8100")),
            api_key=str(values.get("api_key") or os.getenv("ATAGIA_SERVICE_API_KEY", "")),
            user_id=str(values.get("user_id") or "hermes-user"),
            platform_id=str(values.get("platform_id") or "hermes"),
            conversation_id=str(values.get("conversation_id") or "hermes-default-session"),
            character_id=_optional_text(values.get("character_id")),
            user_persona_id=_optional_text(values.get("user_persona_id")),
            mode=str(values.get("mode") or "general_qa"),
            memory_privacy_mode=str(values.get("memory_privacy_mode") or "balanced"),
            fail_open=bool(values.get("fail_open", True)),
            timeout_seconds=float(values.get("timeout_seconds") or 20.0),
        )


class AtagiaMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by the Atagia sidecar API."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = AtagiaConfig.from_dict(config)
        self._queue: Queue[dict[str, Any] | object] = Queue()
        self._worker: threading.Thread | None = None
        self._shutdown = False
        self._last_status: dict[str, Any] = {
            "status": "initialized",
            "last_request": None,
            "last_error": "",
            "last_injected_preview": "",
        }
        self._ensure_worker()

    def is_available(self) -> bool:
        return bool(self.config.enabled and self.config.base_url)

    def initialize(self, config: dict[str, Any] | None = None) -> bool:
        self.save_config(config or {})
        self._ensure_worker()
        self._last_status["status"] = "ready" if self.is_available() else "disabled"
        return self.is_available()

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "base_url": {"type": "string", "default": "http://127.0.0.1:8100"},
                "api_key": {"type": "string", "default": ""},
                "user_id": {"type": "string", "default": "hermes-user"},
                "platform_id": {"type": "string", "default": "hermes"},
                "conversation_id": {"type": "string", "default": "hermes-default-session"},
                "character_id": {"type": "string"},
                "user_persona_id": {"type": "string"},
                "mode": {"type": "string", "default": "general_qa"},
                "memory_privacy_mode": {
                    "type": "string",
                    "enum": ["balanced", "trusted_private"],
                    "default": "balanced",
                },
                "fail_open": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "number", "default": 20.0},
            },
            "required": ["base_url", "user_id", "platform_id"],
        }

    def save_config(self, config: dict[str, Any]) -> dict[str, Any]:
        merged = {**asdict(self.config), **dict(config or {})}
        self.config = AtagiaConfig.from_dict(merged)
        self._last_status["status"] = "configured"
        return asdict(self.config)

    def prefetch(self, query: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Fetch prompt context for a Hermes turn and persist the user message."""
        if not self.is_available():
            return {"system_prompt": "", "available": False}
        identity = self._identity(kwargs)
        message_text = str(
            query
            or kwargs.get("message")
            or kwargs.get("user_message")
            or kwargs.get("prompt")
            or ""
        ).strip()
        if not message_text:
            return {"system_prompt": "", "available": True}
        message_id = str(kwargs.get("message_id") or _message_id(identity["conversation_id"], "user", 1, message_text))
        source_seq = int(kwargs.get("source_seq") or _source_seq(1, "user", message_text))
        payload = {
            **identity,
            "message_text": message_text,
            "message_id": message_id,
            "source_seq": source_seq,
            "ingest_origin": "live_turn",
            "confirmation_strategy": "live_prompt_allowed",
            "memory_privacy_mode": self.config.memory_privacy_mode,
        }
        try:
            result = self._request_json(
                f"/v1/conversations/{identity['conversation_id']}/context",
                payload,
                {
                    "X-Atagia-Message-Id": message_id,
                    "X-Atagia-Source-Seq": str(source_seq),
                    "X-Atagia-Ingest-Origin": "live_turn",
                    "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                    "X-Atagia-Memory-Privacy-Mode": self.config.memory_privacy_mode,
                },
            )
            system_prompt = str(result.get("system_prompt") or "")
            self._last_status.update(
                {
                    "status": "context_prefetched",
                    "last_request": payload,
                    "last_error": "",
                    "last_injected_preview": system_prompt[:1000],
                    "request_message_id": result.get("request_message_id"),
                }
            )
            return {
                "system_prompt": system_prompt,
                "raw_context": result,
                "message_id": message_id,
                "source_seq": source_seq,
                "available": True,
            }
        except Exception as exc:
            self._last_status.update({"status": "failed_open", "last_error": str(exc)})
            if self.config.fail_open:
                return {"system_prompt": "", "available": False, "error": str(exc)}
            raise

    def sync_turn(self, turn: dict[str, Any] | None = None, **kwargs: Any) -> bool:
        """Queue turn persistence without blocking Hermes generation."""
        if not self.is_available():
            return False
        payload = {**dict(turn or {}), **kwargs}
        self._queue.put({"kind": "sync_turn", "payload": payload})
        self._last_status["status"] = "sync_queued"
        return True

    def on_session_end(self, session: dict[str, Any] | None = None, **kwargs: Any) -> bool:
        if not self.is_available():
            return False
        payload = {**dict(session or {}), **kwargs}
        self._queue.put({"kind": "session_end", "payload": payload})
        self._last_status["status"] = "session_backfill_queued"
        return True

    def on_memory_write(self, memory: dict[str, Any] | None = None, **kwargs: Any) -> bool:
        """No-op: curated Hermes memories are not mirrored as fake chat turns yet."""
        self._last_status["status"] = "memory_write_ignored"
        return False

    def shutdown(self) -> None:
        self._shutdown = True
        self._queue.put(_STOP)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._last_status["status"] = "shutdown"

    def status(self) -> dict[str, Any]:
        return dict(self._last_status)

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="atagia-hermes-memory-worker",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            try:
                assert isinstance(item, dict)
                if item.get("kind") == "sync_turn":
                    self._sync_turn_now(dict(item.get("payload") or {}))
                elif item.get("kind") == "session_end":
                    self._backfill_session_now(dict(item.get("payload") or {}))
            except Exception as exc:  # pragma: no cover - status is tested through public methods.
                self._last_status.update({"status": "worker_failed_open", "last_error": str(exc)})
                if not self.config.fail_open:
                    raise
            finally:
                self._queue.task_done()

    def _sync_turn_now(self, payload: dict[str, Any]) -> None:
        identity = self._identity(payload)
        user_message = _optional_text(payload.get("user_message") or payload.get("message"))
        assistant_response = _optional_text(payload.get("assistant_response") or payload.get("response"))
        if user_message and not payload.get("prefetched"):
            self._write_message(identity, "user", user_message, 1, "live_turn", "live_prompt_allowed")
        if assistant_response:
            self._write_message(identity, "assistant", assistant_response, 2, "live_turn", "live_prompt_allowed")
        self._last_status["status"] = "turn_synced"

    def _backfill_session_now(self, payload: dict[str, Any]) -> None:
        identity = self._identity(payload)
        imported = 0
        for index, message in enumerate(_messages_from_payload(payload), start=1):
            if message["role"] not in {"user", "assistant"} or not message["text"]:
                continue
            self._write_message(
                identity,
                message["role"],
                message["text"],
                index,
                "backfill",
                "admin_review_only",
                occurred_at=message.get("occurred_at"),
            )
            imported += 1
        self._last_status.update({"status": "session_backfilled", "imported": imported})

    def _write_message(
        self,
        identity: dict[str, Any],
        role: str,
        text: str,
        index: int,
        ingest_origin: str,
        confirmation_strategy: str,
        *,
        occurred_at: str | None = None,
    ) -> None:
        message_id = _message_id(identity["conversation_id"], role, index, text)
        source_seq = _source_seq(index, role, text)
        path = (
            f"/v1/conversations/{identity['conversation_id']}/responses"
            if role == "assistant"
            else f"/v1/conversations/{identity['conversation_id']}/messages"
        )
        payload = {
            **identity,
            "role": role,
            "text": text,
            "message_id": message_id,
            "source_seq": source_seq,
            "occurred_at": occurred_at,
            "ingest_origin": ingest_origin,
            "confirmation_strategy": confirmation_strategy,
            "memory_privacy_mode": self.config.memory_privacy_mode,
        }
        self._request_json(
            path,
            payload,
            {
                "X-Atagia-Ingest-Origin": ingest_origin,
                "X-Atagia-Confirmation-Strategy": confirmation_strategy,
                "X-Atagia-Memory-Privacy-Mode": self.config.memory_privacy_mode,
            },
        )
        self._last_status["last_request"] = payload

    def _identity(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": str(values.get("user_id") or self.config.user_id),
            "platform_id": str(values.get("platform_id") or self.config.platform_id),
            "conversation_id": str(
                values.get("conversation_id")
                or values.get("session_id")
                or self.config.conversation_id
            ),
            "character_id": _optional_text(values.get("character_id") or self.config.character_id),
            "user_persona_id": _optional_text(values.get("user_persona_id") or self.config.user_persona_id),
            "mode": str(values.get("mode") or self.config.mode),
        }

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.config.base_url.rstrip('/')}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "X-Atagia-User-Id": str(payload["user_id"]),
                "X-Atagia-Conversation-Id": str(payload["conversation_id"]),
                "X-Atagia-Platform-Id": str(payload["platform_id"]),
                **(extra_headers or {}),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        return json.loads(raw) if raw else {}


def _messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = payload.get("messages") or payload.get("transcript") or []
    if not isinstance(raw_messages, list):
        return []
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        messages.append(
            {
                "role": str(message.get("role") or "").lower(),
                "text": str(
                    message.get("content")
                    or message.get("text")
                    or message.get("message")
                    or ""
                ).strip(),
                "occurred_at": _optional_text(
                    message.get("occurred_at")
                    or message.get("created_at")
                    or message.get("timestamp")
                ),
            }
        )
    return messages


def _message_id(conversation_id: str, role: str, index: int, text: str) -> str:
    return f"{conversation_id}:{role}:{index}:{_short_hash(text)}"


def _source_seq(index: int, role: str, text: str) -> int:
    role_offset = 50000 if role == "assistant" else 1
    content_offset = int(_short_hash(text)[:8], 16) % 49999
    return (index * 100000) + role_offset + content_offset


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def wait_for_queue(provider: AtagiaMemoryProvider, timeout_seconds: float = 2.0) -> None:
    """Testing/status helper: wait for queued sync work to settle."""
    deadline = time.monotonic() + timeout_seconds
    while provider._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
