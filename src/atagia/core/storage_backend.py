"""Transient storage backend abstractions."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from atagia.core.ids import generate_prefixed_id
from atagia.models.schemas_jobs import StreamMessage


def extract_context_view_user_id(context_view: dict[str, Any]) -> str | None:
    """Best-effort user identifier extraction for cache invalidation indexes."""
    user_id = context_view.get("user_id")
    if not isinstance(user_id, str):
        return None
    normalized = user_id.strip()
    return normalized or None


def extract_context_view_conversation_id(context_view: dict[str, Any]) -> str | None:
    """Best-effort conversation identifier extraction for cache invalidation indexes."""
    conversation_id = context_view.get("conversation_id")
    if not isinstance(conversation_id, str):
        return None
    normalized = conversation_id.strip()
    return normalized or None


class StorageBackend:
    """Interface for Redis-backed or in-process transient state."""

    async def get_recent_window(self, key: str) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    async def set_recent_window(self, key: str, messages: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    async def get_context_view(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def set_context_view(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        raise NotImplementedError

    async def set_context_view_if_newer(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
    ) -> bool:
        raise NotImplementedError

    async def delete_context_view(self, key: str) -> None:
        raise NotImplementedError

    async def delete_context_views_for_user(self, user_id: str) -> int:
        raise NotImplementedError

    async def delete_context_views_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        raise NotImplementedError

    async def enqueue_job(self, queue_name: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def dequeue_job(
        self,
        queue_name: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        raise NotImplementedError

    async def stream_claim_idle(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        min_idle_ms: int,
        count: int,
    ) -> list[StreamMessage]:
        raise NotImplementedError

    async def stream_ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        raise NotImplementedError

    async def stream_ensure_group(self, stream_name: str, group_name: str) -> None:
        raise NotImplementedError

    async def drain(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for transient stream work to drain when supported."""
        del timeout_seconds
        return False

    async def remember_dedupe(self, key: str, ttl_seconds: int) -> bool:
        raise NotImplementedError

    async def force_dedupe(self, key: str, ttl_seconds: int) -> None:
        """Set or overwrite a dedupe marker unconditionally."""
        raise NotImplementedError

    async def has_dedupe(self, key: str) -> bool:
        raise NotImplementedError

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        raise NotImplementedError

    async def release_lock(self, key: str, token: str) -> None:
        raise NotImplementedError

    async def get_cache_generation(self, key: str) -> int:
        raise NotImplementedError

    async def increment_cache_generation(self, key: str) -> int:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class InProcessBackend(StorageBackend):
    """Single-process, Redis-free backend for local development and tests."""

    _recent_windows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _context_views: dict[str, "_InProcessContextViewEntry"] = field(default_factory=dict)
    _context_view_keys_by_user: dict[str, set[str]] = field(default_factory=dict)
    _context_view_keys_by_conversation: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    _dedupe_keys: dict[str, float] = field(default_factory=dict)
    _locks: dict[str, tuple[float, str]] = field(default_factory=dict)
    _cache_generations: dict[str, int] = field(default_factory=dict)
    _queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    _stream_pending: dict[tuple[str, str], dict[str, dict[str, Any]]] = field(default_factory=dict)
    _stream_groups: set[tuple[str, str]] = field(default_factory=set)
    _pending_job_count: int = 0
    # Keep this critical section trivial: these async methods must not await while
    # holding the lock, or they would block the event loop thread.
    _guard: Lock = field(default_factory=Lock)

    def _purge_expired(self) -> None:
        now = monotonic()
        expired_context_keys = [
            key for key, entry in self._context_views.items() if entry.expires_at <= now
        ]
        for key in expired_context_keys:
            self._delete_context_view_locked(key)

        expired_dedupe_keys = [key for key, expires_at in self._dedupe_keys.items() if expires_at <= now]
        for key in expired_dedupe_keys:
            self._dedupe_keys.pop(key, None)

        expired_lock_keys = [
            key for key, (expires_at, _token) in self._locks.items() if expires_at <= now
        ]
        for key in expired_lock_keys:
            self._locks.pop(key, None)

    def _delete_context_view_locked(self, key: str) -> bool:
        entry = self._context_views.pop(key, None)
        if entry is None:
            return False
        if entry.user_id is not None:
            keys = self._context_view_keys_by_user.get(entry.user_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._context_view_keys_by_user.pop(entry.user_id, None)
        if entry.user_id is not None and entry.conversation_id is not None:
            index_key = (entry.user_id, entry.conversation_id)
            keys = self._context_view_keys_by_conversation.get(index_key)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._context_view_keys_by_conversation.pop(index_key, None)
        return True

    def _store_context_view_locked(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int | None,
    ) -> None:
        self._delete_context_view_locked(key)
        expires_at = monotonic() + ttl_seconds
        user_id = extract_context_view_user_id(context_view)
        conversation_id = extract_context_view_conversation_id(context_view)
        self._context_views[key] = _InProcessContextViewEntry(
            expires_at=expires_at,
            payload=copy.deepcopy(context_view),
            monotonic_seq=monotonic_seq,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if user_id is not None:
            self._context_view_keys_by_user.setdefault(user_id, set()).add(key)
            if conversation_id is not None:
                self._context_view_keys_by_conversation.setdefault(
                    (user_id, conversation_id),
                    set(),
                ).add(key)

    async def get_recent_window(self, key: str) -> list[dict[str, Any]] | None:
        with self._guard:
            value = self._recent_windows.get(key)
            return copy.deepcopy(value) if value is not None else None

    async def set_recent_window(self, key: str, messages: list[dict[str, Any]]) -> None:
        with self._guard:
            self._recent_windows[key] = copy.deepcopy(messages)

    async def get_context_view(self, key: str) -> dict[str, Any] | None:
        with self._guard:
            self._purge_expired()
            entry = self._context_views.get(key)
            return copy.deepcopy(entry.payload) if entry is not None else None

    async def set_context_view(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        with self._guard:
            self._purge_expired()
            self._store_context_view_locked(
                key,
                context_view,
                ttl_seconds,
                monotonic_seq=None,
            )

    async def set_context_view_if_newer(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
    ) -> bool:
        with self._guard:
            self._purge_expired()
            existing = self._context_views.get(key)
            if (
                existing is not None
                and existing.monotonic_seq is not None
                and monotonic_seq <= existing.monotonic_seq
            ):
                return False
            self._store_context_view_locked(
                key,
                context_view,
                ttl_seconds,
                monotonic_seq=monotonic_seq,
            )
            return True

    async def delete_context_view(self, key: str) -> None:
        with self._guard:
            self._purge_expired()
            self._delete_context_view_locked(key)

    async def delete_context_views_for_user(self, user_id: str) -> int:
        with self._guard:
            self._purge_expired()
            keys = list(self._context_view_keys_by_user.get(user_id, set()))
            deleted = 0
            for key in keys:
                if self._delete_context_view_locked(key):
                    deleted += 1
            return deleted

    async def delete_context_views_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        with self._guard:
            self._purge_expired()
            keys = list(self._context_view_keys_by_conversation.get((user_id, conversation_id), set()))
            deleted = 0
            for key in keys:
                if self._delete_context_view_locked(key):
                    deleted += 1
            return deleted

    async def enqueue_job(self, queue_name: str, payload: dict[str, Any]) -> None:
        queue = self._queues.setdefault(queue_name, asyncio.Queue())
        await queue.put(copy.deepcopy(payload))

    async def dequeue_job(
        self,
        queue_name: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        queue = self._queues.setdefault(queue_name, asyncio.Queue())
        if timeout_seconds is not None and timeout_seconds <= 0:
            try:
                payload = queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
            return copy.deepcopy(payload)
        try:
            # None means "wait indefinitely" for the next queued job.
            if timeout_seconds is None:
                payload = await queue.get()
            else:
                payload = await asyncio.wait_for(queue.get(), timeout_seconds)
        except TimeoutError:
            return None
        return copy.deepcopy(payload)

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        message_id = generate_prefixed_id("stm")
        await self.enqueue_job(
            f"stream:{stream_name}",
            {"message_id": message_id, "payload": copy.deepcopy(payload)},
        )
        return message_id

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        del consumer_name  # The in-process fallback does not simulate per-consumer ownership.
        await self.stream_ensure_group(stream_name, group_name)
        messages: list[StreamMessage] = []
        timeout_seconds = None if block_ms is None else max(0.0, block_ms / 1000)
        for index in range(count):
            payload = await self.dequeue_job(
                f"stream:{stream_name}",
                timeout_seconds=timeout_seconds if index == 0 else 0,
            )
            if payload is None:
                break
            message_id = str(payload["message_id"])
            message_payload = copy.deepcopy(payload["payload"])
            with self._guard:
                pending = self._stream_pending.setdefault((stream_name, group_name), {})
                pending[message_id] = {
                    "payload": copy.deepcopy(message_payload),
                    "delivery_count": 1,
                    "last_delivered_at": monotonic(),
                }
                self._pending_job_count += 1
            messages.append(
                StreamMessage(
                    message_id=message_id,
                    payload=message_payload,
                    delivery_count=1,
                )
            )
        return messages

    async def stream_claim_idle(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        min_idle_ms: int,
        count: int,
    ) -> list[StreamMessage]:
        del consumer_name  # The in-process fallback does not simulate per-consumer ownership.
        await self.stream_ensure_group(stream_name, group_name)
        messages: list[StreamMessage] = []
        min_idle_seconds = max(0.0, min_idle_ms / 1000)
        with self._guard:
            now = monotonic()
            pending = self._stream_pending.setdefault((stream_name, group_name), {})
            claimable = [
                (message_id, entry)
                for message_id, entry in pending.items()
                if now - float(entry.get("last_delivered_at", 0.0)) >= min_idle_seconds
            ]
            claimable.sort(key=lambda item: float(item[1].get("last_delivered_at", 0.0)))
            for message_id, entry in claimable[:count]:
                delivery_count = int(entry.get("delivery_count", 1)) + 1
                entry["delivery_count"] = delivery_count
                entry["last_delivered_at"] = now
                messages.append(
                    StreamMessage(
                        message_id=message_id,
                        payload=copy.deepcopy(entry["payload"]),
                        delivery_count=delivery_count,
                    )
                )
        return messages

    async def stream_ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        with self._guard:
            pending = self._stream_pending.setdefault((stream_name, group_name), {})
            removed = pending.pop(message_id, None)
            if removed is not None and self._pending_job_count > 0:
                self._pending_job_count -= 1

    async def stream_ensure_group(self, stream_name: str, group_name: str) -> None:
        with self._guard:
            self._stream_groups.add((stream_name, group_name))

    async def drain(self, timeout_seconds: float = 30.0) -> bool:
        deadline = monotonic() + max(0.0, timeout_seconds)
        while True:
            with self._guard:
                stream_queues_empty = all(
                    queue.empty()
                    for queue_name, queue in self._queues.items()
                    if queue_name.startswith("stream:")
                )
                drained = stream_queues_empty and self._pending_job_count == 0
            if drained:
                return True
            if monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)

    async def remember_dedupe(self, key: str, ttl_seconds: int) -> bool:
        with self._guard:
            self._purge_expired()
            if key in self._dedupe_keys:
                return False
            self._dedupe_keys[key] = monotonic() + ttl_seconds
            return True

    async def force_dedupe(self, key: str, ttl_seconds: int) -> None:
        with self._guard:
            self._dedupe_keys[key] = monotonic() + ttl_seconds

    async def has_dedupe(self, key: str) -> bool:
        with self._guard:
            self._purge_expired()
            return key in self._dedupe_keys

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        with self._guard:
            self._purge_expired()
            if key in self._locks:
                return None
            token = uuid4().hex
            self._locks[key] = (monotonic() + ttl_seconds, token)
            return token

    async def release_lock(self, key: str, token: str) -> None:
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                return
            if entry[1] != token:
                return
            self._locks.pop(key, None)

    async def get_cache_generation(self, key: str) -> int:
        with self._guard:
            return self._cache_generations.get(key, 0)

    async def increment_cache_generation(self, key: str) -> int:
        with self._guard:
            gen = self._cache_generations.get(key, 0) + 1
            self._cache_generations[key] = gen
            return gen

    async def close(self) -> None:
        with self._guard:
            self._recent_windows.clear()
            self._context_views.clear()
            self._context_view_keys_by_user.clear()
            self._context_view_keys_by_conversation.clear()
            self._dedupe_keys.clear()
            self._locks.clear()
            self._cache_generations.clear()
            self._queues.clear()
            self._stream_pending.clear()
            self._stream_groups.clear()
            self._pending_job_count = 0


@dataclass(slots=True)
class _InProcessContextViewEntry:
    """Internal in-process context-view record with TTL and invalidation metadata."""

    expires_at: float
    payload: dict[str, Any]
    monotonic_seq: int | None = None
    user_id: str | None = None
    conversation_id: str | None = None
