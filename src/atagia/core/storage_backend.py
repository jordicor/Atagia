"""Transient storage backend abstractions."""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import isawaitable
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from atagia.core.ids import generate_prefixed_id
from atagia.models.schemas_jobs import StreamMessage


DrainProgressCallback = Callable[
    ["StorageDrainSnapshot"],
    Awaitable[bool | None] | bool | None,
]


@dataclass(frozen=True, slots=True)
class StorageDrainSnapshot:
    """Point-in-time view of transient stream work during a drain."""

    queued_by_stream: dict[str, int] = field(default_factory=dict)
    pending_by_stream: dict[str, int] = field(default_factory=dict)
    pending_job_types: dict[str, int] = field(default_factory=dict)
    active_jobs: tuple[dict[str, Any], ...] = ()
    added_by_stream: dict[str, int] = field(default_factory=dict)
    read_by_stream: dict[str, int] = field(default_factory=dict)
    claimed_by_stream: dict[str, int] = field(default_factory=dict)
    acked_by_stream: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    idle_seconds: float = 0.0
    timeout_seconds: float | None = None
    idle_timeout_seconds: float | None = None

    @property
    def total_queued(self) -> int:
        return sum(self.queued_by_stream.values())

    @property
    def total_pending(self) -> int:
        return sum(self.pending_by_stream.values())

    @property
    def total_added(self) -> int:
        return sum(self.added_by_stream.values())

    @property
    def total_read(self) -> int:
        return sum(self.read_by_stream.values())

    @property
    def total_claimed(self) -> int:
        return sum(self.claimed_by_stream.values())

    @property
    def total_acked(self) -> int:
        return sum(self.acked_by_stream.values())

    @property
    def drained(self) -> bool:
        return self.total_queued == 0 and self.total_pending == 0

    def with_timing(
        self,
        *,
        elapsed_seconds: float,
        idle_seconds: float,
        timeout_seconds: float | None,
        idle_timeout_seconds: float | None,
    ) -> "StorageDrainSnapshot":
        return replace(
            self,
            elapsed_seconds=elapsed_seconds,
            idle_seconds=idle_seconds,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )

    def progress_marker(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        """Stable marker that changes when stream work moves forward."""
        return (
            tuple(sorted(self.queued_by_stream.items())),
            tuple(sorted(self.pending_by_stream.items())),
            tuple(sorted(self.added_by_stream.items())),
            tuple(sorted(self.read_by_stream.items())),
            tuple(sorted(self.claimed_by_stream.items())),
            tuple(sorted(self.acked_by_stream.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "queued_by_stream": dict(sorted(self.queued_by_stream.items())),
            "pending_by_stream": dict(sorted(self.pending_by_stream.items())),
            "pending_job_types": dict(sorted(self.pending_job_types.items())),
            "active_jobs": [dict(job) for job in self.active_jobs],
            "added_by_stream": dict(sorted(self.added_by_stream.items())),
            "read_by_stream": dict(sorted(self.read_by_stream.items())),
            "claimed_by_stream": dict(sorted(self.claimed_by_stream.items())),
            "acked_by_stream": dict(sorted(self.acked_by_stream.items())),
            "total_queued": self.total_queued,
            "total_pending": self.total_pending,
            "total_added": self.total_added,
            "total_read": self.total_read,
            "total_claimed": self.total_claimed,
            "total_acked": self.total_acked,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "idle_seconds": round(self.idle_seconds, 3),
            "timeout_seconds": self.timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "drained": self.drained,
        }


async def emit_drain_progress(
    callback: DrainProgressCallback | None,
    snapshot: StorageDrainSnapshot,
) -> bool:
    if callback is None:
        return False
    result = callback(snapshot)
    if isawaitable(result):
        result = await result
    return bool(result)


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


def _nested_job_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("payload")
    if isinstance(nested, dict) and ("user_id" in nested or "conversation_id" in nested):
        return nested
    return payload


def _job_matches_user(payload: Any, user_id: str) -> bool:
    job = _nested_job_payload(payload)
    if job is None:
        return False
    return str(job.get("user_id") or "") == user_id


def _job_matches_conversation(payload: Any, conversation_id: str) -> bool:
    job = _nested_job_payload(payload)
    if job is None:
        return False
    if str(job.get("conversation_id") or "") == conversation_id:
        return True
    message_ids = job.get("message_ids")
    return False if not isinstance(message_ids, list) else False


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

    async def delete_recent_windows_for_user(self, user_id: str) -> int:
        raise NotImplementedError

    async def delete_recent_window_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        raise NotImplementedError

    async def purge_user_jobs(self, user_id: str) -> int:
        raise NotImplementedError

    async def purge_conversation_jobs(self, user_id: str, conversation_id: str) -> int:
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

    async def drain_snapshot(self) -> StorageDrainSnapshot:
        """Return transient stream drain state when supported."""
        return StorageDrainSnapshot()

    async def drain(
        self,
        timeout_seconds: float = 30.0,
        *,
        idle_timeout_seconds: float | None = None,
        progress_interval_seconds: float = 0.0,
        progress_callback: DrainProgressCallback | None = None,
    ) -> bool:
        """Wait for transient stream work to drain when supported."""
        del timeout_seconds
        del idle_timeout_seconds
        del progress_interval_seconds
        del progress_callback
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
    _stream_add_counts: dict[str, int] = field(default_factory=dict)
    _stream_read_counts: dict[str, int] = field(default_factory=dict)
    _stream_claim_counts: dict[str, int] = field(default_factory=dict)
    _stream_ack_counts: dict[str, int] = field(default_factory=dict)
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

    async def delete_recent_windows_for_user(self, user_id: str) -> int:
        prefix = f"{user_id}:"
        with self._guard:
            keys = [
                key
                for key in self._recent_windows
                if key == user_id or key.startswith(prefix)
            ]
            for key in keys:
                self._recent_windows.pop(key, None)
            return len(keys)

    async def delete_recent_window_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        key = f"{user_id}:{conversation_id}"
        with self._guard:
            existed = key in self._recent_windows
            self._recent_windows.pop(key, None)
            return 1 if existed else 0

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

    async def purge_user_jobs(self, user_id: str) -> int:
        with self._guard:
            return self._purge_jobs_locked(lambda payload: _job_matches_user(payload, user_id))

    async def purge_conversation_jobs(self, user_id: str, conversation_id: str) -> int:
        with self._guard:
            return self._purge_jobs_locked(
                lambda payload: (
                    _job_matches_user(payload, user_id)
                    and _job_matches_conversation(payload, conversation_id)
                )
            )

    def _purge_jobs_locked(self, should_drop: Any) -> int:
        purged = 0
        for queue in self._queues.values():
            retained: list[dict[str, Any]] = []
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                payload = item.get("payload") if isinstance(item, dict) and "payload" in item else item
                if should_drop(payload):
                    purged += 1
                    continue
                retained.append(item)
            for item in retained:
                queue.put_nowait(item)

        for pending in self._stream_pending.values():
            for message_id, entry in list(pending.items()):
                payload = entry.get("payload") if isinstance(entry, dict) else None
                if should_drop(payload):
                    pending.pop(message_id, None)
                    purged += 1
                    if self._pending_job_count > 0:
                        self._pending_job_count -= 1
        return purged

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
        with self._guard:
            self._stream_add_counts[stream_name] = (
                self._stream_add_counts.get(stream_name, 0) + 1
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
                self._stream_read_counts[stream_name] = (
                    self._stream_read_counts.get(stream_name, 0) + 1
                )
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
                self._stream_claim_counts[stream_name] = (
                    self._stream_claim_counts.get(stream_name, 0) + 1
                )
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
            if removed is not None:
                self._stream_ack_counts[stream_name] = (
                    self._stream_ack_counts.get(stream_name, 0) + 1
                )

    async def stream_ensure_group(self, stream_name: str, group_name: str) -> None:
        with self._guard:
            self._stream_groups.add((stream_name, group_name))

    async def drain_snapshot(self) -> StorageDrainSnapshot:
        now = monotonic()
        with self._guard:
            queued_by_stream = {
                queue_name.removeprefix("stream:"): queue.qsize()
                for queue_name, queue in self._queues.items()
                if queue_name.startswith("stream:")
            }
            pending_by_stream: Counter[str] = Counter()
            pending_job_types: Counter[str] = Counter()
            active_job_entries: list[tuple[float, dict[str, Any]]] = []
            for (stream_name, group_name), pending in self._stream_pending.items():
                if not pending:
                    continue
                pending_by_stream[stream_name] += len(pending)
                for message_id, entry in pending.items():
                    payload = entry.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    job_type = str(payload.get("job_type") or "unknown")
                    pending_job_types[job_type] += 1
                    last_delivered_at = float(entry.get("last_delivered_at", 0.0) or 0.0)
                    nested_payload = payload.get("payload")
                    active_job_entries.append(
                        (
                            last_delivered_at,
                            {
                                "stream": stream_name,
                                "group": group_name,
                                "message_id": message_id,
                                "job_id": payload.get("job_id"),
                                "job_type": job_type,
                                "conversation_id": payload.get("conversation_id"),
                                "message_ids": list(payload.get("message_ids") or [])[:5],
                                "payload_message_id": (
                                    nested_payload.get("message_id")
                                    if isinstance(nested_payload, dict)
                                    else None
                                ),
                                "delivery_count": int(entry.get("delivery_count", 1) or 1),
                                "seconds_pending": round(
                                    max(0.0, now - last_delivered_at),
                                    3,
                                ),
                            },
                        )
                    )
            active_job_entries.sort(key=lambda item: item[0])
            return StorageDrainSnapshot(
                queued_by_stream=dict(queued_by_stream),
                pending_by_stream=dict(pending_by_stream),
                pending_job_types=dict(pending_job_types),
                active_jobs=tuple(entry for _, entry in active_job_entries[:8]),
                added_by_stream=dict(self._stream_add_counts),
                read_by_stream=dict(self._stream_read_counts),
                claimed_by_stream=dict(self._stream_claim_counts),
                acked_by_stream=dict(self._stream_ack_counts),
            )

    async def drain(
        self,
        timeout_seconds: float = 30.0,
        *,
        idle_timeout_seconds: float | None = None,
        progress_interval_seconds: float = 0.0,
        progress_callback: DrainProgressCallback | None = None,
    ) -> bool:
        timeout = max(0.0, timeout_seconds)
        idle_timeout = (
            None
            if idle_timeout_seconds is None
            else max(0.0, idle_timeout_seconds)
        )
        started_at = monotonic()
        deadline = started_at + timeout
        last_progress_at = started_at
        last_marker: tuple[tuple[tuple[str, int], ...], ...] | None = None
        progress_interval = max(0.0, progress_interval_seconds)
        next_progress_at = started_at + progress_interval
        while True:
            now = monotonic()
            snapshot = (await self.drain_snapshot()).with_timing(
                elapsed_seconds=now - started_at,
                idle_seconds=now - last_progress_at,
                timeout_seconds=timeout,
                idle_timeout_seconds=idle_timeout,
            )
            marker = snapshot.progress_marker()
            if last_marker is None:
                last_marker = marker
            elif marker != last_marker:
                last_marker = marker
                last_progress_at = now
                snapshot = snapshot.with_timing(
                    elapsed_seconds=now - started_at,
                    idle_seconds=0.0,
                    timeout_seconds=timeout,
                    idle_timeout_seconds=idle_timeout,
                )
            if snapshot.drained:
                return True
            if progress_callback is not None and now >= next_progress_at:
                if await emit_drain_progress(progress_callback, snapshot):
                    last_progress_at = now
                next_progress_at = now + max(progress_interval, 0.01)
            if idle_timeout is not None and monotonic() - last_progress_at >= idle_timeout:
                return False
            if monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

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
            self._stream_add_counts.clear()
            self._stream_read_counts.clear()
            self._stream_claim_counts.clear()
            self._stream_ack_counts.clear()
            self._pending_job_count = 0


@dataclass(slots=True)
class _InProcessContextViewEntry:
    """Internal in-process context-view record with TTL and invalidation metadata."""

    expires_at: float
    payload: dict[str, Any]
    monotonic_seq: int | None = None
    user_id: str | None = None
    conversation_id: str | None = None
