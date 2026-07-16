"""Transient storage backend abstractions."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from inspect import isawaitable
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from atagia.core.ids import generate_prefixed_id
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    REVISE_STREAM_NAME,
    TRANSCRIPT_REBUILD_STREAM_NAME,
    StreamMessage,
)


DrainProgressCallback = Callable[
    ["StorageDrainSnapshot"],
    Awaitable[bool | None] | bool | None,
]

_LIFECYCLE_DIAGNOSTIC_MARKER = "_atagia_lifecycle_diagnostic"
_ATAGIA_JOB_STREAM_NAMES = frozenset(
    {
        COMPACT_STREAM_NAME,
        CONTRACT_STREAM_NAME,
        EVALUATION_STREAM_NAME,
        EXTRACT_STREAM_NAME,
        GRAPH_STREAM_NAME,
        INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
        REVISE_STREAM_NAME,
        TRANSCRIPT_REBUILD_STREAM_NAME,
    }
)
_LEGACY_ATAGIA_LIST_QUEUE_NAMES = frozenset(
    {
        "admin_rebuild_conversation",
        "admin_rebuild_user",
        *(f"dead_letter:{stream_name}" for stream_name in _ATAGIA_JOB_STREAM_NAMES),
    }
)
_OWNERLESS_SHA256_DEDUPE_PREFIXES = frozenset(
    {"contract", "graph", "initial_context_package_refresh"}
)


def _is_twelve_hex_namespace(value: str) -> bool:
    return len(value) == 12 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _is_legacy_extractor_dedupe_key_for_user(key: str, user_id: str) -> bool:
    """Match the one historical generic dedupe shape with an exact user owner."""

    if user_id in _OWNERLESS_SHA256_DEDUPE_PREFIXES:
        return False
    prefix = f"{user_id}:"
    if not key.startswith(prefix):
        return False
    digest = key[len(prefix) :]
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


@dataclass(frozen=True, slots=True)
class RecentWindowIdentity:
    """Exact user and conversation ownership of one recent-window value."""

    user_id: str
    conversation_id: str
    lifecycle_cleanup_key: str
    lifecycle_epoch: str
    cache_revision: int
    derivation_revision: int
    conversation_lifecycle_epoch: str
    conversation_source_revision: int


def _recent_window_identity_is_valid(identity: RecentWindowIdentity) -> bool:
    """Reject runtime values that cannot be canonical cache coordinates."""

    string_coordinates: tuple[object, ...] = (
        identity.user_id,
        identity.conversation_id,
        identity.lifecycle_cleanup_key,
        identity.lifecycle_epoch,
        identity.conversation_lifecycle_epoch,
    )
    if not all(
        isinstance(coordinate, str) and bool(coordinate)
        for coordinate in string_coordinates
    ):
        return False
    revision_coordinates: tuple[object, ...] = (
        identity.cache_revision,
        identity.derivation_revision,
        identity.conversation_source_revision,
    )
    return all(
        isinstance(coordinate, int)
        and not isinstance(coordinate, bool)
        and coordinate >= 0
        for coordinate in revision_coordinates
    )


def _validated_lifecycle_coordinates(
    lifecycle_cleanup_key: str | None,
    lifecycle_epoch: str | None,
) -> tuple[str, str] | None:
    """Return complete lifecycle coordinates or reject a partial identity."""

    if lifecycle_cleanup_key is None and lifecycle_epoch is None:
        return None
    if (
        not isinstance(lifecycle_cleanup_key, str)
        or not lifecycle_cleanup_key
        or not isinstance(lifecycle_epoch, str)
        or not lifecycle_epoch
    ):
        raise ValueError(
            "lifecycle_cleanup_key and lifecycle_epoch must both be non-empty"
        )
    return lifecycle_cleanup_key, lifecycle_epoch


def _validated_job_lock_scope(
    job_id: str | None,
    execution_fence: int | None,
    *,
    lifecycle: tuple[str, str] | None,
) -> tuple[str, int] | None:
    """Return a complete job-lock scope or reject a partial/unsafe one."""

    if job_id is None and execution_fence is None:
        return None
    if lifecycle is None:
        raise ValueError("job-owned locks require lifecycle coordinates")
    if (
        not isinstance(job_id, str)
        or not job_id
        or not isinstance(execution_fence, int)
        or isinstance(execution_fence, bool)
        or execution_fence <= 0
    ):
        raise ValueError(
            "job_id and execution_fence must both be present, with a positive fence"
        )
    return job_id, execution_fence


def build_recent_window_key(user_id: str, conversation_id: str) -> str:
    """Return an injective, versioned key for a user/conversation pair."""

    return "rw:v1:" + json.dumps(
        [user_id, conversation_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _recent_window_identity_can_replace(
    current: RecentWindowIdentity | None,
    incoming: RecentWindowIdentity,
) -> bool:
    """Reject an older writer without preventing a newer lifecycle takeover."""

    if current is None:
        return True
    if (current.user_id, current.conversation_id) != (
        incoming.user_id,
        incoming.conversation_id,
    ):
        return False
    if current.lifecycle_epoch != incoming.lifecycle_epoch:
        return True
    if current.lifecycle_cleanup_key != incoming.lifecycle_cleanup_key:
        return False
    current_user_revision = (
        current.cache_revision,
        current.derivation_revision,
    )
    incoming_user_revision = (
        incoming.cache_revision,
        incoming.derivation_revision,
    )
    if incoming_user_revision != current_user_revision:
        return incoming_user_revision > current_user_revision
    if current.conversation_lifecycle_epoch != incoming.conversation_lifecycle_epoch:
        return False
    return incoming.conversation_source_revision >= current.conversation_source_revision


def _wrap_lifecycle_diagnostic(
    payload: dict[str, Any],
    *,
    lifecycle_cleanup_key: str,
    lifecycle_epoch: str,
    delivery_id: str,
) -> dict[str, Any]:
    """Wrap a list diagnostic with the metadata required for atomic revocation."""

    return {
        _LIFECYCLE_DIAGNOSTIC_MARKER: {
            "delivery_id": delivery_id,
            "lifecycle_cleanup_key": lifecycle_cleanup_key,
            "lifecycle_epoch": lifecycle_epoch,
        },
        "payload": copy.deepcopy(payload),
    }


def _lifecycle_diagnostic_metadata(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    if set(payload) != {_LIFECYCLE_DIAGNOSTIC_MARKER, "payload"}:
        return None
    if not isinstance(payload.get("payload"), dict):
        return None
    metadata = payload.get(_LIFECYCLE_DIAGNOSTIC_MARKER)
    if not isinstance(metadata, dict):
        return None
    if set(metadata) != {
        "delivery_id",
        "lifecycle_cleanup_key",
        "lifecycle_epoch",
    }:
        return None
    delivery_id = metadata.get("delivery_id")
    lifecycle_cleanup_key = metadata.get("lifecycle_cleanup_key")
    lifecycle_epoch = metadata.get("lifecycle_epoch")
    if not all(
        isinstance(value, str) and value
        for value in (delivery_id, lifecycle_cleanup_key, lifecycle_epoch)
    ):
        return None
    return {
        "delivery_id": delivery_id,
        "lifecycle_cleanup_key": lifecycle_cleanup_key,
        "lifecycle_epoch": lifecycle_epoch,
    }


def _is_current_job_notification(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "job_id",
        "dispatch_token",
        "lifecycle_epoch",
        "lifecycle_cleanup_key",
    }
    return set(payload) == required and all(
        isinstance(payload.get(key), str) and payload[key] for key in required
    )


def _legacy_job_envelope_user_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    allowed = {
        "schema_version",
        "job_id",
        "job_type",
        "user_id",
        "parent_job_id",
        "conversation_id",
        "message_ids",
        "transcript_rebuild_id",
        "maintenance_operation_id",
        "payload",
        "created_at",
        "operational_profile",
    }
    required = {"job_id", "job_type", "user_id", "payload"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        return None
    if not all(
        isinstance(payload.get(key), str) and payload[key]
        for key in ("job_id", "job_type", "user_id")
    ):
        return None
    if not isinstance(payload.get("payload"), dict):
        return None
    schema_version = payload.get("schema_version", 1)
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        return None
    message_ids = payload.get("message_ids", [])
    if not isinstance(message_ids, list) or not all(
        isinstance(message_id, str) and message_id for message_id in message_ids
    ):
        return None
    optional_string_fields = (
        "parent_job_id",
        "conversation_id",
        "transcript_rebuild_id",
        "maintenance_operation_id",
        "created_at",
    )
    if any(
        value is not None and not isinstance(value, str)
        for value in (payload.get(field) for field in optional_string_fields)
    ):
        return None
    operational_profile = payload.get("operational_profile")
    if operational_profile is not None and not isinstance(operational_profile, dict):
        return None
    return payload["user_id"]


def _unwrap_lifecycle_diagnostic(payload: Any) -> Any:
    if _lifecycle_diagnostic_metadata(payload) is None:
        return payload
    if not isinstance(payload, dict):  # pragma: no cover - narrowed above.
        return payload
    return payload.get("payload")


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


@dataclass(frozen=True, slots=True)
class LegacyTransientPurgeResult:
    """Counts from a conservative purge of pre-lifecycle transient state."""

    cache_generation_deleted: int = 0
    recent_windows_deleted: int = 0
    context_views_deleted: int = 0
    stream_entries_deleted: int = 0
    queue_entries_deleted: int = 0
    deferred_entries_deleted: int = 0
    legacy_dedupe_deleted: int = 0
    legacy_locks_deleted: int = 0
    malformed_candidates: int = 0

    @property
    def total_deleted(self) -> int:
        """Return the number of obsolete entries removed across all containers."""

        return (
            self.cache_generation_deleted
            + self.recent_windows_deleted
            + self.context_views_deleted
            + self.stream_entries_deleted
            + self.queue_entries_deleted
            + self.deferred_entries_deleted
            + self.legacy_dedupe_deleted
            + self.legacy_locks_deleted
        )

    @property
    def clean(self) -> bool:
        """Whether every candidate could be classified safely."""

        return self.malformed_candidates == 0


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
    if isinstance(nested, dict) and (
        "user_id" in nested or "conversation_id" in nested
    ):
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

    async def set_recent_window_for_lifecycle(
        self,
        key: str,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        """Write and index a window only for the mirrored active lifecycle."""

        raise NotImplementedError

    async def delete_recent_window_if_cache_identity(
        self,
        key: str,
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        """Delete only the recent-window publication owned by this identity."""

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

    async def set_context_view_if_newer_for_lifecycle(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> bool:
        """Atomically fence, publish, and index one lifecycle-owned cache view."""

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

    async def publish_lifecycle_diagnostic(
        self,
        queue_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        """Atomically lifecycle-fence, publish, and index a list diagnostic."""

        raise NotImplementedError

    async def dequeue_job(
        self,
        queue_name: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    async def prepare_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> str:
        """Create a non-publishable mirror or return its existing state."""

        raise NotImplementedError

    async def activate_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> bool:
        """Activate only the exact preparing mirror created by ``nonce``."""

        raise NotImplementedError

    async def publish_job_notification(
        self,
        stream_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        """Atomically validate the active lifecycle, publish, and index delivery."""

        raise NotImplementedError

    async def revoke_lifecycle_and_purge_notifications(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        *,
        group_name: str,
    ) -> int:
        """Atomically revoke a lifecycle and remove its indexed notifications."""

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

    async def stream_ack(
        self, stream_name: str, group_name: str, message_id: str
    ) -> None:
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

    async def remember_dedupe(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
        raise NotImplementedError

    async def force_dedupe(self, key: str, ttl_seconds: int) -> None:
        """Set or overwrite a dedupe marker unconditionally."""
        raise NotImplementedError

    async def has_dedupe(self, key: str) -> bool:
        raise NotImplementedError

    async def acquire_lock(
        self,
        key: str,
        ttl_seconds: int,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> str | None:
        raise NotImplementedError

    async def release_lock(
        self,
        key: str,
        token: str,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> None:
        raise NotImplementedError

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        """Conservatively remove pre-lifecycle state owned by one user."""

        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class InProcessBackend(StorageBackend):
    """Single-process, Redis-free backend for local development and tests."""

    _recent_windows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _recent_window_cache_identities: dict[str, RecentWindowIdentity] = field(
        default_factory=dict
    )
    _recent_window_keys_by_user: dict[str, set[str]] = field(default_factory=dict)
    _context_views: dict[str, "_InProcessContextViewEntry"] = field(
        default_factory=dict
    )
    _context_view_keys_by_user: dict[str, set[str]] = field(default_factory=dict)
    _context_view_keys_by_conversation: dict[tuple[str, str], set[str]] = field(
        default_factory=dict
    )
    _dedupe_keys: dict[str, float] = field(default_factory=dict)
    _locks: dict[str, tuple[float, str]] = field(default_factory=dict)
    _lifecycle_locks: dict[
        tuple[str, str, str],
        tuple[float, str, str | None, int | None],
    ] = field(default_factory=dict)
    _lifecycle_lock_high_waters: dict[tuple[str, str, str, str], tuple[float, int]] = (
        field(default_factory=dict)
    )
    _lifecycle_transient_index: dict[tuple[str, str], set[tuple[str, str]]] = field(
        default_factory=dict
    )
    _queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    _stream_pending: dict[tuple[str, str], dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    _stream_groups: set[tuple[str, str]] = field(default_factory=set)
    _stream_add_counts: dict[str, int] = field(default_factory=dict)
    _stream_read_counts: dict[str, int] = field(default_factory=dict)
    _stream_claim_counts: dict[str, int] = field(default_factory=dict)
    _stream_ack_counts: dict[str, int] = field(default_factory=dict)
    _lifecycle_mirrors: dict[str, str] = field(default_factory=dict)
    _lifecycle_delivery_index: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict
    )
    _stream_delivery_lifecycle: dict[tuple[str, str], str] = field(default_factory=dict)
    _lifecycle_diagnostic_index: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict
    )
    _diagnostic_delivery_lifecycle: dict[tuple[str, str], str] = field(
        default_factory=dict
    )
    _lifecycle_cache_index: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict
    )
    _lifecycle_cache_owner: dict[tuple[str, str], str] = field(default_factory=dict)
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

        expired_dedupe_keys = [
            key for key, expires_at in self._dedupe_keys.items() if expires_at <= now
        ]
        for key in expired_dedupe_keys:
            self._dedupe_keys.pop(key, None)

        expired_lock_keys = [
            key
            for key, (expires_at, _token) in self._locks.items()
            if expires_at <= now
        ]
        for key in expired_lock_keys:
            self._locks.pop(key, None)

        expired_lifecycle_lock_keys = [
            namespaced_key
            for namespaced_key, (
                expires_at,
                _token,
                _job_id,
                _execution_fence,
            ) in self._lifecycle_locks.items()
            if expires_at <= now
        ]
        for cleanup_key, epoch, key in expired_lifecycle_lock_keys:
            self._delete_lifecycle_transient_locked(
                "lock",
                key,
                (cleanup_key, epoch),
            )
        expired_high_water_keys = [
            namespaced_key
            for namespaced_key, (expires_at, _fence) in (
                self._lifecycle_lock_high_waters.items()
            )
            if expires_at <= now
        ]
        for namespaced_key in expired_high_water_keys:
            self._lifecycle_lock_high_waters.pop(namespaced_key, None)

    def _index_lifecycle_transient_locked(
        self,
        kind: str,
        key: str,
        owner: tuple[str, str],
    ) -> None:
        subject = (kind, key)
        self._lifecycle_transient_index.setdefault(owner, set()).add(subject)

    def _remove_lifecycle_transient_index_locked(
        self,
        kind: str,
        key: str,
        owner: tuple[str, str],
    ) -> None:
        subject = (kind, key)
        subjects = self._lifecycle_transient_index.get(owner)
        if subjects is None:
            return
        subjects.discard(subject)
        if not subjects:
            self._lifecycle_transient_index.pop(owner, None)

    def _delete_lifecycle_transient_locked(
        self,
        kind: str,
        key: str,
        owner: tuple[str, str],
    ) -> None:
        self._remove_lifecycle_transient_index_locked(kind, key, owner)
        namespaced_key = (*owner, key)
        if kind == "lock":
            self._lifecycle_locks.pop(namespaced_key, None)

    def _delete_context_view_locked(self, key: str) -> bool:
        self._remove_lifecycle_cache_index_locked("context", key)
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

    def _remove_recent_window_user_index_locked(
        self,
        key: str,
        identity: RecentWindowIdentity,
    ) -> None:
        keys = self._recent_window_keys_by_user.get(identity.user_id)
        if keys is None:
            return
        keys.discard(key)
        if not keys:
            self._recent_window_keys_by_user.pop(identity.user_id, None)

    def _delete_recent_window_locked(self, key: str) -> bool:
        self._remove_lifecycle_cache_index_locked("recent", key)
        identity = self._recent_window_cache_identities.pop(key, None)
        if identity is not None:
            self._remove_recent_window_user_index_locked(key, identity)
        return self._recent_windows.pop(key, None) is not None

    def _index_lifecycle_cache_locked(
        self,
        kind: str,
        key: str,
        lifecycle_cleanup_key: str,
    ) -> None:
        self._remove_lifecycle_cache_index_locked(kind, key)
        subject = (kind, key)
        self._lifecycle_cache_owner[subject] = lifecycle_cleanup_key
        self._lifecycle_cache_index.setdefault(lifecycle_cleanup_key, set()).add(
            subject
        )

    def _remove_lifecycle_cache_index_locked(self, kind: str, key: str) -> None:
        subject = (kind, key)
        owner = self._lifecycle_cache_owner.pop(subject, None)
        if owner is None:
            return
        indexed = self._lifecycle_cache_index.get(owner)
        if indexed is None:
            return
        indexed.discard(subject)
        if not indexed:
            self._lifecycle_cache_index.pop(owner, None)

    def _lifecycle_context_is_current_locked(self, key: str) -> bool:
        subject = ("context", key)
        owner = self._lifecycle_cache_owner.get(subject)
        if owner is None:
            return False
        mirror = self._lifecycle_mirrors.get(owner)
        return (
            isinstance(mirror, str)
            and mirror.startswith("active:")
            and len(mirror) > len("active:")
            and subject in self._lifecycle_cache_index.get(owner, set())
        )

    def _lifecycle_recent_is_current_locked(
        self,
        key: str,
        identity: RecentWindowIdentity,
    ) -> bool:
        if not _recent_window_identity_is_valid(identity):
            return False
        if key != build_recent_window_key(identity.user_id, identity.conversation_id):
            return False
        subject = ("recent", key)
        cleanup_key = identity.lifecycle_cleanup_key
        return (
            self._lifecycle_cache_owner.get(subject) == cleanup_key
            and subject in self._lifecycle_cache_index.get(cleanup_key, set())
            and self._lifecycle_mirrors.get(cleanup_key)
            == f"active:{identity.lifecycle_epoch}"
        )

    def _stream_notification_is_current_locked(
        self,
        stream_name: str,
        message_id: str,
        payload: Any,
    ) -> bool:
        if not _is_current_job_notification(payload):
            return False
        cleanup_key = payload["lifecycle_cleanup_key"]
        return (
            self._lifecycle_mirrors.get(cleanup_key)
            == f"active:{payload['lifecycle_epoch']}"
            and (stream_name, message_id)
            in self._lifecycle_delivery_index.get(cleanup_key, set())
            and self._stream_delivery_lifecycle.get((stream_name, message_id))
            == cleanup_key
        )

    def _diagnostic_is_current_locked(
        self,
        queue_name: str,
        metadata: dict[str, str],
    ) -> bool:
        cleanup_key = metadata["lifecycle_cleanup_key"]
        delivery_id = metadata["delivery_id"]
        return (
            self._lifecycle_mirrors.get(cleanup_key)
            == f"active:{metadata['lifecycle_epoch']}"
            and (queue_name, delivery_id)
            in self._lifecycle_diagnostic_index.get(cleanup_key, set())
            and self._diagnostic_delivery_lifecycle.get((queue_name, delivery_id))
            == cleanup_key
        )

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

    async def set_recent_window_for_lifecycle(
        self,
        key: str,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        incoming_identity = RecentWindowIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
            conversation_lifecycle_epoch=conversation_lifecycle_epoch,
            conversation_source_revision=conversation_source_revision,
        )
        if not _recent_window_identity_is_valid(
            incoming_identity
        ) or key != build_recent_window_key(user_id, conversation_id):
            return False
        with self._guard:
            if (
                self._lifecycle_mirrors.get(lifecycle_cleanup_key)
                != f"active:{lifecycle_epoch}"
            ):
                return False
            current_identity = self._recent_window_cache_identities.get(key)
            if (
                current_identity is not None
                and current_identity.lifecycle_epoch != lifecycle_epoch
                and self._lifecycle_mirrors.get(current_identity.lifecycle_cleanup_key)
                == f"active:{current_identity.lifecycle_epoch}"
            ):
                return False
            if not _recent_window_identity_can_replace(
                current_identity,
                incoming_identity,
            ):
                return False
            if current_identity is not None:
                self._remove_recent_window_user_index_locked(key, current_identity)
            self._recent_windows[key] = copy.deepcopy(messages)
            self._recent_window_cache_identities[key] = incoming_identity
            self._recent_window_keys_by_user.setdefault(user_id, set()).add(key)
            self._index_lifecycle_cache_locked("recent", key, lifecycle_cleanup_key)
            return True

    async def delete_recent_window_if_cache_identity(
        self,
        key: str,
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        candidate_identity = RecentWindowIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
            conversation_lifecycle_epoch=conversation_lifecycle_epoch,
            conversation_source_revision=conversation_source_revision,
        )
        if not _recent_window_identity_is_valid(
            candidate_identity
        ) or key != build_recent_window_key(user_id, conversation_id):
            return False
        with self._guard:
            if self._recent_window_cache_identities.get(key) != candidate_identity:
                return False
            if self._lifecycle_cache_owner.get(("recent", key)) != (
                lifecycle_cleanup_key
            ):
                return False
            self._delete_recent_window_locked(key)
            return True

    async def delete_recent_windows_for_user(self, user_id: str) -> int:
        with self._guard:
            keys = list(self._recent_window_keys_by_user.get(user_id, set()))
            deleted = 0
            for key in keys:
                identity = self._recent_window_cache_identities.get(key)
                if (
                    identity is None
                    or identity.user_id != user_id
                    or key
                    != build_recent_window_key(
                        identity.user_id,
                        identity.conversation_id,
                    )
                ):
                    continue
                deleted += int(self._delete_recent_window_locked(key))
            return deleted

    async def delete_recent_window_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        key = build_recent_window_key(user_id, conversation_id)
        with self._guard:
            identity = self._recent_window_cache_identities.get(key)
            if identity is None or (
                identity.user_id,
                identity.conversation_id,
            ) != (user_id, conversation_id):
                return 0
            return int(self._delete_recent_window_locked(key))

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

    async def set_context_view_if_newer_for_lifecycle(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> bool:
        with self._guard:
            self._purge_expired()
            if (
                self._lifecycle_mirrors.get(lifecycle_cleanup_key)
                != f"active:{lifecycle_epoch}"
            ):
                return False
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
            self._index_lifecycle_cache_locked("context", key, lifecycle_cleanup_key)
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
            keys = list(
                self._context_view_keys_by_conversation.get(
                    (user_id, conversation_id), set()
                )
            )
            deleted = 0
            for key in keys:
                if self._delete_context_view_locked(key):
                    deleted += 1
            return deleted

    async def purge_user_jobs(self, user_id: str) -> int:
        with self._guard:
            return self._purge_jobs_locked(
                lambda payload: _job_matches_user(payload, user_id)
            )

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
                payload = (
                    item.get("payload")
                    if isinstance(item, dict) and "payload" in item
                    else item
                )
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

    async def publish_lifecycle_diagnostic(
        self,
        queue_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        delivery_id = generate_prefixed_id("dlq")
        wrapped = _wrap_lifecycle_diagnostic(
            payload,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            delivery_id=delivery_id,
        )
        with self._guard:
            if (
                self._lifecycle_mirrors.get(lifecycle_cleanup_key)
                != f"active:{lifecycle_epoch}"
            ):
                return None
            queue = self._queues.setdefault(queue_name, asyncio.Queue())
            queue.put_nowait(wrapped)
            self._lifecycle_diagnostic_index.setdefault(
                lifecycle_cleanup_key,
                set(),
            ).add((queue_name, delivery_id))
            self._diagnostic_delivery_lifecycle[(queue_name, delivery_id)] = (
                lifecycle_cleanup_key
            )
        return delivery_id

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
        else:
            try:
                # None means "wait indefinitely" for the next queued job.
                if timeout_seconds is None:
                    payload = await queue.get()
                else:
                    payload = await asyncio.wait_for(queue.get(), timeout_seconds)
            except TimeoutError:
                return None
        metadata = _lifecycle_diagnostic_metadata(payload)
        if metadata is not None:
            delivery_id = metadata["delivery_id"]
            with self._guard:
                cleanup_key = self._diagnostic_delivery_lifecycle.pop(
                    (queue_name, delivery_id),
                    None,
                )
                if cleanup_key is not None:
                    deliveries = self._lifecycle_diagnostic_index.get(cleanup_key)
                    if deliveries is not None:
                        deliveries.discard((queue_name, delivery_id))
                        if not deliveries:
                            self._lifecycle_diagnostic_index.pop(cleanup_key, None)
        return copy.deepcopy(_unwrap_lifecycle_diagnostic(payload))

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

    async def prepare_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> str:
        preparing = f"preparing:{lifecycle_epoch}:{nonce}"
        with self._guard:
            current = self._lifecycle_mirrors.get(lifecycle_cleanup_key)
            if current is None or current.startswith(f"preparing:{lifecycle_epoch}:"):
                self._lifecycle_mirrors[lifecycle_cleanup_key] = preparing
                return preparing
            return current

    async def activate_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> bool:
        expected = f"preparing:{lifecycle_epoch}:{nonce}"
        with self._guard:
            if self._lifecycle_mirrors.get(lifecycle_cleanup_key) != expected:
                return False
            self._lifecycle_mirrors[lifecycle_cleanup_key] = f"active:{lifecycle_epoch}"
            return True

    async def publish_job_notification(
        self,
        stream_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        message_id = generate_prefixed_id("stm")
        with self._guard:
            if (
                self._lifecycle_mirrors.get(lifecycle_cleanup_key)
                != f"active:{lifecycle_epoch}"
            ):
                return None
            queue = self._queues.setdefault(f"stream:{stream_name}", asyncio.Queue())
            queue.put_nowait(
                {
                    "message_id": message_id,
                    "payload": copy.deepcopy(payload),
                }
            )
            self._lifecycle_delivery_index.setdefault(
                lifecycle_cleanup_key,
                set(),
            ).add((stream_name, message_id))
            self._stream_delivery_lifecycle[(stream_name, message_id)] = (
                lifecycle_cleanup_key
            )
            self._stream_add_counts[stream_name] = (
                self._stream_add_counts.get(stream_name, 0) + 1
            )
            return message_id

    async def revoke_lifecycle_and_purge_notifications(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        *,
        group_name: str,
    ) -> int:
        with self._guard:
            self._lifecycle_mirrors[lifecycle_cleanup_key] = (
                f"revoked:{lifecycle_epoch}"
            )
            cache_entries = list(
                self._lifecycle_cache_index.get(lifecycle_cleanup_key, set())
            )
            for kind, key in cache_entries:
                if self._lifecycle_cache_owner.get((kind, key)) != (
                    lifecycle_cleanup_key
                ):
                    continue
                if kind == "context":
                    self._delete_context_view_locked(key)
                elif kind == "recent":
                    identity = self._recent_window_cache_identities.get(key)
                    if identity is None or (
                        identity.lifecycle_cleanup_key,
                        identity.lifecycle_epoch,
                    ) != (lifecycle_cleanup_key, lifecycle_epoch):
                        continue
                    self._delete_recent_window_locked(key)
            transient_entries = list(
                self._lifecycle_transient_index.get(
                    (lifecycle_cleanup_key, lifecycle_epoch),
                    set(),
                )
            )
            for kind, key in transient_entries:
                self._delete_lifecycle_transient_locked(
                    kind,
                    key,
                    (lifecycle_cleanup_key, lifecycle_epoch),
                )
            high_water_keys = [
                namespaced_key
                for namespaced_key in self._lifecycle_lock_high_waters
                if namespaced_key[:2] == (lifecycle_cleanup_key, lifecycle_epoch)
            ]
            for namespaced_key in high_water_keys:
                self._lifecycle_lock_high_waters.pop(namespaced_key, None)
            indexed = self._lifecycle_delivery_index.pop(lifecycle_cleanup_key, set())
            purged = 0
            for stream_name, message_id in indexed:
                queue = self._queues.get(f"stream:{stream_name}")
                if queue is not None:
                    retained: list[dict[str, Any]] = []
                    while True:
                        try:
                            item = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if str(item.get("message_id")) == message_id:
                            purged += 1
                        else:
                            retained.append(item)
                    for item in retained:
                        queue.put_nowait(item)
                pending = self._stream_pending.setdefault((stream_name, group_name), {})
                if pending.pop(message_id, None) is not None:
                    purged += 1
                    if self._pending_job_count > 0:
                        self._pending_job_count -= 1
                self._stream_delivery_lifecycle.pop((stream_name, message_id), None)
            diagnostics = self._lifecycle_diagnostic_index.pop(
                lifecycle_cleanup_key,
                set(),
            )
            for queue_name, delivery_id in diagnostics:
                queue = self._queues.get(queue_name)
                if queue is not None:
                    retained: list[dict[str, Any]] = []
                    while True:
                        try:
                            item = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        metadata = _lifecycle_diagnostic_metadata(item)
                        if (
                            metadata is not None
                            and metadata["delivery_id"] == delivery_id
                        ):
                            purged += 1
                        else:
                            retained.append(item)
                    for item in retained:
                        queue.put_nowait(item)
                self._diagnostic_delivery_lifecycle.pop(
                    (queue_name, delivery_id),
                    None,
                )
            return purged

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        del (
            consumer_name
        )  # The in-process fallback does not simulate per-consumer ownership.
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
        del (
            consumer_name
        )  # The in-process fallback does not simulate per-consumer ownership.
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
            claimable.sort(
                key=lambda item: float(item[1].get("last_delivered_at", 0.0))
            )
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

    async def stream_ack(
        self, stream_name: str, group_name: str, message_id: str
    ) -> None:
        with self._guard:
            pending = self._stream_pending.setdefault((stream_name, group_name), {})
            removed = pending.pop(message_id, None)
            if removed is not None and self._pending_job_count > 0:
                self._pending_job_count -= 1
            if removed is not None:
                self._stream_ack_counts[stream_name] = (
                    self._stream_ack_counts.get(stream_name, 0) + 1
                )
            lifecycle_cleanup_key = self._stream_delivery_lifecycle.pop(
                (stream_name, message_id),
                None,
            )
            if lifecycle_cleanup_key is not None:
                deliveries = self._lifecycle_delivery_index.get(lifecycle_cleanup_key)
                if deliveries is not None:
                    deliveries.discard((stream_name, message_id))
                    if not deliveries:
                        self._lifecycle_delivery_index.pop(lifecycle_cleanup_key, None)

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
                    last_delivered_at = float(
                        entry.get("last_delivered_at", 0.0) or 0.0
                    )
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
                                "message_ids": list(payload.get("message_ids") or [])[
                                    :5
                                ],
                                "payload_message_id": (
                                    nested_payload.get("message_id")
                                    if isinstance(nested_payload, dict)
                                    else None
                                ),
                                "delivery_count": int(
                                    entry.get("delivery_count", 1) or 1
                                ),
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
            None if idle_timeout_seconds is None else max(0.0, idle_timeout_seconds)
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
            if (
                idle_timeout is not None
                and monotonic() - last_progress_at >= idle_timeout
            ):
                return False
            if monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    async def remember_dedupe(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
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

    async def acquire_lock(
        self,
        key: str,
        ttl_seconds: int,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> str | None:
        lifecycle = _validated_lifecycle_coordinates(
            lifecycle_cleanup_key,
            lifecycle_epoch,
        )
        job_scope = _validated_job_lock_scope(
            job_id,
            execution_fence,
            lifecycle=lifecycle,
        )
        with self._guard:
            self._purge_expired()
            if lifecycle is not None:
                cleanup_key, epoch = lifecycle
                if self._lifecycle_mirrors.get(cleanup_key) != f"active:{epoch}":
                    return None
                namespaced_key = (cleanup_key, epoch, key)
                current = self._lifecycle_locks.get(namespaced_key)
                if current is not None:
                    if job_scope is None:
                        return None
                    current_job_id = current[2]
                    current_execution_fence = current[3]
                    if (
                        current_job_id != job_scope[0]
                        or current_execution_fence is None
                        or current_execution_fence >= job_scope[1]
                    ):
                        return None
                elif job_scope is not None:
                    high_water = self._lifecycle_lock_high_waters.get(
                        (*namespaced_key, job_scope[0])
                    )
                    if high_water is not None and high_water[1] >= job_scope[1]:
                        return None
                token = uuid4().hex
                expires_at = monotonic() + ttl_seconds
                self._lifecycle_locks[namespaced_key] = (
                    expires_at,
                    token,
                    job_scope[0] if job_scope is not None else None,
                    job_scope[1] if job_scope is not None else None,
                )
                if job_scope is not None:
                    self._lifecycle_lock_high_waters[
                        (*namespaced_key, job_scope[0])
                    ] = (expires_at, job_scope[1])
                self._index_lifecycle_transient_locked("lock", key, lifecycle)
                return token
            if key in self._locks:
                return None
            token = uuid4().hex
            self._locks[key] = (monotonic() + ttl_seconds, token)
            return token

    async def release_lock(
        self,
        key: str,
        token: str,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> None:
        lifecycle = _validated_lifecycle_coordinates(
            lifecycle_cleanup_key,
            lifecycle_epoch,
        )
        job_scope = _validated_job_lock_scope(
            job_id,
            execution_fence,
            lifecycle=lifecycle,
        )
        with self._guard:
            if lifecycle is not None:
                entry = self._lifecycle_locks.get((*lifecycle, key))
                if entry is None or entry[1] != token:
                    return
                if job_scope is not None and entry[2:] != job_scope:
                    return
                self._delete_lifecycle_transient_locked("lock", key, lifecycle)
                return
            entry = self._locks.get(key)
            if entry is None:
                return
            if entry[1] != token:
                return
            self._locks.pop(key, None)

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        del database_path
        with self._guard:
            self._purge_expired()

            recent_windows_deleted = 0
            malformed_candidates = 0
            for key, value in list(self._recent_windows.items()):
                identity = self._recent_window_cache_identities.get(key)
                if identity is not None:
                    if (
                        not isinstance(identity, RecentWindowIdentity)
                        or not _recent_window_identity_is_valid(identity)
                        or key
                        != build_recent_window_key(
                            identity.user_id,
                            identity.conversation_id,
                        )
                        or not isinstance(value, list)
                    ):
                        malformed_candidates += 1
                        continue
                    if self._lifecycle_recent_is_current_locked(key, identity):
                        continue
                    self._remove_lifecycle_cache_index_locked("recent", key)
                    if identity.user_id == user_id:
                        recent_windows_deleted += int(
                            self._delete_recent_window_locked(key)
                        )
                    continue

                if not isinstance(value, list):
                    malformed_candidates += 1
                    continue
                recent_windows_deleted += int(self._delete_recent_window_locked(key))

            for key, identity in list(self._recent_window_cache_identities.items()):
                if key in self._recent_windows:
                    continue
                if (
                    not isinstance(identity, RecentWindowIdentity)
                    or not _recent_window_identity_is_valid(identity)
                    or key
                    != build_recent_window_key(
                        identity.user_id,
                        identity.conversation_id,
                    )
                ):
                    malformed_candidates += 1
                    continue
                if identity.user_id == user_id:
                    self._delete_recent_window_locked(key)
            indexed_recent = self._recent_window_keys_by_user.get(user_id)
            if indexed_recent is not None:
                indexed_recent.intersection_update(self._recent_windows)
                if not indexed_recent:
                    self._recent_window_keys_by_user.pop(user_id, None)

            legacy_context_keys = [
                key
                for key in self._context_view_keys_by_user.get(user_id, set())
                if not self._lifecycle_context_is_current_locked(key)
            ]
            context_views_deleted = sum(
                int(self._delete_context_view_locked(key))
                for key in legacy_context_keys
            )

            queue_entries_deleted = 0
            stream_entries_deleted = 0

            def classify_owned_payload(payload: Any) -> str:
                if not isinstance(payload, dict):
                    return "malformed"
                owner = payload.get("user_id")
                if not isinstance(owner, str) or not owner:
                    return "malformed"
                return "target" if owner == user_id else "other"

            def classify_legacy_job_payload(payload: Any) -> str:
                owner = _legacy_job_envelope_user_id(payload)
                if owner is None:
                    return "malformed"
                return "target" if owner == user_id else "other"

            def classify_ordinary_queue_item(queue_name: str, item: Any) -> str:
                if not isinstance(item, dict):
                    return "malformed"
                if queue_name == "admin_rebuild_user":
                    candidate = item if set(item) == {"user_id"} else None
                    return classify_owned_payload(candidate)
                elif queue_name == "admin_rebuild_conversation":
                    candidate = (
                        item
                        if set(item) == {"conversation_id", "user_id"}
                        and isinstance(item.get("conversation_id"), str)
                        and bool(item["conversation_id"])
                        else None
                    )
                    return classify_owned_payload(candidate)
                elif queue_name.startswith("dead_letter:"):
                    allowed_shapes = (
                        {"message_id", "delivery_count", "payload", "error"},
                        {
                            "message_id",
                            "delivery_count",
                            "payload",
                            "error",
                            "error_details",
                        },
                    )
                    delivery_count = item.get("delivery_count")
                    candidate = (
                        item.get("payload")
                        if set(item) in allowed_shapes
                        and isinstance(item.get("message_id"), str)
                        and bool(item["message_id"])
                        and isinstance(delivery_count, int)
                        and not isinstance(delivery_count, bool)
                        and delivery_count >= 1
                        and isinstance(item.get("payload"), dict)
                        and isinstance(item.get("error"), str)
                        and (
                            "error_details" not in item
                            or isinstance(item["error_details"], list)
                        )
                        else None
                    )
                    return classify_legacy_job_payload(candidate)
                return "malformed"

            for queue_name in sorted(_LEGACY_ATAGIA_LIST_QUEUE_NAMES):
                queue = self._queues.get(queue_name)
                if queue is None:
                    continue
                retained: list[Any] = []
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    classification = "other"
                    if _is_current_job_notification(item):
                        classification = "malformed"
                    else:
                        metadata = _lifecycle_diagnostic_metadata(item)
                        if metadata is not None:
                            cleanup_key = metadata["lifecycle_cleanup_key"]
                            if self._lifecycle_mirrors.get(cleanup_key) == (
                                f"active:{metadata['lifecycle_epoch']}"
                            ):
                                classification = (
                                    "other"
                                    if self._diagnostic_is_current_locked(
                                        queue_name,
                                        metadata,
                                    )
                                    else "malformed"
                                )
                            else:
                                delivery_id = metadata["delivery_id"]
                                indexed = self._lifecycle_diagnostic_index.get(
                                    cleanup_key
                                )
                                if indexed is not None:
                                    indexed.discard((queue_name, delivery_id))
                                    if not indexed:
                                        self._lifecycle_diagnostic_index.pop(
                                            cleanup_key,
                                            None,
                                        )
                                if (
                                    self._diagnostic_delivery_lifecycle.get(
                                        (queue_name, delivery_id)
                                    )
                                    == cleanup_key
                                ):
                                    self._diagnostic_delivery_lifecycle.pop(
                                        (queue_name, delivery_id),
                                        None,
                                    )
                                classification = classify_owned_payload(item["payload"])
                        elif (
                            isinstance(item, dict)
                            and _LIFECYCLE_DIAGNOSTIC_MARKER in item
                        ):
                            classification = "malformed"
                        else:
                            classification = classify_ordinary_queue_item(
                                queue_name,
                                item,
                            )
                    if classification == "target":
                        queue_entries_deleted += 1
                    else:
                        retained.append(item)
                        if classification == "malformed":
                            malformed_candidates += 1
                for item in retained:
                    queue.put_nowait(item)

            for stream_name in sorted(_ATAGIA_JOB_STREAM_NAMES):
                queue = self._queues.get(f"stream:{stream_name}")
                if queue is None:
                    continue
                retained = []
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if (
                        not isinstance(item, dict)
                        or set(item) != {"message_id", "payload"}
                        or not isinstance(item.get("message_id"), str)
                        or not item["message_id"]
                        or not isinstance(item.get("payload"), dict)
                    ):
                        retained.append(item)
                        malformed_candidates += 1
                        continue
                    message_id = item["message_id"]
                    payload = item["payload"]
                    if _is_current_job_notification(payload):
                        classification = (
                            "other"
                            if self._stream_notification_is_current_locked(
                                stream_name,
                                message_id,
                                payload,
                            )
                            else "malformed"
                        )
                    else:
                        classification = classify_legacy_job_payload(payload)
                    if classification == "target":
                        stream_entries_deleted += 1
                        cleanup_key = self._stream_delivery_lifecycle.pop(
                            (stream_name, message_id),
                            None,
                        )
                        if cleanup_key is not None:
                            indexed = self._lifecycle_delivery_index.get(cleanup_key)
                            if indexed is not None:
                                indexed.discard((stream_name, message_id))
                                if not indexed:
                                    self._lifecycle_delivery_index.pop(
                                        cleanup_key,
                                        None,
                                    )
                    else:
                        retained.append(item)
                        if classification == "malformed":
                            malformed_candidates += 1
                for item in retained:
                    queue.put_nowait(item)

            for (stream_name, _group_name), pending in self._stream_pending.items():
                if stream_name not in _ATAGIA_JOB_STREAM_NAMES:
                    continue
                for message_id, entry in list(pending.items()):
                    if isinstance(entry, dict) and set(entry) == {
                        "payload",
                        "delivery_count",
                        "last_delivered_at",
                    }:
                        payload = entry.get("payload")
                    else:
                        payload = entry
                    if _is_current_job_notification(payload):
                        classification = (
                            "other"
                            if self._stream_notification_is_current_locked(
                                stream_name,
                                message_id,
                                payload,
                            )
                            else "malformed"
                        )
                    else:
                        classification = classify_legacy_job_payload(payload)
                    if classification == "target":
                        pending.pop(message_id, None)
                        stream_entries_deleted += 1
                        if self._pending_job_count > 0:
                            self._pending_job_count -= 1
                        cleanup_key = self._stream_delivery_lifecycle.pop(
                            (stream_name, message_id),
                            None,
                        )
                        if cleanup_key is not None:
                            indexed = self._lifecycle_delivery_index.get(cleanup_key)
                            if indexed is not None:
                                indexed.discard((stream_name, message_id))
                                if not indexed:
                                    self._lifecycle_delivery_index.pop(
                                        cleanup_key,
                                        None,
                                    )
                    elif classification == "malformed":
                        malformed_candidates += 1

            legacy_dedupe_keys = [
                key
                for key in self._dedupe_keys
                if _is_legacy_extractor_dedupe_key_for_user(key, user_id)
            ]
            for key in legacy_dedupe_keys:
                self._dedupe_keys.pop(key, None)
            # All other historical dedupe formats and every generic lock are
            # ownerless or delimiter-ambiguous. They are outside this user's
            # candidate set and must remain untouched.
        return LegacyTransientPurgeResult(
            recent_windows_deleted=recent_windows_deleted,
            context_views_deleted=context_views_deleted,
            stream_entries_deleted=stream_entries_deleted,
            queue_entries_deleted=queue_entries_deleted,
            legacy_dedupe_deleted=len(legacy_dedupe_keys),
            legacy_locks_deleted=0,
            malformed_candidates=malformed_candidates,
        )

    async def close(self) -> None:
        with self._guard:
            self._recent_windows.clear()
            self._recent_window_cache_identities.clear()
            self._recent_window_keys_by_user.clear()
            self._context_views.clear()
            self._context_view_keys_by_user.clear()
            self._context_view_keys_by_conversation.clear()
            self._dedupe_keys.clear()
            self._locks.clear()
            self._lifecycle_locks.clear()
            self._lifecycle_lock_high_waters.clear()
            self._lifecycle_transient_index.clear()
            self._queues.clear()
            self._stream_pending.clear()
            self._stream_groups.clear()
            self._stream_add_counts.clear()
            self._stream_read_counts.clear()
            self._stream_claim_counts.clear()
            self._stream_ack_counts.clear()
            self._lifecycle_mirrors.clear()
            self._lifecycle_delivery_index.clear()
            self._stream_delivery_lifecycle.clear()
            self._lifecycle_diagnostic_index.clear()
            self._diagnostic_delivery_lifecycle.clear()
            self._lifecycle_cache_index.clear()
            self._lifecycle_cache_owner.clear()
            self._pending_job_count = 0


@dataclass(slots=True)
class _InProcessContextViewEntry:
    """Internal in-process context-view record with TTL and invalidation metadata."""

    expires_at: float
    payload: dict[str, Any]
    monotonic_seq: int | None = None
    user_id: str | None = None
    conversation_id: str | None = None
