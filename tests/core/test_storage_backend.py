"""Tests for the in-process storage backend."""

import asyncio

import pytest

from atagia.core import storage_backend as storage_backend_module
from atagia.core.storage_backend import InProcessBackend, build_recent_window_key
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
)

from tests.recent_window_support import stored_recent_window


RECENT_WINDOW_OWNER = {"user_id": "usr_1", "conversation_id": "cnv_1"}
RECENT_WINDOW_KEY = build_recent_window_key(**RECENT_WINDOW_OWNER)
PUBLISHED_RECENT_WINDOW_IDENTITY: dict[str, object] = {
    "lifecycle_cleanup_key": "cleanup_1",
    "lifecycle_epoch": "epoch_1",
    "cache_revision": 0,
    "derivation_revision": 0,
    "conversation_lifecycle_epoch": "conversation_epoch_1",
    "conversation_source_revision": 0,
}


def test_legacy_extractor_dedupe_owner_match_uses_full_user_prefix() -> None:
    digest = "a" * 64

    assert storage_backend_module._is_legacy_extractor_dedupe_key_for_user(
        f"usr:target:{digest}",
        "usr:target",
    )
    assert not storage_backend_module._is_legacy_extractor_dedupe_key_for_user(
        f"usr:target:child:{digest}",
        "usr:target",
    )
    assert not storage_backend_module._is_legacy_extractor_dedupe_key_for_user(
        f"usr:target:{'b' * 63}",
        "usr:target",
    )


@pytest.mark.parametrize(
    "reserved_user_id",
    ("contract", "graph", "initial_context_package_refresh"),
)
def test_legacy_extractor_dedupe_owner_match_rejects_ownerless_hash_collisions(
    reserved_user_id: str,
) -> None:
    assert not storage_backend_module._is_legacy_extractor_dedupe_key_for_user(
        f"{reserved_user_id}:{'a' * 64}",
        reserved_user_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reserved_user_id",
    ("contract", "graph", "initial_context_package_refresh"),
)
async def test_inprocess_legacy_purge_preserves_ownerless_hash_collisions(
    reserved_user_id: str,
) -> None:
    backend = InProcessBackend()
    ambiguous_dedupe_key = f"{reserved_user_id}:{'a' * 64}"
    assert await backend.remember_dedupe(ambiguous_dedupe_key, ttl_seconds=60)

    result = await backend.purge_legacy_transient_state(
        "unused.db",
        reserved_user_id,
    )

    assert result.legacy_dedupe_deleted == 0
    assert result.malformed_candidates == 0
    assert result.clean
    assert await backend.has_dedupe(ambiguous_dedupe_key)


@pytest.mark.asyncio
async def test_recent_window_round_trip_uses_copies() -> None:
    backend = InProcessBackend()
    messages = [{"id": "msg_1", "text": "hello"}]
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_1",
        "epoch_1",
        "nonce_1",
    )

    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        messages,
        **RECENT_WINDOW_OWNER,
        **PUBLISHED_RECENT_WINDOW_IDENTITY,
    )
    fetched = await backend.get_recent_window_for_cache_identity(
        RECENT_WINDOW_KEY,
        **RECENT_WINDOW_OWNER,
        **PUBLISHED_RECENT_WINDOW_IDENTITY,
    )

    assert fetched == messages
    assert fetched is not messages
    fetched[0]["text"] = "changed"
    assert (
        await backend.get_recent_window_for_cache_identity(
            RECENT_WINDOW_KEY,
            **RECENT_WINDOW_OWNER,
            **PUBLISHED_RECENT_WINDOW_IDENTITY,
        )
    ) == messages


@pytest.mark.parametrize(
    "mismatch",
    [
        {"lifecycle_epoch": "epoch_2"},
        {"lifecycle_cleanup_key": "cleanup_2"},
        {"cache_revision": 1},
        {"derivation_revision": 1},
        {"conversation_lifecycle_epoch": "conversation_epoch_2"},
        {"conversation_source_revision": 1},
    ],
)
@pytest.mark.asyncio
async def test_recent_window_read_refuses_a_foreign_cache_identity(
    mismatch: dict[str, object],
) -> None:
    """The read enforces the identity contract the write does.

    A key-only read cannot tell a window that describes the reader's canonical
    state from one an older turn or an older lifecycle published, so every
    coordinate the writer fenced on has to fence the read too. A mismatch is an
    honest miss, which sends the reader to SQLite instead of serving a
    transcript that is not the one it asked for.
    """

    backend = InProcessBackend()
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"id": "msg_1", "text": "hello"}],
        **RECENT_WINDOW_OWNER,
        **PUBLISHED_RECENT_WINDOW_IDENTITY,
    )

    assert (
        await backend.get_recent_window_for_cache_identity(
            RECENT_WINDOW_KEY,
            **RECENT_WINDOW_OWNER,
            **{**PUBLISHED_RECENT_WINDOW_IDENTITY, **mismatch},
        )
        is None
    )
    # The refusal is a read fence, not a delete: the entry the identity's real
    # owner is entitled to survives the foreign read.
    assert await stored_recent_window(backend, RECENT_WINDOW_KEY) == [
        {"id": "msg_1", "text": "hello"}
    ]


@pytest.mark.asyncio
async def test_recent_window_read_refuses_another_conversations_key() -> None:
    """A key that does not derive from the requested pair is never served."""

    backend = InProcessBackend()
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"id": "msg_1", "text": "hello"}],
        **RECENT_WINDOW_OWNER,
        **PUBLISHED_RECENT_WINDOW_IDENTITY,
    )

    assert (
        await backend.get_recent_window_for_cache_identity(
            RECENT_WINDOW_KEY,
            user_id="usr_2",
            conversation_id="cnv_1",
            **PUBLISHED_RECENT_WINDOW_IDENTITY,
        )
        is None
    )
    assert (
        await backend.get_recent_window_for_cache_identity(
            RECENT_WINDOW_KEY,
            **RECENT_WINDOW_OWNER,
            **{**PUBLISHED_RECENT_WINDOW_IDENTITY, "cache_revision": -1},
        )
        is None
    )


@pytest.mark.asyncio
async def test_context_view_ttl_dedupe_and_locking() -> None:
    backend = InProcessBackend()

    await backend.set_context_view("ctx:1", {"items": ["one"]}, ttl_seconds=1)
    assert await backend.get_context_view("ctx:1") == {"items": ["one"]}
    await asyncio.sleep(1.05)
    assert await backend.get_context_view("ctx:1") is None

    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is True
    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is False
    await asyncio.sleep(1.05)
    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is True

    first_lock = await backend.acquire_lock("lock:1", ttl_seconds=1)
    assert first_lock is not None
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is None
    await backend.release_lock("lock:1", "wrong-token")
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is None
    await backend.release_lock("lock:1", first_lock)
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is not None


@pytest.mark.asyncio
async def test_lifecycle_locks_are_namespaced_purged_and_fenced() -> None:
    backend = InProcessBackend()
    cleanup_key = "cleanup_old"
    lifecycle_epoch = "epoch_old"
    coordinates = {
        "lifecycle_cleanup_key": cleanup_key,
        "lifecycle_epoch": lifecycle_epoch,
    }

    assert (
        await backend.acquire_lock(
            "job:lock",
            ttl_seconds=60,
            **coordinates,
        )
        is None
    )
    await backend.prepare_lifecycle_mirror(cleanup_key, lifecycle_epoch, "nonce_old")
    assert await backend.activate_lifecycle_mirror(
        cleanup_key,
        lifecycle_epoch,
        "nonce_old",
    )
    old_lock_token = await backend.acquire_lock(
        "job:lock",
        ttl_seconds=60,
        **coordinates,
    )
    assert old_lock_token is not None
    fenced_token = await backend.acquire_lock(
        "job:fenced-lock",
        ttl_seconds=60,
        job_id="job_fenced",
        execution_fence=4,
        **coordinates,
    )
    assert fenced_token is not None
    await backend.release_lock(
        "job:fenced-lock",
        fenced_token,
        job_id="job_fenced",
        execution_fence=4,
        **coordinates,
    )
    assert backend._lifecycle_lock_high_waters
    paused = asyncio.Event()
    resume = asyncio.Event()

    async def stale_resume() -> str | None:
        paused.set()
        await resume.wait()
        return await backend.acquire_lock(
            "stale:lock",
            ttl_seconds=60,
            **coordinates,
        )

    stale_task = asyncio.create_task(stale_resume())
    await paused.wait()
    await backend.revoke_lifecycle_and_purge_notifications(
        cleanup_key,
        lifecycle_epoch,
        group_name="workers",
    )
    assert backend._lifecycle_locks == {}
    assert backend._lifecycle_lock_high_waters == {}
    assert backend._lifecycle_transient_index == {}
    resume.set()
    assert await stale_task is None

    next_coordinates = {
        "lifecycle_cleanup_key": "cleanup_new",
        "lifecycle_epoch": "epoch_new",
    }
    await backend.prepare_lifecycle_mirror(
        "cleanup_new",
        "epoch_new",
        "nonce_new",
    )
    assert await backend.activate_lifecycle_mirror(
        "cleanup_new",
        "epoch_new",
        "nonce_new",
    )
    new_lock_token = await backend.acquire_lock(
        "job:lock",
        ttl_seconds=60,
        **next_coordinates,
    )
    assert new_lock_token is not None
    await backend.release_lock("job:lock", old_lock_token, **coordinates)
    assert (
        await backend.acquire_lock(
            "job:lock",
            ttl_seconds=60,
            **next_coordinates,
        )
        is None
    )


@pytest.mark.asyncio
async def test_job_fenced_lock_preserves_high_water_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(storage_backend_module, "monotonic", lambda: now[0])
    backend = InProcessBackend()
    coordinates = {
        "lifecycle_cleanup_key": "cleanup_fenced",
        "lifecycle_epoch": "epoch_fenced",
    }
    await backend.prepare_lifecycle_mirror(
        "cleanup_fenced",
        "epoch_fenced",
        "nonce_fenced",
    )
    assert await backend.activate_lifecycle_mirror(
        "cleanup_fenced",
        "epoch_fenced",
        "nonce_fenced",
    )

    fence_7 = await backend.acquire_lock(
        "projection",
        ttl_seconds=10,
        job_id="job_a",
        execution_fence=7,
        **coordinates,
    )
    assert fence_7 is not None
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_a",
            execution_fence=7,
            **coordinates,
        )
        is None
    )
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_b",
            execution_fence=99,
            **coordinates,
        )
        is None
    )
    fence_8 = await backend.acquire_lock(
        "projection",
        ttl_seconds=10,
        job_id="job_a",
        execution_fence=8,
        **coordinates,
    )
    assert fence_8 is not None
    await backend.release_lock(
        "projection",
        fence_7,
        job_id="job_a",
        execution_fence=7,
        **coordinates,
    )
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_a",
            execution_fence=8,
            **coordinates,
        )
        is None
    )
    await backend.release_lock(
        "projection",
        fence_8,
        job_id="job_a",
        execution_fence=8,
        **coordinates,
    )
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_a",
            execution_fence=8,
            **coordinates,
        )
        is None
    )
    lifecycle_only = await backend.acquire_lock(
        "projection",
        ttl_seconds=10,
        **coordinates,
    )
    assert lifecycle_only is not None
    await backend.release_lock("projection", lifecycle_only, **coordinates)
    job_b = await backend.acquire_lock(
        "projection",
        ttl_seconds=10,
        job_id="job_b",
        execution_fence=1,
        **coordinates,
    )
    assert job_b is not None
    await backend.release_lock(
        "projection",
        job_b,
        job_id="job_b",
        execution_fence=1,
        **coordinates,
    )
    fence_9 = await backend.acquire_lock(
        "projection",
        ttl_seconds=10,
        job_id="job_a",
        execution_fence=9,
        **coordinates,
    )
    assert fence_9 is not None
    await backend.release_lock("projection", fence_8, **coordinates)
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_a",
            execution_fence=9,
            **coordinates,
        )
        is None
    )

    now[0] += 11
    assert (
        await backend.acquire_lock(
            "projection",
            ttl_seconds=10,
            job_id="job_a",
            execution_fence=9,
            **coordinates,
        )
        is not None
    )

    with pytest.raises(ValueError):
        await backend.acquire_lock("invalid", 10, job_id="job_a")
    with pytest.raises(ValueError):
        await backend.acquire_lock("invalid", 10, job_id="job_a", **coordinates)
    with pytest.raises(ValueError):
        await backend.acquire_lock(
            "invalid",
            10,
            job_id="job_a",
            execution_fence=True,
            **coordinates,
        )


@pytest.mark.asyncio
async def test_legacy_transient_purge_preserves_current_inprocess_state() -> None:
    backend = InProcessBackend()
    backend._recent_windows["usr_target:cnv_legacy"] = [{"text": "target"}]
    backend._recent_windows["usr:target:cnv_collision"] = [
        {"text": "ambiguous legacy key"}
    ]
    backend._recent_windows["usr_other:cnv_legacy"] = [{"text": "other"}]
    await backend.set_context_view(
        "legacy-target",
        {"user_id": "usr_target", "value": "legacy"},
        ttl_seconds=60,
    )
    await backend.set_context_view(
        "legacy-other",
        {"user_id": "usr_other", "value": "other"},
        ttl_seconds=60,
    )
    await backend.prepare_lifecycle_mirror("cleanup_current", "epoch_current", "nonce")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_current",
        "epoch_current",
        "nonce",
    )
    current_recent_owner = {
        "user_id": "usr_target",
        "conversation_id": "cnv_current",
    }
    current_recent_key = build_recent_window_key(**current_recent_owner)
    assert await backend.set_recent_window_for_lifecycle(
        current_recent_key,
        [{"text": "current"}],
        **current_recent_owner,
        lifecycle_cleanup_key="cleanup_current",
        lifecycle_epoch="epoch_current",
        cache_revision=1,
        derivation_revision=1,
        conversation_lifecycle_epoch="cnv_epoch_current",
        conversation_source_revision=1,
    )
    assert await backend.set_context_view_if_newer_for_lifecycle(
        "current-target",
        {"user_id": "usr_target", "value": "current"},
        ttl_seconds=60,
        monotonic_seq=1,
        lifecycle_cleanup_key="cleanup_current",
        lifecycle_epoch="epoch_current",
    )
    await backend.set_context_view(
        "corrupt-owner-target",
        {"user_id": "usr_target", "value": "corrupt-owner"},
        ttl_seconds=60,
    )
    backend._lifecycle_cache_owner[("context", "corrupt-owner-target")] = (
        "cleanup_corrupt"
    )
    backend._lifecycle_mirrors["cleanup_corrupt"] = "active:epoch_corrupt"
    target_extractor_dedupe = f"usr_target:{'a' * 64}"
    other_extractor_dedupe = f"usr_other:{'b' * 64}"
    opaque_near_match = f"usr_target:{'c' * 63}"
    assert await backend.remember_dedupe(target_extractor_dedupe, 60)
    assert await backend.remember_dedupe(other_extractor_dedupe, 60)
    assert await backend.remember_dedupe(opaque_near_match, 60)
    assert await backend.remember_dedupe("old-worker-marker", 60)
    assert await backend.remember_dedupe("lifecycle:cooldown:0123456789ab", 60)
    assert await backend.acquire_lock("old-worker-lock", 60) is not None
    current_lock = await backend.acquire_lock("lifecycle:lock:0123456789ab", 60)
    assert current_lock is not None

    result = await backend.purge_legacy_transient_state("unused.db", "usr_target")

    assert result.recent_windows_deleted == 3
    assert result.context_views_deleted == 2
    assert result.legacy_dedupe_deleted == 1
    assert result.legacy_locks_deleted == 0
    assert result.malformed_candidates == 0
    assert result.total_deleted == 6
    assert result.clean
    assert await backend.get_context_view("legacy-target") is None
    assert backend._recent_windows.keys() == {current_recent_key}
    assert await stored_recent_window(backend, current_recent_key) == [{"text": "current"}]
    assert await backend.get_context_view("legacy-other") == {
        "user_id": "usr_other",
        "value": "other",
    }
    assert await backend.get_context_view("current-target") == {
        "user_id": "usr_target",
        "value": "current",
    }
    assert await backend.get_context_view("corrupt-owner-target") is None
    assert await backend.has_dedupe("lifecycle:cooldown:0123456789ab")
    assert not await backend.has_dedupe(target_extractor_dedupe)
    assert await backend.has_dedupe(other_extractor_dedupe)
    assert await backend.has_dedupe(opaque_near_match)
    assert await backend.has_dedupe("old-worker-marker")
    assert await backend.acquire_lock("old-worker-lock", ttl_seconds=60) is None
    assert (
        await backend.acquire_lock("lifecycle:lock:0123456789ab", ttl_seconds=60)
        is None
    )
    second = await backend.purge_legacy_transient_state("other.db", "usr_target")
    assert second.total_deleted == 0
    assert second.malformed_candidates == 0
    assert second.clean


@pytest.mark.asyncio
async def test_legacy_transient_purge_classifies_inprocess_queues_and_streams() -> None:
    backend = InProcessBackend()
    target = {
        "job_id": "target",
        "job_type": "run_evaluation",
        "payload": {},
        "user_id": "usr_target",
    }
    other = {
        "job_id": "other",
        "job_type": "run_evaluation",
        "payload": {},
        "user_id": "usr_other",
    }
    current = {
        "dispatch_token": "dispatch_current",
        "job_id": "job_current",
        "lifecycle_cleanup_key": "cleanup_current",
        "lifecycle_epoch": "epoch_current",
    }
    await backend.prepare_lifecycle_mirror(
        "cleanup_current",
        "epoch_current",
        "nonce_current",
    )
    assert await backend.activate_lifecycle_mirror(
        "cleanup_current",
        "epoch_current",
        "nonce_current",
    )

    admin_queue = "admin_rebuild_user"
    arbitrary_wrapper = {"payload": target}
    for item in ({"user_id": "usr_other"}, {"user_id": "usr_target"}, current):
        await backend.enqueue_job(admin_queue, item)
    await backend.enqueue_job(admin_queue, arbitrary_wrapper)
    conversation_queue = "admin_rebuild_conversation"
    await backend.enqueue_job(
        conversation_queue,
        {"conversation_id": "cnv_target", "user_id": "usr_target"},
    )
    await backend.enqueue_job(
        conversation_queue,
        {"conversation_id": "cnv_other", "user_id": "usr_other"},
    )
    dead_letter_queue = f"dead_letter:{EXTRACT_STREAM_NAME}"
    dead_target = {
        "delivery_count": 3,
        "error": "legacy failure",
        "message_id": "message_target",
        "payload": target,
    }
    dead_other = {
        "delivery_count": 4,
        "error": "legacy failure",
        "error_details": [],
        "message_id": "message_other",
        "payload": other,
    }
    await backend.enqueue_job(dead_letter_queue, dead_target)
    await backend.enqueue_job(dead_letter_queue, dead_other)
    current_diagnostic_id = await backend.publish_lifecycle_diagnostic(
        dead_letter_queue,
        {"user_id": "usr_target", "value": "current diagnostic"},
        lifecycle_cleanup_key="cleanup_current",
        lifecycle_epoch="epoch_current",
    )
    assert current_diagnostic_id is not None
    unproven_diagnostic_id = await backend.publish_lifecycle_diagnostic(
        dead_letter_queue,
        {"user_id": "usr_target", "value": "unproven diagnostic"},
        lifecycle_cleanup_key="cleanup_current",
        lifecycle_epoch="epoch_current",
    )
    assert unproven_diagnostic_id is not None
    backend._lifecycle_diagnostic_index["cleanup_current"].discard(
        (dead_letter_queue, unproven_diagnostic_id)
    )
    foreign_queue = "foreign"
    foreign_item = {"payload": target}
    await backend.enqueue_job(foreign_queue, foreign_item)

    queued_stream = EVALUATION_STREAM_NAME
    await backend.stream_add(queued_stream, target)
    await backend.stream_add(queued_stream, other)
    assert (
        await backend.publish_job_notification(
            queued_stream,
            current,
            lifecycle_cleanup_key="cleanup_current",
            lifecycle_epoch="epoch_current",
        )
        is not None
    )
    await backend.stream_add(queued_stream, current)
    await backend.stream_add(queued_stream, {"unattributed": "payload"})

    pending_stream = CONTRACT_STREAM_NAME

    async def add_and_read_pending(payload: dict[str, object]) -> str:
        await backend.stream_add(pending_stream, payload)
        messages = await backend.stream_read(
            pending_stream,
            "workers",
            "consumer",
            count=1,
            block_ms=0,
        )
        assert len(messages) == 1
        return messages[0].message_id

    pending_target_id = await add_and_read_pending(target)
    pending_other_id = await add_and_read_pending(other)
    current_pending_id = await backend.publish_job_notification(
        pending_stream,
        current,
        lifecycle_cleanup_key="cleanup_current",
        lifecycle_epoch="epoch_current",
    )
    assert current_pending_id is not None
    current_pending_read = await backend.stream_read(
        pending_stream,
        "workers",
        "consumer",
        count=1,
        block_ms=0,
    )
    assert [message.message_id for message in current_pending_read] == [
        current_pending_id
    ]
    unindexed_current_pending_id = await add_and_read_pending(current)

    result = await backend.purge_legacy_transient_state(
        "unused.db",
        "usr_target",
    )

    assert result.queue_entries_deleted == 3
    assert result.stream_entries_deleted == 2
    assert result.malformed_candidates == 6
    assert result.total_deleted == 5
    assert not result.clean
    assert list(backend._queues[admin_queue]._queue) == [
        {"user_id": "usr_other"},
        current,
        arbitrary_wrapper,
    ]
    assert list(backend._queues[conversation_queue]._queue) == [
        {"conversation_id": "cnv_other", "user_id": "usr_other"}
    ]
    dead_items = list(backend._queues[dead_letter_queue]._queue)
    assert dead_items[0] == dead_other
    assert len(dead_items) == 3
    assert list(backend._queues[foreign_queue]._queue) == [foreign_item]
    queued_payloads = [
        item["payload"] for item in backend._queues[f"stream:{queued_stream}"]._queue
    ]
    assert queued_payloads == [
        other,
        current,
        current,
        {"unattributed": "payload"},
    ]
    pending = backend._stream_pending[(pending_stream, "workers")]
    assert pending_target_id not in pending
    assert set(pending) == {
        pending_other_id,
        current_pending_id,
        unindexed_current_pending_id,
    }
    assert backend._pending_job_count == 3

    second = await backend.purge_legacy_transient_state(
        "other.db",
        "usr_target",
    )
    assert second.total_deleted == 0
    assert second.malformed_candidates == 6
    assert not second.clean


@pytest.mark.asyncio
async def test_delete_context_view_removes_single_entry() -> None:
    backend = InProcessBackend()

    await backend.set_context_view(
        "ctx:1",
        {"user_id": "usr_1", "items": ["one"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:2",
        {"user_id": "usr_1", "items": ["two"]},
        ttl_seconds=10,
    )

    await backend.delete_context_view("ctx:1")

    assert await backend.get_context_view("ctx:1") is None
    assert await backend.get_context_view("ctx:2") == {
        "user_id": "usr_1",
        "items": ["two"],
    }


@pytest.mark.asyncio
async def test_delete_context_views_for_user_wipes_only_indexed_entries() -> None:
    backend = InProcessBackend()

    await backend.set_context_view(
        "ctx:1",
        {"user_id": "usr_1", "items": ["one"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:2",
        {"user_id": "usr_1", "items": ["two"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:3",
        {"user_id": "usr_2", "items": ["three"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:legacy",
        {"items": ["legacy"]},
        ttl_seconds=10,
    )

    deleted = await backend.delete_context_views_for_user("usr_1")

    assert deleted == 2
    assert await backend.get_context_view("ctx:1") is None
    assert await backend.get_context_view("ctx:2") is None
    assert await backend.get_context_view("ctx:3") == {
        "user_id": "usr_2",
        "items": ["three"],
    }
    assert await backend.get_context_view("ctx:legacy") == {"items": ["legacy"]}


@pytest.mark.asyncio
async def test_set_context_view_if_newer_rejects_older_publish() -> None:
    backend = InProcessBackend()

    first = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "older"},
        ttl_seconds=10,
        monotonic_seq=3,
    )
    second = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "stale"},
        ttl_seconds=10,
        monotonic_seq=2,
    )
    third = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "newer"},
        ttl_seconds=10,
        monotonic_seq=4,
    )

    assert first is True
    assert second is False
    assert third is True
    assert await backend.get_context_view("ctx:1") == {
        "user_id": "usr_1",
        "value": "newer",
    }


@pytest.mark.asyncio
async def test_job_queue_round_trip_and_timeout() -> None:
    backend = InProcessBackend()

    assert await backend.dequeue_job("ingest", timeout_seconds=0) is None
    await backend.enqueue_job("ingest", {"job_id": "job_1"})
    await backend.enqueue_job("ingest", {"job_id": "job_2"})

    assert await backend.dequeue_job("ingest", timeout_seconds=0) == {"job_id": "job_1"}
    assert await backend.dequeue_job("ingest", timeout_seconds=0.1) == {
        "job_id": "job_2"
    }
    assert await backend.dequeue_job("ingest", timeout_seconds=0.05) is None


@pytest.mark.asyncio
async def test_stream_pending_messages_can_be_reclaimed_before_ack() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add("atagia:test", {"job_id": "job_1"})

    first_read = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    second_read = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    reclaimed = await backend.stream_claim_idle(
        "atagia:test",
        "workers",
        "consumer-2",
        min_idle_ms=0,
        count=1,
    )

    assert [message.message_id for message in first_read] == [message_id]
    assert second_read == []
    assert [message.message_id for message in reclaimed] == [message_id]
    assert reclaimed[0].delivery_count == 2

    await backend.stream_ack("atagia:test", "workers", message_id)
    assert (
        await backend.stream_claim_idle(
            "atagia:test",
            "workers",
            "consumer-2",
            min_idle_ms=0,
            count=1,
        )
        == []
    )


@pytest.mark.asyncio
async def test_lifecycle_gated_notification_publish_fails_closed_and_revokes_atomically() -> (
    None
):
    backend = InProcessBackend()
    notification = {
        "job_id": "job_1",
        "dispatch_token": "dispatch_1",
        "lifecycle_epoch": "epoch_1",
        "lifecycle_cleanup_key": "cleanup_1",
    }

    assert (
        await backend.publish_job_notification(
            "atagia:test",
            notification,
            lifecycle_cleanup_key="cleanup_1",
            lifecycle_epoch="epoch_1",
        )
        is None
    )

    state = await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert state == "preparing:epoch_1:nonce_1"
    assert (
        await backend.publish_job_notification(
            "atagia:test",
            notification,
            lifecycle_cleanup_key="cleanup_1",
            lifecycle_epoch="epoch_1",
        )
        is None
    )
    assert (
        await backend.activate_lifecycle_mirror(
            "cleanup_1",
            "epoch_1",
            "nonce_1",
        )
        is True
    )

    message_id = await backend.publish_job_notification(
        "atagia:test",
        notification,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
    )
    assert message_id is not None
    assert (
        await backend.revoke_lifecycle_and_purge_notifications(
            "cleanup_1",
            "epoch_1",
            group_name="workers",
        )
        == 1
    )
    assert (
        await backend.stream_read(
            "atagia:test",
            "workers",
            "consumer-1",
            count=1,
            block_ms=0,
        )
        == []
    )
    assert (
        await backend.publish_job_notification(
            "atagia:test",
            notification,
            lifecycle_cleanup_key="cleanup_1",
            lifecycle_epoch="epoch_1",
        )
        is None
    )


@pytest.mark.asyncio
async def test_lifecycle_diagnostic_publish_linearizes_with_revoke_and_purge() -> None:
    backend = InProcessBackend()
    cleanup_key = "cleanup_dead_letter"
    lifecycle_epoch = "epoch_dead_letter"
    queue_name = "dead_letter:atagia:test"
    diagnostic = {
        "job_id": "job_dead_letter",
        "user_id": "usr_dead_letter",
        "lifecycle_epoch": lifecycle_epoch,
        "error_class": "InjectedError",
    }
    await backend.prepare_lifecycle_mirror(cleanup_key, lifecycle_epoch, "nonce")
    assert await backend.activate_lifecycle_mirror(
        cleanup_key,
        lifecycle_epoch,
        "nonce",
    )
    assert (
        await backend.publish_lifecycle_diagnostic(
            queue_name,
            diagnostic,
            lifecycle_cleanup_key=cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
        )
        is not None
    )

    paused = asyncio.Event()
    resume = asyncio.Event()

    async def paused_publish() -> str | None:
        paused.set()
        await resume.wait()
        return await backend.publish_lifecycle_diagnostic(
            queue_name,
            diagnostic,
            lifecycle_cleanup_key=cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
        )

    publish_task = asyncio.create_task(paused_publish())
    await paused.wait()
    assert (
        await backend.revoke_lifecycle_and_purge_notifications(
            cleanup_key,
            lifecycle_epoch,
            group_name="workers",
        )
        == 1
    )
    resume.set()
    assert await publish_task is None
    assert await backend.dequeue_job(queue_name, timeout_seconds=0) is None


@pytest.mark.asyncio
async def test_lifecycle_gated_cache_writes_are_purged_and_cannot_resume() -> None:
    backend = InProcessBackend()
    context = {
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "value": "old lifecycle data",
    }

    assert not await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "before mirror"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=0,
        derivation_revision=0,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=0,
    )
    assert not await backend.set_context_view_if_newer_for_lifecycle(
        "ctx:old",
        context,
        ttl_seconds=60,
        monotonic_seq=1,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
    )

    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_1",
        "epoch_1",
        "nonce_1",
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old lifecycle data"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=0,
        derivation_revision=0,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=1,
    )
    assert await backend.set_context_view_if_newer_for_lifecycle(
        "ctx:old",
        context,
        ttl_seconds=60,
        monotonic_seq=1,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
    )

    await backend.revoke_lifecycle_and_purge_notifications(
        "cleanup_1",
        "epoch_1",
        group_name="workers",
    )
    assert await stored_recent_window(backend, RECENT_WINDOW_KEY) is None
    assert await backend.get_context_view("ctx:old") is None
    assert not await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "resumed stale writer"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=0,
        derivation_revision=0,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=1,
    )
    assert not await backend.set_context_view_if_newer_for_lifecycle(
        "ctx:old",
        context,
        ttl_seconds=60,
        monotonic_seq=2,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
    )


@pytest.mark.asyncio
async def test_recent_window_conditional_cleanup_includes_derivation_revision() -> None:
    backend = InProcessBackend()
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_1",
        "epoch_1",
        "nonce_1",
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old selected transcript"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=4,
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "new selected transcript"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=8,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=5,
    )

    assert not await backend.delete_recent_window_if_cache_identity(
        RECENT_WINDOW_KEY,
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=4,
    )
    assert await stored_recent_window(backend, RECENT_WINDOW_KEY) == [
        {"text": "new selected transcript"}
    ]
    assert await backend.delete_recent_window_if_cache_identity(
        RECENT_WINDOW_KEY,
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=8,
        conversation_lifecycle_epoch="conversation_epoch_1",
        conversation_source_revision=5,
    )


@pytest.mark.asyncio
async def test_recent_window_source_revision_blocks_old_overwrite_and_cleanup() -> None:
    backend = InProcessBackend()
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_1",
        "epoch_1",
        "nonce_1",
    )
    common_identity = {
        **RECENT_WINDOW_OWNER,
        "lifecycle_cleanup_key": "cleanup_1",
        "lifecycle_epoch": "epoch_1",
        "cache_revision": 3,
        "derivation_revision": 7,
        "conversation_lifecycle_epoch": "conversation_epoch_1",
    }
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old conversation source"}],
        **common_identity,
        conversation_source_revision=4,
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "new conversation source"}],
        **common_identity,
        conversation_source_revision=5,
    )

    assert not await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old writer resumed"}],
        **common_identity,
        conversation_source_revision=4,
    )
    assert not await backend.delete_recent_window_if_cache_identity(
        RECENT_WINDOW_KEY,
        **common_identity,
        conversation_source_revision=4,
    )
    assert await stored_recent_window(backend, RECENT_WINDOW_KEY) == [
        {"text": "new conversation source"}
    ]


@pytest.mark.asyncio
async def test_recent_window_new_conversation_epoch_takes_monotonic_ownership() -> None:
    backend = InProcessBackend()
    await backend.prepare_lifecycle_mirror("cleanup_1", "epoch_1", "nonce_1")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_1",
        "epoch_1",
        "nonce_1",
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old conversation epoch"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_old",
        conversation_source_revision=9,
    )
    assert await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "new conversation epoch"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=4,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_new",
        conversation_source_revision=0,
    )

    assert not await backend.set_recent_window_for_lifecycle(
        RECENT_WINDOW_KEY,
        [{"text": "old epoch resumed"}],
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_old",
        conversation_source_revision=10,
    )
    assert not await backend.delete_recent_window_if_cache_identity(
        RECENT_WINDOW_KEY,
        **RECENT_WINDOW_OWNER,
        lifecycle_cleanup_key="cleanup_1",
        lifecycle_epoch="epoch_1",
        cache_revision=3,
        derivation_revision=7,
        conversation_lifecycle_epoch="conversation_epoch_old",
        conversation_source_revision=9,
    )
    assert await stored_recent_window(backend, RECENT_WINDOW_KEY) == [
        {"text": "new conversation epoch"}
    ]


@pytest.mark.asyncio
async def test_recent_window_keys_and_user_cleanup_are_exact_for_colon_ids() -> None:
    backend = InProcessBackend()
    first_owner = {"user_id": "account:blue", "conversation_id": "thread"}
    second_owner = {"user_id": "account", "conversation_id": "blue:thread"}
    first_key = build_recent_window_key(**first_owner)
    second_key = build_recent_window_key(**second_owner)
    assert first_key != second_key

    for suffix in ("first", "second"):
        await backend.prepare_lifecycle_mirror(
            f"cleanup_{suffix}",
            f"epoch_{suffix}",
            f"nonce_{suffix}",
        )
        assert await backend.activate_lifecycle_mirror(
            f"cleanup_{suffix}",
            f"epoch_{suffix}",
            f"nonce_{suffix}",
        )

    assert await backend.set_recent_window_for_lifecycle(
        first_key,
        [{"text": "first"}],
        **first_owner,
        lifecycle_cleanup_key="cleanup_first",
        lifecycle_epoch="epoch_first",
        cache_revision=0,
        derivation_revision=0,
        conversation_lifecycle_epoch="conversation_epoch_first",
        conversation_source_revision=0,
    )
    assert await backend.set_recent_window_for_lifecycle(
        second_key,
        [{"text": "second"}],
        **second_owner,
        lifecycle_cleanup_key="cleanup_second",
        lifecycle_epoch="epoch_second",
        cache_revision=0,
        derivation_revision=0,
        conversation_lifecycle_epoch="conversation_epoch_second",
        conversation_source_revision=0,
    )

    assert await backend.delete_recent_windows_for_user("account") == 1
    assert await stored_recent_window(backend, second_key) is None
    assert await stored_recent_window(backend, first_key) == [{"text": "first"}]
    assert (
        await backend.delete_recent_window_for_conversation(
            first_owner["user_id"],
            first_owner["conversation_id"],
        )
        == 1
    )
    assert await stored_recent_window(backend, first_key) is None


@pytest.mark.asyncio
async def test_recent_window_key_identity_mismatch_is_non_mutating() -> None:
    backend = InProcessBackend()
    owner = {"user_id": "owner:one", "conversation_id": "conversation:one"}
    key = build_recent_window_key(**owner)
    await backend.prepare_lifecycle_mirror("cleanup_owner", "epoch_owner", "nonce")
    assert await backend.activate_lifecycle_mirror(
        "cleanup_owner",
        "epoch_owner",
        "nonce",
    )
    identity = {
        "lifecycle_cleanup_key": "cleanup_owner",
        "lifecycle_epoch": "epoch_owner",
        "cache_revision": 0,
        "derivation_revision": 0,
        "conversation_lifecycle_epoch": "conversation_epoch_owner",
        "conversation_source_revision": 0,
    }
    assert not await backend.set_recent_window_for_lifecycle(
        key,
        [{"text": "wrong owner"}],
        user_id="owner",
        conversation_id="one:conversation:one",
        **identity,
    )
    assert await stored_recent_window(backend, key) is None

    assert await backend.set_recent_window_for_lifecycle(
        key,
        [{"text": "owned"}],
        **owner,
        **identity,
    )
    assert not await backend.delete_recent_window_if_cache_identity(
        key,
        user_id="owner",
        conversation_id="one:conversation:one",
        **identity,
    )
    assert await stored_recent_window(backend, key) == [{"text": "owned"}]


@pytest.mark.asyncio
async def test_stream_drain_waits_for_pending_ack() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add("atagia:test", {"job_id": "job_1"})
    messages = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )

    assert [message.message_id for message in messages] == [message_id]

    async def _ack_later() -> None:
        await asyncio.sleep(0.05)
        await backend.stream_ack("atagia:test", "workers", message_id)

    ack_task = asyncio.create_task(_ack_later())
    try:
        assert await backend.drain(timeout_seconds=0.5) is True
    finally:
        await ack_task


@pytest.mark.asyncio
async def test_stream_drain_snapshot_reports_queue_pending_and_ack_progress() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add(
        "atagia:test",
        {
            "job_id": "job_1",
            "job_type": "extract_memory_candidates",
            "conversation_id": "conv_1",
            "message_ids": ["msg_1"],
            "payload": {"message_id": "msg_1"},
        },
    )

    queued_snapshot = await backend.drain_snapshot()
    assert queued_snapshot.total_queued == 1
    assert queued_snapshot.queued_by_stream == {"atagia:test": 1}
    assert queued_snapshot.total_pending == 0

    await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    pending_snapshot = await backend.drain_snapshot()
    assert pending_snapshot.total_queued == 0
    assert pending_snapshot.pending_by_stream == {"atagia:test": 1}
    assert pending_snapshot.pending_job_types == {"extract_memory_candidates": 1}
    assert pending_snapshot.active_jobs[0]["job_id"] == "job_1"
    assert pending_snapshot.active_jobs[0]["payload_message_id"] == "msg_1"

    await backend.stream_ack("atagia:test", "workers", message_id)
    drained_snapshot = await backend.drain_snapshot()
    assert drained_snapshot.drained is True
    assert drained_snapshot.acked_by_stream == {"atagia:test": 1}


@pytest.mark.asyncio
async def test_stream_drain_idle_timeout_resets_when_progress_callback_reports_progress() -> (
    None
):
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add("atagia:test", {"job_id": "job_1"})
    await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    callback_calls = 0

    async def report_progress_once(_snapshot) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            await backend.stream_ack("atagia:test", "workers", message_id)
            return True
        return False

    assert (
        await backend.drain(
            timeout_seconds=0.5,
            idle_timeout_seconds=0.05,
            progress_interval_seconds=0.01,
            progress_callback=report_progress_once,
        )
        is True
    )
    assert callback_calls >= 1


@pytest.mark.asyncio
async def test_stream_drain_times_out_when_pending_work_remains() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    await backend.stream_add("atagia:test", {"job_id": "job_1"})
    await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )

    assert await backend.drain(timeout_seconds=0.05) is False
