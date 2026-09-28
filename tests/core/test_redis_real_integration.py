"""Real-Redis gates for lifecycle fencing and atomic list cleanup."""

from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path
import traceback
from typing import Any

import pytest

from atagia.core import json_utils
from atagia.core.clock import SystemClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.redis_client import ATAGIA_QUEUE_PREFIX, RedisBackend
from atagia.core.repositories import UserRepository
from atagia.core.storage_backend import (
    InProcessBackend,
    StorageBackend,
    build_recent_window_key,
)
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    DurableJobNotification,
    JobEnvelope,
    JobType,
)
from atagia.services.lifecycle_mirror_reconciler import (
    reconcile_active_lifecycle_mirror,
)
from tests.redis_real_support import RealRedisServer, running_redis_server
from tests.recent_window_support import stored_recent_window


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
REAL_RECENT_OWNER = {"user_id": "usr_1", "conversation_id": "conv_1"}
REAL_RECENT_KEY = build_recent_window_key(**REAL_RECENT_OWNER)

INVALID_RECENT_WINDOW_IDENTITIES: tuple[
    tuple[str, dict[str, object]],
    ...,
] = (
    ("empty_user_id", {"user_id": ""}),
    ("missing_user_id", {"user_id": None}),
    ("empty_conversation_id", {"conversation_id": ""}),
    ("missing_conversation_id", {"conversation_id": None}),
    ("empty_cleanup_key", {"lifecycle_cleanup_key": ""}),
    ("missing_cleanup_key", {"lifecycle_cleanup_key": None}),
    ("empty_lifecycle_epoch", {"lifecycle_epoch": ""}),
    ("missing_lifecycle_epoch", {"lifecycle_epoch": None}),
    ("empty_conversation_epoch", {"conversation_lifecycle_epoch": ""}),
    ("missing_conversation_epoch", {"conversation_lifecycle_epoch": None}),
    ("negative_cache_revision", {"cache_revision": -1}),
    ("fractional_cache_revision", {"cache_revision": 2.5}),
    ("boolean_cache_revision", {"cache_revision": True}),
    ("negative_derivation_revision", {"derivation_revision": -1}),
    ("fractional_derivation_revision", {"derivation_revision": 2.5}),
    ("boolean_derivation_revision", {"derivation_revision": True}),
    ("negative_conversation_revision", {"conversation_source_revision": -1}),
    ("fractional_conversation_revision", {"conversation_source_revision": 3.5}),
    ("boolean_conversation_revision", {"conversation_source_revision": True}),
)


@pytest.fixture
def real_redis_server(tmp_path: Path):
    with running_redis_server(tmp_path) as server:
        yield server


def _run_process_operation(
    operation: str,
    redis_url: str,
    queue_name: str,
    barrier: multiprocessing.synchronize.Barrier,
    result_queue: multiprocessing.queues.Queue,
    payload: dict[str, Any] | None = None,
) -> None:
    """Execute one queue operation in an independent interpreter process."""

    async def run() -> int | dict[str, Any] | None:
        backend = RedisBackend(redis_url)
        try:
            await asyncio.to_thread(barrier.wait, 5)
            if operation == "purge":
                return await backend.purge_user_jobs("usr_target")
            if operation == "enqueue":
                if payload is None:
                    raise AssertionError("enqueue requires a payload")
                await backend.enqueue_job(queue_name, payload)
                return payload
            if operation == "dequeue":
                return await backend.dequeue_job(queue_name, timeout_seconds=0)
            raise AssertionError(f"Unsupported process operation: {operation}")
        finally:
            await backend.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _start_operation_pair(
    *,
    redis_url: str,
    queue_name: str,
    first_operation: str,
    second_operation: str,
    second_payload: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Release two child processes at the same explicit barrier."""

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    first_results = context.Queue()
    second_results = context.Queue()
    first = context.Process(
        target=_run_process_operation,
        args=(
            first_operation,
            redis_url,
            queue_name,
            barrier,
            first_results,
        ),
    )
    second = context.Process(
        target=_run_process_operation,
        args=(
            second_operation,
            redis_url,
            queue_name,
            barrier,
            second_results,
            second_payload,
        ),
    )
    first.start()
    second.start()
    barrier.wait(timeout=5)
    first.join(timeout=10)
    second.join(timeout=10)
    if first.is_alive():
        first.kill()
        first.join(timeout=5)
    if second.is_alive():
        second.kill()
        second.join(timeout=5)
    assert first.exitcode == 0
    assert second.exitcode == 0
    first_status, first_value = first_results.get(timeout=2)
    second_status, second_value = second_results.get(timeout=2)
    assert first_status == "ok", first_value
    assert second_status == "ok", second_value
    return first_value, second_value


@pytest.mark.asyncio
async def test_real_redis_lifecycle_locks_reject_stale_resume(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    revoker = RedisBackend(real_redis_server.url)
    cleanup_key = "cleanup_transient_old"
    lifecycle_epoch = "epoch_transient_old"
    coordinates = {
        "lifecycle_cleanup_key": cleanup_key,
        "lifecycle_epoch": lifecycle_epoch,
    }
    try:
        await backend.prepare_lifecycle_mirror(
            cleanup_key,
            lifecycle_epoch,
            "nonce_old",
        )
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
        assert list(real_redis_server.client.scan_iter("lifecycle_lock_fence:*"))

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
        await revoker.revoke_lifecycle_and_purge_notifications(
            cleanup_key,
            lifecycle_epoch,
            group_name="atagia-workers",
        )
        assert list(real_redis_server.client.scan_iter("lifecycle_lock:*")) == []
        assert list(real_redis_server.client.scan_iter("lifecycle_lock_fence:*")) == []
        assert (
            list(real_redis_server.client.scan_iter("lifecycle_lock_entries:*")) == []
        )
        resume.set()
        assert await stale_task is None

        next_coordinates = {
            "lifecycle_cleanup_key": "cleanup_transient_new",
            "lifecycle_epoch": "epoch_transient_new",
        }
        await backend.prepare_lifecycle_mirror(
            "cleanup_transient_new",
            "epoch_transient_new",
            "nonce_new",
        )
        assert await backend.activate_lifecycle_mirror(
            "cleanup_transient_new",
            "epoch_transient_new",
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
    finally:
        await backend.close()
        await revoker.close()


@pytest.mark.asyncio
async def test_real_redis_job_fenced_lock_keeps_high_water_after_release(
    real_redis_server: RealRedisServer,
) -> None:
    first = RedisBackend(real_redis_server.url)
    second = RedisBackend(real_redis_server.url)
    coordinates = {
        "lifecycle_cleanup_key": "cleanup_job_fence",
        "lifecycle_epoch": "epoch_job_fence",
    }
    try:
        await first.prepare_lifecycle_mirror(
            "cleanup_job_fence",
            "epoch_job_fence",
            "nonce_job_fence",
        )
        assert await first.activate_lifecycle_mirror(
            "cleanup_job_fence",
            "epoch_job_fence",
            "nonce_job_fence",
        )
        malformed_high_water_key = first._lifecycle_lock_high_water_key(
            "malformed-fence",
            "cleanup_job_fence",
            "epoch_job_fence",
            "job_malformed",
        )
        real_redis_server.client.set(malformed_high_water_key, "not-an-integer")
        assert (
            await second.acquire_lock(
                "malformed-fence",
                ttl_seconds=60,
                job_id="job_malformed",
                execution_fence=1,
                **coordinates,
            )
            is None
        )
        fence_7 = await first.acquire_lock(
            "projection",
            ttl_seconds=60,
            job_id="job_a",
            execution_fence=7,
            **coordinates,
        )
        assert fence_7 is not None
        assert (
            await second.acquire_lock(
                "projection",
                ttl_seconds=60,
                job_id="job_a",
                execution_fence=7,
                **coordinates,
            )
            is None
        )
        assert (
            await second.acquire_lock(
                "projection",
                ttl_seconds=60,
                job_id="job_b",
                execution_fence=99,
                **coordinates,
            )
            is None
        )
        fence_8 = await second.acquire_lock(
            "projection",
            ttl_seconds=60,
            job_id="job_a",
            execution_fence=8,
            **coordinates,
        )
        assert fence_8 is not None

        await first.release_lock(
            "projection",
            fence_7,
            job_id="job_a",
            execution_fence=7,
            **coordinates,
        )
        assert (
            await first.acquire_lock(
                "projection",
                ttl_seconds=60,
                job_id="job_a",
                execution_fence=8,
                **coordinates,
            )
            is None
        )
        await second.release_lock(
            "projection",
            fence_8,
            job_id="job_a",
            execution_fence=8,
            **coordinates,
        )
        assert (
            await first.acquire_lock(
                "projection",
                ttl_seconds=60,
                job_id="job_a",
                execution_fence=8,
                **coordinates,
            )
            is None
        )
        job_b = await first.acquire_lock(
            "projection",
            ttl_seconds=60,
            job_id="job_b",
            execution_fence=100,
            **coordinates,
        )
        assert job_b is not None
        await first.release_lock(
            "projection",
            job_b,
            job_id="job_b",
            execution_fence=100,
            **coordinates,
        )
        lifecycle_only_after_job = await second.acquire_lock(
            "projection",
            ttl_seconds=60,
            **coordinates,
        )
        assert lifecycle_only_after_job is not None
        await second.release_lock(
            "projection",
            lifecycle_only_after_job,
            **coordinates,
        )
        fence_9 = await first.acquire_lock(
            "projection",
            ttl_seconds=60,
            job_id="job_a",
            execution_fence=9,
            **coordinates,
        )
        assert fence_9 is not None
        await second.release_lock("projection", fence_8, **coordinates)
        assert (
            await second.acquire_lock(
                "projection",
                ttl_seconds=60,
                job_id="job_a",
                execution_fence=9,
                **coordinates,
            )
            is None
        )

        lifecycle_only = await first.acquire_lock(
            "cache-guard",
            ttl_seconds=60,
            **coordinates,
        )
        assert lifecycle_only is not None
        assert (
            await second.acquire_lock(
                "cache-guard",
                ttl_seconds=60,
                job_id="job_a",
                execution_fence=100,
                **coordinates,
            )
            is None
        )
        await first.release_lock("cache-guard", lifecycle_only, **coordinates)
        assert (
            await second.acquire_lock(
                "cache-guard",
                ttl_seconds=60,
                **coordinates,
            )
            is not None
        )
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_real_redis_lifecycle_mirror_reset_and_revocation_are_fail_closed(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "lifecycle-mirror.db")
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = SystemClock()
    await UserRepository(connection, clock).create_user("usr_1")
    identity = await UserLifecycleRepository(connection, clock).get_active_identity(
        "usr_1"
    )
    assert identity is not None
    backend = RedisBackend(real_redis_server.url)
    stream_name = "atagia:real-lifecycle"
    try:
        notification = {
            "job_id": "job_1",
            "dispatch_token": "dispatch_1",
            "lifecycle_epoch": identity.lifecycle_epoch,
            "lifecycle_cleanup_key": identity.lifecycle_cleanup_key,
        }
        context_view = {
            "user_id": "usr_1",
            "conversation_id": "conv_1",
            "items": ["current"],
        }

        # Missing mirrors never authorize publications or cache writes.
        assert (
            await backend.publish_job_notification(
                stream_name,
                notification,
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                lifecycle_epoch=identity.lifecycle_epoch,
            )
            is None
        )
        assert not await backend.set_recent_window_for_lifecycle(
            REAL_RECENT_KEY,
            [{"role": "user", "text": "current"}],
            **REAL_RECENT_OWNER,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
            cache_revision=identity.cache_revision,
            derivation_revision=identity.derivation_revision,
            conversation_lifecycle_epoch="conversation_epoch_1",
            conversation_source_revision=1,
        )
        assert not await backend.set_context_view_if_newer_for_lifecycle(
            "ctx_1",
            context_view,
            ttl_seconds=60,
            monotonic_seq=1,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
        )

        # Revocation linearized after prepare prevents that exact nonce activating.
        nonce = "prepare_barrier_nonce"
        assert (
            await backend.prepare_lifecycle_mirror(
                identity.lifecycle_cleanup_key,
                identity.lifecycle_epoch,
                nonce,
            )
            == f"preparing:{identity.lifecycle_epoch}:{nonce}"
        )
        assert (
            await backend.publish_job_notification(
                stream_name,
                notification,
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                lifecycle_epoch=identity.lifecycle_epoch,
            )
            is None
        )
        await backend.revoke_lifecycle_and_purge_notifications(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            group_name="atagia-workers",
        )
        assert not await backend.activate_lifecycle_mirror(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            nonce,
        )
        assert not await reconcile_active_lifecycle_mirror(
            connection,
            backend,
            user_id="usr_1",
            lifecycle_epoch=identity.lifecycle_epoch,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        )

        # A Redis reset pauses publication until the dedicated SQLite recheck.
        real_redis_server.client.flushall()
        assert (
            await backend.publish_job_notification(
                stream_name,
                notification,
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                lifecycle_epoch=identity.lifecycle_epoch,
            )
            is None
        )
        assert await reconcile_active_lifecycle_mirror(
            connection,
            backend,
            user_id="usr_1",
            lifecycle_epoch=identity.lifecycle_epoch,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        )
        delivery_id = await backend.publish_job_notification(
            stream_name,
            notification,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
        )
        assert delivery_id is not None
        assert await backend.set_recent_window_for_lifecycle(
            REAL_RECENT_KEY,
            [{"role": "user", "text": "current"}],
            **REAL_RECENT_OWNER,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
            cache_revision=identity.cache_revision,
            derivation_revision=identity.derivation_revision,
            conversation_lifecycle_epoch="conversation_epoch_1",
            conversation_source_revision=1,
        )
        assert await backend.set_recent_window_for_lifecycle(
            REAL_RECENT_KEY,
            [{"role": "user", "text": "new conversation source"}],
            **REAL_RECENT_OWNER,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
            cache_revision=identity.cache_revision,
            derivation_revision=identity.derivation_revision,
            conversation_lifecycle_epoch="conversation_epoch_1",
            conversation_source_revision=2,
        )
        assert not await backend.set_recent_window_for_lifecycle(
            REAL_RECENT_KEY,
            [{"role": "user", "text": "old writer resumed"}],
            **REAL_RECENT_OWNER,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
            cache_revision=identity.cache_revision,
            derivation_revision=identity.derivation_revision,
            conversation_lifecycle_epoch="conversation_epoch_1",
            conversation_source_revision=1,
        )
        assert not await backend.delete_recent_window_if_cache_identity(
            REAL_RECENT_KEY,
            **REAL_RECENT_OWNER,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
            cache_revision=identity.cache_revision,
            derivation_revision=identity.derivation_revision,
            conversation_lifecycle_epoch="conversation_epoch_1",
            conversation_source_revision=1,
        )
        assert await stored_recent_window(backend, REAL_RECENT_KEY) == [
            {"role": "user", "text": "new conversation source"}
        ]
        assert await backend.set_context_view_if_newer_for_lifecycle(
            "ctx_1",
            context_view,
            ttl_seconds=60,
            monotonic_seq=1,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
            lifecycle_epoch=identity.lifecycle_epoch,
        )
        context_index_ttl = real_redis_server.client.ttl(
            f"lifecycle_context_entries:{identity.lifecycle_cleanup_key}"
        )
        assert 0 < context_index_ttl <= 60
        assert (
            real_redis_server.client.ttl(
                f"lifecycle_recent_entries:{identity.lifecycle_cleanup_key}"
            )
            == -1
        )

        purged = await backend.revoke_lifecycle_and_purge_notifications(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            group_name="atagia-workers",
        )
        assert purged == 1
        assert real_redis_server.client.xlen(stream_name) == 0
        assert await stored_recent_window(backend, REAL_RECENT_KEY) is None
        assert await backend.get_context_view("ctx_1") is None
        assert (
            await backend.publish_job_notification(
                stream_name,
                notification,
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                lifecycle_epoch=identity.lifecycle_epoch,
            )
            is None
        )

        # Even an inconsistent active row may not reconcile while cleanup is linked.
        real_redis_server.client.flushall()
        await connection.execute(
            """
            UPDATE user_lifecycles
            SET erasure_cleanup_id = 'cleanup_pending_barrier'
            WHERE user_id = ? AND lifecycle_epoch = ?
            """,
            ("usr_1", identity.lifecycle_epoch),
        )
        await connection.commit()
        assert not await reconcile_active_lifecycle_mirror(
            connection,
            backend,
            user_id="usr_1",
            lifecycle_epoch=identity.lifecycle_epoch,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_real_redis_context_owner_prevents_stale_revoke_and_legacy_purge(
    real_redis_server: RealRedisServer,
) -> None:
    first = RedisBackend(real_redis_server.url)
    second = RedisBackend(real_redis_server.url)
    client = real_redis_server.client
    context_key = "context_takeover"
    user_id = "usr_context_takeover"
    old_cleanup = "cleanup_context_old"
    old_epoch = "epoch_context_old"
    new_cleanup = "cleanup_context_new"
    new_epoch = "epoch_context_new"
    member = f"c\x1f{context_key}"
    try:
        for cleanup_key, lifecycle_epoch, nonce in (
            (old_cleanup, old_epoch, "nonce_old"),
            (new_cleanup, new_epoch, "nonce_new"),
        ):
            await first.prepare_lifecycle_mirror(
                cleanup_key,
                lifecycle_epoch,
                nonce,
            )
            assert await first.activate_lifecycle_mirror(
                cleanup_key,
                lifecycle_epoch,
                nonce,
            )

        assert await first.set_context_view_if_newer_for_lifecycle(
            context_key,
            {"user_id": user_id, "value": "old"},
            ttl_seconds=300,
            monotonic_seq=1,
            lifecycle_cleanup_key=old_cleanup,
            lifecycle_epoch=old_epoch,
        )
        client.set(f"job_lifecycle:{old_cleanup}", f"revoked:{old_epoch}")
        assert await second.set_context_view_if_newer_for_lifecycle(
            context_key,
            {"user_id": user_id, "value": "new"},
            ttl_seconds=300,
            monotonic_seq=2,
            lifecycle_cleanup_key=new_cleanup,
            lifecycle_epoch=new_epoch,
        )
        assert not client.exists(f"lifecycle_context_entries:{old_cleanup}")
        client.sadd(f"lifecycle_context_entries:{old_cleanup}", member)

        await first.revoke_lifecycle_and_purge_notifications(
            old_cleanup,
            old_epoch,
            group_name="atagia-workers",
        )

        assert await first.get_context_view(context_key) == {
            "user_id": user_id,
            "value": "new",
        }
        assert client.get(f"context_view_lifecycle_owner:{context_key}") == new_cleanup
        assert client.sismember(f"lifecycle_context_entries:{new_cleanup}", member)
        assert client.sismember(f"context_view_user:{user_id}", context_key)

        race_key = "context_legacy_race"
        await first.set_context_view(
            race_key,
            {"user_id": user_id, "value": "legacy"},
            ttl_seconds=300,
        )
        purge_result, published = await asyncio.gather(
            first.purge_legacy_transient_state("/database.db", user_id),
            second.set_context_view_if_newer_for_lifecycle(
                race_key,
                {"user_id": user_id, "value": "current"},
                ttl_seconds=300,
                monotonic_seq=1,
                lifecycle_cleanup_key=new_cleanup,
                lifecycle_epoch=new_epoch,
            ),
        )
        assert purge_result.clean
        assert published
        assert await first.get_context_view(race_key) == {
            "user_id": user_id,
            "value": "current",
        }
        assert client.get(f"context_view_lifecycle_owner:{race_key}") == new_cleanup
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_real_redis_recent_window_takeover_survives_old_lifecycle_revoke(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    owner = {"user_id": "usr_takeover", "conversation_id": "conv_takeover"}
    logical_key = build_recent_window_key(**owner)
    member = f"r\x1f{logical_key}"
    old_cleanup_key = "cleanup_recent_old"
    old_lifecycle_epoch = "epoch_recent_old"
    new_cleanup_key = "cleanup_recent_new"
    new_lifecycle_epoch = "epoch_recent_new"
    old_index_key = f"lifecycle_recent_entries:{old_cleanup_key}"
    new_index_key = f"lifecycle_recent_entries:{new_cleanup_key}"
    identity_key = f"recent_window_cache_identity:{logical_key}"
    user_index_key = f"recent_window_user:{owner['user_id']}"
    try:
        await backend.prepare_lifecycle_mirror(
            old_cleanup_key,
            old_lifecycle_epoch,
            "old_nonce",
        )
        assert await backend.activate_lifecycle_mirror(
            old_cleanup_key,
            old_lifecycle_epoch,
            "old_nonce",
        )
        await backend.prepare_lifecycle_mirror(
            new_cleanup_key,
            new_lifecycle_epoch,
            "new_nonce",
        )
        assert await backend.activate_lifecycle_mirror(
            new_cleanup_key,
            new_lifecycle_epoch,
            "new_nonce",
        )
        assert await backend.set_recent_window_for_lifecycle(
            logical_key,
            [{"role": "user", "text": "old owner"}],
            **owner,
            lifecycle_cleanup_key=old_cleanup_key,
            lifecycle_epoch=old_lifecycle_epoch,
            cache_revision=4,
            derivation_revision=2,
            conversation_lifecycle_epoch="conversation_epoch_old",
            conversation_source_revision=9,
        )
        assert real_redis_server.client.sismember(old_index_key, member)
        assert real_redis_server.client.sismember(user_index_key, logical_key)

        # Two active owners are ambiguous, so a different lifecycle cannot
        # steal the key until the previous owner's mirror is no longer active.
        assert not await backend.set_recent_window_for_lifecycle(
            logical_key,
            [{"role": "assistant", "text": "new owner"}],
            **owner,
            lifecycle_cleanup_key=new_cleanup_key,
            lifecycle_epoch=new_lifecycle_epoch,
            cache_revision=0,
            derivation_revision=0,
            conversation_lifecycle_epoch="conversation_epoch_new",
            conversation_source_revision=1,
        )
        real_redis_server.client.set(
            f"job_lifecycle:{old_cleanup_key}",
            f"revoked:{old_lifecycle_epoch}",
        )
        assert await backend.set_recent_window_for_lifecycle(
            logical_key,
            [{"role": "assistant", "text": "new owner"}],
            **owner,
            lifecycle_cleanup_key=new_cleanup_key,
            lifecycle_epoch=new_lifecycle_epoch,
            cache_revision=0,
            derivation_revision=0,
            conversation_lifecycle_epoch="conversation_epoch_new",
            conversation_source_revision=1,
        )
        assert not real_redis_server.client.exists(old_index_key)
        assert real_redis_server.client.sismember(new_index_key, member)
        assert real_redis_server.client.sismember(user_index_key, logical_key)
        current_identity = json_utils.loads(real_redis_server.client.get(identity_key))
        assert current_identity["lifecycle_cleanup_key"] == new_cleanup_key
        assert current_identity["lifecycle_epoch"] == new_lifecycle_epoch

        # Exercise the revoke-side ownership check even if a stale old index
        # member survives or is reintroduced by inconsistent external state.
        real_redis_server.client.sadd(old_index_key, member)
        await backend.revoke_lifecycle_and_purge_notifications(
            old_cleanup_key,
            old_lifecycle_epoch,
            group_name="atagia-workers",
        )
        assert await stored_recent_window(backend, logical_key) == [
            {"role": "assistant", "text": "new owner"}
        ]
        assert json_utils.loads(real_redis_server.client.get(identity_key)) == (
            current_identity
        )
        assert not real_redis_server.client.exists(old_index_key)
        assert real_redis_server.client.sismember(new_index_key, member)
        assert real_redis_server.client.sismember(user_index_key, logical_key)

        # A malformed legacy/corrupt owner identity is never overwritten by
        # guessing missing ownership coordinates.
        malformed_owner = {
            "user_id": "usr_takeover",
            "conversation_id": "conv_incomplete_identity",
        }
        malformed_key = build_recent_window_key(**malformed_owner)
        real_redis_server.client.set(
            f"recent_window:{malformed_key}",
            '[{"role":"user","text":"preserve on malformed identity"}]',
        )
        real_redis_server.client.set(
            f"recent_window_cache_identity:{malformed_key}",
            (
                '{"lifecycle_cleanup_key":"cleanup_incomplete",'
                '"lifecycle_epoch":"epoch_incomplete"}'
            ),
        )
        assert not await backend.set_recent_window_for_lifecycle(
            malformed_key,
            [{"role": "assistant", "text": "must not replace"}],
            **malformed_owner,
            lifecycle_cleanup_key=new_cleanup_key,
            lifecycle_epoch=new_lifecycle_epoch,
            cache_revision=1,
            derivation_revision=0,
            conversation_lifecycle_epoch="conversation_epoch_new",
            conversation_source_revision=2,
        )
        assert await stored_recent_window(backend, malformed_key) == [
            {"role": "user", "text": "preserve on malformed identity"}
        ]

        await backend.revoke_lifecycle_and_purge_notifications(
            new_cleanup_key,
            new_lifecycle_epoch,
            group_name="atagia-workers",
        )
        assert await stored_recent_window(backend, logical_key) is None
        assert not real_redis_server.client.exists(user_index_key)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_recent_window_identity_validation_has_inprocess_redis_parity(
    real_redis_server: RealRedisServer,
) -> None:
    inprocess_backend = InProcessBackend()
    redis_backend = RedisBackend(real_redis_server.url)
    backends: tuple[tuple[str, StorageBackend], ...] = (
        ("inprocess", inprocess_backend),
        ("redis", redis_backend),
    )
    try:
        for backend_name, backend in backends:
            for case_name, invalid_coordinates in INVALID_RECENT_WINDOW_IDENTITIES:
                user_id = f"usr_invalid:{backend_name}"
                conversation_id = f"conversation:{case_name}"
                logical_key = build_recent_window_key(user_id, conversation_id)
                cleanup_key = f"cleanup_{backend_name}_{case_name}"
                lifecycle_epoch = f"epoch_{backend_name}_{case_name}"
                valid_identity: dict[str, object] = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "lifecycle_cleanup_key": cleanup_key,
                    "lifecycle_epoch": lifecycle_epoch,
                    "cache_revision": 2,
                    "derivation_revision": 2,
                    "conversation_lifecycle_epoch": (
                        f"conversation_epoch_{backend_name}_{case_name}"
                    ),
                    "conversation_source_revision": 3,
                }
                invalid_identity = {
                    **valid_identity,
                    **invalid_coordinates,
                }
                await backend.prepare_lifecycle_mirror(
                    cleanup_key,
                    lifecycle_epoch,
                    f"nonce_{backend_name}_{case_name}",
                )
                assert await backend.activate_lifecycle_mirror(
                    cleanup_key,
                    lifecycle_epoch,
                    f"nonce_{backend_name}_{case_name}",
                )

                assert not await backend.set_recent_window_for_lifecycle(
                    logical_key,
                    [{"role": "user", "text": "invalid write"}],
                    **invalid_identity,
                )
                assert await stored_recent_window(backend, logical_key) is None

                expected_window = [{"role": "assistant", "text": "valid owner remains"}]
                assert await backend.set_recent_window_for_lifecycle(
                    logical_key,
                    expected_window,
                    **valid_identity,
                )
                assert not await backend.delete_recent_window_if_cache_identity(
                    logical_key,
                    **invalid_identity,
                )
                assert await stored_recent_window(backend, logical_key) == expected_window
                # The read is fenced on the same coordinates as the write and
                # the conditional delete, on both backends: an identity that
                # cannot publish cannot read either.
                assert (
                    await backend.get_recent_window_for_cache_identity(
                        logical_key,
                        **invalid_identity,
                    )
                    is None
                )
                assert (
                    await backend.get_recent_window_for_cache_identity(
                        logical_key,
                        **{**valid_identity, "cache_revision": 3},
                    )
                    is None
                )
                assert (
                    await backend.get_recent_window_for_cache_identity(
                        logical_key,
                        **valid_identity,
                    )
                    == expected_window
                )
    finally:
        await redis_backend.close()
        await inprocess_backend.close()


@pytest.mark.asyncio
async def test_recent_window_colon_ids_have_exact_backend_parity(
    real_redis_server: RealRedisServer,
) -> None:
    inprocess_backend = InProcessBackend()
    redis_backend = RedisBackend(real_redis_server.url)
    backends: tuple[tuple[str, StorageBackend], ...] = (
        ("inprocess", inprocess_backend),
        ("redis", redis_backend),
    )
    try:
        for backend_name, backend in backends:
            first_owner = {
                "user_id": f"account:{backend_name}",
                "conversation_id": "thread",
            }
            second_owner = {
                "user_id": "account",
                "conversation_id": f"{backend_name}:thread",
            }
            first_key = build_recent_window_key(**first_owner)
            second_key = build_recent_window_key(**second_owner)
            assert first_key != second_key

            for suffix in ("first", "second"):
                cleanup_key = f"cleanup_{backend_name}_{suffix}"
                lifecycle_epoch = f"epoch_{backend_name}_{suffix}"
                nonce = f"nonce_{backend_name}_{suffix}"
                await backend.prepare_lifecycle_mirror(
                    cleanup_key,
                    lifecycle_epoch,
                    nonce,
                )
                assert await backend.activate_lifecycle_mirror(
                    cleanup_key,
                    lifecycle_epoch,
                    nonce,
                )

            first_identity = {
                "lifecycle_cleanup_key": f"cleanup_{backend_name}_first",
                "lifecycle_epoch": f"epoch_{backend_name}_first",
                "cache_revision": 0,
                "derivation_revision": 0,
                "conversation_lifecycle_epoch": "conversation_epoch_first",
                "conversation_source_revision": 0,
            }
            second_identity = {
                "lifecycle_cleanup_key": f"cleanup_{backend_name}_second",
                "lifecycle_epoch": f"epoch_{backend_name}_second",
                "cache_revision": 0,
                "derivation_revision": 0,
                "conversation_lifecycle_epoch": "conversation_epoch_second",
                "conversation_source_revision": 0,
            }
            assert await backend.set_recent_window_for_lifecycle(
                first_key,
                [{"text": "first"}],
                **first_owner,
                **first_identity,
            )
            assert await backend.set_recent_window_for_lifecycle(
                second_key,
                [{"text": "second"}],
                **second_owner,
                **second_identity,
            )

            assert not await backend.set_recent_window_for_lifecycle(
                first_key,
                [{"text": "mismatched"}],
                user_id="account",
                conversation_id=f"{backend_name}:thread",
                **first_identity,
            )
            assert not await backend.delete_recent_window_if_cache_identity(
                first_key,
                user_id="account",
                conversation_id=f"{backend_name}:thread",
                **first_identity,
            )
            assert await stored_recent_window(backend, first_key) == [{"text": "first"}]

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
    finally:
        await redis_backend.close()
        await inprocess_backend.close()


@pytest.mark.asyncio
async def test_real_redis_dead_letter_publish_linearizes_with_revoke_and_purge(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    cleanup_key = "cleanup_dead_letter_race"
    lifecycle_epoch = "epoch_dead_letter_race"
    queue_name = "dead_letter:atagia:race"
    diagnostic = {
        "job_id": "job_dead_letter_race",
        "user_id": "usr_dead_letter_race",
        "lifecycle_epoch": lifecycle_epoch,
        "error_class": "InjectedError",
    }
    try:
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
        purged = await backend.revoke_lifecycle_and_purge_notifications(
            cleanup_key,
            lifecycle_epoch,
            group_name="atagia-workers",
        )
        resume.set()

        assert purged == 1
        assert await publish_task is None
        assert real_redis_server.client.llen(f"{ATAGIA_QUEUE_PREFIX}{queue_name}") == 0
        assert (
            real_redis_server.client.exists(f"job_lifecycle_diagnostics:{cleanup_key}")
            == 0
        )
        assert await backend.dequeue_job(queue_name, timeout_seconds=0) is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_real_redis_atomic_list_purge_preserves_concurrent_operations(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    queue_name = "atomic-purge"
    queue_key = f"{ATAGIA_QUEUE_PREFIX}{queue_name}"
    keep_before = {"job_id": "keep_before", "user_id": "usr_keep"}
    target = {"job_id": "drop", "user_id": "usr_target"}
    keep_after = {"job_id": "keep_after", "user_id": "usr_other"}
    concurrent = {"job_id": "keep_concurrent", "user_id": "usr_keep"}
    foreign_queue_key = "queue:foreign"
    foreign_queue_raw = json_utils.dumps(
        {"payload": {"job_id": "foreign", "user_id": "usr_target"}},
        sort_keys=True,
    )
    try:
        await backend.enqueue_job(queue_name, keep_before)
        real_redis_server.client.rpush(queue_key, "{malformed-json")
        await backend.enqueue_job(queue_name, target)
        await backend.enqueue_job(queue_name, keep_after)
        real_redis_server.client.rpush(foreign_queue_key, foreign_queue_raw)

        purged, enqueued = _start_operation_pair(
            redis_url=real_redis_server.url,
            queue_name=queue_name,
            first_operation="purge",
            second_operation="enqueue",
            second_payload=concurrent,
        )
        assert purged == 1
        assert enqueued == concurrent
        raw_items = real_redis_server.client.lrange(queue_key, 0, -1)
        assert raw_items[1] == "{malformed-json"
        decoded = [
            item if item == "{malformed-json" else __import__("json").loads(item)
            for item in raw_items
        ]
        assert decoded == [keep_before, "{malformed-json", keep_after, concurrent]
        assert real_redis_server.client.lrange(foreign_queue_key, 0, -1) == [
            foreign_queue_raw
        ]
        assert await backend.purge_user_jobs("usr_target") == 0

        dead_letter_queue = "dead_letter:atagia:extract"
        await backend.enqueue_job(
            dead_letter_queue,
            {
                "job_id": "dead_target",
                "user_id": "usr_target",
                "conversation_id": "cnv_private",
                "error_class": "RuntimeError",
            },
        )
        assert await backend.purge_user_jobs("usr_target") == 1
        assert (
            real_redis_server.client.llen(f"{ATAGIA_QUEUE_PREFIX}{dead_letter_queue}")
            == 0
        )

        # Whether purge or dequeue linearizes first, the popped item never returns.
        dequeue_queue = "atomic-dequeue"
        dequeue_key = f"{ATAGIA_QUEUE_PREFIX}{dequeue_queue}"
        first = {"job_id": "first", "user_id": "usr_keep"}
        middle_target = {"job_id": "middle", "user_id": "usr_target"}
        last = {"job_id": "last", "user_id": "usr_keep"}
        for payload in (first, middle_target, last):
            await backend.enqueue_job(dequeue_queue, payload)
        purged, dequeued = _start_operation_pair(
            redis_url=real_redis_server.url,
            queue_name=dequeue_queue,
            first_operation="purge",
            second_operation="dequeue",
        )
        assert purged == 1
        assert dequeued == first
        remaining = real_redis_server.client.lrange(dequeue_key, 0, -1)
        assert [__import__("json").loads(item) for item in remaining] == [last]

        # Large and empty lists exercise the same atomic script without special paths.
        large_queue = "atomic-large"
        expected_kept: list[dict[str, str]] = []
        for index in range(600):
            payload = {
                "job_id": f"large_{index}",
                "user_id": "usr_target" if index % 3 == 0 else "usr_keep",
            }
            await backend.enqueue_job(large_queue, payload)
            if payload["user_id"] != "usr_target":
                expected_kept.append(payload)
        assert await backend.purge_user_jobs("usr_target") == 200
        large_raw = real_redis_server.client.lrange(
            f"{ATAGIA_QUEUE_PREFIX}{large_queue}",
            0,
            -1,
        )
        assert [__import__("json").loads(item) for item in large_raw] == expected_kept
        assert await backend.purge_user_jobs("usr_missing") == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_real_redis_stream_poll_semantics_and_group_reset_recovery(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    stream_name = "atagia:reset-recovery"
    try:
        await backend.stream_ensure_group(stream_name, "atagia-workers")
        real_redis_server.client.flushall()
        await backend.stream_add(stream_name, {"job_id": "after-reset"})

        # XAUTOCLAIM must recreate a group destroyed by reset without a restart.
        assert (
            await backend.stream_claim_idle(
                stream_name,
                "atagia-workers",
                "worker-reset",
                min_idle_ms=0,
                count=1,
            )
            == []
        )
        messages = await asyncio.wait_for(
            backend.stream_read(
                stream_name,
                "atagia-workers",
                "worker-reset",
                count=1,
                block_ms=0,
            ),
            timeout=0.5,
        )
        assert [message.payload for message in messages] == [{"job_id": "after-reset"}]
        await backend.stream_ack(
            stream_name,
            "atagia-workers",
            messages[0].message_id,
        )

        # Zero is nonblocking, while None preserves the abstract forever-blocking API.
        assert (
            await asyncio.wait_for(
                backend.stream_read(
                    stream_name,
                    "atagia-workers",
                    "worker-reset",
                    count=1,
                    block_ms=0,
                ),
                timeout=0.5,
            )
            == []
        )
        blocking_read = asyncio.create_task(
            backend.stream_read(
                stream_name,
                "atagia-workers",
                "worker-reset",
                count=1,
                block_ms=None,
            )
        )
        await asyncio.sleep(0)
        await backend.stream_add(stream_name, {"job_id": "unblock"})
        unblocked = await asyncio.wait_for(blocking_read, timeout=1.0)
        assert [message.payload for message in unblocked] == [{"job_id": "unblock"}]
        await backend.stream_ack(
            stream_name,
            "atagia-workers",
            unblocked[0].message_id,
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_real_redis_legacy_transient_purge_is_exact_and_idempotent(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    client = real_redis_server.client
    target_user = "usr:target"
    other_user = "usr_other"
    legacy_target = JobEnvelope(
        job_id="job_legacy_target",
        job_type=JobType.RUN_EVALUATION,
        user_id=target_user,
    ).model_dump(mode="json")
    legacy_other = JobEnvelope(
        job_id="job_legacy_other",
        job_type=JobType.RUN_EVALUATION,
        user_id=other_user,
    ).model_dump(mode="json")
    current_notification = DurableJobNotification(
        job_id="job_current",
        dispatch_token="dispatch_current",
        lifecycle_epoch="epoch_current",
        lifecycle_cleanup_key="cleanup_current",
    ).model_dump(mode="json")
    target_raw = json_utils.dumps(legacy_target, sort_keys=True)
    other_raw = json_utils.dumps(legacy_other, sort_keys=True)
    current_raw = json_utils.dumps(current_notification, sort_keys=True)
    dead_target_raw = json_utils.dumps(
        {
            "delivery_count": 3,
            "error": "legacy failure",
            "message_id": "legacy_target_message",
            "payload": legacy_target,
        },
        sort_keys=True,
    )
    dead_other_raw = json_utils.dumps(
        {
            "delivery_count": 4,
            "error": "legacy failure",
            "error_details": [],
            "message_id": "legacy_other_message",
            "payload": legacy_other,
        },
        sort_keys=True,
    )
    try:
        client.set(f"cachegen:{target_user}", "1")
        client.set(f"cachegen:aaaaaaaaaaaa:{target_user}", "1")
        client.set(f"cachegen:bbbbbbbbbbbb:{target_user}", "1")
        client.set(f"cachegen:cccccccccccc:{other_user}", "1")
        client.set(f"cachegen:not-hex-value:{target_user}", "1")

        legacy_recent_keys = (
            "recent_window:a:conversation",
            "recent_window:a:b:conversation",
            "recent_window:rw:v1:legacy-looking",
        )
        for recent_key in legacy_recent_keys:
            client.set(recent_key, "[]")

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
        current_recent_owner = {
            "user_id": target_user,
            "conversation_id": "conversation_current",
        }
        current_recent_key = build_recent_window_key(**current_recent_owner)
        assert await backend.set_recent_window_for_lifecycle(
            current_recent_key,
            [{"role": "user", "text": "current"}],
            **current_recent_owner,
            lifecycle_cleanup_key="cleanup_current",
            lifecycle_epoch="epoch_current",
            cache_revision=1,
            derivation_revision=2,
            conversation_lifecycle_epoch="conversation_epoch_current",
            conversation_source_revision=3,
        )
        await backend.prepare_lifecycle_mirror(
            "cleanup_stale_recent",
            "epoch_stale_recent",
            "nonce_stale_recent",
        )
        assert await backend.activate_lifecycle_mirror(
            "cleanup_stale_recent",
            "epoch_stale_recent",
            "nonce_stale_recent",
        )
        stale_recent_owner = {
            "user_id": target_user,
            "conversation_id": "conversation_stale_recent",
        }
        stale_recent_key = build_recent_window_key(**stale_recent_owner)
        assert await backend.set_recent_window_for_lifecycle(
            stale_recent_key,
            [{"role": "user", "text": "stale"}],
            **stale_recent_owner,
            lifecycle_cleanup_key="cleanup_stale_recent",
            lifecycle_epoch="epoch_stale_recent",
            cache_revision=1,
            derivation_revision=1,
            conversation_lifecycle_epoch="conversation_epoch_stale",
            conversation_source_revision=1,
        )
        client.srem(
            "lifecycle_recent_entries:cleanup_stale_recent",
            f"r\x1f{stale_recent_key}",
        )
        await backend.prepare_lifecycle_mirror(
            "cleanup_missing_recent",
            "epoch_missing_recent",
            "nonce_missing_recent",
        )
        assert await backend.activate_lifecycle_mirror(
            "cleanup_missing_recent",
            "epoch_missing_recent",
            "nonce_missing_recent",
        )
        missing_recent_owner = {
            "user_id": target_user,
            "conversation_id": "conversation_missing_recent",
        }
        missing_recent_key = build_recent_window_key(**missing_recent_owner)
        assert await backend.set_recent_window_for_lifecycle(
            missing_recent_key,
            [{"role": "user", "text": "missing value"}],
            **missing_recent_owner,
            lifecycle_cleanup_key="cleanup_missing_recent",
            lifecycle_epoch="epoch_missing_recent",
            cache_revision=1,
            derivation_revision=1,
            conversation_lifecycle_epoch="conversation_epoch_missing",
            conversation_source_revision=1,
        )
        client.delete(f"recent_window:{missing_recent_key}")
        missing_recent_without_identity = "missing_recent_without_identity"
        client.sadd(
            f"recent_window_user:{target_user}",
            missing_recent_without_identity,
        )
        orphan_recent_owner = {
            "user_id": target_user,
            "conversation_id": "conversation_orphan_identity",
        }
        orphan_recent_key = build_recent_window_key(**orphan_recent_owner)
        assert await backend.set_recent_window_for_lifecycle(
            orphan_recent_key,
            [{"role": "user", "text": "orphan identity"}],
            **orphan_recent_owner,
            lifecycle_cleanup_key="cleanup_missing_recent",
            lifecycle_epoch="epoch_missing_recent",
            cache_revision=1,
            derivation_revision=1,
            conversation_lifecycle_epoch="conversation_epoch_orphan",
            conversation_source_revision=1,
        )
        client.delete(f"recent_window:{orphan_recent_key}")
        client.srem(f"recent_window_user:{target_user}", orphan_recent_key)

        await backend.set_context_view(
            "context_legacy_target",
            {"user_id": target_user, "value": "legacy"},
            ttl_seconds=300,
        )
        await backend.set_context_view(
            "context_legacy_other",
            {"user_id": other_user, "value": "other"},
            ttl_seconds=300,
        )
        client.set(
            "context_view:context_orphan_target",
            json_utils.dumps(
                {"user_id": target_user, "value": "orphan"},
                sort_keys=True,
            ),
        )
        client.set(
            "context_view:context_orphan_other",
            json_utils.dumps(
                {"user_id": other_user, "value": "orphan-other"},
                sort_keys=True,
            ),
        )
        client.set(
            "context_view:context_corrupt_owner_target",
            json_utils.dumps(
                {"user_id": target_user, "value": "corrupt-owner"},
                sort_keys=True,
            ),
        )
        client.set(
            "context_view_lifecycle_owner:context_corrupt_owner_target",
            "cleanup_corrupt_owner",
        )
        client.set("job_lifecycle:cleanup_corrupt_owner", "active:epoch_corrupt")
        owner_only_context = "context_owner_only_target"
        owner_only_conversation = "conversation_owner_only"
        owner_only_cleanup_key = "cleanup_owner_only"
        client.set(f"context_view_owner:{owner_only_context}", target_user)
        client.set(f"context_view_seq:{owner_only_context}", "7")
        client.set(
            f"context_view_conversation_owner:{owner_only_context}",
            owner_only_conversation,
        )
        client.sadd(
            f"context_view_conversation:{owner_only_conversation}",
            owner_only_context,
        )
        client.set(
            f"context_view_lifecycle_owner:{owner_only_context}",
            owner_only_cleanup_key,
        )
        client.sadd(
            f"lifecycle_context_entries:{owner_only_cleanup_key}",
            f"c\x1f{owner_only_context}",
        )
        assert await backend.set_context_view_if_newer_for_lifecycle(
            "context_current_target",
            {"user_id": target_user, "value": "current"},
            ttl_seconds=300,
            monotonic_seq=1,
            lifecycle_cleanup_key="cleanup_current",
            lifecycle_epoch="epoch_current",
        )
        client.sadd(f"context_view_user:{target_user}", "missing_context")
        client.sadd(f"context_view_user:{target_user}", "context_orphan_other")

        stream_name = EVALUATION_STREAM_NAME
        target_message_id = client.xadd(stream_name, {"payload": target_raw})
        other_message_id = client.xadd(stream_name, {"payload": other_raw})
        current_message_id = client.xadd(stream_name, {"payload": current_raw})
        client.sadd(
            "job_lifecycle_deliveries:cleanup_current",
            f"{stream_name}\x1f{current_message_id}",
        )
        client.hset(
            f"job_delivery_owner:{stream_name}",
            current_message_id,
            "cleanup_current",
        )
        client.xgroup_create(stream_name, "legacy-workers", id="0")
        delivered = client.xreadgroup(
            "legacy-workers",
            "legacy-consumer",
            {stream_name: ">"},
        )
        assert delivered
        client.xack(
            stream_name,
            "legacy-workers",
            target_message_id,
            other_message_id,
            current_message_id,
        )

        admin_user_queue = "queue:admin_rebuild_user"
        admin_user_other_before = json_utils.dumps(
            {"user_id": other_user},
            sort_keys=True,
        )
        admin_user_target = json_utils.dumps(
            {"user_id": target_user},
            sort_keys=True,
        )
        admin_user_other_after = json_utils.dumps(
            {"user_id": "usr_other_after"},
            sort_keys=True,
        )
        client.rpush(
            admin_user_queue,
            admin_user_other_before,
            admin_user_target,
            admin_user_other_after,
        )
        admin_conversation_queue = "queue:admin_rebuild_conversation"
        admin_conversation_target = json_utils.dumps(
            {"conversation_id": "cnv_target", "user_id": target_user},
            sort_keys=True,
        )
        admin_conversation_other = json_utils.dumps(
            {"conversation_id": "cnv_other", "user_id": other_user},
            sort_keys=True,
        )
        client.rpush(
            admin_conversation_queue,
            admin_conversation_target,
            admin_conversation_other,
        )

        dead_letter_key = f"queue:dead_letter:{EVALUATION_STREAM_NAME}"
        current_diagnostic = json_utils.dumps(
            {
                "_atagia_lifecycle_diagnostic": {
                    "delivery_id": "delivery_current",
                    "lifecycle_cleanup_key": "cleanup_current",
                    "lifecycle_epoch": "epoch_current",
                },
                "payload": {"user_id": target_user, "value": "current"},
            },
            sort_keys=True,
        )
        stale_diagnostic = json_utils.dumps(
            {
                "_atagia_lifecycle_diagnostic": {
                    "delivery_id": "delivery_stale",
                    "lifecycle_cleanup_key": "cleanup_stale_diagnostic",
                    "lifecycle_epoch": "epoch_stale_diagnostic",
                },
                "payload": {"user_id": target_user, "value": "stale"},
            },
            sort_keys=True,
        )
        client.rpush(
            dead_letter_key,
            dead_target_raw,
            dead_other_raw,
            current_diagnostic,
            stale_diagnostic,
        )
        client.sadd(
            "job_lifecycle_diagnostics:cleanup_current",
            f"{dead_letter_key}\x1f{current_diagnostic}",
        )
        client.expire(dead_letter_key, 300)
        foreign_queue_key = "queue:foreign"
        foreign_queue_raw = json_utils.dumps(
            {"payload": {"user_id": target_user}},
            sort_keys=True,
        )
        foreign_dead_letter_key = "queue:dead_letter:atagia:foreign"
        foreign_atagia_key = "atagia:foreign"
        client.rpush(foreign_queue_key, foreign_queue_raw)
        client.rpush(foreign_dead_letter_key, dead_target_raw)
        client.rpush(foreign_atagia_key, target_raw)

        deferred_key = f"stream_deferred:{EXTRACT_STREAM_NAME}"
        foreign_deferred_key = "stream_deferred:foreign-application"
        deferred_target = json_utils.dumps(
            {"id": "deferred_target", "payload_json": target_raw},
            sort_keys=True,
        )
        deferred_other = json_utils.dumps(
            {"id": "deferred_other", "payload_json": other_raw},
            sort_keys=True,
        )
        deferred_payload_target = json_utils.dumps(
            {"id": "deferred_payload_target", "payload": legacy_target},
            sort_keys=True,
        )
        deferred_payload_other = json_utils.dumps(
            {"id": "deferred_payload_other", "payload": legacy_other},
            sort_keys=True,
        )
        client.zadd(
            deferred_key,
            {
                deferred_target: 1,
                deferred_other: 2,
                deferred_payload_target: 3,
                deferred_payload_other: 4,
            },
        )
        client.zadd(foreign_deferred_key, {deferred_target: 1})

        target_extractor_dedupe_key = f"dedupe:{target_user}:{'a' * 64}"
        other_extractor_dedupe_key = f"dedupe:{other_user}:{'b' * 64}"
        client.set(target_extractor_dedupe_key, "1")
        client.set(other_extractor_dedupe_key, "1")
        client.set("dedupe:old-worker-marker", "1")
        client.set(f"dedupe:lifecycle:cooldown:{target_user}", "legacy-collision")
        client.set("dedupe:lifecycle:cooldown:0123456789ab", "current")
        client.set("lock:old-worker-lock", "legacy-token")
        client.set(f"lock:lifecycle:lock:{target_user}", "legacy-collision")
        client.set("lock:lifecycle:lock:0123456789ab", "current-token")

        result = await backend.purge_legacy_transient_state(
            "/different/database/path.db",
            target_user,
        )

        assert result.cache_generation_deleted == 4
        assert result.recent_windows_deleted == 4
        assert result.context_views_deleted == 3
        assert result.stream_entries_deleted == 1
        assert result.queue_entries_deleted == 4
        assert result.deferred_entries_deleted == 2
        assert result.legacy_dedupe_deleted == 1
        assert result.legacy_locks_deleted == 0
        assert result.malformed_candidates == 0
        assert result.total_deleted == 19
        assert result.clean

        assert not client.exists(f"cachegen:{target_user}")
        assert not client.exists(f"cachegen:aaaaaaaaaaaa:{target_user}")
        assert not client.exists(f"cachegen:bbbbbbbbbbbb:{target_user}")
        assert client.exists(f"cachegen:cccccccccccc:{other_user}")
        assert not client.exists(f"cachegen:not-hex-value:{target_user}")
        assert all(not client.exists(key) for key in legacy_recent_keys)
        assert await stored_recent_window(backend, current_recent_key) == [
            {"role": "user", "text": "current"}
        ]
        assert await stored_recent_window(backend, stale_recent_key) is None
        assert not client.exists(f"recent_window_cache_identity:{missing_recent_key}")
        assert not client.sismember(
            "lifecycle_recent_entries:cleanup_missing_recent",
            f"r\x1f{missing_recent_key}",
        )
        assert not client.sismember(
            f"recent_window_user:{target_user}",
            missing_recent_key,
        )
        assert not client.sismember(
            f"recent_window_user:{target_user}",
            missing_recent_without_identity,
        )
        assert not client.exists(f"recent_window_cache_identity:{orphan_recent_key}")
        assert not client.sismember(
            "lifecycle_recent_entries:cleanup_missing_recent",
            f"r\x1f{orphan_recent_key}",
        )
        assert await backend.get_context_view("context_legacy_target") is None
        assert await backend.get_context_view("context_orphan_target") is None
        assert await backend.get_context_view("context_orphan_other") == {
            "user_id": other_user,
            "value": "orphan-other",
        }
        assert await backend.get_context_view("context_corrupt_owner_target") is None
        assert not client.exists(
            "context_view_lifecycle_owner:context_corrupt_owner_target"
        )
        assert not client.exists(f"context_view_owner:{owner_only_context}")
        assert not client.exists(f"context_view_seq:{owner_only_context}")
        assert not client.exists(
            f"context_view_conversation_owner:{owner_only_context}"
        )
        assert not client.exists(f"context_view_lifecycle_owner:{owner_only_context}")
        assert not client.sismember(
            f"context_view_user:{target_user}", owner_only_context
        )
        assert not client.sismember(
            f"context_view_conversation:{owner_only_conversation}",
            owner_only_context,
        )
        assert not client.sismember(
            f"lifecycle_context_entries:{owner_only_cleanup_key}",
            f"c\x1f{owner_only_context}",
        )
        assert await backend.get_context_view("context_legacy_other") == {
            "user_id": other_user,
            "value": "other",
        }
        assert await backend.get_context_view("context_current_target") == {
            "user_id": target_user,
            "value": "current",
        }
        assert not client.sismember(
            f"context_view_user:{target_user}", "missing_context"
        )
        assert not client.sismember(
            f"context_view_user:{target_user}", "context_orphan_other"
        )
        assert client.xrange(stream_name) == [
            (other_message_id, {"payload": other_raw}),
            (current_message_id, {"payload": current_raw}),
        ]
        assert client.sismember(
            "job_lifecycle_deliveries:cleanup_current",
            f"{stream_name}\x1f{current_message_id}",
        )
        assert (
            client.hget(f"job_delivery_owner:{stream_name}", current_message_id)
            == "cleanup_current"
        )
        assert client.lrange(dead_letter_key, 0, -1) == [
            dead_other_raw,
            current_diagnostic,
        ]
        assert client.lrange(foreign_queue_key, 0, -1) == [foreign_queue_raw]
        assert client.lrange(foreign_dead_letter_key, 0, -1) == [dead_target_raw]
        assert client.lrange(foreign_atagia_key, 0, -1) == [target_raw]
        assert client.lrange(admin_user_queue, 0, -1) == [
            admin_user_other_before,
            admin_user_other_after,
        ]
        assert client.lrange(admin_conversation_queue, 0, -1) == [
            admin_conversation_other,
        ]
        assert 0 < client.ttl(dead_letter_key) <= 300
        assert client.zrange(deferred_key, 0, -1) == [
            deferred_other,
            deferred_payload_other,
        ]
        assert client.zrange(foreign_deferred_key, 0, -1) == [deferred_target]
        assert client.exists("dedupe:lifecycle:cooldown:0123456789ab")
        assert client.exists("lock:lifecycle:lock:0123456789ab")
        assert not client.exists(target_extractor_dedupe_key)
        assert client.exists(other_extractor_dedupe_key)
        assert client.exists("dedupe:old-worker-marker")
        assert client.exists(f"dedupe:lifecycle:cooldown:{target_user}")
        assert client.exists("lock:old-worker-lock")
        assert client.exists(f"lock:lifecycle:lock:{target_user}")

        second = await backend.purge_legacy_transient_state(
            "/yet/another/database.db",
            target_user,
        )
        assert second.total_deleted == 0
        assert second.malformed_candidates == 0
        assert second.clean
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reserved_user_id",
    ("contract", "graph", "initial_context_package_refresh"),
)
async def test_real_redis_legacy_purge_preserves_ownerless_hash_collisions(
    real_redis_server: RealRedisServer,
    reserved_user_id: str,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    client = real_redis_server.client
    ambiguous_dedupe_key = f"dedupe:{reserved_user_id}:{'a' * 64}"
    client.set(ambiguous_dedupe_key, "1")
    try:
        result = await backend.purge_legacy_transient_state(
            "/database.db",
            reserved_user_id,
        )

        assert result.legacy_dedupe_deleted == 0
        assert result.malformed_candidates == 0
        assert result.clean
        assert client.get(ambiguous_dedupe_key) == "1"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_real_redis_legacy_transient_purge_preserves_malformed_candidates(
    real_redis_server: RealRedisServer,
) -> None:
    backend = RedisBackend(real_redis_server.url)
    client = real_redis_server.client
    target_raw = json_utils.dumps(
        JobEnvelope(
            job_id="job_malformed_target",
            job_type=JobType.RUN_EVALUATION,
            user_id="usr_target",
        ).model_dump(mode="json"),
        sort_keys=True,
    )
    notification_like_raw = json_utils.dumps(
        {
            "job_id": "job_current_malformed",
            "dispatch_token": "dispatch_current_malformed",
            "lifecycle_epoch": "epoch_current_malformed",
            "lifecycle_cleanup_key": "cleanup_current_malformed",
            "unexpected": "field",
        },
        sort_keys=True,
    )
    exact_current_notification_raw = json_utils.dumps(
        {
            "dispatch_token": "dispatch_unindexed",
            "job_id": "job_unindexed",
            "lifecycle_cleanup_key": "cleanup_unindexed",
            "lifecycle_epoch": "epoch_unindexed",
        },
        sort_keys=True,
    )
    invalid_diagnostic_outer = json_utils.dumps(
        {
            "_atagia_lifecycle_diagnostic": {
                "delivery_id": "delivery_invalid_outer",
                "lifecycle_cleanup_key": "cleanup_invalid_outer",
                "lifecycle_epoch": "epoch_invalid_outer",
            },
            "payload": {"user_id": "usr_target"},
            "unexpected": "field",
        },
        sort_keys=True,
    )
    invalid_diagnostic_metadata = json_utils.dumps(
        {
            "_atagia_lifecycle_diagnostic": {
                "delivery_id": "delivery_invalid_metadata",
                "lifecycle_cleanup_key": "cleanup_invalid_metadata",
                "lifecycle_epoch": "epoch_invalid_metadata",
                "unexpected": "field",
            },
            "payload": {"user_id": "usr_target"},
        },
        sort_keys=True,
    )
    active_unproven_diagnostic = json_utils.dumps(
        {
            "_atagia_lifecycle_diagnostic": {
                "delivery_id": "delivery_active_unproven",
                "lifecycle_cleanup_key": "cleanup_active_unproven",
                "lifecycle_epoch": "epoch_active_unproven",
            },
            "payload": {"user_id": "usr_target"},
        },
        sort_keys=True,
    )
    unattributed_raw = json_utils.dumps(
        {"unattributed": "candidate"},
        sort_keys=True,
    )
    stream_name = EXTRACT_STREAM_NAME
    dead_letter_key = f"queue:dead_letter:{EXTRACT_STREAM_NAME}"
    legacy_queue_key = "queue:admin_rebuild_user"
    valid_other_queue_raw = json_utils.dumps(
        {"user_id": "usr_other"},
        sort_keys=True,
    )
    deferred_key = f"stream_deferred:{EXTRACT_STREAM_NAME}"
    wrong_type_keys = (
        EVALUATION_STREAM_NAME,
        f"queue:dead_letter:{CONTRACT_STREAM_NAME}",
        "queue:admin_rebuild_user_wrong_type",
        f"stream_deferred:{CONTRACT_STREAM_NAME}",
    )
    deferred_members = {
        "not-json": 1,
        json_utils.dumps(
            {"id": "invalid_inner", "payload_json": "not-json"},
            sort_keys=True,
        ): 2,
        json_utils.dumps({"payload_json": target_raw}, sort_keys=True): 3,
        json_utils.dumps({"id": "", "payload_json": target_raw}, sort_keys=True): 4,
        json_utils.dumps({"id": 5, "payload_json": target_raw}, sort_keys=True): 5,
        json_utils.dumps(
            {
                "id": "notification_with_extra",
                "payload_json": notification_like_raw,
            },
            sort_keys=True,
        ): 6,
        json_utils.dumps(
            {"id": "unattributed", "payload_json": unattributed_raw},
            sort_keys=True,
        ): 7,
        json_utils.dumps(
            {
                "id": "current_notification",
                "payload_json": exact_current_notification_raw,
            },
            sort_keys=True,
        ): 8,
        json_utils.dumps(
            {
                "id": "both_payload_forms",
                "payload": json_utils.loads(target_raw),
                "payload_json": target_raw,
            },
            sort_keys=True,
        ): 9,
        json_utils.dumps(
            {
                "id": "extra_outer_field",
                "payload": json_utils.loads(target_raw),
                "unexpected": "field",
            },
            sort_keys=True,
        ): 10,
    }
    try:
        client.set("cachegen:", "unexpected-shape")
        client.set("cachegen:usr_other", "unexpected-shape")
        client.set("cachegen:not-hex-value:usr_other", "unexpected-shape")
        client.xadd(stream_name, {"payload": "not-json"})
        client.xadd(stream_name, {"payload": notification_like_raw})
        client.xadd(stream_name, {"payload": unattributed_raw})
        client.xadd(stream_name, {"payload": exact_current_notification_raw})
        client.set(
            "job_lifecycle:cleanup_unindexed",
            "active:epoch_unindexed",
        )
        client.xadd(
            stream_name,
            {"payload": target_raw, "unexpected": "field"},
        )
        client.execute_command(
            "XADD",
            stream_name,
            "*",
            "payload",
            unattributed_raw,
            "payload",
            target_raw,
        )
        client.rpush(
            dead_letter_key,
            "not-json",
            invalid_diagnostic_outer,
            invalid_diagnostic_metadata,
            unattributed_raw,
            active_unproven_diagnostic,
        )
        client.set(
            "job_lifecycle:cleanup_active_unproven",
            "active:epoch_active_unproven",
        )
        client.rpush(
            legacy_queue_key,
            "not-json",
            unattributed_raw,
            valid_other_queue_raw,
        )
        client.zadd(deferred_key, deferred_members)
        client.set("context_view:malformed-orphan", "not-json")
        client.rpush("context_view:wrong-type", "unexpected-type")
        client.rpush("context_view_owner:wrong-type", "usr_target")
        client.set("context_view_owner:empty-owner", "")
        client.rpush(
            "recent_window_cache_identity:wrong-type",
            "unexpected-type",
        )
        client.set("recent_window_cache_identity:invalid-json", "not-json")
        for wrong_type_key in wrong_type_keys:
            client.set(wrong_type_key, "unexpected-type")

        first = await backend.purge_legacy_transient_state(
            "/database.db",
            "usr_target",
        )

        assert first.total_deleted == 0
        assert first.malformed_candidates == 35
        assert not first.clean
        assert client.xlen(stream_name) == 6
        assert client.lrange(dead_letter_key, 0, -1) == [
            "not-json",
            invalid_diagnostic_outer,
            invalid_diagnostic_metadata,
            unattributed_raw,
            active_unproven_diagnostic,
        ]
        assert client.lrange(legacy_queue_key, 0, -1) == [
            "not-json",
            unattributed_raw,
            valid_other_queue_raw,
        ]
        assert client.zcard(deferred_key) == 10
        assert client.get("context_view:malformed-orphan") == "not-json"
        assert client.lrange("context_view:wrong-type", 0, -1) == ["unexpected-type"]
        assert client.lrange("context_view_owner:wrong-type", 0, -1) == ["usr_target"]
        assert client.get("context_view_owner:empty-owner") == ""
        assert client.lrange(
            "recent_window_cache_identity:wrong-type",
            0,
            -1,
        ) == ["unexpected-type"]
        assert client.get("recent_window_cache_identity:invalid-json") == "not-json"
        assert client.get("cachegen:") == "unexpected-shape"
        assert client.get("cachegen:usr_other") == "unexpected-shape"
        assert client.get("cachegen:not-hex-value:usr_other") == "unexpected-shape"
        assert all(client.get(key) == "unexpected-type" for key in wrong_type_keys)

        second = await backend.purge_legacy_transient_state(
            "/database.db",
            "usr_target",
        )
        assert second.total_deleted == 0
        assert second.malformed_candidates == 35
        assert not second.clean
    finally:
        await backend.close()
