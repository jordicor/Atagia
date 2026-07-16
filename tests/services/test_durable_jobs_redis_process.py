"""Real Redis and independent-process gates for durable jobs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from typing import Any

import pytest

from atagia.core.clock import SystemClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.redis_client import ATAGIA_QUEUE_PREFIX, RedisBackend
from atagia.core.repositories import UserRepository
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    REVISE_STREAM_NAME,
    TRANSCRIPT_REBUILD_STREAM_NAME,
    ClaimedJob,
    JobEnvelope,
    JobRunStatus,
    JobType,
    WORKER_GROUP_NAME,
)
from atagia.services.durable_job_dispatcher import DurableJobDispatcher
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.worker_effect_fence import (
    StaleJobEffectFenceError,
    WorkerEffectFence,
)
from atagia.services.worker_job_lease import WorkerJobLease
from tests.redis_real_support import RealRedisServer, running_redis_server


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
STREAM_BY_TYPE = {
    JobType.EXTRACT_MEMORY_CANDIDATES: EXTRACT_STREAM_NAME,
    JobType.PROJECT_CONTRACT: CONTRACT_STREAM_NAME,
    JobType.REVISE_BELIEFS: REVISE_STREAM_NAME,
    JobType.COMPACT_SUMMARIES: COMPACT_STREAM_NAME,
    JobType.SYNC_GRAPH: GRAPH_STREAM_NAME,
    JobType.RUN_EVALUATION: EVALUATION_STREAM_NAME,
    JobType.REFRESH_INITIAL_CONTEXT_PACKAGE: INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    JobType.REBUILD_SELECTED_TRANSCRIPT: TRANSCRIPT_REBUILD_STREAM_NAME,
}
LEASE_SECONDS = 0.6
HEARTBEAT_SECONDS = 0.1
VISIBILITY_SECONDS = 5.0


@pytest.fixture
def real_redis_server(tmp_path: Path):
    with running_redis_server(tmp_path) as server:
        yield server


def _runtime_settings(redis_url: str) -> Settings:
    return replace(
        Settings.from_env(),
        storage_backend="redis",
        redis_url=redis_url,
        workers_enabled=True,
        service_process_count=2,
        worker_dispatch_visibility_seconds=VISIBILITY_SECONDS,
        worker_dispatch_sweep_interval_seconds=0.05,
        worker_execution_lease_seconds=LEASE_SECONDS,
        worker_execution_heartbeat_seconds=HEARTBEAT_SECONDS,
        worker_stream_reclaim_idle_seconds=VISIBILITY_SECONDS,
        worker_retry_backoff_initial_seconds=0.1,
        worker_retry_backoff_max_seconds=0.2,
    )


def _serialize_claim(claim: ClaimedJob) -> dict[str, Any]:
    return {
        "notification_message_id": claim.notification_message_id,
        "envelope": claim.envelope.model_dump(mode="json"),
        "owner_id": claim.owner_id,
        "attempt_count": claim.attempt_count,
        "execution_fence": claim.execution_fence,
        "lifecycle_epoch": claim.lifecycle_epoch,
        "lifecycle_cleanup_key": claim.lifecycle_cleanup_key,
        "derivation_revision": claim.derivation_revision,
    }


def _deserialize_claim(payload: dict[str, Any]) -> ClaimedJob:
    return ClaimedJob(
        notification_message_id=str(payload["notification_message_id"]),
        envelope=JobEnvelope.model_validate(payload["envelope"]),
        owner_id=str(payload["owner_id"]),
        attempt_count=int(payload["attempt_count"]),
        execution_fence=int(payload["execution_fence"]),
        lifecycle_epoch=str(payload["lifecycle_epoch"]),
        lifecycle_cleanup_key=str(payload["lifecycle_cleanup_key"]),
        derivation_revision=int(payload["derivation_revision"]),
    )


async def _read_claimable_notification(
    backend: RedisBackend,
    tracking: JobTrackingService,
    *,
    stream_name: str,
    owner_id: str,
) -> tuple[Any, ClaimedJob]:
    """Discard stale pending tokens before reading the current notification."""

    await backend.stream_ensure_group(stream_name, WORKER_GROUP_NAME)
    for _ in range(4):
        messages = await backend.stream_claim_idle(
            stream_name,
            WORKER_GROUP_NAME,
            owner_id,
            min_idle_ms=0,
            count=1,
        )
        if not messages:
            messages = await backend.stream_read(
                stream_name,
                WORKER_GROUP_NAME,
                owner_id,
                count=1,
                block_ms=2_000,
            )
        if not messages:
            continue
        message = messages[0]
        claim = await tracking.claim_notification(message, owner_id=owner_id)
        if claim is not None:
            return message, claim
        await backend.stream_ack(
            stream_name,
            WORKER_GROUP_NAME,
            message.message_id,
        )
    raise AssertionError("No current durable notification became claimable")


def _worker_process(
    database_path: str,
    redis_url: str,
    stream_name: str,
    owner_id: str,
    release_event: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.queues.Queue,
    *,
    heartbeat: bool,
) -> None:
    """Claim a job and pause at a process-safe barrier before its domain effect."""

    async def run() -> None:
        clock = SystemClock()
        job_connection = await open_connection(database_path)
        effect_connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        tracking = JobTrackingService(
            job_connection,
            clock,
            workers_enabled=True,
            settings=_runtime_settings(redis_url),
        )
        try:
            message, claim = await _read_claimable_notification(
                backend,
                tracking,
                stream_name=stream_name,
                owner_id=owner_id,
            )
            result_queue.put(("claimed", _serialize_claim(claim)))
            if not heartbeat:
                release_event.wait(timeout=30)
                return
            lease = WorkerJobLease(
                tracking,
                claim,
                effect_fence=WorkerEffectFence(effect_connection, clock),
            )
            async with lease:
                released = await asyncio.to_thread(release_event.wait, 30)
                if not released:
                    raise TimeoutError("worker release barrier timed out")
                await effect_connection.execute(
                    """
                    INSERT INTO job_effect_probe(job_id, owner_id)
                    VALUES (?, ?)
                    """,
                    (claim.envelope.job_id, owner_id),
                )
                await effect_connection.commit()
                await lease.succeed()
            await backend.stream_ack(
                stream_name,
                WORKER_GROUP_NAME,
                message.message_id,
            )
            result_queue.put(("done", _serialize_claim(claim)))
        finally:
            await backend.close()
            await effect_connection.close()
            await job_connection.close()

    try:
        asyncio.run(run())
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _contender_process(
    database_path: str,
    redis_url: str,
    stream_name: str,
    owner_id: str,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    """Attempt to claim another owner's still-healthy Redis pending entry."""

    async def run() -> bool:
        connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        tracking = JobTrackingService(
            connection,
            SystemClock(),
            workers_enabled=True,
            settings=_runtime_settings(redis_url),
        )
        try:
            messages = await backend.stream_claim_idle(
                stream_name,
                WORKER_GROUP_NAME,
                owner_id,
                min_idle_ms=0,
                count=1,
            )
            if not messages:
                raise AssertionError("Expected one pending Redis notification")
            claim = await tracking.claim_notification(messages[0], owner_id=owner_id)
            await backend.stream_ack(
                stream_name,
                WORKER_GROUP_NAME,
                messages[0].message_id,
            )
            return claim is not None
        finally:
            await backend.close()
            await connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _dispatcher_process(
    database_path: str,
    redis_url: str,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    async def run() -> dict[str, int]:
        connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        try:
            result = await DurableJobDispatcher(
                connection,
                SystemClock(),
                storage_backend=backend,
                target_backend="redis",
                visibility_seconds=VISIBILITY_SECONDS,
                sweep_interval_seconds=0.05,
                batch_size=100,
            ).dispatch_once()
            return {
                "claimed": result.claimed,
                "published": result.published,
                "delivery_failed": result.delivery_failed,
                "recovered_leases": result.recovered_leases,
            }
        finally:
            await backend.close()
            await connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _stale_effect_process(
    database_path: str,
    claim_payload: dict[str, Any],
    result_queue: multiprocessing.queues.Queue,
) -> None:
    async def run() -> dict[str, bool]:
        claim = _deserialize_claim(claim_payload)
        clock = SystemClock()
        effect_connection = await open_connection(database_path)
        job_connection = await open_connection(database_path)
        fenced = False
        try:
            try:
                async with WorkerEffectFence(effect_connection, clock).activate(claim):
                    await effect_connection.execute(
                        """
                        INSERT INTO job_effect_probe(job_id, owner_id)
                        VALUES (?, 'stale-owner')
                        """,
                        (claim.envelope.job_id,),
                    )
                    await effect_connection.commit()
            except StaleJobEffectFenceError:
                fenced = True
            terminalized = await JobRunRepository(job_connection, clock).finish_claim(
                claim,
                status=JobRunStatus.SUCCEEDED,
            )
            return {"effect_fenced": fenced, "terminalized": terminalized}
        finally:
            await effect_connection.close()
            await job_connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _claim_all_process(
    database_path: str,
    redis_url: str,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    async def run() -> int:
        connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        tracking = JobTrackingService(
            connection,
            SystemClock(),
            workers_enabled=True,
            settings=_runtime_settings(redis_url),
        )
        completed = 0
        try:
            for job_type, stream_name in STREAM_BY_TYPE.items():
                message, claim = await _read_claimable_notification(
                    backend,
                    tracking,
                    stream_name=stream_name,
                    owner_id=f"matrix-{job_type.value}",
                )
                if claim.envelope.job_type is not job_type:
                    raise AssertionError("Recovered envelope has the wrong JobType")
                if not await tracking.finish_claim_succeeded(claim):
                    raise AssertionError("Current matrix claim could not finish")
                await backend.stream_ack(
                    stream_name,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                completed += 1
            return completed
        finally:
            await backend.close()
            await connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _transition_process(
    database_path: str,
    redis_url: str,
    stream_name: str,
    owner_id: str,
    action: str,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    """Apply one retry/defer/dead-letter transition in a fresh process."""

    async def run() -> dict[str, Any]:
        clock = SystemClock()
        job_connection = await open_connection(database_path)
        effect_connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        tracking = JobTrackingService(
            job_connection,
            clock,
            workers_enabled=True,
            settings=_runtime_settings(redis_url),
        )
        try:
            message, claim = await _read_claimable_notification(
                backend,
                tracking,
                stream_name=stream_name,
                owner_id=owner_id,
            )
            private_error = RuntimeError(
                "PRIVATE_PROVIDER_RETRY_TEXT_MUST_NOT_SURVIVE_TERMINAL"
            )
            lease = WorkerJobLease(
                tracking,
                claim,
                effect_fence=WorkerEffectFence(effect_connection, clock),
            )
            ack_after_transition = True
            async with lease:
                if action == "retry":
                    await lease.retry(private_error)
                elif action == "defer":
                    await lease.defer(
                        private_error,
                        deferred_until=clock.now() + timedelta(seconds=0.15),
                    )
                elif action == "dead_letter":
                    assert await lease.dead_letter(
                        backend,
                        stream_name=stream_name,
                        group_name=WORKER_GROUP_NAME,
                        message=message,
                        exc=private_error,
                    )
                    ack_after_transition = False
                else:
                    raise AssertionError(f"Unsupported transition action: {action}")
            if ack_after_transition:
                await backend.stream_ack(
                    stream_name,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
            return _serialize_claim(claim)
        finally:
            await backend.close()
            await effect_connection.close()
            await job_connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _cancel_lifecycle_process(
    database_path: str,
    redis_url: str,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    """Revoke one lifecycle, cancel its durable jobs, and purge Redis."""

    async def run() -> dict[str, int]:
        clock = SystemClock()
        connection = await open_connection(database_path)
        backend = RedisBackend(redis_url)
        try:
            cursor = await connection.execute(
                """
                SELECT lifecycle_epoch, lifecycle_cleanup_key
                FROM user_lifecycles
                WHERE user_id = 'usr_1'
                """
            )
            lifecycle = await cursor.fetchone()
            assert lifecycle is not None
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                """
                UPDATE user_lifecycles
                SET state = 'cleanup_pending',
                    revoked_at = ?,
                    updated_at = ?
                WHERE user_id = 'usr_1'
                  AND lifecycle_epoch = ?
                  AND state = 'active'
                """,
                (
                    clock.now().isoformat(),
                    clock.now().isoformat(),
                    str(lifecycle["lifecycle_epoch"]),
                ),
            )
            cancelled = await JobRunRepository(
                connection,
                clock,
            ).cancel_jobs_with_inactive_lifecycle(commit=False)
            await connection.commit()
            purged = await backend.revoke_lifecycle_and_purge_notifications(
                str(lifecycle["lifecycle_cleanup_key"]),
                str(lifecycle["lifecycle_epoch"]),
                group_name=WORKER_GROUP_NAME,
            )
            return {"cancelled": cancelled, "purged": purged}
        finally:
            await backend.close()
            await connection.close()

    try:
        result_queue.put(("ok", asyncio.run(run())))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


async def _initialize_jobs(
    database_path: str,
    jobs: list[tuple[str, JobType]],
) -> None:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = SystemClock()
    try:
        await connection.execute(
            """
            CREATE TABLE job_effect_probe(
                _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                owner_id TEXT NOT NULL
            )
            """
        )
        await connection.commit()
        await UserRepository(connection, clock).create_user("usr_1")
        repository = JobRunRepository(connection, clock)
        for job_id, job_type in jobs:
            await repository.create_durable_job(
                stream_name=STREAM_BY_TYPE[job_type],
                target_backend="redis",
                envelope=JobEnvelope(
                    job_id=job_id,
                    job_type=job_type,
                    user_id="usr_1",
                    payload={"private_text": f"private payload for {job_id}"},
                ),
                source_token_estimate=None,
                size_bucket=None,
            )
    finally:
        await connection.close()


def _start_process(
    context: multiprocessing.context.BaseContext,
    target,
    args: tuple[Any, ...],
    *,
    kwargs: dict[str, Any] | None = None,
):
    process = context.Process(target=target, args=args, kwargs=kwargs or {})
    process.start()
    return process


def _receive(result_queue, *, expected_status: str = "ok"):
    status, payload = result_queue.get(timeout=15)
    assert status == expected_status, payload
    return payload


def _join_success(process) -> None:
    process.join(timeout=15)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    assert process.exitcode == 0


async def _job_row(database_path: str, job_id: str) -> dict[str, Any]:
    connection = await open_connection(database_path)
    try:
        row = await JobRunRepository(connection, SystemClock()).get_job(job_id)
        assert row is not None
        return row
    finally:
        await connection.close()


async def _wait_for_heartbeats(
    database_path: str,
    job_id: str,
    *,
    initial_value: str,
    count: int,
) -> list[str]:
    observed: list[str] = []
    deadline = time.monotonic() + 5.0
    while len(observed) < count:
        row = await _job_row(database_path, job_id)
        value = str(row["last_heartbeat_at"])
        if value != initial_value and (not observed or value != observed[-1]):
            observed.append(value)
        if time.monotonic() >= deadline:
            raise AssertionError(f"Observed only {len(observed)} lease heartbeats")
        await asyncio.sleep(0.02)
    return observed


def _dispatch_in_process(
    context: multiprocessing.context.BaseContext,
    database_path: str,
    redis_url: str,
) -> dict[str, int]:
    results = context.Queue()
    process = _start_process(
        context,
        _dispatcher_process,
        (database_path, redis_url, results),
    )
    payload = _receive(results)
    _join_success(process)
    return payload


def _transition_in_process(
    context: multiprocessing.context.BaseContext,
    database_path: str,
    redis_url: str,
    stream_name: str,
    *,
    owner_id: str,
    action: str,
) -> ClaimedJob:
    results = context.Queue()
    process = _start_process(
        context,
        _transition_process,
        (
            database_path,
            redis_url,
            stream_name,
            owner_id,
            action,
            results,
        ),
    )
    claim = _deserialize_claim(_receive(results))
    _join_success(process)
    return claim


async def _wait_until_due(database_path: str, job_id: str) -> dict[str, Any]:
    row = await _job_row(database_path, job_id)
    due_at = datetime.fromisoformat(str(row["deferred_until"]))
    remaining = max(
        0.0,
        (due_at - datetime.now(tz=timezone.utc)).total_seconds(),
    )
    await asyncio.sleep(remaining + 0.03)
    return row


@pytest.mark.asyncio
async def test_real_redis_healthy_slow_job_renews_lease_across_processes(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "healthy-lease.db")
    await _initialize_jobs(
        database_path,
        [("job_healthy", JobType.RUN_EVALUATION)],
    )
    context = multiprocessing.get_context("spawn")
    assert _dispatch_in_process(context, database_path, real_redis_server.url) == {
        "claimed": 1,
        "published": 1,
        "delivery_failed": 0,
        "recovered_leases": 0,
    }

    release = context.Event()
    worker_results = context.Queue()
    worker = _start_process(
        context,
        _worker_process,
        (
            database_path,
            real_redis_server.url,
            EVALUATION_STREAM_NAME,
            "healthy-owner",
            release,
            worker_results,
        ),
        kwargs={"heartbeat": True},
    )
    claim_payload = _receive(worker_results, expected_status="claimed")
    claim = _deserialize_claim(claim_payload)
    row = await _job_row(database_path, "job_healthy")
    initial_heartbeat = str(row["last_heartbeat_at"])
    observed = await _wait_for_heartbeats(
        database_path,
        "job_healthy",
        initial_value=initial_heartbeat,
        count=3,
    )
    assert len(set(observed)) == 3

    contender_results = context.Queue()
    contender = _start_process(
        context,
        _contender_process,
        (
            database_path,
            real_redis_server.url,
            EVALUATION_STREAM_NAME,
            "contender-owner",
            contender_results,
        ),
    )
    assert _receive(contender_results) is False
    _join_success(contender)

    recovery_connection = await open_connection(database_path)
    try:
        assert (
            await JobRunRepository(
                recovery_connection, SystemClock()
            ).recover_expired_execution_leases()
            == 0
        )
    finally:
        await recovery_connection.close()

    release.set()
    completed_claim = _deserialize_claim(
        _receive(worker_results, expected_status="done")
    )
    _join_success(worker)
    assert completed_claim.execution_fence == claim.execution_fence
    row = await _job_row(database_path, "job_healthy")
    assert row["status"] == JobRunStatus.SUCCEEDED.value
    assert row["attempt_count"] == 1
    assert row["execution_fence"] == 1
    assert row["recovery_envelope_json"] is None

    verify = await open_connection(database_path)
    try:
        cursor = await verify.execute(
            "SELECT job_id, owner_id FROM job_effect_probe ORDER BY _rowid"
        )
        assert [dict(item) for item in await cursor.fetchall()] == [
            {"job_id": "job_healthy", "owner_id": "healthy-owner"}
        ]
    finally:
        await verify.close()
    groups = real_redis_server.client.xinfo_groups(EVALUATION_STREAM_NAME)
    assert len(groups) == 1
    assert int(groups[0]["pending"]) == 0
    assert real_redis_server.client.xlen(EVALUATION_STREAM_NAME) == 0


@pytest.mark.asyncio
async def test_real_redis_killed_owner_takeover_fences_stale_domain_effect(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "killed-owner.db")
    await _initialize_jobs(
        database_path,
        [("job_takeover", JobType.COMPACT_SUMMARIES)],
    )
    context = multiprocessing.get_context("spawn")
    first_dispatch = _dispatch_in_process(context, database_path, real_redis_server.url)
    assert first_dispatch["published"] == 1

    never_release = context.Event()
    owner_results = context.Queue()
    owner = _start_process(
        context,
        _worker_process,
        (
            database_path,
            real_redis_server.url,
            COMPACT_STREAM_NAME,
            "killed-owner",
            never_release,
            owner_results,
        ),
        kwargs={"heartbeat": False},
    )
    stale_claim_payload = _receive(owner_results, expected_status="claimed")
    stale_claim = _deserialize_claim(stale_claim_payload)
    owner.kill()
    owner.join(timeout=5)
    assert owner.exitcode is not None and owner.exitcode != 0

    expired_row = await _job_row(database_path, "job_takeover")
    lease_deadline = datetime.fromisoformat(
        str(expired_row["execution_lease_expires_at"])
    )
    remaining = max(
        0.0,
        (lease_deadline - datetime.now(tz=timezone.utc)).total_seconds(),
    )
    await asyncio.sleep(remaining + 0.05)

    takeover_dispatch = _dispatch_in_process(
        context, database_path, real_redis_server.url
    )
    assert takeover_dispatch["recovered_leases"] == 1
    assert takeover_dispatch["published"] == 1

    takeover_release = context.Event()
    takeover_results = context.Queue()
    takeover = _start_process(
        context,
        _worker_process,
        (
            database_path,
            real_redis_server.url,
            COMPACT_STREAM_NAME,
            "takeover-owner",
            takeover_release,
            takeover_results,
        ),
        kwargs={"heartbeat": True},
    )
    takeover_claim = _deserialize_claim(
        _receive(takeover_results, expected_status="claimed")
    )
    assert takeover_claim.execution_fence > stale_claim.execution_fence

    stale_results = context.Queue()
    stale = _start_process(
        context,
        _stale_effect_process,
        (database_path, stale_claim_payload, stale_results),
    )
    stale_outcome = _receive(stale_results)
    _join_success(stale)
    assert stale_outcome == {"effect_fenced": True, "terminalized": False}

    takeover_release.set()
    _receive(takeover_results, expected_status="done")
    _join_success(takeover)
    final_row = await _job_row(database_path, "job_takeover")
    assert final_row["status"] == JobRunStatus.SUCCEEDED.value
    assert final_row["attempt_count"] == 2
    assert final_row["dispatch_attempt_count"] == 2
    assert final_row["execution_fence"] == takeover_claim.execution_fence
    assert final_row["recovery_envelope_json"] is None

    verify = await open_connection(database_path)
    try:
        cursor = await verify.execute(
            "SELECT job_id, owner_id FROM job_effect_probe ORDER BY _rowid"
        )
        assert [dict(item) for item in await cursor.fetchall()] == [
            {"job_id": "job_takeover", "owner_id": "takeover-owner"}
        ]
    finally:
        await verify.close()
    groups = real_redis_server.client.xinfo_groups(COMPACT_STREAM_NAME)
    assert int(groups[0]["pending"]) == 0
    assert int(groups[0]["lag"]) == 0
    assert real_redis_server.client.xlen(COMPACT_STREAM_NAME) == 0


@pytest.mark.asyncio
async def test_real_redis_restart_recovers_every_registered_job_type(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "job-type-matrix.db")
    jobs = [(f"job_{job_type.value}", job_type) for job_type in JobType]
    await _initialize_jobs(database_path, jobs)
    context = multiprocessing.get_context("spawn")
    dispatched = _dispatch_in_process(context, database_path, real_redis_server.url)
    assert dispatched["claimed"] == len(JobType)
    assert dispatched["published"] == len(JobType)

    results = context.Queue()
    claimant = _start_process(
        context,
        _claim_all_process,
        (database_path, real_redis_server.url, results),
    )
    assert _receive(results) == len(JobType)
    _join_success(claimant)

    reopened = await open_connection(database_path)
    try:
        repository = JobRunRepository(reopened, SystemClock())
        for job_id, _job_type in jobs:
            row = await repository.get_job(job_id)
            assert row is not None
            assert row["status"] == JobRunStatus.SUCCEEDED.value
            assert row["attempt_count"] == 1
            assert row["recovery_envelope_json"] is None
            assert row["envelope_schema_version"] is None
    finally:
        await reopened.close()
    for stream_name in STREAM_BY_TYPE.values():
        assert real_redis_server.client.xlen(stream_name) == 0


@pytest.mark.asyncio
async def test_real_redis_retry_defer_and_dead_letter_survive_process_restarts(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "retry-defer-dead-letter.db")
    job_id = "job_retry_defer_dead_letter"
    await _initialize_jobs(
        database_path,
        [(job_id, JobType.RUN_EVALUATION)],
    )
    context = multiprocessing.get_context("spawn")

    assert (
        _dispatch_in_process(
            context,
            database_path,
            real_redis_server.url,
        )["published"]
        == 1
    )
    retry_claim = _transition_in_process(
        context,
        database_path,
        real_redis_server.url,
        EVALUATION_STREAM_NAME,
        owner_id="retry-owner",
        action="retry",
    )
    retry_row = await _wait_until_due(database_path, job_id)
    assert retry_claim.attempt_count == 1
    assert retry_row["status"] == JobRunStatus.RETRYING.value

    assert (
        _dispatch_in_process(
            context,
            database_path,
            real_redis_server.url,
        )["published"]
        == 1
    )
    defer_claim = _transition_in_process(
        context,
        database_path,
        real_redis_server.url,
        EVALUATION_STREAM_NAME,
        owner_id="defer-owner",
        action="defer",
    )
    deferred_row = await _wait_until_due(database_path, job_id)
    assert defer_claim.attempt_count == 2
    assert deferred_row["status"] == JobRunStatus.DEFERRED.value

    assert (
        _dispatch_in_process(
            context,
            database_path,
            real_redis_server.url,
        )["published"]
        == 1
    )
    dead_letter_claim = _transition_in_process(
        context,
        database_path,
        real_redis_server.url,
        EVALUATION_STREAM_NAME,
        owner_id="dead-letter-owner",
        action="dead_letter",
    )
    assert dead_letter_claim.attempt_count == 3

    final_row = await _job_row(database_path, job_id)
    dead_letter_raw = real_redis_server.client.lindex(
        f"{ATAGIA_QUEUE_PREFIX}dead_letter:{EVALUATION_STREAM_NAME}",
        0,
    )
    assert dead_letter_raw is not None
    wrapped_dead_letter = json.loads(dead_letter_raw)
    dead_letter = wrapped_dead_letter["payload"]
    assert final_row["status"] == JobRunStatus.DEAD_LETTERED.value
    assert final_row["attempt_count"] == 3
    assert final_row["execution_fence"] == 3
    assert final_row["error_class"] == "RuntimeError"
    assert final_row["error_message"] is None
    assert final_row["recovery_envelope_json"] is None
    assert dead_letter["error_class"] == "RuntimeError"
    assert "error" not in dead_letter
    assert "error_details" not in dead_letter
    assert "PRIVATE_PROVIDER_RETRY_TEXT" not in json.dumps(
        {"row": final_row, "dead_letter": dead_letter},
        sort_keys=True,
        default=str,
    )
    groups = real_redis_server.client.xinfo_groups(EVALUATION_STREAM_NAME)
    assert int(groups[0]["pending"]) == 0
    assert int(groups[0]["lag"]) == 0
    assert real_redis_server.client.xlen(EVALUATION_STREAM_NAME) == 0


@pytest.mark.asyncio
async def test_real_redis_lifecycle_cancellation_purges_unclaimed_notification(
    real_redis_server: RealRedisServer,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "lifecycle-cancellation.db")
    job_id = "job_cancel_before_claim"
    await _initialize_jobs(
        database_path,
        [(job_id, JobType.SYNC_GRAPH)],
    )
    context = multiprocessing.get_context("spawn")
    assert (
        _dispatch_in_process(
            context,
            database_path,
            real_redis_server.url,
        )["published"]
        == 1
    )
    assert real_redis_server.client.xlen(GRAPH_STREAM_NAME) == 1

    results = context.Queue()
    canceller = _start_process(
        context,
        _cancel_lifecycle_process,
        (database_path, real_redis_server.url, results),
    )
    assert _receive(results) == {"cancelled": 1, "purged": 1}
    _join_success(canceller)

    row = await _job_row(database_path, job_id)
    assert row["status"] == JobRunStatus.CANCELLED.value
    assert row["recovery_envelope_json"] is None
    assert row["dispatch_token"] is None
    assert row["execution_owner"] is None
    assert real_redis_server.client.xlen(GRAPH_STREAM_NAME) == 0
    assert real_redis_server.client.xinfo_groups(GRAPH_STREAM_NAME) == []
