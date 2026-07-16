"""Tests for durable worker-job run tracking."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy2
import sqlite3
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import (
    MigrationManager,
    close_connection,
    initialize_database,
    open_connection,
)
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    ClaimedJob,
    DurableJobNotification,
    JobEnvelope,
    JobRunStatus,
    JobType,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc))
    return connection, clock


async def _seed_scope(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
) -> None:
    await UserRepository(connection, clock).create_user("usr_1")
    await connection.execute(
        """
        INSERT INTO assistant_modes(
            id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "coding_debug",
            "Coding Debug",
            "hash_1",
            "{}",
            clock.now().isoformat(),
            clock.now().isoformat(),
        ),
    )
    await connection.commit()
    await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        None,
        "coding_debug",
        "Tracked chat",
    )


def _envelope(job_id: str, job_type: JobType, message_id: str) -> JobEnvelope:
    return JobEnvelope(
        job_id=job_id,
        job_type=job_type,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload={"message_id": message_id, "message_text": "private source"},
    )


async def _create(
    repository: JobRunRepository,
    *,
    envelope: JobEnvelope,
    stream_name: str,
    metadata: dict[str, int] | None = None,
) -> dict[str, object]:
    return await repository.create_durable_job(
        stream_name=stream_name,
        target_backend="inprocess",
        envelope=envelope,
        source_token_estimate=128,
        size_bucket="small",
        metadata=metadata,
    )


async def _dispatch_and_claim(
    repository: JobRunRepository,
    *,
    owner: str,
) -> ClaimedJob:
    rows = await repository.claim_dispatchable_jobs(
        target_backend="inprocess",
        limit=1,
        visibility_seconds=30,
    )
    assert len(rows) == 1
    row = rows[0]
    notification = DurableJobNotification(
        job_id=str(row["job_id"]),
        dispatch_token=str(row["dispatch_token"]),
        lifecycle_epoch=str(row["lifecycle_epoch"]),
        lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
    )
    claim = await repository.claim_notification(
        f"delivery_{row['job_id']}",
        notification,
        owner_id=owner,
        lease_seconds=30,
    )
    assert claim is not None
    return claim


class _ObserveImmediate:
    """Signal immediately before a connection waits for its writer lock."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.immediate_attempted = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
            self.immediate_attempted.set()
        return await self._connection.execute(sql, *args, **kwargs)


async def _run_after_writer_waits(
    *,
    blocker: aiosqlite.Connection,
    observed: _ObserveImmediate,
    clock: FrozenClock,
    operation: Coroutine[Any, Any, Any],
) -> Any:
    await blocker.execute("BEGIN IMMEDIATE")
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(observed.immediate_attempted.wait(), timeout=2.0)
        clock.advance(seconds=2)
        await blocker.commit()
        return await asyncio.wait_for(task, timeout=2.0)
    finally:
        if blocker.in_transaction:
            await blocker.rollback()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _file_connections(
    tmp_path: Path,
    filename: str,
) -> tuple[aiosqlite.Connection, aiosqlite.Connection, FrozenClock]:
    database_path = str(tmp_path / filename)
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc))
    await _seed_scope(connection, clock)
    blocker = await open_connection(database_path)
    return connection, blocker, clock


async def _claimed_file_job(
    tmp_path: Path,
    *,
    filename: str,
    job_id: str,
) -> tuple[
    aiosqlite.Connection,
    aiosqlite.Connection,
    FrozenClock,
    ClaimedJob,
]:
    connection, blocker, clock = await _file_connections(tmp_path, filename)
    repository = JobRunRepository(connection, clock)
    await _create(
        repository,
        envelope=_envelope(
            job_id,
            JobType.EXTRACT_MEMORY_CANDIDATES,
            f"msg_{job_id}",
        ),
        stream_name=EXTRACT_STREAM_NAME,
    )
    claim = await _dispatch_and_claim(repository, owner=f"worker_{job_id}")
    await connection.execute(
        "UPDATE worker_job_runs SET execution_lease_expires_at = ? WHERE job_id = ?",
        (
            (clock.now() + timedelta(seconds=1)).isoformat(),
            claim.envelope.job_id,
        ),
    )
    await connection.commit()
    return connection, blocker, clock, claim


@pytest.mark.asyncio
async def test_migration_0055_aborts_atomically_until_legacy_jobs_are_drained(
    tmp_path: Path,
) -> None:
    migration_manager = MigrationManager(MIGRATIONS_DIR)
    upgrade_migrations = tmp_path / "migrations-through-0055"
    upgrade_migrations.mkdir()
    migration_0055 = None
    for migration in migration_manager.discover():
        if migration.version <= 54:
            copy2(migration.path, upgrade_migrations / migration.path.name)
        elif migration.version == 55:
            migration_0055 = migration
    assert migration_0055 is not None

    database_path = str(tmp_path / "legacy-worker-upgrade.db")
    timestamp = "2026-07-13T10:00:00+00:00"
    connection = await initialize_database(database_path, upgrade_migrations)
    try:
        await connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES ('usr_legacy_jobs', NULL, ?, ?, NULL)
            """,
            (timestamp, timestamp),
        )
        await connection.executemany(
            """
            INSERT INTO worker_job_runs(
                job_id,
                stream_name,
                job_type,
                user_id,
                status,
                attempt_count,
                queued_at,
                metadata_json
            ) VALUES (?, 'atagia:legacy', 'extract_memory_candidates',
                      'usr_legacy_jobs', ?, ?, ?, ?)
            """,
            (
                ("job_legacy_queued", "queued", 1, timestamp, '{"marker":"queued"}'),
                ("job_legacy_running", "running", 2, timestamp, '{"marker":"running"}'),
                (
                    "job_legacy_retrying",
                    "retrying",
                    3,
                    timestamp,
                    '{"marker":"retrying"}',
                ),
                (
                    "job_legacy_succeeded",
                    "succeeded",
                    4,
                    timestamp,
                    '{"marker":"succeeded"}',
                ),
            ),
        )
        await connection.commit()
        before_rows = [
            dict(row)
            for row in await (
                await connection.execute(
                    """
                    SELECT _rowid, job_id, status, attempt_count, metadata_json
                    FROM worker_job_runs
                    ORDER BY _rowid
                    """
                )
            ).fetchall()
        ]
    finally:
        await close_connection(connection)

    copy2(migration_0055.path, upgrade_migrations / migration_0055.path.name)
    failed_upgrade = await open_connection(database_path)
    try:
        with pytest.raises(
            sqlite3.IntegrityError,
            match=r"migration 0055 requires all legacy worker_job_runs to be terminal",
        ):
            await MigrationManager(upgrade_migrations).apply_all(failed_upgrade)

        versions = await MigrationManager(upgrade_migrations).applied_versions(
            failed_upgrade
        )
        assert max(versions) == 54
        assert 55 not in versions
        assert (
            await (
                await failed_upgrade.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'user_lifecycles'"
                )
            ).fetchone()
            is None
        )
        columns = {
            str(row["name"])
            for row in await (
                await failed_upgrade.execute("PRAGMA table_info(worker_job_runs)")
            ).fetchall()
        }
        assert "target_backend" not in columns
        assert "lifecycle_epoch" not in columns
        assert [
            dict(row)
            for row in await (
                await failed_upgrade.execute(
                    """
                    SELECT _rowid, job_id, status, attempt_count, metadata_json
                    FROM worker_job_runs
                    ORDER BY _rowid
                    """
                )
            ).fetchall()
        ] == before_rows
        foreign_keys = await (
            await failed_upgrade.execute("PRAGMA foreign_keys")
        ).fetchone()
        assert foreign_keys is not None
        assert int(foreign_keys[0]) == 1
    finally:
        await close_connection(failed_upgrade)

    drained = await open_connection(database_path)
    try:
        await drained.execute(
            """
            UPDATE worker_job_runs
            SET status = 'failed', finished_at = ?
            WHERE status IN ('queued', 'running', 'retrying')
            """,
            (timestamp,),
        )
        await drained.commit()
        applied = await MigrationManager(upgrade_migrations).apply_all(drained)
        assert [migration.version for migration in applied] == [55]
    finally:
        await close_connection(drained)

    reopened = await open_connection(database_path)
    try:
        versions = await MigrationManager(upgrade_migrations).applied_versions(reopened)
        assert max(versions) == 55
        lifecycle = await (
            await reopened.execute(
                """
                SELECT lifecycle_epoch, lifecycle_cleanup_key, state
                FROM user_lifecycles
                WHERE user_id = 'usr_legacy_jobs'
                """
            )
        ).fetchone()
        assert lifecycle is not None
        assert lifecycle["state"] == "active"
        rows = await (
            await reopened.execute(
                """
                SELECT _rowid, job_id, status, attempt_count, metadata_json,
                       target_backend, lifecycle_epoch, lifecycle_cleanup_key,
                       envelope_schema_version, recovery_envelope_json
                FROM worker_job_runs
                ORDER BY _rowid
                """
            )
        ).fetchall()
        assert [int(row["_rowid"]) for row in rows] == [
            int(row["_rowid"]) for row in before_rows
        ]
        assert [str(row["job_id"]) for row in rows] == [
            "job_legacy_queued",
            "job_legacy_running",
            "job_legacy_retrying",
            "job_legacy_succeeded",
        ]
        assert [str(row["status"]) for row in rows] == [
            "failed",
            "failed",
            "failed",
            "succeeded",
        ]
        assert [int(row["attempt_count"]) for row in rows] == [1, 2, 3, 4]
        assert [str(row["metadata_json"]) for row in rows] == [
            str(row["metadata_json"]) for row in before_rows
        ]
        assert {str(row["target_backend"]) for row in rows} == {"legacy_terminal"}
        assert {str(row["lifecycle_epoch"]) for row in rows} == {
            str(lifecycle["lifecycle_epoch"])
        }
        assert {str(row["lifecycle_cleanup_key"]) for row in rows} == {
            str(lifecycle["lifecycle_cleanup_key"])
        }
        assert all(row["envelope_schema_version"] is None for row in rows)
        assert all(row["recovery_envelope_json"] is None for row in rows)
        assert (
            await (await reopened.execute("PRAGMA foreign_key_check")).fetchall() == []
        )
    finally:
        await close_connection(reopened)


@pytest.mark.asyncio
async def test_job_run_repository_tracks_fenced_progress_and_retries() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)
        extract_envelope = _envelope(
            "job_extract_msg_1",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_1",
        )
        extract = await _create(
            repository,
            envelope=extract_envelope,
            stream_name=EXTRACT_STREAM_NAME,
            metadata={"message_count": 1},
        )
        assert extract["status"] == JobRunStatus.QUEUED.value
        assert extract["recovery_envelope_json"] == extract_envelope.model_dump(
            mode="json"
        )

        extract_claim = await _dispatch_and_claim(repository, owner="extract-worker")
        clock.advance(seconds=2)
        assert await repository.finish_claim(
            extract_claim,
            status=JobRunStatus.SUCCEEDED,
            metadata={"memory_count": 2},
        )

        contract_envelope = _envelope(
            "job_contract_msg_1",
            JobType.PROJECT_CONTRACT,
            "msg_1",
        )
        await _create(
            repository,
            envelope=contract_envelope,
            stream_name=CONTRACT_STREAM_NAME,
        )

        progress = await repository.source_message_progress(
            user_id="usr_1",
            conversation_id="cnv_1",
            window_start="2026-05-02T12:00:00+00:00",
        )
        assert progress == {
            "tracked_source_messages": 1,
            "processed_source_messages": 0,
            "pending_source_messages": 1,
        }
        counts = await repository.status_counts(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert {(row["status"], row["job_type"]): row["count"] for row in counts} == {
            (JobRunStatus.QUEUED.value, JobType.PROJECT_CONTRACT.value): 1,
            (JobRunStatus.SUCCEEDED.value, JobType.EXTRACT_MEMORY_CANDIDATES.value): 1,
        }

        completed = await repository.get_job(extract_envelope.job_id)
        assert completed is not None
        assert completed["duration_ms"] is not None
        assert completed["metadata_json"]["message_count"] == 1
        assert completed["metadata_json"]["memory_count"] == 2
        assert completed["recovery_envelope_json"] is None
        assert completed["envelope_schema_version"] is None

        first_claim = await _dispatch_and_claim(repository, owner="contract-worker-1")
        first_due = clock.now() + timedelta(seconds=60)
        assert await repository.release_claim_for_retry(
            first_claim,
            error_class="TransientLLMError",
            error_message="provider unavailable",
            deferred_until=first_due.isoformat(),
        )
        deferred = await repository.get_job(contract_envelope.job_id)
        assert deferred is not None
        assert deferred["status"] == JobRunStatus.DEFERRED.value
        assert deferred["transient_defer_count"] == 1
        assert deferred["first_deferred_at"] == clock.now().isoformat()
        assert deferred["last_deferred_at"] == clock.now().isoformat()

        assert (
            await repository.claim_dispatchable_jobs(
                target_backend="inprocess",
                limit=1,
                visibility_seconds=30,
            )
            == []
        )
        clock.advance(seconds=60)
        second_claim = await _dispatch_and_claim(repository, owner="contract-worker-2")
        second_due = clock.now() + timedelta(seconds=60)
        assert await repository.release_claim_for_retry(
            second_claim,
            error_class="TransientLLMError",
            error_message="provider unavailable",
            deferred_until=second_due.isoformat(),
        )
        second_deferred = await repository.get_job(contract_envelope.job_id)
        assert second_deferred is not None
        assert second_deferred["transient_defer_count"] == 2
        assert second_deferred["first_deferred_at"] == deferred["first_deferred_at"]
        assert second_deferred["last_deferred_at"] == clock.now().isoformat()

        clock.advance(seconds=60)
        final_claim = await _dispatch_and_claim(repository, owner="contract-worker-3")
        assert await repository.finish_claim(
            final_claim,
            status=JobRunStatus.SUCCEEDED,
        )
        succeeded = await repository.get_job(contract_envelope.job_id)
        assert succeeded is not None
        assert succeeded["status"] == JobRunStatus.SUCCEEDED.value
        assert succeeded["attempt_count"] == 3
        assert succeeded["error_class"] is None
        assert succeeded["error_message"] is None
        assert succeeded["deferred_until"] is None
        assert succeeded["transient_defer_count"] == 2
        assert succeeded["recovery_envelope_json"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_duplicate_job_id_requires_the_exact_same_envelope() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)
        envelope = _envelope(
            "job_same",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_1",
        )
        first = await _create(
            repository,
            envelope=envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )
        duplicate = await _create(
            repository,
            envelope=envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )
        assert duplicate["job_id"] == first["job_id"]

        conflicting = envelope.model_copy(update={"payload": {"message_text": "other"}})
        with pytest.raises(ValueError, match="different envelope"):
            await _create(
                repository,
                envelope=conflicting,
                stream_name=EXTRACT_STREAM_NAME,
            )
    finally:
        await connection.close()


@pytest.mark.parametrize("status", list(JobRunStatus))
@pytest.mark.asyncio
async def test_source_message_dedupe_covers_every_nonfailed_logical_state(
    status: JobRunStatus,
) -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)
        envelope = _envelope(
            f"job_dedupe_{status.value}",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_dedupe",
        )
        await _create(
            repository,
            envelope=envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )
        await connection.execute(
            "UPDATE worker_job_runs SET status = ? WHERE job_id = ?",
            (status.value, envelope.job_id),
        )
        await connection.commit()

        exists = await repository.source_message_job_exists(
            user_id="usr_1",
            source_message_id="msg_dedupe",
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
        )
        assert exists is (
            status
            in {
                JobRunStatus.QUEUED,
                JobRunStatus.AWAITING_CLAIM,
                JobRunStatus.RUNNING,
                JobRunStatus.RETRYING,
                JobRunStatus.DEFERRED,
                JobRunStatus.SUCCEEDED,
                JobRunStatus.SKIPPED,
            }
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_target_backend_guard_rejects_only_stranding_nonterminal_jobs() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)
        envelope = _envelope(
            "job_backend_guard",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_backend_guard",
        )
        await _create(
            repository,
            envelope=envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )

        assert await repository.nonterminal_count() == 1
        await repository.assert_target_backend_compatible("inprocess")
        with pytest.raises(
            RuntimeError,
            match=r"Cannot switch durable job backend.*inprocess=1",
        ):
            await repository.assert_target_backend_compatible("redis")

        claim = await _dispatch_and_claim(repository, owner="backend-guard-worker")
        assert await repository.finish_claim(
            claim,
            status=JobRunStatus.SUCCEEDED,
        )
        assert await repository.nonterminal_count() == 0
        await repository.assert_target_backend_compatible("redis")
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_durable_child_rechecks_parent_lease_after_writer_wait(
    tmp_path: Path,
) -> None:
    connection, blocker, clock = await _file_connections(
        tmp_path,
        "create-child-lock-first.db",
    )
    try:
        repository = JobRunRepository(connection, clock)
        parent_envelope = _envelope(
            "job_parent_lock_first",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_parent_lock_first",
        )
        await _create(
            repository,
            envelope=parent_envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )
        parent_claim = await _dispatch_and_claim(
            repository,
            owner="parent-lock-first-worker",
        )
        await connection.execute(
            "UPDATE worker_job_runs SET execution_lease_expires_at = ? WHERE job_id = ?",
            (
                (clock.now() + timedelta(seconds=1)).isoformat(),
                parent_envelope.job_id,
            ),
        )
        await connection.commit()
        child_envelope = _envelope(
            "job_child_lock_first",
            JobType.PROJECT_CONTRACT,
            "msg_child_lock_first",
        ).model_copy(update={"parent_job_id": parent_envelope.job_id})
        observed = _ObserveImmediate(connection)

        with pytest.raises(RuntimeError, match="Cannot create durable job"):
            await _run_after_writer_waits(
                blocker=blocker,
                observed=observed,
                clock=clock,
                operation=JobRunRepository(observed, clock).create_durable_job(
                    stream_name=CONTRACT_STREAM_NAME,
                    target_backend="inprocess",
                    envelope=child_envelope,
                    source_token_estimate=128,
                    size_bucket="small",
                    parent_claim=parent_claim,
                ),
            )

        assert await repository.get_job(child_envelope.job_id) is None
    finally:
        await blocker.close()
        await connection.close()


@pytest.mark.asyncio
async def test_claim_notification_rechecks_visibility_after_writer_wait(
    tmp_path: Path,
) -> None:
    connection, blocker, clock = await _file_connections(
        tmp_path,
        "claim-notification-lock-first.db",
    )
    try:
        repository = JobRunRepository(connection, clock)
        envelope = _envelope(
            "job_notification_lock_first",
            JobType.EXTRACT_MEMORY_CANDIDATES,
            "msg_notification_lock_first",
        )
        await _create(
            repository,
            envelope=envelope,
            stream_name=EXTRACT_STREAM_NAME,
        )
        rows = await repository.claim_dispatchable_jobs(
            target_backend="inprocess",
            limit=1,
            visibility_seconds=30,
        )
        assert len(rows) == 1
        row = rows[0]
        await connection.execute(
            "UPDATE worker_job_runs SET dispatch_visibility_deadline = ? WHERE job_id = ?",
            (
                (clock.now() + timedelta(seconds=1)).isoformat(),
                envelope.job_id,
            ),
        )
        await connection.commit()
        notification = DurableJobNotification(
            job_id=envelope.job_id,
            dispatch_token=str(row["dispatch_token"]),
            lifecycle_epoch=str(row["lifecycle_epoch"]),
            lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
        )
        observed = _ObserveImmediate(connection)

        claim = await _run_after_writer_waits(
            blocker=blocker,
            observed=observed,
            clock=clock,
            operation=JobRunRepository(observed, clock).claim_notification(
                "delivery_notification_lock_first",
                notification,
                owner_id="notification-lock-first-worker",
                lease_seconds=30,
            ),
        )

        assert claim is None
        stored = await repository.get_job(envelope.job_id)
        assert stored is not None
        assert stored["status"] == JobRunStatus.AWAITING_CLAIM.value
        assert stored["attempt_count"] == 0
    finally:
        await blocker.close()
        await connection.close()


@pytest.mark.asyncio
async def test_heartbeat_claim_rechecks_execution_lease_after_writer_wait(
    tmp_path: Path,
) -> None:
    connection, blocker, clock, claim = await _claimed_file_job(
        tmp_path,
        filename="heartbeat-lock-first.db",
        job_id="job_heartbeat_lock_first",
    )
    try:
        observed = _ObserveImmediate(connection)
        heartbeat_succeeded = await _run_after_writer_waits(
            blocker=blocker,
            observed=observed,
            clock=clock,
            operation=JobRunRepository(observed, clock).heartbeat_claim(
                claim,
                lease_seconds=30,
            ),
        )

        assert not heartbeat_succeeded
        stored = await JobRunRepository(connection, clock).get_job(
            claim.envelope.job_id
        )
        assert stored is not None
        assert stored["status"] == JobRunStatus.RUNNING.value
        assert stored["execution_owner"] == claim.owner_id
    finally:
        await blocker.close()
        await connection.close()


@pytest.mark.asyncio
async def test_finish_claim_rechecks_execution_lease_after_writer_wait(
    tmp_path: Path,
) -> None:
    connection, blocker, clock, claim = await _claimed_file_job(
        tmp_path,
        filename="finish-lock-first.db",
        job_id="job_finish_lock_first",
    )
    try:
        observed = _ObserveImmediate(connection)
        finish_succeeded = await _run_after_writer_waits(
            blocker=blocker,
            observed=observed,
            clock=clock,
            operation=JobRunRepository(observed, clock).finish_claim(
                claim,
                status=JobRunStatus.SUCCEEDED,
            ),
        )

        assert not finish_succeeded
        stored = await JobRunRepository(connection, clock).get_job(
            claim.envelope.job_id
        )
        assert stored is not None
        assert stored["status"] == JobRunStatus.RUNNING.value
        assert stored["finished_at"] is None
    finally:
        await blocker.close()
        await connection.close()


@pytest.mark.asyncio
async def test_release_claim_for_retry_rechecks_execution_lease_after_writer_wait(
    tmp_path: Path,
) -> None:
    connection, blocker, clock, claim = await _claimed_file_job(
        tmp_path,
        filename="retry-lock-first.db",
        job_id="job_retry_lock_first",
    )
    try:
        observed = _ObserveImmediate(connection)
        release_succeeded = await _run_after_writer_waits(
            blocker=blocker,
            observed=observed,
            clock=clock,
            operation=JobRunRepository(observed, clock).release_claim_for_retry(
                claim,
                error_class="TransientProviderError",
                error_message="not persisted",
            ),
        )

        assert not release_succeeded
        stored = await JobRunRepository(connection, clock).get_job(
            claim.envelope.job_id
        )
        assert stored is not None
        assert stored["status"] == JobRunStatus.RUNNING.value
        assert stored["error_class"] is None
    finally:
        await blocker.close()
        await connection.close()
