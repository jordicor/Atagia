"""Concurrency tests for SQLite worker effects and fenced child creation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import (
    ConversationRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_jobs import (
    DurableJobNotification,
    JobEnvelope,
    JobRunStatus,
    JobType,
    StreamMessage,
)
from atagia.services.job_execution_context import (
    StaleParentJobFenceError,
    bind_job_claim,
    reset_job_claim,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.worker_effect_fence import (
    StaleJobEffectFenceError,
    WorkerEffectFence,
)
from atagia.services.worker_job_lease import WorkerJobLease

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)


def _envelope(
    job_id: str,
    *,
    parent_job_id: str | None = None,
    conversation_id: str | None = None,
) -> JobEnvelope:
    return JobEnvelope(
        job_id=job_id,
        job_type=JobType.RUN_EVALUATION,
        user_id="usr_1",
        parent_job_id=parent_job_id,
        conversation_id=conversation_id,
        payload={"metrics": ["system"]},
    )


async def _create_parent(repository: JobRunRepository) -> JobEnvelope:
    envelope = _envelope("job_parent")
    await repository.create_durable_job(
        stream_name="atagia:evaluate",
        target_backend="inprocess",
        envelope=envelope,
        source_token_estimate=None,
        size_bucket=None,
    )
    return envelope


async def _claim_parent(
    repository: JobRunRepository,
    *,
    owner_id: str,
) -> object:
    return await _claim_job(repository, job_id="job_parent", owner_id=owner_id)


async def _claim_job(
    repository: JobRunRepository,
    *,
    job_id: str,
    owner_id: str,
) -> object:
    rows = await repository.claim_dispatchable_jobs(
        target_backend="inprocess",
        limit=20,
        visibility_seconds=30,
    )
    row = next(row for row in rows if row["job_id"] == job_id)
    notification = DurableJobNotification(
        job_id=job_id,
        dispatch_token=str(row["dispatch_token"]),
        lifecycle_epoch=str(row["lifecycle_epoch"]),
        lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
    )
    claim = await repository.claim_notification(
        f"delivery-{owner_id}",
        notification,
        owner_id=owner_id,
        lease_seconds=10,
    )
    assert claim is not None
    return claim


async def _create_conversation(
    connection,
    clock: FrozenClock,
    conversation_id: str,
) -> None:
    await connection.execute(
        """
        INSERT OR IGNORE INTO assistant_modes(
            id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
        ) VALUES ('general_qa', 'General QA', 'hash', '{}', ?, ?)
        """,
        (clock.now().isoformat(), clock.now().isoformat()),
    )
    await connection.commit()
    await ConversationRepository(connection, clock).create_conversation(
        conversation_id,
        "usr_1",
        None,
        "general_qa",
        conversation_id,
    )


@pytest.mark.asyncio
async def test_domain_statement_is_atomic_with_execution_and_lifecycle_fence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "fenced-effects.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        jobs = JobRunRepository(job_connection, clock)
        await _create_parent(jobs)
        old_claim = await _claim_parent(jobs, owner_id="old-owner")
        effect_fence = WorkerEffectFence(domain_connection, clock)

        async with effect_fence.activate(old_claim):
            await WorkspaceRepository(domain_connection, clock).create_workspace(
                "wrk_current",
                "usr_1",
                "Current owner effect",
            )

        clock.advance(seconds=11)
        assert await jobs.recover_expired_execution_leases() == 1
        new_claim = await _claim_parent(jobs, owner_id="new-owner")
        assert new_claim.execution_fence > old_claim.execution_fence

        with pytest.raises(StaleJobEffectFenceError):
            async with effect_fence.activate(old_claim):
                await WorkspaceRepository(domain_connection, clock).create_workspace(
                    "wrk_stale",
                    "usr_1",
                    "Stale owner effect",
                )
        assert (
            await WorkspaceRepository(domain_connection, clock).get_workspace(
                "wrk_stale",
                "usr_1",
            )
            is None
        )

        await job_connection.execute(
            "UPDATE user_lifecycles SET state = 'erasing' WHERE user_id = ?",
            ("usr_1",),
        )
        await job_connection.commit()
        with pytest.raises(StaleJobEffectFenceError):
            async with effect_fence.activate(new_claim):
                await WorkspaceRepository(domain_connection, clock).create_workspace(
                    "wrk_revoked",
                    "usr_1",
                    "Revoked lifecycle effect",
                )
        assert (
            await WorkspaceRepository(domain_connection, clock).get_workspace(
                "wrk_revoked",
                "usr_1",
            )
            is None
        )
    finally:
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_paused_job_loses_every_fence_after_derivation_revision_bump(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "derivation-revision-fence.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        jobs = JobRunRepository(job_connection, clock)
        await _create_parent(jobs)
        claim = await _claim_parent(jobs, owner_id="paused-owner")
        lifecycle_repository = UserLifecycleRepository(job_connection, clock)
        identity = await lifecycle_repository.get_active_identity("usr_1")
        assert identity is not None

        with pytest.raises(StaleJobEffectFenceError):
            async with WorkerEffectFence(domain_connection, clock).activate(claim):
                bumped_revision = await lifecycle_repository.bump_derivation_revision(
                    "usr_1",
                    expected_lifecycle_epoch=identity.lifecycle_epoch,
                )
                assert bumped_revision == claim.derivation_revision + 1
                await WorkspaceRepository(
                    domain_connection,
                    clock,
                ).create_workspace(
                    "wrk_stale_revision",
                    "usr_1",
                    "Stale derivation effect",
                )

        assert not await jobs.claim_is_current(claim)
        assert not await jobs.heartbeat_claim(claim, lease_seconds=10)
        assert not await jobs.finish_claim(
            claim,
            status=JobRunStatus.SUCCEEDED,
        )
        assert not await jobs.release_claim_for_retry(
            claim,
            error_class="RevisionChanged",
            error_message="canonical source changed",
        )
        assert (
            await WorkspaceRepository(domain_connection, clock).get_workspace(
                "wrk_stale_revision",
                "usr_1",
            )
            is None
        )
    finally:
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_awaiting_claim_job_keeps_enqueue_revision_and_is_cancelled() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        repository = JobRunRepository(connection, clock)
        envelope = await _create_parent(repository)
        rows = await repository.claim_dispatchable_jobs(
            target_backend="inprocess",
            limit=1,
            visibility_seconds=30,
        )
        assert len(rows) == 1
        awaiting = rows[0]
        enqueue_revision = int(awaiting["derivation_revision"])
        identity = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity("usr_1")
        assert identity is not None
        bumped_revision = await UserLifecycleRepository(
            connection,
            clock,
        ).bump_derivation_revision(
            "usr_1",
            expected_lifecycle_epoch=identity.lifecycle_epoch,
        )
        assert bumped_revision == enqueue_revision + 1

        claim = await repository.claim_notification(
            "delivery-stale-enqueue",
            DurableJobNotification(
                job_id=envelope.job_id,
                dispatch_token=str(awaiting["dispatch_token"]),
                lifecycle_epoch=str(awaiting["lifecycle_epoch"]),
                lifecycle_cleanup_key=str(awaiting["lifecycle_cleanup_key"]),
            ),
            owner_id="must-not-own",
            lease_seconds=30,
        )

        assert claim is None
        cancelled = await repository.get_job(envelope.job_id)
        assert cancelled is not None
        assert cancelled["status"] == JobRunStatus.CANCELLED.value
        assert cancelled["derivation_revision"] == enqueue_revision
        assert cancelled["execution_owner"] is None
        assert cancelled["recovery_envelope_json"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_child_creation_requires_the_current_parent_fence() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        repository = JobRunRepository(connection, clock)
        await _create_parent(repository)
        old_claim = await _claim_parent(repository, owner_id="old-owner")
        tracking = JobTrackingService(connection, clock, workers_enabled=False)

        token = bind_job_claim(old_claim)
        try:
            child = _envelope("job_child_current", parent_job_id="job_parent")
            await tracking.enqueue_job(backend, "atagia:evaluate", child)
            with pytest.raises(StaleParentJobFenceError, match="must capture"):
                await tracking.enqueue_job(
                    backend,
                    "atagia:evaluate",
                    _envelope("job_child_missing_parent"),
                )
        finally:
            reset_job_claim(token)
        assert await repository.get_job("job_child_current") is not None

        lifecycle_repository = UserLifecycleRepository(connection, clock)
        identity = await lifecycle_repository.get_active_identity("usr_1")
        assert identity is not None
        bumped_revision = await lifecycle_repository.bump_derivation_revision(
            "usr_1",
            expected_lifecycle_epoch=identity.lifecycle_epoch,
        )
        assert bumped_revision == old_claim.derivation_revision + 1
        stale_revision_token = bind_job_claim(old_claim)
        try:
            with pytest.raises(StaleParentJobFenceError, match="no longer current"):
                await tracking.enqueue_job(
                    backend,
                    "atagia:evaluate",
                    _envelope(
                        "job_child_stale_revision",
                        parent_job_id="job_parent",
                    ),
                )
        finally:
            reset_job_claim(stale_revision_token)
        assert await repository.get_job("job_child_stale_revision") is None

        assert await repository.cancel_jobs_with_stale_derivation() == 2
        cancelled_parent = await repository.get_job("job_parent")
        cancelled_child = await repository.get_job("job_child_current")
        assert cancelled_parent is not None
        assert cancelled_child is not None
        assert cancelled_parent["status"] == JobRunStatus.CANCELLED.value
        assert cancelled_child["status"] == JobRunStatus.CANCELLED.value

        stale_token = bind_job_claim(old_claim)
        try:
            with pytest.raises(StaleParentJobFenceError, match="no longer current"):
                await tracking.enqueue_job(
                    backend,
                    "atagia:evaluate",
                    _envelope("job_child_stale", parent_job_id="job_parent"),
                )
        finally:
            reset_job_claim(stale_token)
        assert await repository.get_job("job_child_stale") is None
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_retried_and_running_jobs_never_adopt_new_derivation_revision(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "derivation-revision-reclaim.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        await _create_conversation(job_connection, clock, "cnv_paused")
        await _create_conversation(job_connection, clock, "cnv_retry")
        jobs = JobRunRepository(job_connection, clock)

        paused_envelope = _envelope(
            "job_paused_chat",
            conversation_id="cnv_paused",
        )
        await jobs.create_durable_job(
            stream_name="atagia:evaluate",
            target_backend="inprocess",
            envelope=paused_envelope,
            source_token_estimate=None,
            size_bucket=None,
        )
        paused_claim = await _claim_job(
            jobs,
            job_id=paused_envelope.job_id,
            owner_id="paused-chat-owner",
        )

        retry_envelope = _envelope(
            "job_retry_chat",
            conversation_id="cnv_retry",
        )
        await jobs.create_durable_job(
            stream_name="atagia:evaluate",
            target_backend="inprocess",
            envelope=retry_envelope,
            source_token_estimate=None,
            size_bucket=None,
        )
        first_retry_claim = await _claim_job(
            jobs,
            job_id=retry_envelope.job_id,
            owner_id="retry-chat-owner-1",
        )
        assert await jobs.release_claim_for_retry(
            first_retry_claim,
            error_class="CanonicalReevaluation",
            error_message="retry from canonical state",
            deferred_until=clock.now().isoformat(),
            is_transient_defer=False,
        )

        lifecycle_repository = UserLifecycleRepository(job_connection, clock)
        identity = await lifecycle_repository.get_active_identity("usr_1")
        assert identity is not None
        bumped_revision = await lifecycle_repository.bump_derivation_revision(
            "usr_1",
            expected_lifecycle_epoch=identity.lifecycle_epoch,
        )
        assert bumped_revision == paused_claim.derivation_revision + 1
        assert not await jobs.claim_is_current(paused_claim)

        assert await jobs.cancel_jobs_with_stale_derivation() == 2
        cancelled_paused = await jobs.get_job(paused_envelope.job_id)
        cancelled_retry = await jobs.get_job(retry_envelope.job_id)
        assert cancelled_paused is not None
        assert cancelled_retry is not None
        assert cancelled_paused["status"] == JobRunStatus.CANCELLED.value
        assert cancelled_retry["status"] == JobRunStatus.CANCELLED.value
        assert (
            cancelled_retry["derivation_revision"]
            == first_retry_claim.derivation_revision
        )

        fresh_envelope = _envelope(
            "job_fresh_chat",
            conversation_id="cnv_retry",
        )
        await jobs.create_durable_job(
            stream_name="atagia:evaluate",
            target_backend="inprocess",
            envelope=fresh_envelope,
            source_token_estimate=None,
            size_bucket=None,
        )
        fresh_claim = await _claim_job(
            jobs,
            job_id=fresh_envelope.job_id,
            owner_id="fresh-chat-owner",
        )
        assert fresh_claim.derivation_revision == bumped_revision

        async with WorkerEffectFence(domain_connection, clock).activate(fresh_claim):
            await WorkspaceRepository(domain_connection, clock).create_workspace(
                "wrk_fresh_revision",
                "usr_1",
                "Fresh derivation effect",
            )
        assert await jobs.finish_claim(
            fresh_claim,
            status=JobRunStatus.SUCCEEDED,
        )
        assert (
            await WorkspaceRepository(domain_connection, clock).get_workspace(
                "wrk_fresh_revision",
                "usr_1",
            )
            is not None
        )
        stored = await jobs.get_job(fresh_envelope.job_id)
        assert stored is not None
        assert stored["status"] == JobRunStatus.SUCCEEDED.value
        assert stored["derivation_revision"] == bumped_revision
    finally:
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_child_enqueue_uses_domain_transaction_without_cross_connection_deadlock(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "child-enqueue-lock-order.db")
    seed_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    parent_job_connection = await open_connection(database_path)
    competing_job_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    competing_task: asyncio.Task[None] | None = None
    try:
        await competing_job_connection.execute("PRAGMA busy_timeout = 1000")
        await UserRepository(seed_connection, clock).create_user("usr_1")
        parent_repository = JobRunRepository(parent_job_connection, clock)
        await _create_parent(parent_repository)
        parent_claim = await _claim_parent(
            parent_repository,
            owner_id="parent-owner",
        )
        parent_tracking = JobTrackingService(
            parent_job_connection,
            clock,
            workers_enabled=False,
            child_job_connection=domain_connection,
        )
        competing_tracking = JobTrackingService(
            competing_job_connection,
            clock,
            workers_enabled=False,
        )
        competing_entered_repository = asyncio.Event()
        original_create = competing_tracking._repository.create_durable_job

        async def observed_competing_create(**kwargs):
            competing_entered_repository.set()
            return await original_create(**kwargs)

        competing_tracking._repository.create_durable_job = observed_competing_create

        async with WorkerEffectFence(domain_connection, clock).activate(parent_claim):
            await domain_connection.execute("BEGIN IMMEDIATE")
            timestamp = clock.now().isoformat()
            await domain_connection.execute(
                """
                INSERT INTO workspaces(
                    id, user_id, name, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, '{}', ?, ?)
                """,
                (
                    "wrk_atomic_child",
                    "usr_1",
                    "Atomic child effect",
                    timestamp,
                    timestamp,
                ),
            )

            competing_task = asyncio.create_task(
                competing_tracking.enqueue_job(
                    backend,
                    "atagia:evaluate",
                    _envelope("job_competing_root"),
                    dispatch=False,
                )
            )
            await competing_entered_repository.wait()
            await asyncio.sleep(0)
            assert not competing_task.done()

            token = bind_job_claim(parent_claim)
            try:
                refresh_generation = await asyncio.wait_for(
                    parent_tracking.reserve_icp_refresh_generation("usr_1"),
                    timeout=1.0,
                )
                await asyncio.wait_for(
                    parent_tracking.enqueue_job(
                        backend,
                        "atagia:evaluate",
                        _envelope(
                            "job_atomic_child",
                            parent_job_id="job_parent",
                        ),
                        dispatch=False,
                    ),
                    timeout=1.0,
                )
            finally:
                reset_job_claim(token)
            assert refresh_generation >= 1

        await asyncio.wait_for(competing_task, timeout=2.0)
        competing_task = None
        verifier = JobRunRepository(seed_connection, clock)
        assert await verifier.get_job("job_atomic_child") is not None
        assert await verifier.get_job("job_competing_root") is not None
        assert (
            await WorkspaceRepository(seed_connection, clock).get_workspace(
                "wrk_atomic_child",
                "usr_1",
            )
            is not None
        )
    finally:
        if competing_task is not None:
            competing_task.cancel()
            await asyncio.gather(competing_task, return_exceptions=True)
        await backend.close()
        await competing_job_connection.close()
        await parent_job_connection.close()
        await domain_connection.close()
        await seed_connection.close()


class _PausingDiagnosticBackend(InProcessBackend):
    def __init__(self) -> None:
        super().__init__()
        self.publish_entered = asyncio.Event()
        self.resume_publish = asyncio.Event()

    async def publish_lifecycle_diagnostic(self, *args, **kwargs):
        self.publish_entered.set()
        await self.resume_publish.wait()
        return await super().publish_lifecycle_diagnostic(*args, **kwargs)


@pytest.mark.asyncio
async def test_paused_dead_letter_linearizes_before_revocation_and_is_purged(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "dead-letter-revocation.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    revocation_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    backend = _PausingDiagnosticBackend()
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        lifecycle = await UserLifecycleRepository(
            job_connection,
            clock,
        ).get_active_identity("usr_1")
        assert lifecycle is not None
        await _create_parent(JobRunRepository(job_connection, clock))
        claim = await _claim_parent(
            JobRunRepository(job_connection, clock),
            owner_id="paused-owner",
        )
        await backend.prepare_lifecycle_mirror(
            lifecycle.lifecycle_cleanup_key,
            lifecycle.lifecycle_epoch,
            "nonce",
        )
        assert await backend.activate_lifecycle_mirror(
            lifecycle.lifecycle_cleanup_key,
            lifecycle.lifecycle_epoch,
            "nonce",
        )
        message = StreamMessage(
            message_id="delivery-paused",
            payload=DurableJobNotification(
                job_id=claim.envelope.job_id,
                dispatch_token="dispatch-consumed",
                lifecycle_epoch=lifecycle.lifecycle_epoch,
                lifecycle_cleanup_key=lifecycle.lifecycle_cleanup_key,
            ).model_dump(mode="json"),
        )
        lease = WorkerJobLease(
            JobTrackingService(job_connection, clock, workers_enabled=False),
            claim,
            effect_fence=WorkerEffectFence(domain_connection, clock),
        )
        dead_letter_task = asyncio.create_task(
            lease.dead_letter(
                backend,
                stream_name="atagia:evaluate",
                group_name="atagia-workers",
                message=message,
                exc=RuntimeError("safe injected failure"),
            )
        )
        await backend.publish_entered.wait()

        revocation_started = asyncio.Event()

        async def revoke() -> None:
            revocation_started.set()
            await revocation_connection.execute("BEGIN IMMEDIATE")
            await revocation_connection.execute(
                """
                UPDATE user_lifecycles
                SET state = 'cleanup_pending', revoked_at = ?, updated_at = ?
                WHERE user_id = ? AND lifecycle_epoch = ?
                """,
                (
                    clock.now().isoformat(),
                    clock.now().isoformat(),
                    "usr_1",
                    lifecycle.lifecycle_epoch,
                ),
            )
            await JobRunRepository(
                revocation_connection,
                clock,
            ).cancel_jobs_with_inactive_lifecycle(commit=False)
            await revocation_connection.commit()
            await backend.revoke_lifecycle_and_purge_notifications(
                lifecycle.lifecycle_cleanup_key,
                lifecycle.lifecycle_epoch,
                group_name="atagia-workers",
            )

        revocation_task = asyncio.create_task(revoke())
        await revocation_started.wait()
        await asyncio.sleep(0)
        assert not revocation_task.done()
        backend.resume_publish.set()

        assert await dead_letter_task
        await revocation_task
        stored = await JobRunRepository(job_connection, clock).get_job("job_parent")
        assert stored is not None
        assert stored["status"] == "dead_lettered"
        assert (
            await backend.dequeue_job(
                "dead_letter:atagia:evaluate",
                timeout_seconds=0,
            )
            is None
        )
    finally:
        await backend.close()
        await revocation_connection.close()
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_dead_letter_does_not_publish_after_derivation_revision_bump(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "dead-letter-derivation-revision.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        lifecycle_repository = UserLifecycleRepository(job_connection, clock)
        lifecycle = await lifecycle_repository.get_active_identity("usr_1")
        assert lifecycle is not None
        await _create_parent(JobRunRepository(job_connection, clock))
        claim = await _claim_parent(
            JobRunRepository(job_connection, clock),
            owner_id="stale-dead-letter-owner",
        )
        await backend.prepare_lifecycle_mirror(
            lifecycle.lifecycle_cleanup_key,
            lifecycle.lifecycle_epoch,
            "nonce",
        )
        assert await backend.activate_lifecycle_mirror(
            lifecycle.lifecycle_cleanup_key,
            lifecycle.lifecycle_epoch,
            "nonce",
        )
        bumped_revision = await lifecycle_repository.bump_derivation_revision(
            "usr_1",
            expected_lifecycle_epoch=lifecycle.lifecycle_epoch,
        )
        assert bumped_revision == claim.derivation_revision + 1

        message = StreamMessage(
            message_id="delivery-stale-revision",
            payload=DurableJobNotification(
                job_id=claim.envelope.job_id,
                dispatch_token="dispatch-consumed",
                lifecycle_epoch=lifecycle.lifecycle_epoch,
                lifecycle_cleanup_key=lifecycle.lifecycle_cleanup_key,
            ).model_dump(mode="json"),
        )
        lease = WorkerJobLease(
            JobTrackingService(job_connection, clock, workers_enabled=False),
            claim,
            effect_fence=WorkerEffectFence(domain_connection, clock),
        )
        assert not await lease.dead_letter(
            backend,
            stream_name="atagia:evaluate",
            group_name="atagia-workers",
            message=message,
            exc=RuntimeError("safe injected failure"),
        )
        assert (
            await backend.dequeue_job(
                "dead_letter:atagia:evaluate",
                timeout_seconds=0,
            )
            is None
        )
    finally:
        await backend.close()
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_write_coverage_guard_flags_table_created_after_fence_init(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "coverage-guard.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        jobs = JobRunRepository(job_connection, clock)
        await _create_parent(jobs)
        claim = await _claim_parent(jobs, owner_id="guard-owner")
        effect_fence = WorkerEffectFence(domain_connection, clock)

        # The migrated production schema (including its FTS5 virtual and
        # shadow tables) must satisfy the guard.
        async with effect_fence.activate(claim):
            pass

        await job_connection.execute(
            "CREATE TABLE rogue_domain_table(id TEXT PRIMARY KEY)"
        )
        await job_connection.commit()
        with pytest.raises(RuntimeError, match="rogue_domain_table"):
            await effect_fence.assert_write_coverage()
        with pytest.raises(RuntimeError, match="rogue_domain_table"):
            async with effect_fence.activate(claim):
                raise AssertionError("activation must fail before any effect")
    finally:
        await domain_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_write_coverage_guard_exempts_virtual_module_tables(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "coverage-exempt.db")
    job_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    domain_connection = await open_connection(database_path)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    try:
        await UserRepository(job_connection, clock).create_user("usr_1")
        jobs = JobRunRepository(job_connection, clock)
        await _create_parent(jobs)
        claim = await _claim_parent(jobs, owner_id="exempt-owner")
        effect_fence = WorkerEffectFence(domain_connection, clock)
        async with effect_fence.activate(claim):
            pass

        # A virtual table created after fence initialization is exempt, along
        # with its shadow tables and any plain backing table the module owns
        # by name prefix (sqlite-vec's *_vector_chunksNN pattern).
        await job_connection.execute("CREATE VIRTUAL TABLE rogue_fts USING fts5(body)")
        await job_connection.execute(
            "CREATE TABLE rogue_fts_vector_chunks00(id TEXT PRIMARY KEY)"
        )
        await job_connection.commit()
        await effect_fence.assert_write_coverage()
        async with effect_fence.activate(claim):
            pass

        await job_connection.execute("CREATE TABLE truly_rogue(id TEXT PRIMARY KEY)")
        await job_connection.commit()
        with pytest.raises(RuntimeError, match="truly_rogue"):
            await effect_fence.assert_write_coverage()
    finally:
        await domain_connection.close()
        await job_connection.close()
