"""Durable dispatch, restart recovery, lease, and fencing tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import UserRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CONTRACT_STREAM_NAME,
    EVALUATION_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    REVISE_STREAM_NAME,
    TRANSCRIPT_REBUILD_STREAM_NAME,
    DurableJobNotification,
    JobEnvelope,
    JobRunStatus,
    JobType,
    WORKER_GROUP_NAME,
)
from atagia.services.durable_job_dispatcher import DurableJobDispatcher

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


class FailOncePublishBackend(InProcessBackend):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def publish_job_notification(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise ConnectionError("injected transient publish failure")
        return await super().publish_job_notification(*args, **kwargs)


async def _database(path: str = ":memory:"):
    connection = await initialize_database(path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await UserRepository(connection, clock).create_user("usr_1")
    return connection, clock


async def _create_job(
    repository: JobRunRepository,
    job_type: JobType,
    *,
    job_id: str,
) -> JobEnvelope:
    envelope = JobEnvelope(
        job_id=job_id,
        job_type=job_type,
        user_id="usr_1",
        payload={"private_text": f"content for {job_id}"},
    )
    await repository.create_durable_job(
        stream_name=STREAM_BY_TYPE[job_type],
        target_backend="inprocess",
        envelope=envelope,
        source_token_estimate=None,
        size_bucket=None,
    )
    return envelope


def _dispatcher(connection, clock, backend):
    return DurableJobDispatcher(
        connection,
        clock,
        storage_backend=backend,
        target_backend="inprocess",
        visibility_seconds=10,
        sweep_interval_seconds=0.1,
        batch_size=100,
    )


@pytest.mark.asyncio
async def test_dispatcher_recovers_every_registered_job_type_without_exposing_envelopes() -> (
    None
):
    connection, clock = await _database()
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    try:
        for job_type in JobType:
            await _create_job(repository, job_type, job_id=f"job_{job_type.value}")

        result = await _dispatcher(connection, clock, backend).dispatch_once()

        assert result.claimed == len(JobType)
        assert result.published == len(JobType)
        for job_type in JobType:
            messages = await backend.stream_read(
                STREAM_BY_TYPE[job_type],
                WORKER_GROUP_NAME,
                "worker-1",
                count=1,
                block_ms=0,
            )
            assert len(messages) == 1
            assert set(messages[0].payload) == {
                "job_id",
                "dispatch_token",
                "lifecycle_epoch",
                "lifecycle_cleanup_key",
            }
            assert "private_text" not in str(messages[0].payload)
            claim = await repository.claim_notification(
                messages[0].message_id,
                DurableJobNotification.model_validate(messages[0].payload),
                owner_id="worker-1",
                lease_seconds=30,
            )
            assert claim is not None
            assert claim.envelope.job_type is job_type
            assert claim.envelope.payload["private_text"].startswith("content for")
            assert await repository.finish_claim(
                claim,
                status=JobRunStatus.SUCCEEDED,
            )
            row = await repository.get_job(claim.envelope.job_id)
            assert row is not None
            assert row["recovery_envelope_json"] is None
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_restart_matrix_recovers_every_job_type_from_every_nonterminal_state() -> (
    None
):
    connection, clock = await _database()
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    nonterminal_states = (
        JobRunStatus.QUEUED,
        JobRunStatus.AWAITING_CLAIM,
        JobRunStatus.RUNNING,
        JobRunStatus.RETRYING,
        JobRunStatus.DEFERRED,
    )
    expired_at = "2026-07-13T09:59:00+00:00"
    try:
        expected_job_ids: dict[JobType, set[str]] = {
            job_type: set() for job_type in JobType
        }
        for job_type in JobType:
            for state in nonterminal_states:
                job_id = f"job_matrix_{job_type.value}_{state.value}"
                expected_job_ids[job_type].add(job_id)
                await _create_job(
                    repository,
                    job_type,
                    job_id=job_id,
                )
                if state is JobRunStatus.QUEUED:
                    continue
                await connection.execute(
                    """
                    UPDATE worker_job_runs
                    SET status = ?,
                        dispatch_token = ?,
                        dispatch_visibility_deadline = ?,
                        execution_owner = ?,
                        execution_fence = ?,
                        execution_lease_expires_at = ?,
                        deferred_until = ?
                    WHERE job_id = ?
                    """,
                    (
                        state.value,
                        (
                            f"dispatch_{job_id}"
                            if state is JobRunStatus.AWAITING_CLAIM
                            else None
                        ),
                        (expired_at if state is JobRunStatus.AWAITING_CLAIM else None),
                        "expired-owner" if state is JobRunStatus.RUNNING else None,
                        1 if state is JobRunStatus.RUNNING else 0,
                        expired_at if state is JobRunStatus.RUNNING else None,
                        (
                            expired_at
                            if state in {JobRunStatus.RETRYING, JobRunStatus.DEFERRED}
                            else None
                        ),
                        job_id,
                    ),
                )
        await connection.commit()

        result = await _dispatcher(connection, clock, backend).dispatch_once()
        expected_count = len(JobType) * len(nonterminal_states)
        assert result.recovered_leases == len(JobType)
        assert result.claimed == expected_count
        assert result.published == expected_count

        for job_type in JobType:
            messages = await backend.stream_read(
                STREAM_BY_TYPE[job_type],
                WORKER_GROUP_NAME,
                f"matrix-{job_type.value}",
                count=len(nonterminal_states),
                block_ms=0,
            )
            assert len(messages) == len(nonterminal_states)
            claimed_ids: set[str] = set()
            for message in messages:
                claim = await repository.claim_notification(
                    message.message_id,
                    DurableJobNotification.model_validate(message.payload),
                    owner_id=f"matrix-{job_type.value}",
                    lease_seconds=30,
                )
                assert claim is not None
                assert claim.envelope.job_type is job_type
                claimed_ids.add(claim.envelope.job_id)
                assert await repository.finish_claim(
                    claim,
                    status=JobRunStatus.SUCCEEDED,
                )
                await backend.stream_ack(
                    STREAM_BY_TYPE[job_type],
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
            assert claimed_ids == expected_job_ids[job_type]
        assert await repository.nonterminal_count() == 0
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_visibility_expiry_replaces_token_and_stale_notification_cannot_execute() -> (
    None
):
    connection, clock = await _database()
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    try:
        await _create_job(
            repository,
            JobType.EXTRACT_MEMORY_CANDIDATES,
            job_id="job_visibility",
        )
        dispatcher = _dispatcher(connection, clock, backend)
        assert (await dispatcher.dispatch_once()).published == 1
        first = (
            await backend.stream_read(
                EXTRACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                "worker-1",
                count=1,
                block_ms=0,
            )
        )[0]
        first_notification = DurableJobNotification.model_validate(first.payload)

        clock.advance(seconds=11)
        assert (await dispatcher.dispatch_once()).published == 1
        assert (
            await repository.claim_notification(
                first.message_id,
                first_notification,
                owner_id="worker-stale",
                lease_seconds=30,
            )
            is None
        )
        second = (
            await backend.stream_read(
                EXTRACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                "worker-2",
                count=1,
                block_ms=0,
            )
        )[0]
        second_notification = DurableJobNotification.model_validate(second.payload)
        assert second_notification.dispatch_token != first_notification.dispatch_token
        assert (
            await repository.claim_notification(
                second.message_id,
                second_notification,
                owner_id="worker-2",
                lease_seconds=30,
            )
            is not None
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_transient_publish_failure_recovers_without_dispatcher_restart() -> None:
    connection, clock = await _database()
    backend = FailOncePublishBackend()
    repository = JobRunRepository(connection, clock)
    try:
        await _create_job(
            repository,
            JobType.PROJECT_CONTRACT,
            job_id="job_publish_retry",
        )
        dispatcher = _dispatcher(connection, clock, backend)
        first = await dispatcher.dispatch_once()
        assert first.delivery_failed == 1
        row = await repository.get_job("job_publish_retry")
        assert row is not None
        assert row["status"] == JobRunStatus.AWAITING_CLAIM.value

        clock.advance(seconds=11)
        second = await dispatcher.dispatch_once()
        assert second.published == 1
        assert await backend.stream_read(
            CONTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "worker-1",
            count=1,
            block_ms=0,
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_renewable_lease_prevents_takeover_and_fences_stale_owner() -> None:
    connection, clock = await _database()
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    try:
        await _create_job(
            repository,
            JobType.COMPACT_SUMMARIES,
            job_id="job_lease",
        )
        dispatcher = _dispatcher(connection, clock, backend)
        await dispatcher.dispatch_once()
        first_message = (
            await backend.stream_read(
                COMPACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                "worker-1",
                count=1,
                block_ms=0,
            )
        )[0]
        first_claim = await repository.claim_notification(
            first_message.message_id,
            DurableJobNotification.model_validate(first_message.payload),
            owner_id="worker-1",
            lease_seconds=10,
        )
        assert first_claim is not None

        clock.advance(seconds=8)
        assert await repository.heartbeat_claim(first_claim, lease_seconds=10)
        clock.advance(seconds=8)
        assert await repository.recover_expired_execution_leases() == 0
        clock.advance(seconds=3)
        assert await repository.recover_expired_execution_leases() == 1
        assert (await dispatcher.dispatch_once()).published == 1
        second_message = (
            await backend.stream_read(
                COMPACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                "worker-2",
                count=1,
                block_ms=0,
            )
        )[0]
        second_claim = await repository.claim_notification(
            second_message.message_id,
            DurableJobNotification.model_validate(second_message.payload),
            owner_id="worker-2",
            lease_seconds=10,
        )
        assert second_claim is not None
        assert second_claim.execution_fence > first_claim.execution_fence
        assert not await repository.finish_claim(
            first_claim,
            status=JobRunStatus.SUCCEEDED,
        )
        assert await repository.finish_claim(
            second_claim,
            status=JobRunStatus.SUCCEEDED,
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_terminal_transition_clears_recovery_envelope_after_reopen(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "durable-jobs.db")
    connection, clock = await _database(database_path)
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    await _create_job(repository, JobType.RUN_EVALUATION, job_id="job_terminal")
    await _dispatcher(connection, clock, backend).dispatch_once()
    message = (
        await backend.stream_read(
            EVALUATION_STREAM_NAME,
            WORKER_GROUP_NAME,
            "worker-1",
            count=1,
            block_ms=0,
        )
    )[0]
    claim = await repository.claim_notification(
        message.message_id,
        DurableJobNotification.model_validate(message.payload),
        owner_id="worker-1",
        lease_seconds=30,
    )
    assert claim is not None
    private_sentinel = "PRIVATE_PROVIDER_ECHO_MUST_NOT_PERSIST"
    assert await repository.finish_claim(
        claim,
        status=JobRunStatus.FAILED,
        error_class="InjectedProviderError",
        error_message=f"provider echoed {private_sentinel}",
    )
    await connection.close()

    reopened = await open_connection(database_path)
    try:
        row = await JobRunRepository(reopened, clock).get_job("job_terminal")
        assert row is not None
        assert row["status"] == JobRunStatus.FAILED.value
        assert row["recovery_envelope_json"] is None
        assert row["attempt_count"] == 1
        assert row["execution_fence"] == 1
        assert row["error_class"] == "InjectedProviderError"
        assert row["error_message"] is None
        assert private_sentinel not in repr(row)
    finally:
        await reopened.close()
        await backend.close()


@pytest.mark.parametrize(
    "transition",
    ["heartbeat", "retry", "succeed", "fail"],
)
@pytest.mark.asyncio
async def test_expired_owner_cannot_revive_or_terminalize_without_a_sweep(
    transition: str,
) -> None:
    connection, clock = await _database()
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    try:
        await _create_job(
            repository,
            JobType.RUN_EVALUATION,
            job_id=f"job_expired_{transition}",
        )
        await _dispatcher(connection, clock, backend).dispatch_once()
        message = (
            await backend.stream_read(
                EVALUATION_STREAM_NAME,
                WORKER_GROUP_NAME,
                "expired-owner",
                count=1,
                block_ms=0,
            )
        )[0]
        claim = await repository.claim_notification(
            message.message_id,
            DurableJobNotification.model_validate(message.payload),
            owner_id="expired-owner",
            lease_seconds=10,
        )
        assert claim is not None
        before = await repository.get_job(claim.envelope.job_id)
        assert before is not None
        clock.advance(seconds=11)

        if transition == "heartbeat":
            changed = await repository.heartbeat_claim(claim, lease_seconds=10)
        elif transition == "retry":
            changed = await repository.release_claim_for_retry(
                claim,
                error_class="InjectedProviderError",
                error_message="expired provider call",
            )
        elif transition == "succeed":
            changed = await repository.finish_claim(
                claim,
                status=JobRunStatus.SUCCEEDED,
            )
        else:
            changed = await repository.finish_claim(
                claim,
                status=JobRunStatus.FAILED,
                error_class="InjectedProviderError",
                error_message="expired provider call",
            )

        assert not changed
        after = await repository.get_job(claim.envelope.job_id)
        assert after is not None
        assert after["status"] == JobRunStatus.RUNNING.value
        assert after["execution_owner"] == before["execution_owner"]
        assert after["execution_fence"] == before["execution_fence"]
        assert (
            after["execution_lease_expires_at"] == before["execution_lease_expires_at"]
        )
        assert after["recovery_envelope_json"] is not None
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_terminal_deterministic_child_is_idempotent_after_database_reopen(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "terminal-child.db")
    connection, clock = await _database(database_path)
    backend = InProcessBackend()
    repository = JobRunRepository(connection, clock)
    await _create_job(
        repository,
        JobType.EXTRACT_MEMORY_CANDIDATES,
        job_id="job_parent",
    )
    await _dispatcher(connection, clock, backend).dispatch_once()
    parent_message = (
        await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "parent-owner",
            count=1,
            block_ms=0,
        )
    )[0]
    parent_claim = await repository.claim_notification(
        parent_message.message_id,
        DurableJobNotification.model_validate(parent_message.payload),
        owner_id="parent-owner",
        lease_seconds=30,
    )
    assert parent_claim is not None
    child_envelope = JobEnvelope(
        job_id="job_deterministic_child",
        job_type=JobType.COMPACT_SUMMARIES,
        user_id="usr_1",
        parent_job_id=parent_claim.envelope.job_id,
        message_ids=["msg_source"],
        payload={"private_text": "child payload"},
    )
    await repository.create_durable_job(
        stream_name=COMPACT_STREAM_NAME,
        target_backend="inprocess",
        envelope=child_envelope,
        source_token_estimate=None,
        size_bucket=None,
        parent_claim=parent_claim,
    )
    await _dispatcher(connection, clock, backend).dispatch_once()
    child_message = (
        await backend.stream_read(
            COMPACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "child-owner",
            count=1,
            block_ms=0,
        )
    )[0]
    child_claim = await repository.claim_notification(
        child_message.message_id,
        DurableJobNotification.model_validate(child_message.payload),
        owner_id="child-owner",
        lease_seconds=30,
    )
    assert child_claim is not None
    assert await repository.finish_claim(
        child_claim,
        status=JobRunStatus.SUCCEEDED,
    )
    await connection.close()

    reopened = await open_connection(database_path)
    try:
        reopened_repository = JobRunRepository(reopened, clock)
        existing = await reopened_repository.create_durable_job(
            stream_name=COMPACT_STREAM_NAME,
            target_backend="inprocess",
            envelope=child_envelope,
            source_token_estimate=None,
            size_bucket=None,
            parent_claim=parent_claim,
        )
        assert existing["status"] == JobRunStatus.SUCCEEDED.value
        assert existing["recovery_envelope_json"] is None

        conflicting_identity = child_envelope.model_copy(
            update={"message_ids": ["different_source"]}
        )
        with pytest.raises(ValueError, match="different job identity"):
            await reopened_repository.create_durable_job(
                stream_name=COMPACT_STREAM_NAME,
                target_backend="inprocess",
                envelope=conflicting_identity,
                source_token_estimate=None,
                size_bucket=None,
                parent_claim=parent_claim,
            )
    finally:
        await reopened.close()
        await backend.close()
