"""Tests for durable worker-job run tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.models.schemas_jobs import CONTRACT_STREAM_NAME, EXTRACT_STREAM_NAME, JobRunStatus, JobType

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _connection_and_clock() -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc))
    return connection, clock


async def _insert_assistant_mode(connection: aiosqlite.Connection, mode_id: str = "coding_debug") -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            mode_id,
            "Coding Debug",
            "hash_1",
            "{}",
            "2026-05-02T12:00:00+00:00",
            "2026-05-02T12:00:00+00:00",
        ),
    )
    await connection.commit()


async def _seed_scope(connection: aiosqlite.Connection, clock: FrozenClock) -> None:
    await UserRepository(connection, clock).create_user("usr_1")
    await _insert_assistant_mode(connection)
    await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        None,
        "coding_debug",
        "Tracked chat",
    )


@pytest.mark.asyncio
async def test_job_run_repository_tracks_progress_and_status_transitions() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)

        extract = await repository.create_queued_job(
            job_id="job_extract_msg_1",
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id="usr_1",
            conversation_id="cnv_1",
            source_message_ids=["msg_1"],
            source_token_estimate=128,
            size_bucket="small",
            metadata={"message_count": 1},
        )
        contract = await repository.create_queued_job(
            job_id="job_contract_msg_1",
            stream_name=CONTRACT_STREAM_NAME,
            job_type=JobType.PROJECT_CONTRACT.value,
            user_id="usr_1",
            conversation_id="cnv_1",
            source_message_ids=["msg_1"],
            source_token_estimate=128,
            size_bucket="small",
        )

        assert extract["status"] == JobRunStatus.QUEUED.value
        assert contract["status"] == JobRunStatus.QUEUED.value
        assert await repository.oldest_nonterminal_queued_at(
            user_id="usr_1",
            conversation_id="cnv_1",
        ) == "2026-05-02T12:00:00+00:00"

        await repository.mark_running("job_extract_msg_1", attempt_count=1)
        clock.advance(seconds=2)
        await repository.mark_succeeded("job_extract_msg_1", metadata={"memory_count": 2})

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
        counts = await repository.status_counts(user_id="usr_1", conversation_id="cnv_1")
        assert {
            (row["status"], row["job_type"]): row["count"]
            for row in counts
        } == {
            (JobRunStatus.QUEUED.value, JobType.PROJECT_CONTRACT.value): 1,
            (JobRunStatus.SUCCEEDED.value, JobType.EXTRACT_MEMORY_CANDIDATES.value): 1,
        }

        completed = await repository.get_job("job_extract_msg_1")
        assert completed is not None
        assert completed["duration_ms"] is not None
        assert completed["metadata_json"]["message_count"] == 1
        assert completed["metadata_json"]["memory_count"] == 2

        await repository.mark_running("job_contract_msg_1", attempt_count=1)
        await repository.mark_retrying(
            "job_contract_msg_1",
            attempt_count=2,
            error_class="TransientError",
            error_message="temporary backend issue",
            deferred_until="2026-05-02T12:01:00+00:00",
        )

        retrying = await repository.get_job("job_contract_msg_1")
        assert retrying is not None
        assert retrying["status"] == JobRunStatus.RETRYING.value
        assert retrying["attempt_count"] == 2
        assert retrying["error_class"] == "TransientError"
        assert retrying["deferred_until"] == "2026-05-02T12:01:00+00:00"
        clock.advance(seconds=1)
        first_deferred = await repository.mark_deferred(
            "job_contract_msg_1",
            attempt_count=1,
            error_class="TransientLLMError",
            error_message="provider unavailable",
            deferred_until="2026-05-02T12:02:00+00:00",
        )
        assert first_deferred["transient_defer_count"] == 1
        assert first_deferred["first_deferred_at"] == "2026-05-02T12:00:03+00:00"
        assert first_deferred["last_deferred_at"] == "2026-05-02T12:00:03+00:00"
        clock.advance(seconds=1)
        second_deferred = await repository.mark_deferred(
            "job_contract_msg_1",
            attempt_count=1,
            error_class="TransientLLMError",
            error_message="provider unavailable",
            deferred_until="2026-05-02T12:03:00+00:00",
        )
        assert second_deferred["transient_defer_count"] == 2
        assert second_deferred["first_deferred_at"] == "2026-05-02T12:00:03+00:00"
        assert second_deferred["last_deferred_at"] == "2026-05-02T12:00:04+00:00"
        await repository.mark_succeeded("job_contract_msg_1")
        succeeded_after_retry = await repository.get_job("job_contract_msg_1")
        assert succeeded_after_retry is not None
        assert succeeded_after_retry["status"] == JobRunStatus.SUCCEEDED.value
        assert succeeded_after_retry["error_class"] is None
        assert succeeded_after_retry["error_message"] is None
        assert succeeded_after_retry["deferred_until"] is None
        assert succeeded_after_retry["transient_defer_count"] == 2
    finally:
        await connection.close()
