"""Tests for memory-processing status summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobRunStatus,
    JobType,
    MessageJobPayload,
    StreamMessage,
    WorkerControlMode,
)
from atagia.services.chat_support import enqueue_message_jobs
from atagia.services.job_tracking_service import JobTrackingService, render_memory_processing_status_block
from atagia.services.llm_client import TransientLLMError
from atagia.services.worker_control_service import WorkerControlService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class RecordingBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.added: list[tuple[str, dict[str, Any]]] = []

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        if self.fail:
            raise RuntimeError("stream unavailable")
        self.added.append((stream_name, payload))
        return f"stm_{len(self.added)}"


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


def _message_job(
    job_id: str,
    message_id: str,
    text: str,
    *,
    created_at: datetime | None = None,
    user_persona_id: str | None = None,
    platform_id: str = "default",
    character_id: str | None = None,
) -> JobEnvelope:
    return JobEnvelope(
        job_id=job_id,
        job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload=MessageJobPayload(
            message_id=message_id,
            message_text=text,
            role="user",
            assistant_mode_id="coding_debug",
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
        ).model_dump(mode="json"),
        created_at=created_at or datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "sqlite_path": ":memory:",
        "migrations_path": str(MIGRATIONS_DIR),
        "manifests_path": str(MANIFESTS_DIR),
        "storage_backend": "inprocess",
        "redis_url": "redis://localhost:6379/0",
        "openai_api_key": None,
        "openrouter_api_key": None,
        "openrouter_site_url": "http://localhost",
        "openrouter_app_name": "Atagia",
        "llm_chat_model": None,
        "service_mode": False,
        "service_api_key": None,
        "admin_api_key": None,
        "workers_enabled": True,
        "debug": False,
    }
    values.update(overrides)
    return Settings(**values)


def _stream_message(envelope: JobEnvelope, *, delivery_count: int = 1) -> StreamMessage:
    return StreamMessage(
        message_id=f"stm_{envelope.job_id}",
        payload=envelope.model_dump(mode="json"),
        delivery_count=delivery_count,
    )


@pytest.mark.asyncio
async def test_job_tracking_service_summarizes_pending_work_and_estimate() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = JobRunRepository(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)

        for index in range(3):
            job_id = f"job_completed_{index}"
            await repository.create_queued_job(
                job_id=job_id,
                stream_name=EXTRACT_STREAM_NAME,
                job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
                user_id="usr_1",
                conversation_id="cnv_1",
                source_message_ids=[f"msg_completed_{index}"],
                source_token_estimate=128,
                size_bucket="small",
            )
            await repository.mark_running(job_id, attempt_count=1)
            clock.advance(seconds=4)
            await repository.mark_succeeded(job_id)

        await service.create_queued_job(
            EXTRACT_STREAM_NAME,
            _message_job(
                "job_pending",
                "msg_pending",
                "Please remember this long update.",
                created_at=clock.now(),
            ),
        )

        status = await service.get_status(user_id="usr_1", conversation_id="cnv_1")

        assert status.processing is True
        assert status.status == "queued"
        assert status.pending_jobs == 1
        assert status.running_jobs == 0
        assert status.pending_source_messages == 1
        assert status.processed_source_messages == 0
        assert status.tracked_source_messages == 1
        assert status.pending_jobs_by_type == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 1,
        }
        assert status.estimate.confidence == "low"
        assert status.estimate.basis == "historical_jobs"
        assert status.estimate.estimate_range_seconds is not None
        assert status.global_queue_state == "normal"

        prompt_block = render_memory_processing_status_block(status)
        assert "Memory Processing Status" in prompt_block
        assert "Processed source messages in current window: 0/1" in prompt_block
        assert "Pending work: extract_memory_candidates=1" in prompt_block
        assert "Rough remaining time" in prompt_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_enqueue_message_jobs_records_success_and_enqueue_failure() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)

        success_backend = RecordingBackend()
        await enqueue_message_jobs(
            storage_backend=success_backend,
            jobs=[(EXTRACT_STREAM_NAME, _message_job("job_success", "msg_success", "hello"))],
            job_tracking_service=service,
        )
        assert success_backend.added[0][0] == EXTRACT_STREAM_NAME
        success = await repository.get_job("job_success")
        assert success is not None
        assert success["status"] == JobRunStatus.QUEUED.value

        with pytest.raises(RuntimeError, match="stream unavailable"):
            await enqueue_message_jobs(
                storage_backend=RecordingBackend(fail=True),
                jobs=[(CONTRACT_STREAM_NAME, _message_job("job_failed", "msg_failed", "hello"))],
                job_tracking_service=service,
            )

        failed = await repository.get_job("job_failed")
        assert failed is not None
        assert failed["status"] == JobRunStatus.FAILED.value
        assert failed["error_class"] == "RuntimeError"
        assert failed["error_message"] == "stream unavailable"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_enqueue_message_jobs_skips_source_work_when_paused() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)
        worker_control = WorkerControlService(connection, clock)
        await worker_control.set_mode(WorkerControlMode.PAUSE_NEW_JOBS, reason="backup")

        backend = RecordingBackend()
        job_ids = await enqueue_message_jobs(
            storage_backend=backend,
            jobs=[(EXTRACT_STREAM_NAME, _message_job("job_paused", "msg_paused", "hello"))],
            job_tracking_service=service,
            worker_control_service=worker_control,
        )

        assert job_ids == []
        assert backend.added == []
        assert await repository.get_job("job_paused") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_auto_hard_pauses_after_failure_storm() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        settings = _settings(
            worker_circuit_breaker_failure_threshold=3,
            worker_circuit_breaker_window_seconds=60,
            worker_circuit_breaker_min_failure_ratio=0.75,
        )
        service = JobTrackingService(
            connection,
            clock,
            workers_enabled=True,
            settings=settings,
        )
        worker_control = WorkerControlService(connection, clock)

        for index in range(3):
            envelope = _message_job(
                f"job_failure_{index}",
                f"msg_failure_{index}",
                "provider call failed",
            )
            await service.create_queued_job(EXTRACT_STREAM_NAME, envelope)
            message = _stream_message(envelope, delivery_count=1)
            await service.mark_running(message)
            await service.mark_retrying(
                message,
                TransientLLMError("provider unavailable"),
            )

        state = await worker_control.get_state()

        assert state.mode is WorkerControlMode.HARD_PAUSE
        assert state.updated_by == "worker_circuit_breaker"
        assert state.reason is not None
        assert "Auto hard pause" in state.reason
        assert "TransientLLMError=3" in state.reason
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_circuit_breaker_respects_failure_ratio() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        settings = _settings(
            worker_circuit_breaker_failure_threshold=3,
            worker_circuit_breaker_window_seconds=60,
            worker_circuit_breaker_min_failure_ratio=0.8,
        )
        service = JobTrackingService(
            connection,
            clock,
            workers_enabled=True,
            settings=settings,
        )
        worker_control = WorkerControlService(connection, clock)

        for index in range(3):
            envelope = _message_job(
                f"job_success_{index}",
                f"msg_success_{index}",
                "healthy job",
            )
            await service.create_queued_job(EXTRACT_STREAM_NAME, envelope)
            message = _stream_message(envelope, delivery_count=1)
            await service.mark_running(message)
            await service.mark_succeeded(message)

        for index in range(3):
            envelope = _message_job(
                f"job_mixed_failure_{index}",
                f"msg_mixed_failure_{index}",
                "provider call failed",
            )
            await service.create_queued_job(EXTRACT_STREAM_NAME, envelope)
            message = _stream_message(envelope, delivery_count=1)
            await service.mark_running(message)
            await service.mark_retrying(
                message,
                TransientLLMError("provider unavailable"),
            )

        state = await worker_control.get_state()

        assert state.mode is WorkerControlMode.ACTIVE
        assert state.reason is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_filters_non_admin_user_status_by_namespace() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)

        await service.create_queued_job(
            EXTRACT_STREAM_NAME,
            _message_job(
                "job_persona_a",
                "msg_persona_a",
                "Persona A work.",
                user_persona_id="persona_a",
                platform_id="web",
                character_id="char_a",
            ),
        )
        await service.create_queued_job(
            EXTRACT_STREAM_NAME,
            _message_job(
                "job_persona_b",
                "msg_persona_b",
                "Persona B work.",
                user_persona_id="persona_b",
                platform_id="web",
                character_id="char_b",
            ),
        )

        stored = await repository.get_job("job_persona_a")
        assert stored is not None
        assert stored["user_persona_id"] == "persona_a"
        assert stored["platform_id"] == "web"
        assert stored["character_id"] == "char_a"
        assert stored["policy_snapshot_json"]["remember_across_chats"] is True

        with pytest.raises(ValueError, match="conversation_id or platform_id"):
            await service.get_status(user_id="usr_1")
        with pytest.raises(ValueError, match="chat-local"):
            await service.get_status(
                user_id="usr_1",
                platform_id="web",
                remember_across_chats=False,
            )

        status = await service.get_status(
            user_id="usr_1",
            user_persona_id="persona_a",
            platform_id="web",
            character_id="char_a",
        )

        assert status.pending_jobs == 1
        assert status.pending_jobs_by_type == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 1,
        }
        assert status.global_pending_jobs == 1
    finally:
        await connection.close()
