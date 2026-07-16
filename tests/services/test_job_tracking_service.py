"""Tests for durable job tracking and memory-processing status summaries."""

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
from atagia.core.storage_backend import InProcessBackend
from atagia.models.schemas_jobs import (
    EXTRACT_STREAM_NAME,
    ClaimedJob,
    JobEnvelope,
    JobRunStatus,
    JobType,
    MessageJobPayload,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerControlMode,
)
from atagia.models.schemas_memory import (
    OperationalProfileSnapshot,
    OperationalRiskLevel,
    OperationalSignals,
)
from atagia.services.chat_support import enqueue_message_jobs
from atagia.services.durable_job_dispatcher import DurableJobDispatcher
from atagia.services.job_tracking_service import (
    JobTrackingService,
    render_memory_processing_status_block,
)
from atagia.services.llm_client import TransientLLMError
from atagia.services.worker_control_service import WorkerControlService

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


class FailOncePublishBackend(InProcessBackend):
    """Inject one notification-delivery failure without losing durable work."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def publish_job_notification(self, *args: Any, **kwargs: Any) -> str | None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("stream unavailable")
        return await super().publish_job_notification(*args, **kwargs)


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


def _operational_snapshot(token: str) -> OperationalProfileSnapshot:
    return OperationalProfileSnapshot(
        profile_id="normal",
        signals=OperationalSignals(),
        risk_level=OperationalRiskLevel.NORMAL,
        authorized=True,
        profile_hash=f"profile-{token}",
        token=token,
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


async def _enqueue_and_claim(
    service: JobTrackingService,
    backend: InProcessBackend,
    envelope: JobEnvelope,
    *,
    owner_id: str,
) -> tuple[StreamMessage, ClaimedJob]:
    await service.enqueue_job(backend, EXTRACT_STREAM_NAME, envelope)
    messages = await backend.stream_read(
        EXTRACT_STREAM_NAME,
        WORKER_GROUP_NAME,
        owner_id,
        count=1,
        block_ms=0,
    )
    assert len(messages) == 1
    claim = await service.claim_notification(messages[0], owner_id=owner_id)
    assert claim is not None
    return messages[0], claim


async def _ack(
    backend: InProcessBackend,
    message: StreamMessage,
) -> None:
    await backend.stream_ack(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME, message.message_id)


@pytest.mark.asyncio
async def test_job_tracking_service_summarizes_pending_work_and_estimate() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)

        for index in range(3):
            message, claim = await _enqueue_and_claim(
                service,
                backend,
                _message_job(
                    f"job_completed_{index}",
                    f"msg_completed_{index}",
                    "Completed source.",
                    created_at=clock.now(),
                ),
                owner_id=f"worker-completed-{index}",
            )
            clock.advance(seconds=4)
            assert await service.finish_claim_succeeded(claim)
            await _ack(backend, message)

        await service.enqueue_job(
            backend,
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
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_enqueue_failure_preserves_job_until_continuous_dispatch_recovers() -> (
    None
):
    connection, clock = await _connection_and_clock()
    backend = FailOncePublishBackend()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)
        envelope = _message_job("job_recover", "msg_recover", "hello")

        await enqueue_message_jobs(
            storage_backend=backend,
            jobs=[(EXTRACT_STREAM_NAME, envelope)],
            job_tracking_service=service,
            initial_context_package_refresh_enabled=False,
        )
        stored = await repository.get_job(envelope.job_id)
        assert stored is not None
        assert stored["status"] == JobRunStatus.AWAITING_CLAIM.value
        assert stored["error_class"] is None

        clock.advance(seconds=31)
        dispatcher = DurableJobDispatcher(
            connection,
            clock,
            storage_backend=backend,
            target_backend="inprocess",
            visibility_seconds=30,
            sweep_interval_seconds=1,
            batch_size=10,
        )
        result = await dispatcher.dispatch_once()
        assert result.published == 1
        messages = await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "worker-recovered",
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
        assert "message_text" not in str(messages[0].payload)
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_stores_full_operational_profile_snapshot() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)
        envelope = _message_job("job_operational", "msg_operational", "Operational job")
        envelope = envelope.model_copy(
            update={"operational_profile": _operational_snapshot("custom-token")}
        )

        await service.enqueue_job(backend, EXTRACT_STREAM_NAME, envelope)
        stored = await repository.get_job(envelope.job_id)
        assert stored is not None
        assert stored["metadata_json"]["operational_profile"] == "normal"
        assert stored["metadata_json"]["operational_profile_snapshot"]["token"] == (
            "custom-token"
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_enqueue_message_jobs_skips_source_work_when_paused() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)
        worker_control = WorkerControlService(connection, clock)
        await worker_control.set_mode(WorkerControlMode.PAUSE_NEW_JOBS, reason="backup")

        job_ids = await enqueue_message_jobs(
            storage_backend=backend,
            jobs=[
                (EXTRACT_STREAM_NAME, _message_job("job_paused", "msg_paused", "hello"))
            ],
            job_tracking_service=service,
            worker_control_service=worker_control,
        )
        assert job_ids == []
        assert await repository.get_job("job_paused") is None
        assert (
            await backend.stream_read(
                EXTRACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                "worker-paused",
                count=1,
                block_ms=0,
            )
            == []
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_auto_hard_pauses_after_failure_storm() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
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
            message, claim = await _enqueue_and_claim(
                service,
                backend,
                _message_job(
                    f"job_failure_{index}",
                    f"msg_failure_{index}",
                    "provider call failed",
                ),
                owner_id=f"worker-failure-{index}",
            )
            assert await service.finish_claim_failed(
                claim,
                TransientLLMError("provider unavailable"),
            )
            await _ack(backend, message)

        state = await worker_control.get_state()
        assert state.mode is WorkerControlMode.HARD_PAUSE
        assert state.updated_by == "worker_circuit_breaker"
        assert state.reason is not None
        assert "Auto hard pause" in state.reason
        assert "TransientLLMError=3" in state.reason
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_circuit_breaker_respects_failure_ratio() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
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
            message, claim = await _enqueue_and_claim(
                service,
                backend,
                _message_job(f"job_success_{index}", f"msg_success_{index}", "healthy"),
                owner_id=f"worker-success-{index}",
            )
            assert await service.finish_claim_succeeded(claim)
            await _ack(backend, message)

        for index in range(3):
            message, claim = await _enqueue_and_claim(
                service,
                backend,
                _message_job(
                    f"job_mixed_failure_{index}",
                    f"msg_mixed_failure_{index}",
                    "provider call failed",
                ),
                owner_id=f"worker-mixed-{index}",
            )
            assert await service.finish_claim_failed(
                claim,
                TransientLLMError("provider unavailable"),
            )
            await _ack(backend, message)

        state = await worker_control.get_state()
        assert state.mode is WorkerControlMode.ACTIVE
        assert state.reason is None
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_job_tracking_filters_non_admin_user_status_by_namespace() -> None:
    connection, clock = await _connection_and_clock()
    backend = InProcessBackend()
    try:
        await _seed_scope(connection, clock)
        service = JobTrackingService(connection, clock, workers_enabled=True)
        repository = JobRunRepository(connection, clock)

        await service.enqueue_job(
            backend,
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
        await service.enqueue_job(
            backend,
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
        await backend.close()
        await connection.close()
