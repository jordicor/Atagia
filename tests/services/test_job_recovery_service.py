"""Tests for recovering durable jobs into transient in-process streams."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.operational_profile import OperationalProfileLoader
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    JobRunStatus,
    JobType,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import (
    OperationalProfileSnapshot,
    OperationalRiskLevel,
    OperationalSignals,
)
from atagia.services.job_recovery_service import JobRecoveryService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
OPERATIONAL_PROFILES_DIR = Path(__file__).resolve().parents[2] / "operational_profiles"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-recovery.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        operational_profiles_path=str(OPERATIONAL_PROFILES_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=True,
        debug=False,
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


async def _insert_assistant_mode(connection) -> None:
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "coding_debug",
            "Coding Debug",
            "hash_1",
            "{}",
            "2026-05-05T12:00:00+00:00",
            "2026-05-05T12:00:00+00:00",
        ),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_recovery_requeues_nonterminal_jobs_into_inprocess_streams(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        await _insert_assistant_mode(connection)
        conversation = await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Tracked chat",
            platform_id="aurvek",
        )
        messages = MessageRepository(connection, clock)
        await messages.create_message(
            "aurvek:msg:1",
            "cnv_1",
            "user",
            1,
            "First message.",
        )
        await messages.create_message(
            "aurvek:msg:2",
            "cnv_1",
            "user",
            2,
            "Second message.",
        )
        jobs = JobRunRepository(connection, clock)
        await jobs.create_queued_job(
            job_id="job_extract",
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id="usr_1",
            conversation_id=str(conversation["id"]),
            source_message_ids=["aurvek:msg:1"],
            source_token_estimate=12,
            size_bucket="small",
            metadata={"operational_profile": "normal"},
            platform_id="aurvek",
        )
        await jobs.create_queued_job(
            job_id="job_contract",
            stream_name=CONTRACT_STREAM_NAME,
            job_type=JobType.PROJECT_CONTRACT.value,
            user_id="usr_1",
            conversation_id=str(conversation["id"]),
            source_message_ids=["aurvek:msg:2"],
            source_token_estimate=12,
            size_bucket="small",
            metadata={"operational_profile": "normal"},
            platform_id="aurvek",
        )
        await jobs.mark_running("job_contract", attempt_count=1)

        result = await JobRecoveryService(
            connection,
            clock,
            settings=settings,
            storage_backend=backend,
            operational_profile_loader=OperationalProfileLoader(
                settings.operational_profiles_dir()
            ),
        ).recover_inprocess_stream_jobs()

        assert result.recovered_jobs == 2
        assert result.recovered_by_type == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 1,
            JobType.PROJECT_CONTRACT.value: 1,
        }
        extract_job = await jobs.get_job("job_extract")
        contract_job = await jobs.get_job("job_contract")
        assert extract_job is not None
        assert contract_job is not None
        assert extract_job["status"] == JobRunStatus.QUEUED.value
        assert contract_job["status"] == JobRunStatus.QUEUED.value
        assert contract_job["started_at"] is None
        assert contract_job["last_heartbeat_at"] is None
        assert contract_job["metadata_json"]["inprocess_recovered_at"] == (
            "2026-05-05T12:00:00+00:00"
        )

        extract_messages = await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test",
            count=10,
            block_ms=0,
        )
        contract_messages = await backend.stream_read(
            CONTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test",
            count=10,
            block_ms=0,
        )
        assert [message.payload["job_id"] for message in extract_messages] == [
            "job_extract"
        ]
        assert [message.payload["job_id"] for message in contract_messages] == [
            "job_contract"
        ]
        assert extract_messages[0].payload["payload"]["message_text"] == "First message."
        assert contract_messages[0].payload["payload"]["message_text"] == "Second message."
        assert contract_messages[0].payload["payload"]["recent_messages"] == [
            {
                "id": "aurvek:msg:1",
                "role": "user",
                "content": "First message.",
                "seq": 1,
                "occurred_at": None,
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_recovery_preserves_future_deferred_jobs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        await _insert_assistant_mode(connection)
        conversation = await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Tracked chat",
            platform_id="aurvek",
        )
        messages = MessageRepository(connection, clock)
        await messages.create_message(
            "aurvek:msg:1",
            "cnv_1",
            "user",
            1,
            "First message.",
        )
        jobs = JobRunRepository(connection, clock)
        await jobs.create_queued_job(
            job_id="job_extract_deferred",
            stream_name=EXTRACT_STREAM_NAME,
            job_type=JobType.EXTRACT_MEMORY_CANDIDATES.value,
            user_id="usr_1",
            conversation_id=str(conversation["id"]),
            source_message_ids=["aurvek:msg:1"],
            source_token_estimate=12,
            size_bucket="small",
            metadata={"operational_profile": "normal"},
            platform_id="aurvek",
        )
        await jobs.mark_running("job_extract_deferred", attempt_count=1)
        await jobs.mark_retrying(
            "job_extract_deferred",
            attempt_count=1,
            error_class="TransientLLMError",
            error_message="provider unavailable",
            deferred_until="2026-05-05T12:02:00+00:00",
        )

        result = await JobRecoveryService(
            connection,
            clock,
            settings=settings,
            storage_backend=backend,
            operational_profile_loader=OperationalProfileLoader(
                settings.operational_profiles_dir()
            ),
        ).recover_inprocess_stream_jobs()
        immediate = await backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test",
            count=10,
            block_ms=0,
        )
        recovered_job = await jobs.get_job("job_extract_deferred")

        assert result.recovered_jobs == 1
        assert immediate == []
        assert len(backend._stream_deferred[EXTRACT_STREAM_NAME]) == 1
        assert recovered_job is not None
        assert recovered_job["status"] == JobRunStatus.RETRYING.value
        assert recovered_job["deferred_until"] == "2026-05-05T12:02:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_recovery_requeues_initial_context_package_refresh_jobs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc))
    backend = InProcessBackend()
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        await _insert_assistant_mode(connection)
        conversation = await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Tracked chat",
            platform_id="aurvek",
        )
        jobs = JobRunRepository(connection, clock)
        await jobs.create_queued_job(
            job_id="job_initial_context",
            stream_name=INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value,
            user_id="usr_1",
            conversation_id=str(conversation["id"]),
            source_message_ids=[],
            source_token_estimate=None,
            size_bucket=None,
            metadata={
                "reason": "backfill",
                "package_kind": "all",
                "retrieval_profile_id": "coding_debug",
                "privacy_enforcement": "enforce",
                "operational_profile": "normal",
                "operational_profile_snapshot": _operational_snapshot(
                    "custom-recovery-token"
                ).model_dump(mode="json"),
            },
            platform_id="aurvek",
        )
        await jobs.mark_running("job_initial_context", attempt_count=1)

        result = await JobRecoveryService(
            connection,
            clock,
            settings=settings,
            storage_backend=backend,
            operational_profile_loader=OperationalProfileLoader(
                settings.operational_profiles_dir()
            ),
        ).recover_inprocess_stream_jobs()

        assert result.recovered_jobs == 1
        assert result.recovered_by_type == {
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 1,
        }
        recovered_job = await jobs.get_job("job_initial_context")
        assert recovered_job is not None
        assert recovered_job["status"] == JobRunStatus.QUEUED.value

        messages = await backend.stream_read(
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test",
            count=10,
            block_ms=0,
        )
        assert [message.payload["job_id"] for message in messages] == [
            "job_initial_context"
        ]
        payload = messages[0].payload["payload"]
        assert payload["reason"] == "backfill"
        assert payload["retrieval_profile_id"] == "coding_debug"
        assert messages[0].payload["operational_profile"]["profile_id"] == "normal"
        assert messages[0].payload["operational_profile"]["token"] == (
            "custom-recovery-token"
        )
    finally:
        await connection.close()
