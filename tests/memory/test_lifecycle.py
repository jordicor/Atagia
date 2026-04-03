"""Integration tests for memory lifecycle management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.lifecycle import MemoryLifecycleManager
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    contracts = ContractDimensionRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    return connection, clock, memories, contracts


async def _create_memory_at(
    memories: MemoryObjectRepository,
    clock: FrozenClock,
    *,
    created_at: datetime,
    memory_id: str,
    object_type: MemoryObjectType,
    scope: MemoryScope,
    canonical_text: str,
    confidence: float,
    vitality: float,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    assistant_mode_id: str | None = "coding_debug",
    conversation_id: str | None = "cnv_1",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    original = clock.current
    clock.current = created_at
    try:
        return await memories.create_memory_object(
            user_id="usr_1",
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            object_type=object_type,
            scope=scope,
            canonical_text=canonical_text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=confidence,
            vitality=vitality,
            privacy_level=0,
            status=status,
            memory_id=memory_id,
            payload=payload,
        )
    finally:
        clock.current = original


@pytest.mark.asyncio
async def test_vitality_decay_reduces_old_active_memory() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        memory = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_decay",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers short explanations",
            confidence=0.85,
            vitality=0.8,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()
        updated = await memories.get_memory_object(str(memory["id"]), "usr_1")

        assert result.decayed_count == 1
        assert updated is not None
        assert updated["vitality"] == pytest.approx(0.72)
        assert updated["updated_at"] == clock.now().isoformat()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_recent_memory_is_not_decayed() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        memory = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=2),
            memory_id="mem_recent",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Recently updated memory",
            confidence=0.75,
            vitality=0.6,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()
        updated = await memories.get_memory_object(str(memory["id"]), "usr_1")

        assert result.decayed_count == 0
        assert updated is not None
        assert updated["vitality"] == pytest.approx(0.6)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_archive_moves_low_value_non_evidence_and_preserves_summary() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        memory = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_archive",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers blunt answers",
            confidence=0.2,
            vitality=0.04,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()
        updated = await memories.get_memory_object(str(memory["id"]), "usr_1")

        assert result.archived_count == 1
        assert updated is not None
        assert updated["status"] == MemoryStatus.ARCHIVED.value
        assert updated["payload_json"]["archived_summary"] == "User prefers blunt answers"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_low_value_evidence_is_preserved() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        memory = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_evidence",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="User mentioned retries failing",
            confidence=0.2,
            vitality=0.04,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()
        updated = await memories.get_memory_object(str(memory["id"]), "usr_1")

        assert result.archived_count == 0
        assert result.skipped_evidence_count == 1
        assert updated is not None
        assert updated["status"] == MemoryStatus.ACTIVE.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_expired_ephemeral_objects_are_deleted() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=2),
            memory_id="mem_ephemeral",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.EPHEMERAL_SESSION,
            canonical_text="Temporary retry experiment",
            confidence=0.8,
            vitality=0.2,
            status=MemoryStatus.ARCHIVED,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        assert result.deleted_count == 1
        assert await memories.get_memory_object("mem_ephemeral", "usr_1") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_expired_review_required_objects_are_deleted() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=8),
            memory_id="mem_review",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Weak inference awaiting review",
            confidence=0.2,
            vitality=0.1,
            status=MemoryStatus.REVIEW_REQUIRED,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        assert result.deleted_count == 1
        assert await memories.get_memory_object("mem_review", "usr_1") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_non_expired_review_required_objects_survive() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=2),
            memory_id="mem_review_recent",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Recent weak inference",
            confidence=0.2,
            vitality=0.1,
            status=MemoryStatus.REVIEW_REQUIRED,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        assert result.deleted_count == 0
        assert await memories.get_memory_object("mem_review_recent", "usr_1") is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_dimensions_cleanup_after_archive() -> None:
    connection, clock, memories, contracts = await _build_runtime()
    try:
        memory = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_contract_source",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers concise answers",
            confidence=0.2,
            vitality=0.04,
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.ASSISTANT_MODE,
            dimension_name="directness",
            value_json={"label": "concise"},
            confidence=0.8,
            source_memory_id=str(memory["id"]),
        )

        await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        assert await contracts.count_for_context("usr_1", "coding_debug", None, None) == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_dimensions_fall_back_to_best_remaining_active_memory() -> None:
    connection, clock, memories, contracts = await _build_runtime()
    try:
        archived_source = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_contract_primary",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User strongly prefers direct answers",
            confidence=0.2,
            vitality=0.04,
            conversation_id=None,
            payload={
                "dimension_name": "directness",
                "value_json": {"label": "very_direct"},
            },
        )
        fallback_source = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=1),
            memory_id="mem_contract_fallback",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers direct answers",
            confidence=0.6,
            vitality=0.4,
            conversation_id=None,
            payload={
                "dimension_name": "directness",
                "value_json": {"label": "direct"},
            },
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.ASSISTANT_MODE,
            dimension_name="directness",
            value_json={"label": "very_direct"},
            confidence=0.9,
            source_memory_id=str(archived_source["id"]),
        )

        await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        projected_rows = await contracts.list_for_context("usr_1", "coding_debug", None, None)
        assert len(projected_rows) == 1
        assert projected_rows[0]["source_memory_id"] == str(fallback_source["id"])
        assert projected_rows[0]["value_json"] == {"label": "direct"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_dry_run_reports_changes_without_committing() -> None:
    connection, clock, memories, _contracts = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_dry_run",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="User prefers directness",
            confidence=0.2,
            vitality=0.04,
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle(dry_run=True)
        memory = await memories.get_memory_object("mem_dry_run", "usr_1")

        assert result.decayed_count == 1
        assert result.archived_count == 1
        assert memory is not None
        assert memory["status"] == MemoryStatus.ACTIVE.value
        assert memory["vitality"] == pytest.approx(0.04)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_full_lifecycle_cycle_runs_all_steps_in_order() -> None:
    connection, clock, memories, contracts = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_decay_only",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Useful stable belief",
            confidence=0.8,
            vitality=0.8,
        )
        archived_source = await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_archive_then_cleanup",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Low-value contract memory",
            confidence=0.2,
            vitality=0.04,
        )
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_evidence_keep",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Evidence to preserve",
            confidence=0.2,
            vitality=0.04,
        )
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=2),
            memory_id="mem_ephemeral_delete",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.EPHEMERAL_SESSION,
            canonical_text="Disposable session fact",
            confidence=0.7,
            vitality=0.2,
            status=MemoryStatus.ARCHIVED,
        )
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=8),
            memory_id="mem_review_delete",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Expired review candidate",
            confidence=0.2,
            vitality=0.1,
            status=MemoryStatus.REVIEW_REQUIRED,
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.ASSISTANT_MODE,
            dimension_name="directness",
            value_json={"label": "low_value"},
            confidence=0.6,
            source_memory_id=str(archived_source["id"]),
        )

        result = await MemoryLifecycleManager(connection, clock, _settings()).run_cycle()

        assert result.decayed_count == 3
        assert result.archived_count == 1
        assert result.deleted_count == 2
        assert result.skipped_evidence_count == 1
        assert await memories.get_memory_object("mem_ephemeral_delete", "usr_1") is None
        assert await memories.get_memory_object("mem_review_delete", "usr_1") is None
        assert await memories.get_memory_object("mem_archive_then_cleanup", "usr_1") is not None
        assert (
            await memories.get_memory_object("mem_archive_then_cleanup", "usr_1")
        )["status"] == MemoryStatus.ARCHIVED.value
        assert await contracts.count_for_context("usr_1", "coding_debug", None, None) == 0
    finally:
        await connection.close()
