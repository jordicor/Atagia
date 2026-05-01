"""Tests for embedding cleanup during memory lifecycle changes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.consent_repository import PendingMemoryConfirmationRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.lifecycle import MemoryLifecycleManager
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryCategory, MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class TrackingEmbeddingIndex:
    vector_limit = 1

    def __init__(self) -> None:
        self.deleted_memory_ids: list[str] = []

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise AssertionError("upsert() is not used in lifecycle embedding tests")

    async def search(self, query: str, user_id: str, top_k: int):
        raise AssertionError("search() is not used in lifecycle embedding tests")

    async def delete(self, memory_id: str) -> None:
        self.deleted_memory_ids.append(memory_id)


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    return connection, clock, memories


async def _create_memory_at(
    memories: MemoryObjectRepository,
    clock: FrozenClock,
    *,
    created_at: datetime,
    memory_id: str,
    object_type: MemoryObjectType,
    scope: MemoryScope,
    status: MemoryStatus,
    confidence: float,
    vitality: float,
) -> None:
    original = clock.current
    clock.current = created_at
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=object_type,
            scope=scope,
            canonical_text=f"Memory {memory_id}",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=confidence,
            vitality=vitality,
            privacy_level=0,
            status=status,
            memory_id=memory_id,
        )
    finally:
        clock.current = original


@pytest.mark.asyncio
async def test_hard_delete_also_deletes_embedding() -> None:
    connection, clock, memories = await _build_runtime()
    embedding_index = TrackingEmbeddingIndex()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=8),
            memory_id="mem_review_delete",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            status=MemoryStatus.REVIEW_REQUIRED,
            confidence=0.2,
            vitality=0.1,
        )

        result = await MemoryLifecycleManager(
            connection,
            clock,
            _settings(),
            embedding_index=embedding_index,
        ).run_cycle()

        assert result.deleted_count == 1
        assert embedding_index.deleted_memory_ids == ["mem_review_delete"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_archive_also_deletes_embedding() -> None:
    connection, clock, memories = await _build_runtime()
    embedding_index = TrackingEmbeddingIndex()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=10),
            memory_id="mem_archive_keep_embedding",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            scope=MemoryScope.ASSISTANT_MODE,
            status=MemoryStatus.ACTIVE,
            confidence=0.2,
            vitality=0.04,
        )

        result = await MemoryLifecycleManager(
            connection,
            clock,
            _settings(),
            embedding_index=embedding_index,
        ).run_cycle()
        archived = await memories.get_memory_object("mem_archive_keep_embedding", "usr_1")

        assert result.archived_count == 1
        assert archived is not None
        assert archived["status"] == MemoryStatus.ARCHIVED.value
        assert embedding_index.deleted_memory_ids == ["mem_archive_keep_embedding"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_dry_run_skips_embedding_delete_side_effects() -> None:
    connection, clock, memories = await _build_runtime()
    embedding_index = TrackingEmbeddingIndex()
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=8),
            memory_id="mem_dry_run_keep_embedding",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            status=MemoryStatus.REVIEW_REQUIRED,
            confidence=0.2,
            vitality=0.1,
        )

        result = await MemoryLifecycleManager(
            connection,
            clock,
            _settings(),
            embedding_index=embedding_index,
        ).run_cycle(dry_run=True)

        stored = await memories.get_memory_object("mem_dry_run_keep_embedding", "usr_1")
        assert result.deleted_count == 1
        assert stored is not None
        assert embedding_index.deleted_memory_ids == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_expired_pending_confirmation_also_deletes_embedding_and_marker() -> None:
    connection, clock, memories = await _build_runtime()
    embedding_index = TrackingEmbeddingIndex()
    confirmations = PendingMemoryConfirmationRepository(connection, clock)
    try:
        await _create_memory_at(
            memories,
            clock,
            created_at=clock.now() - timedelta(days=8),
            memory_id="mem_pending_expired",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            confidence=0.9,
            vitality=0.4,
        )
        await confirmations.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id="mem_pending_expired",
            category=MemoryCategory.UNKNOWN,
            created_at=(clock.now() - timedelta(days=8)).isoformat(),
        )

        result = await MemoryLifecycleManager(
            connection,
            clock,
            _settings(),
            embedding_index=embedding_index,
        ).run_cycle()

        stored = await memories.get_memory_object("mem_pending_expired", "usr_1")
        assert result.declined_count == 1
        assert stored is not None
        assert stored["status"] == MemoryStatus.DECLINED.value
        assert embedding_index.deleted_memory_ids == ["mem_pending_expired"]
        assert await confirmations.get_marker_for_memory("usr_1", "mem_pending_expired") is None
    finally:
        await connection.close()
