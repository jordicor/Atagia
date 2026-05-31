"""Lifecycle invariants for user communication language profiles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    LanguageProfileSourceRef,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    ObservedUserLanguage,
    UserCommunicationProfile,
)
from atagia.services.lifecycle_service import (
    DELETE_CONVERSATION_CONFIRMATION,
    ERASE_ALL_DATA_CONFIRMATION,
    ConversationLifecycleService,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class _NoopEmbeddingIndex:
    async def delete(self, memory_id: str) -> None:
        del memory_id


async def _seed_connection():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await WorkspaceRepository(connection, clock).create_workspace("wrk_1", "usr_1", "Workspace")
    await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Chat",
        platform_id="mac",
        active_mind_id="mind_1",
    )
    return connection, clock


def _runtime(clock: FrozenClock) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(erasure_purge_streams=True),
        clock=clock,
        llm_client=None,
        storage_backend=InProcessBackend(),
        database_path=":memory:",
        artifact_blob_store=None,
        embedding_index=_NoopEmbeddingIndex(),
    )


def _context(*, source_message_id: str = "msg_1") -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id=source_message_id,
        workspace_id="wrk_1",
        assistant_mode_id="coding_debug",
        platform_id="mac",
        character_id="wrk_1",
        active_mind_id="mind_1",
    )


def _profile(*, source_message_id: str = "msg_1") -> UserCommunicationProfile:
    return UserCommunicationProfile(
        observed_user_languages=[
            ObservedUserLanguage(
                language_code="es",
                message_count=1,
                last_seen_at="2026-05-20T12:00:00+00:00",
                source_refs=[
                    LanguageProfileSourceRef(
                        source_kind="source_message",
                        conversation_id="cnv_1",
                        source_message_id=source_message_id,
                    )
                ],
                confidence=0.9,
            )
        ]
    )


def _memory_source_profile(*, memory_id: str = "mem_language_source") -> UserCommunicationProfile:
    return UserCommunicationProfile(
        observed_user_languages=[
            ObservedUserLanguage(
                language_code="ca",
                message_count=1,
                last_seen_at="2026-05-20T12:00:00+00:00",
                source_refs=[
                    LanguageProfileSourceRef(
                        source_kind="memory_object",
                        memory_id=memory_id,
                    )
                ],
                confidence=0.9,
            )
        ]
    )


async def _create_source_memory(
    connection,
    clock: FrozenClock,
    *,
    memory_id: str = "mem_language_source",
) -> None:
    await MemoryObjectRepository(connection, clock).create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="The user writes in Catalan in this source memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id=memory_id,
        language_codes=["en"],
    )


@pytest.mark.asyncio
async def test_archiving_conversation_marks_user_language_profile_stale() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(),
            _profile(),
            scope=MemoryScope.CHARACTER,
        )

        await ConversationLifecycleService(_runtime(clock)).archive_conversation(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        assert await repository.get_user_language_profile_for_context(_context()) is None
        row = await repository.get_profile_row_by_target(
            _context(),
            scope=MemoryScope.CHARACTER,
        )
        assert row["stale"] == 1
        assert row["stale_reason"] == "source_conversation_archived"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_archiving_conversation_marks_memory_sourced_language_profile_stale() -> None:
    connection, clock = await _seed_connection()
    try:
        await _create_source_memory(connection, clock)
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(),
            _memory_source_profile(),
            scope=MemoryScope.CHARACTER,
        )

        await ConversationLifecycleService(_runtime(clock)).archive_conversation(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        assert await repository.get_user_language_profile_for_context(_context()) is None
        row = await repository.get_profile_row_by_target(
            _context(),
            scope=MemoryScope.CHARACTER,
        )
        assert row["stale"] == 1
        assert row["stale_reason"] == "source_conversation_archived"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_deleting_conversation_marks_memory_sourced_language_profile_stale() -> None:
    connection, clock = await _seed_connection()
    try:
        await _create_source_memory(connection, clock)
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(),
            _memory_source_profile(),
            scope=MemoryScope.CHARACTER,
        )

        await ConversationLifecycleService(_runtime(clock)).delete_conversation(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
            confirmation=DELETE_CONVERSATION_CONFIRMATION,
        )

        assert await repository.get_user_language_profile_for_context(_context()) is None
        row = await repository.get_profile_row_by_target(
            _context(),
            scope=MemoryScope.CHARACTER,
        )
        assert row["stale"] == 1
        assert row["stale_reason"] == "source_conversation_deleted"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_right_to_erasure_deletes_user_language_profiles_before_user_row() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(),
            _profile(),
            scope=MemoryScope.CHARACTER,
        )

        report = await ConversationLifecycleService(_runtime(clock)).erase_user_data(
            connection,
            user_id="usr_1",
            confirmation=ERASE_ALL_DATA_CONFIRMATION,
        )

        assert report.user_id == "usr_1"
        profile_count = await _count(
            connection,
            "SELECT COUNT(*) AS count FROM user_communication_profiles WHERE user_id = ?",
            ("usr_1",),
        )
        user_count = await _count(
            connection,
            "SELECT COUNT(*) AS count FROM users WHERE id = ?",
            ("usr_1",),
        )
        assert profile_count == 0
        assert user_count == 0
    finally:
        await connection.close()


async def _count(connection, query: str, parameters: tuple[str, ...]) -> int:
    cursor = await connection.execute(query, parameters)
    row = await cursor.fetchone()
    return int(row["count"])
