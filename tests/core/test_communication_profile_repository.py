"""Tests for non-FTS user communication profile storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository, WorkspaceRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    LanguageProfileSourceRef,
    MemoryScope,
    ObservedUserLanguage,
    UserCommunicationProfile,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _seed_connection():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await UserRepository(connection, clock).create_user("usr_other")
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


def _context(
    *,
    user_id: str = "usr_1",
    conversation_id: str = "cnv_1",
    source_message_id: str = "msg_1",
    remember_across_chats: bool = True,
    character_id: str | None = "wrk_1",
    workspace_id: str | None = "wrk_1",
    platform_id: str = "mac",
    active_mind_id: str | None = "mind_1",
    incognito: bool = False,
) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id=user_id,
        conversation_id=conversation_id,
        source_message_id=source_message_id,
        workspace_id=workspace_id,
        assistant_mode_id="coding_debug",
        platform_id=platform_id,
        character_id=character_id,
        active_mind_id=active_mind_id,
        remember_across_chats=remember_across_chats,
        incognito=incognito,
    )


def _profile(
    *,
    source_message_id: str = "msg_1",
    conversation_id: str = "cnv_1",
    memory_id: str | None = None,
) -> UserCommunicationProfile:
    source_ref = (
        LanguageProfileSourceRef(
            source_kind="memory_object",
            memory_id=memory_id,
        )
        if memory_id is not None
        else LanguageProfileSourceRef(
            source_kind="source_message",
            source_message_id=source_message_id,
            conversation_id=conversation_id,
        )
    )
    return UserCommunicationProfile(
        observed_user_languages=[
            ObservedUserLanguage(
                language_code="es",
                message_count=1,
                last_seen_at="2026-05-20T12:00:00+00:00",
                source_refs=[source_ref],
                confidence=0.9,
            )
        ]
    )


@pytest.mark.asyncio
async def test_user_communication_profiles_are_dedicated_non_fts_rows() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(),
            _profile(),
            scope=MemoryScope.CHARACTER,
        )

        cursor = await connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE name LIKE 'user_communication_profiles%fts%'
            """
        )
        assert await cursor.fetchall() == []

        loaded = await repository.get_user_language_profile_for_context(_context())
        assert loaded is not None
        assert [row.language_code for row in loaded.observed_user_languages] == ["es"]

        other_user = await repository.get_user_language_profile_for_context(
            _context(user_id="usr_other")
        )
        assert other_user is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_visibility_prefers_chat_before_character_and_respects_stale() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        context = _context()
        await repository.upsert_user_language_profile(
            context,
            _profile(source_message_id="msg_character"),
            scope=MemoryScope.CHARACTER,
        )
        await repository.upsert_user_language_profile(
            context.model_copy(update={"remember_across_chats": False}),
            _profile(source_message_id="msg_chat"),
            scope=MemoryScope.CHAT,
        )

        loaded = await repository.get_user_language_profile_for_context(context)
        assert loaded is not None
        assert loaded.observed_user_languages[0].source_refs[0].source_message_id == "msg_chat"

        changed = await repository.mark_stale_for_source_message(
            user_id="usr_1",
            source_message_id="msg_chat",
            reason="source_message_deleted",
        )
        assert changed == 1

        loaded_after_stale = await repository.get_user_language_profile_for_context(context)
        assert loaded_after_stale is not None
        assert loaded_after_stale.observed_user_languages[0].source_refs[0].source_message_id == "msg_character"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_scope_falls_back_to_user_when_no_character_target_exists() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        context = _context(
            workspace_id=None,
            character_id=None,
            platform_id="mac",
            active_mind_id=None,
        )
        assert repository.target_scope_for_context(context) is MemoryScope.USER

        await repository.upsert_user_language_profile(
            context,
            _profile(),
            scope=MemoryScope.USER,
        )
        loaded = await repository.get_user_language_profile_for_context(context)
        assert loaded is not None
        assert loaded.observed_user_languages[0].language_code == "es"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_staleness_covers_conversation_and_memory_sources() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        await repository.upsert_user_language_profile(
            _context(source_message_id="msg_memory"),
            _profile(memory_id="mem_1"),
            scope=MemoryScope.CHARACTER,
        )

        memory_stale = await repository.mark_stale_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
            reason="source_memory_deleted",
        )
        assert memory_stale == 1
        assert await repository.get_user_language_profile_for_context(_context()) is None

        await repository.upsert_user_language_profile(
            _context(source_message_id="msg_conversation"),
            _profile(source_message_id="msg_conversation"),
            scope=MemoryScope.CHARACTER,
        )
        conversation_stale = await repository.mark_stale_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
            reason="source_conversation_deleted",
        )
        assert conversation_stale == 1
        assert await repository.get_user_language_profile_for_context(_context()) is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_legacy_invalid_language_codes_do_not_drop_entire_profile() -> None:
    connection, clock = await _seed_connection()
    try:
        repository = CommunicationProfileRepository(connection, clock)
        context = _context()
        await repository.upsert_user_language_profile(
            context,
            _profile(),
            scope=MemoryScope.CHARACTER,
        )
        row = await repository.get_profile_row_by_target(
            context,
            scope=MemoryScope.CHARACTER,
        )
        payload = dict(row["profile_json"])
        payload["observed_user_languages"] = [
            *payload["observed_user_languages"],
            {
                **payload["observed_user_languages"][0],
                "language_code": "jp",
            },
        ]
        payload["explicit_language_preferences"] = [
            {
                "language_code": "zz",
                "preference_kind": "default_answer_language",
                "context_label": "ordinary_chat",
                "source_refs": payload["observed_user_languages"][0]["source_refs"],
                "confidence": 0.8,
            },
            {
                "language_code": "ca",
                "preference_kind": "default_answer_language",
                "context_label": "ordinary_chat",
                "source_refs": payload["observed_user_languages"][0]["source_refs"],
                "confidence": 0.9,
            },
        ]
        await connection.execute(
            """
            UPDATE user_communication_profiles
            SET profile_json = ?
            WHERE id = ?
            """,
            (json_utils.dumps(payload, sort_keys=True), row["id"]),
        )
        await connection.commit()

        loaded = await repository.get_user_language_profile_for_context(context)

        assert loaded is not None
        assert [row.language_code for row in loaded.observed_user_languages] == ["es"]
        assert [
            row.language_code for row in loaded.explicit_language_preferences
        ] == ["ca"]
    finally:
        await connection.close()
