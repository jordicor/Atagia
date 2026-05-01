"""Tests for query-time language profile aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Primary Workspace")
    await workspaces.create_workspace("wrk_2", "usr_1", "Other Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Coding")
    await conversations.create_conversation("cnv_2", "usr_1", "wrk_2", "personal_assistant", "Personal")
    return connection, clock, memories, search


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    language_codes: list[str] | None,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    assistant_mode_id: str = "coding_debug",
    workspace_id: str | None = "wrk_1",
    conversation_id: str | None = "cnv_1",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=MemoryObjectType.EVIDENCE,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=privacy_level,
        intimacy_boundary=intimacy_boundary,
        intimacy_boundary_confidence=0.9 if intimacy_boundary is not IntimacyBoundary.ORDINARY else 0.0,
        status=status,
        language_codes=language_codes,
        memory_id=memory_id,
    )


@pytest.mark.asyncio
async def test_aggregate_retrievable_language_mix_filters_by_scope_status_privacy_and_null_codes() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_en_1",
            canonical_text="english dosage memory",
            language_codes=["en"],
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_es_1",
            canonical_text="spanish dosage memory",
            language_codes=["es"],
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_es_2",
            canonical_text="workspace spanish memory",
            language_codes=["es"],
            scope=MemoryScope.WORKSPACE,
            conversation_id=None,
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_fr_pending",
            canonical_text="pending french memory",
            language_codes=["fr"],
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_de_private",
            canonical_text="private german memory",
            language_codes=["de"],
            privacy_level=3,
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_it_other_mode",
            canonical_text="other mode italian memory",
            language_codes=["it"],
            assistant_mode_id="personal_assistant",
            workspace_id="wrk_2",
            conversation_id="cnv_2",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_null_codes",
            canonical_text="no language metadata",
            language_codes=None,
        )

        profile = await search.aggregate_retrievable_language_mix(
            user_id="usr_1",
            scope_filter=[
                MemoryScope.CONVERSATION,
                MemoryScope.WORKSPACE,
                MemoryScope.ASSISTANT_MODE,
                MemoryScope.GLOBAL_USER,
            ],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "es",
                "memory_count": 2,
                "last_seen_at": "2026-04-05T12:00:02+00:00",
            },
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_language_mix_respects_limit_and_tiebreaks_by_last_seen() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_en",
            canonical_text="english memory",
            language_codes=["en"],
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_es",
            canonical_text="spanish memory",
            language_codes=["es"],
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_fr",
            canonical_text="french memory",
            language_codes=["fr"],
        )

        profile = await search.aggregate_retrievable_language_mix(
            user_id="usr_1",
            scope_filter=[
                MemoryScope.CONVERSATION,
                MemoryScope.WORKSPACE,
                MemoryScope.ASSISTANT_MODE,
                MemoryScope.GLOBAL_USER,
            ],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            limit=2,
        )

        assert [row["language_code"] for row in profile] == ["fr", "es"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_language_mix_returns_empty_on_cold_start() -> None:
    connection, _clock, _memories, search = await _build_runtime()
    try:
        profile = await search.aggregate_retrievable_language_mix(
            user_id="usr_1",
            scope_filter=[
                MemoryScope.CONVERSATION,
                MemoryScope.WORKSPACE,
                MemoryScope.ASSISTANT_MODE,
                MemoryScope.GLOBAL_USER,
            ],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_language_mix_can_include_authorized_intimacy_context() -> None:
    connection, _clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_private_es",
            canonical_text="private spanish continuity",
            language_codes=["es"],
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
        )

        ordinary_profile = await search.aggregate_retrievable_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=2,
        )
        authorized_profile = await search.aggregate_retrievable_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=2,
            allow_intimacy_context=True,
        )

        assert ordinary_profile == []
        assert authorized_profile == [
            {
                "language_code": "es",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
    finally:
        await connection.close()
