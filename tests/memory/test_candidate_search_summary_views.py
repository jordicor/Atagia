"""Tests for retrieval of mirrored conversation-chunk summary views."""

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
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalPlan,
    SummaryViewKind,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime() -> tuple[object, MemoryObjectRepository, CandidateSearch]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "general_qa", "Chat")
    return connection, memories, search


def _plan(*, retrieval_levels: list[int]) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="general_qa",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["pottery"],
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=2,
        retrieval_levels=retrieval_levels,
        consequence_search_enabled=False,
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=False,
    )


@pytest.mark.asyncio
async def test_level_zero_retrieval_includes_conversation_chunk_mirrors() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Melanie showed Caroline a finished pottery plate.",
            index_text="pottery plate",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_evidence",
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_chunk_1",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=0,
            summary_text="Melanie signed up for a pottery class yesterday.",
            source_object_ids=["mem_evidence"],
            created_at="2026-04-05T12:00:00+00:00",
            updated_at="2026-04-05T12:00:00+00:00",
            index_text="conversation chunk: Melanie signed up for a pottery class yesterday.",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            confidence=0.68,
            stability=0.74,
            vitality=0.18,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={},
        )

        candidates = await search.search(_plan(retrieval_levels=[0]), "usr_1")

        candidate_ids = [candidate["id"] for candidate in candidates]

        assert "sum_mem_sum_chunk_1" in candidate_ids
        assert "mem_evidence" in candidate_ids
        chunk_candidate = next(candidate for candidate in candidates if candidate["id"] == "sum_mem_sum_chunk_1")
        assert chunk_candidate["payload_json"]["summary_kind"] == SummaryViewKind.CONVERSATION_CHUNK.value
        assert chunk_candidate["payload_json"]["hierarchy_level"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_nonzero_retrieval_levels_exclude_conversation_chunk_mirrors() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_chunk_1",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=0,
            summary_text="Melanie signed up for a pottery class yesterday.",
            source_object_ids=[],
            created_at="2026-04-05T12:00:00+00:00",
            updated_at="2026-04-05T12:00:00+00:00",
            index_text="conversation chunk: Melanie signed up for a pottery class yesterday.",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            confidence=0.68,
            stability=0.74,
            vitality=0.18,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={},
        )

        candidates = await search.search(_plan(retrieval_levels=[1]), "usr_1")

        assert candidates == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_level_zero_retrieval_excludes_malformed_conversation_chunk_mirrors() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_chunk_bad_level",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=1,
            summary_text="Malformed chunk summary that should never appear at level 0.",
            source_object_ids=[],
            created_at="2026-04-05T12:00:00+00:00",
            updated_at="2026-04-05T12:00:00+00:00",
            index_text="conversation chunk malformed pottery",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            confidence=0.68,
            stability=0.74,
            vitality=0.18,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={},
        )

        candidates = await search.search(_plan(retrieval_levels=[0]), "usr_1")

        assert [candidate["id"] for candidate in candidates] == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_level_zero_retrieval_keeps_user_id_isolation_for_conflicting_chunk_mirrors() -> None:
    connection, memories, search = await _build_runtime()
    try:
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_chunk_usr1",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=0,
            summary_text="User one pottery summary.",
            source_object_ids=[],
            created_at="2026-04-05T12:00:00+00:00",
            updated_at="2026-04-05T12:00:00+00:00",
            index_text="pottery summary for user one",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            confidence=0.68,
            stability=0.74,
            vitality=0.18,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={},
        )
        await memories.upsert_summary_mirror(
            user_id="usr_2",
            summary_view_id="sum_chunk_usr2",
            summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
            hierarchy_level=0,
            summary_text="User two conflicting pottery summary.",
            source_object_ids=[],
            created_at="2026-04-05T12:00:00+00:00",
            updated_at="2026-04-05T12:00:00+00:00",
            index_text="pottery summary for user two",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            confidence=0.68,
            stability=0.74,
            vitality=0.18,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={},
        )

        candidates = await search.search(_plan(retrieval_levels=[0]), "usr_1")
        candidate_ids = [candidate["id"] for candidate in candidates]

        assert "sum_mem_sum_chunk_usr1" in candidate_ids
        assert "sum_mem_sum_chunk_usr2" not in candidate_ids
    finally:
        await connection.close()
