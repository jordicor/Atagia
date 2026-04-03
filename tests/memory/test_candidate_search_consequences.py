"""Tests for consequence-chain candidate search integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.consequence_repository import ConsequenceRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    NeedTrigger,
    RetrievalPlan,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 2, 15, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    chains = ConsequenceRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    return connection, memories, chains, search


async def _seed_chain(
    memories: MemoryObjectRepository,
    chains: ConsequenceRepository,
    *,
    include_tendency: bool = True,
) -> tuple[str, str, str | None]:
    action = await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Suggested a large refactor to simplify the code path.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        payload={"source_message_ids": ["msg_assistant_1"]},
        memory_id="mem_action",
    )
    outcome = await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Regressions followed after the refactor suggestion.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        payload={"source_message_ids": ["msg_user_1"]},
        memory_id="mem_outcome",
    )
    tendency = None
    if include_tendency:
        tendency = await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.CONSEQUENCE_CHAIN,
            scope=MemoryScope.WORKSPACE,
            canonical_text="Prefer incremental patches in this workspace.",
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.64,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            payload={"source_message_ids": ["msg_user_1", "msg_assistant_1"]},
            memory_id="mem_tendency",
        )
    await chains.create_chain(
        {
            "id": "chn_1",
            "user_id": "usr_1",
            "workspace_id": "wrk_1",
            "conversation_id": "cnv_1",
            "assistant_mode_id": "coding_debug",
            "action_memory_id": str(action["id"]),
            "outcome_memory_id": str(outcome["id"]),
            "tendency_belief_id": None if tendency is None else str(tendency["id"]),
            "confidence": 0.8,
            "status": "active",
            "created_at": "2026-04-02T15:00:00+00:00",
            "updated_at": "2026-04-02T15:00:00+00:00",
        }
    )
    return str(action["id"]), str(outcome["id"]), None if tendency is None else str(tendency["id"])


def _plan(need_type: NeedTrigger | None) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["regressions"],
        scope_filter=[MemoryScope.WORKSPACE, MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        require_evidence_regrounding=False,
        need_driven_boosts={} if need_type is None else {need_type: 1.0},
        skip_retrieval=False,
    )


@pytest.mark.asyncio
async def test_consequence_chains_surface_for_follow_up_failure_need_signal() -> None:
    connection, memories, chains, search = await _build_runtime()
    try:
        _, _, tendency_id = await _seed_chain(memories, chains)

        candidates = await search.search(_plan(NeedTrigger.FOLLOW_UP_FAILURE), "usr_1")

        assert tendency_id in [candidate["id"] for candidate in candidates]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_consequence_chains_surface_for_loop_need_signal() -> None:
    connection, memories, chains, search = await _build_runtime()
    try:
        _, _, tendency_id = await _seed_chain(memories, chains)

        candidates = await search.search(_plan(NeedTrigger.LOOP), "usr_1")

        assert tendency_id in [candidate["id"] for candidate in candidates]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_consequence_chains_do_not_surface_for_other_need_signals() -> None:
    connection, memories, chains, search = await _build_runtime()
    try:
        _, _, tendency_id = await _seed_chain(memories, chains)

        candidates = await search.search(_plan(NeedTrigger.AMBIGUITY), "usr_1")

        assert tendency_id not in [candidate["id"] for candidate in candidates]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_consequence_chains_without_tendency_surface_outcome_memory() -> None:
    connection, memories, chains, search = await _build_runtime()
    try:
        _, outcome_id, tendency_id = await _seed_chain(memories, chains, include_tendency=False)

        candidates = await search.search(_plan(NeedTrigger.FOLLOW_UP_FAILURE), "usr_1")

        assert tendency_id is None
        assert outcome_id in [candidate["id"] for candidate in candidates]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_consequence_chain_search_rejects_invalid_match_column() -> None:
    connection, _memories, _chains, search = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="Invalid match column"):
            await search._search_consequence_chain_matches(  # noqa: SLF001
                plan=_plan(NeedTrigger.FOLLOW_UP_FAILURE),
                user_id="usr_1",
                fts_query="regressions",
                match_column="bad_column",
            )
    finally:
        await connection.close()
