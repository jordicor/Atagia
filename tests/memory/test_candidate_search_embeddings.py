"""Tests for embedding-backed candidate search."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    RetrievalPlan,
    VerbatimPinTargetKind,
)
from atagia.services.embeddings import EmbeddingMatch

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class FakeEmbeddingIndex:
    def __init__(self, matches: list[EmbeddingMatch], vector_limit: int = 100) -> None:
        self.matches = list(matches)
        self.calls: list[tuple[str, str, int]] = []
        self._vector_limit = vector_limit

    @property
    def vector_limit(self) -> int:
        return self._vector_limit

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise AssertionError("upsert() is not used in candidate search tests")

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        self.calls.append((query, user_id, top_k))
        return list(self.matches)

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used in candidate search tests")


async def _build_runtime(
    matches: list[EmbeddingMatch],
    settings: Settings | None = None,
    *,
    embedding_vector_limit: int = 100,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    embedding_index = FakeEmbeddingIndex(matches, vector_limit=embedding_vector_limit)
    search = CandidateSearch(connection, clock, embedding_index=embedding_index, settings=settings)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await workspaces.create_workspace("wrk_3", "usr_1", "Secondary Workspace")
    await workspaces.create_workspace("wrk_2", "usr_2", "Other Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "User One")
    await conversations.create_conversation("cnv_other", "usr_1", "wrk_1", "coding_debug", "User One Other")
    await conversations.create_conversation("cnv_2", "usr_2", "wrk_2", "coding_debug", "User Two")
    return connection, memories, search, embedding_index


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    user_id: str,
    canonical_text: str,
    scope: MemoryScope,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    assistant_mode_id: str | None = "coding_debug",
    temporal_type: str = "unknown",
    valid_from: str | None = None,
    valid_to: str | None = None,
    user_persona_id: str | None = None,
    platform_id: str | None = None,
    character_id: str | None = None,
    sensitivity: MemorySensitivity | None = None,
    platform_locked: bool = False,
    platform_id_lock: str | None = None,
    scope_canonical: str | None = None,
    index_text: str | None = None,
) -> None:
    await memories.create_memory_object(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=MemoryObjectType.EVIDENCE,
        scope=scope,
        canonical_text=canonical_text,
        index_text=index_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        temporal_type=temporal_type,
        valid_from=valid_from,
        valid_to=valid_to,
        memory_id=memory_id,
        user_persona_id=user_persona_id,
        platform_id=platform_id,
        character_id=character_id,
        sensitivity=sensitivity,
        platform_locked=platform_locked,
        platform_id_lock=platform_id_lock,
        scope_canonical=scope_canonical,
    )


def _plan(
    *,
    vector_limit: int = 5,
    max_candidates: int = 10,
    fts_queries: list[str] | None = None,
) -> RetrievalPlan:
    resolved_fts_queries = fts_queries or ["retry loop"]
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        user_persona_id=None,
        platform_id="default",
        character_id="wrk_1",
        fts_queries=resolved_fts_queries,
        sub_query_plans=[
            {
                "text": resolved_fts_queries[0],
                "fts_queries": resolved_fts_queries,
            }
        ],
        query_type="default",
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.WORKSPACE, MemoryScope.ASSISTANT_MODE],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=vector_limit,
        max_candidates=max_candidates,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=False,
    )


@pytest.mark.asyncio
async def test_search_by_embedding_returns_candidates_with_similarity_scores() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [EmbeddingMatch(memory_id="mem_1", score=0.88)]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            user_id="usr_1",
            canonical_text="Retry loop workaround",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert len(candidates) == 1
        assert candidates[0]["id"] == "mem_1"
        assert candidates[0]["similarity_score"] == pytest.approx(0.88)
        assert candidates[0]["position_rank"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_applies_phase7_namespace_platform_and_sensitivity_filters() -> None:
    matches = [
        EmbeddingMatch(memory_id="mem_chat", score=0.99, position_rank=1),
        EmbeddingMatch(memory_id="mem_character", score=0.98, position_rank=2),
        EmbeddingMatch(memory_id="mem_user", score=0.97, position_rank=3),
        EmbeddingMatch(memory_id="mem_other_chat", score=0.96, position_rank=4),
        EmbeddingMatch(memory_id="mem_other_character", score=0.95, position_rank=5),
        EmbeddingMatch(memory_id="mem_private", score=0.94, position_rank=6),
        EmbeddingMatch(memory_id="mem_locked_other_platform", score=0.93, position_rank=7),
    ]
    connection, memories, search, embedding_index = await _build_runtime(matches)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_chat",
            user_id="usr_1",
            canonical_text="chat namespace memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            scope_canonical=MemoryScope.CHAT.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_character",
            user_id="usr_1",
            canonical_text="character namespace memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_1",
            conversation_id=None,
            character_id="wrk_1",
            scope_canonical=MemoryScope.CHARACTER.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_user",
            user_id="usr_1",
            canonical_text="user namespace memory",
            scope=MemoryScope.GLOBAL_USER,
            workspace_id=None,
            conversation_id=None,
            scope_canonical=MemoryScope.USER.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_chat",
            user_id="usr_1",
            canonical_text="wrong chat memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_other",
            scope_canonical=MemoryScope.CHAT.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_character",
            user_id="usr_1",
            canonical_text="wrong character memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_3",
            conversation_id=None,
            character_id="wrk_3",
            scope_canonical=MemoryScope.CHARACTER.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_private",
            user_id="usr_1",
            canonical_text="private memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            sensitivity=MemorySensitivity.PRIVATE,
            scope_canonical=MemoryScope.CHAT.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_locked_other_platform",
            user_id="usr_1",
            canonical_text="locked other platform memory",
            scope=MemoryScope.GLOBAL_USER,
            workspace_id=None,
            conversation_id=None,
            platform_id="ios",
            platform_locked=True,
            platform_id_lock="ios",
            scope_canonical=MemoryScope.USER.value,
        )

        candidates = await search.search_by_embedding(
            query_text="namespace",
            user_id="usr_1",
            plan=_plan(vector_limit=10, max_candidates=10),
            embedding_index=embedding_index,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_chat",
            "mem_character",
            "mem_user",
        ]

        trusted_candidates = await search.search_by_embedding(
            query_text="namespace",
            user_id="usr_1",
            plan=_plan(vector_limit=10, max_candidates=10).model_copy(
                update={"allow_private_sensitivity": True}
            ),
            embedding_index=embedding_index,
        )

        assert [candidate["id"] for candidate in trusted_candidates] == [
            "mem_chat",
            "mem_character",
            "mem_user",
            "mem_private",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_allows_private_sensitivity_only_when_plan_allows_it() -> None:
    connection, memories, search, _embedding_index = await _build_runtime([])
    try:
        for memory_id, sensitivity in (
            ("mem_public", MemorySensitivity.PUBLIC),
            ("mem_private", MemorySensitivity.PRIVATE),
            ("mem_secret", MemorySensitivity.SECRET),
        ):
            await _seed_memory(
                memories,
                memory_id=memory_id,
                user_id="usr_1",
                canonical_text=f"{memory_id} sensitivity signal",
                index_text="sensitivity signal",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                sensitivity=sensitivity,
                scope_canonical=MemoryScope.CHAT.value,
            )

        normal_candidates = await search.search(
            _plan(vector_limit=0, fts_queries=["sensitivity signal"]).model_copy(
                update={"privacy_ceiling": 3}
            ),
            "usr_1",
        )
        trusted_candidates = await search.search(
            _plan(vector_limit=0, fts_queries=["sensitivity signal"]).model_copy(
                update={
                    "privacy_ceiling": 3,
                    "allow_private_sensitivity": True,
                }
            ),
            "usr_1",
        )

        assert {candidate["id"] for candidate in normal_candidates} == {"mem_public"}
        assert {candidate["id"] for candidate in trusted_candidates} == {
            "mem_public",
            "mem_private",
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pin_search_allows_private_sensitivity_only_when_plan_allows_it() -> None:
    connection, _memories, search, _embedding_index = await _build_runtime([])
    try:
        pins = VerbatimPinRepository(connection, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))
        for pin_id, sensitivity, privacy_level in (
            ("pin_public", MemorySensitivity.PUBLIC, 0),
            ("pin_private", MemorySensitivity.PRIVATE, 2),
            ("pin_secret", MemorySensitivity.SECRET, 3),
        ):
            await pins.create_verbatim_pin(
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id=f"msg_{pin_id}",
                pin_id=pin_id,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                canonical_text=f"{pin_id} exact recall token",
                index_text="exact recall token",
                privacy_level=privacy_level,
                sensitivity=sensitivity,
                created_by="usr_1",
                scope_canonical=MemoryScope.CHAT.value,
                platform_id="default",
            )

        normal_candidates = await search.search(
            _plan(vector_limit=0, fts_queries=["exact recall token"]).model_copy(
                update={
                    "privacy_ceiling": 3,
                    "raw_context_access_mode": "verbatim",
                    "scope_filter": [MemoryScope.CONVERSATION],
                }
            ),
            "usr_1",
        )
        trusted_candidates = await search.search(
            _plan(vector_limit=0, fts_queries=["exact recall token"]).model_copy(
                update={
                    "privacy_ceiling": 3,
                    "raw_context_access_mode": "verbatim",
                    "scope_filter": [MemoryScope.CONVERSATION],
                    "allow_private_sensitivity": True,
                }
            ),
            "usr_1",
        )

        assert {candidate["id"] for candidate in normal_candidates} == {"pin_public"}
        assert {candidate["id"] for candidate in trusted_candidates} == {
            "pin_public",
            "pin_private",
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_incognito_only_allows_active_chat() -> None:
    connection, memories, search, embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_chat", score=0.99),
            EmbeddingMatch(memory_id="mem_character", score=0.98),
            EmbeddingMatch(memory_id="mem_user", score=0.97),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_chat",
            user_id="usr_1",
            canonical_text="chat memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            scope_canonical=MemoryScope.CHAT.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_character",
            user_id="usr_1",
            canonical_text="character memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_1",
            conversation_id=None,
            character_id="wrk_1",
            scope_canonical=MemoryScope.CHARACTER.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_user",
            user_id="usr_1",
            canonical_text="user memory",
            scope=MemoryScope.GLOBAL_USER,
            workspace_id=None,
            conversation_id=None,
            scope_canonical=MemoryScope.USER.value,
        )
        plan = _plan(vector_limit=10).model_copy(update={"incognito": True})

        candidates = await search.search_by_embedding(
            query_text="memory",
            user_id="usr_1",
            plan=plan,
            embedding_index=embedding_index,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_chat"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_fts_search_applies_phase7_namespace_filters_before_selection() -> None:
    connection, memories, search, _embedding_index = await _build_runtime([])
    try:
        await _seed_memory(
            memories,
            memory_id="mem_chat",
            user_id="usr_1",
            canonical_text="namespace fts visible chat",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            scope_canonical=MemoryScope.CHAT.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_character",
            user_id="usr_1",
            canonical_text="namespace fts visible character",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_1",
            conversation_id=None,
            character_id="wrk_1",
            scope_canonical=MemoryScope.CHARACTER.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_legacy_workspace",
            user_id="usr_1",
            canonical_text="namespace fts legacy workspace",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_1",
            conversation_id=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_private",
            user_id="usr_1",
            canonical_text="namespace fts private chat",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            sensitivity=MemorySensitivity.PRIVATE,
            scope_canonical=MemoryScope.CHAT.value,
        )

        candidates = await search.search(
            _plan(vector_limit=0, fts_queries=["namespace fts"]),
            "usr_1",
        )

        assert {candidate["id"] for candidate in candidates} == {"mem_chat", "mem_character"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_filters_by_user_and_scope() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_ok", score=0.9, position_rank=1),
            EmbeddingMatch(memory_id="mem_other_user", score=0.8, position_rank=2),
            EmbeddingMatch(memory_id="mem_other_workspace", score=0.7, position_rank=3),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_ok",
            user_id="usr_1",
            canonical_text="Retry loop workaround",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_user",
            user_id="usr_2",
            canonical_text="Other user memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_2",
            conversation_id="cnv_2",
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_workspace",
            user_id="usr_1",
            canonical_text="Wrong workspace memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_3",
            conversation_id=None,
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_ok"]
        assert candidates[0]["position_rank"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_returns_empty_when_vector_limit_is_zero() -> None:
    connection, _memories, search, embedding_index = await _build_runtime(
        [EmbeddingMatch(memory_id="mem_1", score=0.88)]
    )
    try:
        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=0),
            embedding_index=embedding_index,
        )

        assert candidates == []
        assert embedding_index.calls == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_preserves_backend_rank_after_plan_filtering() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_filtered", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_survivor", score=0.91, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_filtered",
            user_id="usr_1",
            canonical_text="Filtered workspace memory",
            scope=MemoryScope.WORKSPACE,
            workspace_id="wrk_3",
            conversation_id=None,
        )
        await _seed_memory(
            memories,
            memory_id="mem_survivor",
            user_id="usr_1",
            canonical_text="Retry loop survivor",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_survivor"]
        assert candidates[0]["position_rank"] == 2
        assert candidates[0]["embedding_position_rank"] == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_enforces_vector_limit_after_overfetch() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_2", score=0.91, position_rank=2),
            EmbeddingMatch(memory_id="mem_3", score=0.90, position_rank=3),
        ]
    )
    try:
        for index in range(1, 4):
            await _seed_memory(
                memories,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                canonical_text=f"retry loop candidate {index}",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
            )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=1),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_respects_backend_vector_limit_cap() -> None:
    connection, memories, search, embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_2", score=0.91, position_rank=2),
            EmbeddingMatch(memory_id="mem_3", score=0.90, position_rank=3),
        ],
        embedding_vector_limit=2,
    )
    try:
        for index in range(1, 4):
            await _seed_memory(
                memories,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                canonical_text=f"retry loop candidate {index}",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
            )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=5),
            embedding_index=search._embedding_index,  # noqa: SLF001
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_1", "mem_2"]
        assert embedding_index.calls == [("retry loop", "usr_1", 8)]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_overfetches_and_demotes_stale_ephemeral_hits() -> None:
    connection, memories, search, embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_stale_ephemeral", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_current", score=0.91, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_stale_ephemeral",
            user_id="usr_1",
            canonical_text="retry loop stale state",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            temporal_type="ephemeral",
            valid_from="2026-04-03T09:00:00+00:00",
        )
        await _seed_memory(
            memories,
            memory_id="mem_current",
            user_id="usr_1",
            canonical_text="retry loop durable memory",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            temporal_type="permanent",
        )

        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=1),
            embedding_index=embedding_index,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_current"]
        assert embedding_index.calls == [("retry loop", "usr_1", 4)]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_by_embedding_uses_configured_overfetch_multiplier() -> None:
    settings = replace(Settings.from_env(), embedding_search_overfetch_multiplier=2)
    connection, _memories, search, embedding_index = await _build_runtime([], settings=settings)
    try:
        candidates = await search.search_by_embedding(
            query_text="retry loop",
            user_id="usr_1",
            plan=_plan(vector_limit=3),
            embedding_index=embedding_index,
        )

        assert candidates == []
        assert embedding_index.calls == [("retry loop", "usr_1", 6)]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_merged_fts_and_embedding_search_deduplicates_correctly() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.95, position_rank=1),
            EmbeddingMatch(memory_id="mem_2", score=0.82, position_rank=2),
        ]
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            user_id="usr_1",
            canonical_text="retry loop fastapi websocket",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_2",
            user_id="usr_1",
            canonical_text="Fallback memory for semantic match",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(_plan(), "usr_1")

        assert [candidate["id"] for candidate in candidates] == ["mem_1", "mem_2"]
        assert [candidate["id"] for candidate in candidates].count("mem_1") == 1
        assert candidates[0]["retrieval_sources"] == ["fts", "embedding"]
        assert "fts_rank" in candidates[0]
        assert candidates[0]["embedding_similarity_score"] == pytest.approx(0.95)
        assert "embedding_distance" in candidates[0]
        assert candidates[1]["retrieval_sources"] == ["embedding"]
        assert candidates[0]["rrf_score"] > candidates[1]["rrf_score"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_merged_search_respects_max_candidates_cap() -> None:
    connection, memories, search, _embedding_index = await _build_runtime(
        [
            EmbeddingMatch(memory_id="mem_1", score=0.91),
            EmbeddingMatch(memory_id="mem_2", score=0.88),
            EmbeddingMatch(memory_id="mem_3", score=0.87),
        ]
    )
    try:
        for index in range(1, 4):
            await _seed_memory(
                memories,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                canonical_text=f"semantic memory {index}",
                scope=MemoryScope.CONVERSATION,
                workspace_id="wrk_1",
                conversation_id="cnv_1",
            )

            candidates = await search.search(_plan(vector_limit=5, max_candidates=2), "usr_1")

        assert len(candidates) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rrf_uses_best_fts_rank_once_across_query_rewrites() -> None:
    connection, memories, search, _embedding_index = await _build_runtime([])
    try:
        await _seed_memory(
            memories,
            memory_id="mem_shared",
            user_id="usr_1",
            canonical_text="retry loop fastapi websocket",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )
        await _seed_memory(
            memories,
            memory_id="mem_broad",
            user_id="usr_1",
            canonical_text="retry loop fallback",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
        )

        candidates = await search.search(
            _plan(fts_queries=["retry loop fastapi", "retry loop"], vector_limit=0),
            "usr_1",
        )

        shared = next(candidate for candidate in candidates if candidate["id"] == "mem_shared")

        assert shared["retrieval_sources"] == ["fts"]
        assert shared["channel_ranks"] == {
            "verbatim_pin": None,
            "artifact_chunk": None,
            "fts": 1,
            "embedding": None,
            "consequence": None,
            "verbatim_evidence_search": None,
        }
        assert shared["matched_sub_queries"] == ["retry loop fastapi"]
        assert shared["subquery_ranks"]["retry loop fastapi"] >= 1
        assert shared["rrf_score_raw"] == pytest.approx(
            1.0 / (60 + int(shared["position_rank"]))
        )
        assert shared["rrf_score"] == pytest.approx(
            shared["rrf_score_raw"] / (1.0 / 61.0)
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_fts_bm25_column_weights_prefer_canonical_text_matches() -> None:
    settings = replace(
        Settings.from_env(),
        memory_fts_canonical_bm25_weight=5.0,
        memory_fts_index_bm25_weight=0.1,
    )
    connection, memories, search, _embedding_index = await _build_runtime([], settings=settings)
    try:
        await _seed_memory(
            memories,
            memory_id="mem_canonical",
            user_id="usr_1",
            canonical_text="quasar",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            index_text="unrelated",
        )
        await _seed_memory(
            memories,
            memory_id="mem_index",
            user_id="usr_1",
            canonical_text="unrelated",
            scope=MemoryScope.CONVERSATION,
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            index_text="quasar",
        )

        candidates = await search.search(_plan(fts_queries=["quasar"], vector_limit=0), "usr_1")

        assert [candidate["id"] for candidate in candidates[:2]] == ["mem_canonical", "mem_index"]
        assert candidates[0]["fts_rank"] < candidates[1]["fts_rank"]
    finally:
        await connection.close()


def test_rank_fusion_prefers_exact_single_hit_for_default_queries() -> None:
    search = CandidateSearch(None, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))

    _raw_exact, exact_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=3,
        query_type="default",
    )
    _raw_generic, generic_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1, 2],
        max_lists=3,
        query_type="default",
    )

    assert exact_score > generic_score


def test_rank_fusion_keeps_coverage_bias_for_broad_list_queries() -> None:
    search = CandidateSearch(None, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))

    _raw_exact, exact_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=3,
        query_type="broad_list",
    )
    _raw_generic, generic_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1, 2],
        max_lists=3,
        query_type="broad_list",
    )

    assert generic_score > exact_score


def test_rank_fusion_applies_query_type_channel_weighting() -> None:
    search = CandidateSearch(None, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc)))

    _raw_fts, exact_fts_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=1,
        query_type="slot_fill",
        exact_recall_mode=True,
        channel_ranks={"fts": 1},
    )
    _raw_embedding, exact_embedding_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=1,
        query_type="slot_fill",
        exact_recall_mode=True,
        channel_ranks={"embedding": 1},
    )
    _raw_broad_embedding, broad_embedding_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=1,
        query_type="broad_list",
        channel_ranks={"embedding": 1},
    )
    _raw_broad_verbatim, broad_verbatim_score = search._compute_rank_fusion_scores(  # noqa: SLF001
        [1],
        max_lists=1,
        query_type="broad_list",
        channel_ranks={"verbatim_evidence_search": 1},
    )

    assert exact_fts_score > exact_embedding_score
    assert broad_embedding_score > broad_verbatim_score
