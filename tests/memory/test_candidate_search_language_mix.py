"""Tests for query-time language profile aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    MindTopology,
    SpaceBoundaryMode,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class FakeLanguageProfileEmbeddingIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    @property
    def vector_limit(self) -> int:
        return 10

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise AssertionError("upsert() is not used by language profile aggregation")

    async def search(self, query: str, user_id: str, top_k: int) -> list[object]:
        self.calls.append((query, user_id, top_k))
        return []

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used by language profile aggregation")


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
    user_id: str = "usr_1",
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
    character_id: str | None = None,
    scope_canonical: str | None = None,
    platform_id: str | None = None,
    platform_locked: bool = False,
    platform_id_lock: str | None = None,
    space_id: str | None = None,
    space_boundary_mode: str | None = None,
    memory_owner_id: str | None = None,
    embodiment_id: str | None = None,
    realm_id: str | None = None,
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
        privacy_level=privacy_level,
        intimacy_boundary=intimacy_boundary,
        intimacy_boundary_confidence=0.9 if intimacy_boundary is not IntimacyBoundary.ORDINARY else 0.0,
        status=status,
        language_codes=language_codes,
        memory_id=memory_id,
        character_id=character_id,
        scope_canonical=scope_canonical,
        platform_id=platform_id,
        platform_locked=platform_locked,
        platform_id_lock=platform_id_lock,
        space_id=space_id,
        space_boundary_mode=space_boundary_mode,
        memory_owner_id=memory_owner_id,
        embodiment_id=embodiment_id,
        realm_id=realm_id,
    )


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_filters_by_scope_status_privacy_and_reports_unknown_codes() -> None:
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

        profile = await search.aggregate_retrievable_content_language_mix(
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
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:06+00:00",
            },
            {
                "language_code": "es",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:01+00:00",
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
async def test_aggregate_retrievable_content_language_mix_uses_canonical_character_scope() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_legacy_workspace",
            canonical_text="legacy workspace language",
            language_codes=["es"],
            scope=MemoryScope.WORKSPACE,
            conversation_id=None,
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_character",
            canonical_text="canonical character language",
            language_codes=["ca"],
            scope=MemoryScope.WORKSPACE,
            conversation_id=None,
            character_id="wrk_1",
            scope_canonical=MemoryScope.CHARACTER.value,
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.WORKSPACE],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            character_id="wrk_1",
        )

        assert profile == [
            {
                "language_code": "ca",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:01+00:00",
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_limit_and_tiebreaks_by_last_seen() -> None:
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

        profile = await search.aggregate_retrievable_content_language_mix(
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
async def test_aggregate_retrievable_content_language_mix_returns_empty_on_cold_start() -> None:
    connection, _clock, _memories, search = await _build_runtime()
    try:
        profile = await search.aggregate_retrievable_content_language_mix(
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
async def test_aggregate_retrievable_content_language_mix_reports_unknown_for_missing_language_metadata() -> None:
    connection, _clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_missing_language_metadata",
            canonical_text="eligible memory without language metadata",
            language_codes=None,
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_ignores_legacy_invalid_language_codes() -> None:
    connection, _clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_legacy_invalid_only",
            canonical_text="legacy row with invalid metadata",
            language_codes=["en"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_legacy_mixed",
            canonical_text="legacy row with mixed metadata",
            language_codes=["ca"],
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '["jp"]'
            WHERE id = 'mem_legacy_invalid_only'
            """
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '["zz", "ca"]'
            WHERE id = 'mem_legacy_mixed'
            """
        )
        await connection.commit()

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "ca",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_treats_malformed_legacy_json_as_unknown() -> None:
    connection, _clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_valid_en",
            canonical_text="valid english metadata",
            language_codes=["en"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_malformed_language_metadata",
            canonical_text="legacy row with malformed metadata",
            language_codes=["ca"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_scalar_language_metadata",
            canonical_text="legacy row with scalar metadata",
            language_codes=["ca"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_object_language_metadata",
            canonical_text="legacy row with object metadata",
            language_codes=["ca"],
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = 'not json'
            WHERE id = 'mem_malformed_language_metadata'
            """
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '"en"'
            WHERE id = 'mem_scalar_language_metadata'
            """
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '{"language_code": "es"}'
            WHERE id = 'mem_object_language_metadata'
            """
        )
        await connection.commit()

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "unknown",
                "memory_count": 3,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
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
async def test_phase8_language_mix_uses_base_metadata_not_surfaces_or_index_text() -> None:
    connection, clock, memories, _search = await _build_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(connection, clock)
    fake_embedding_index = FakeLanguageProfileEmbeddingIndex()
    search_with_embedding_index = CandidateSearch(
        connection,
        clock,
        embedding_index=fake_embedding_index,
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_en_with_spanish_surface",
            canonical_text="english canonical profile source",
            index_text="indice espanol que no debe contar",
            language_codes=["en"],
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_en_with_spanish_surface",
            surface_type="alias",
            surface_text="superficie espanola que no debe contar",
            alias_kind="translation",
            language_code="es",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_unknown_with_spanish_surface",
            canonical_text="canonical source without language metadata",
            index_text="texte francais qui ne doit pas compter",
            language_codes=None,
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_unknown_with_spanish_surface",
            surface_type="alias",
            surface_text="otra superficie espanola que no debe contar",
            alias_kind="translation",
            language_code="es",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_private_es_blocked",
            canonical_text="blocked spanish memory",
            language_codes=["es"],
            privacy_level=3,
        )

        profile = await search_with_embedding_index.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:01+00:00",
            },
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
        ]
        assert fake_embedding_index.calls == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_hides_non_public_even_with_intimacy_context() -> None:
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

        ordinary_profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=2,
        )
        authorized_profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=2,
            allow_intimacy_context=True,
        )

        assert ordinary_profile == []
        assert authorized_profile == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_filters_by_user_id() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        users = UserRepository(connection, clock)
        workspaces = WorkspaceRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        await users.create_user("usr_2")
        await workspaces.create_workspace("wrk_user_2", "usr_2", "Other User Workspace")
        await conversations.create_conversation(
            "cnv_user_2",
            "usr_2",
            "wrk_user_2",
            "coding_debug",
            "Other User Chat",
        )
        await _seed_memory(
            memories,
            memory_id="mem_visible_en",
            canonical_text="visible english memory",
            language_codes=["en"],
        )
        await _seed_memory(
            memories,
            user_id="usr_2",
            memory_id="mem_other_user_es",
            canonical_text="other user spanish memory",
            language_codes=["es"],
            workspace_id="wrk_user_2",
            conversation_id="cnv_user_2",
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
        )

        assert profile == [
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_platform_visibility() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_default_en",
            canonical_text="default platform memory",
            language_codes=["en"],
            platform_id="default",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_locked_ios_es",
            canonical_text="ios locked memory",
            language_codes=["es"],
            platform_id="ios",
            platform_locked=True,
            platform_id_lock="ios",
        )

        default_profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            platform_id="default",
        )
        ios_profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            platform_id="ios",
        )

        assert [row["language_code"] for row in default_profile] == ["en"]
        assert [row["language_code"] for row in ios_profile] == ["es", "en"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_active_space_visibility() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active_space_en",
            canonical_text="active space english memory",
            language_codes=["en"],
            space_id="space_active",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_other_space_es",
            canonical_text="other space spanish memory",
            language_codes=["es"],
            space_id="space_other",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            active_space_id="space_active",
            active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
        )

        assert [row["language_code"] for row in profile] == ["en"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_active_mind_visibility() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active_mind_en",
            canonical_text="active mind english memory",
            language_codes=["en"],
            memory_owner_id="mind_active",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_other_mind_es",
            canonical_text="other mind spanish memory",
            language_codes=["es"],
            memory_owner_id="mind_other",
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            active_mind_id="mind_active",
            mind_topology=MindTopology.UNIMIND,
        )

        assert [row["language_code"] for row in profile] == ["en"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_active_embodiment_visibility() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active_body_en",
            canonical_text="active body english memory",
            language_codes=["en"],
            embodiment_id="body_active",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_other_body_es",
            canonical_text="other body spanish memory",
            language_codes=["es"],
            embodiment_id="body_other",
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            active_embodiment_id="body_active",
        )

        assert [row["language_code"] for row in profile] == ["en"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_aggregate_retrievable_content_language_mix_respects_active_realm_visibility() -> None:
    connection, clock, memories, search = await _build_runtime()
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active_realm_en",
            canonical_text="active realm english memory",
            language_codes=["en"],
            realm_id="realm_active",
        )
        clock.advance(seconds=1)
        await _seed_memory(
            memories,
            memory_id="mem_other_realm_es",
            canonical_text="other realm spanish memory",
            language_codes=["es"],
            realm_id="realm_other",
        )

        profile = await search.aggregate_retrievable_content_language_mix(
            user_id="usr_1",
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            privacy_ceiling=1,
            active_realm_id="realm_active",
        )

        assert [row["language_code"] for row in profile] == ["en"]
    finally:
        await connection.close()
