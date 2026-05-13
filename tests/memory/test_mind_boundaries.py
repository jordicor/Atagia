"""Tests for Mind perspective boundary behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.mind_repository import (
    DEFAULT_MIND_ID,
    DEFAULT_OVERSEER_MIND_ID,
    MindNotFoundError,
    MindRepository,
)
from atagia.core.overseer_grant_repository import OverseerGrantRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    MindKind,
    MindTopology,
    OverseerGrantKind,
    OverseerGrantTargetKind,
    PresenceKind,
    PlannedSubQuery,
    RetrievalPlan,
    ScoredCandidate,
    VerbatimPinTargetKind,
)
from atagia.services.context_cache_service import ContextCacheService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))


def _plan(
    *,
    active_mind_id: str | None,
    mind_topology: MindTopology | None,
    raw_context_access_mode: str = "normal",
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="alpha",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        platform_id="default",
        active_mind_id=active_mind_id,
        mind_topology=mind_topology or MindTopology.UNIMIND,
        fts_queries=["alpha"],
        sub_query_plans=[PlannedSubQuery(text="alpha", fts_queries=["alpha"])],
        scope_filter=[MemoryScope.USER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=20,
        max_context_items=20,
        privacy_ceiling=3,
        retrieval_levels=[0],
        raw_context_access_mode=raw_context_access_mode,
    )


def _resolved_policy():
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    return PolicyResolver().resolve(manifest, None, None)


async def _create_user_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    text: str,
    memory_owner_id: str | None,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return await memories.create_memory_object(
        user_id="usr_1",
        object_type=object_type,
        scope=MemoryScope.USER,
        canonical_text=text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        sensitivity=MemorySensitivity.PUBLIC,
        memory_id=memory_id,
        payload=payload or {},
        memory_owner_id=memory_owner_id,
        source_mind_id=memory_owner_id,
    )


async def _setup_connection():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = _clock()
    await UserRepository(connection, clock).create_user("usr_1")
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (clock.now().isoformat(), clock.now().isoformat()),
    )
    await connection.commit()
    return connection, clock


@pytest.mark.asyncio
async def test_explicit_mind_id_must_already_exist_for_owner_user() -> None:
    connection, clock = await _setup_connection()
    try:
        users = UserRepository(connection, clock)
        await users.create_user("usr_2")
        minds = MindRepository(connection, clock)

        await minds.resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            kind=MindKind.OWNED_AI,
            display_name="Alpha",
            source_kind="test",
            source_id="mind_alpha",
        )

        resolved = await minds.resolve_active_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            active_presence_id=None,
            topology=MindTopology.MULTI_MIND,
        )
        assert resolved["id"] == "mind_alpha"

        with pytest.raises(MindNotFoundError, match="Mind not found for user"):
            await minds.resolve_active_mind(
                owner_user_id="usr_1",
                mind_id="missing_mind",
                active_presence_id=None,
                topology=MindTopology.MULTI_MIND,
            )
        assert await minds.get_mind(
            owner_user_id="usr_1",
            mind_id="missing_mind",
        ) is None

        with pytest.raises(MindNotFoundError, match="Mind not found for user"):
            await minds.resolve_active_mind(
                owner_user_id="usr_2",
                mind_id="mind_alpha",
                active_presence_id=None,
                topology=MindTopology.MULTI_MIND,
            )
        assert await minds.get_mind(
            owner_user_id="usr_2",
            mind_id="mind_alpha",
        ) is None

        default_mind = await minds.resolve_active_mind(
            owner_user_id="usr_2",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.UNIMIND,
        )
        assert default_mind["id"] == DEFAULT_MIND_ID

        derived_presence_mind = await minds.resolve_active_mind(
            owner_user_id="usr_2",
            mind_id=None,
            active_presence_id="presence_alpha",
            active_presence_kind=PresenceKind.OWNED_FACET,
            active_presence_display_name="Presence Alpha",
            topology=MindTopology.MULTI_MIND,
        )
        assert derived_presence_mind["id"] == "presence_alpha"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ojocentauri_resolves_overseer_mind_and_rejects_non_overseer() -> None:
    connection, clock = await _setup_connection()
    try:
        minds = MindRepository(connection, clock)
        overseer = await minds.resolve_active_mind(
            owner_user_id="usr_1",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.OJOCENTAURI,
        )
        assert overseer["id"] == DEFAULT_OVERSEER_MIND_ID
        assert overseer["kind"] == MindKind.OVERSEER.value

        await minds.resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            kind=MindKind.OWNED_AI,
            display_name="Alpha",
            source_kind="test",
            source_id="mind_alpha",
        )
        with pytest.raises(MindNotFoundError, match="Overseer Mind not found"):
            await minds.resolve_active_mind(
                owner_user_id="usr_1",
                mind_id="mind_alpha",
                active_presence_id=None,
                topology=MindTopology.OJOCENTAURI,
            )
        with pytest.raises(ValueError, match="overseer_mind_id"):
            await OverseerGrantRepository(connection, clock).upsert_grant(
                owner_user_id="usr_1",
                overseer_mind_id="mind_alpha",
                target_kind=OverseerGrantTargetKind.MIND,
                target_id="mind_beta",
                grant_kind=OverseerGrantKind.READ,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ojocentauri_grants_fail_closed_for_unknown_targets() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        minds = MindRepository(connection, clock)
        grants = OverseerGrantRepository(connection, clock)
        await minds.resolve_active_mind(
            owner_user_id="usr_1",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.OJOCENTAURI,
        )
        await _create_user_memory(
            memories,
            memory_id="mem_overseer",
            text="alpha overseer coordination note",
            memory_owner_id=DEFAULT_OVERSEER_MIND_ID,
        )
        await _create_user_memory(
            memories,
            memory_id="mem_ghost",
            text="alpha ghost mind memory",
            memory_owner_id="ghost_mind",
        )

        unknown_targets = (
            (OverseerGrantTargetKind.MIND, "ghost_mind"),
            (OverseerGrantTargetKind.SPACE, "ghost_space"),
            (OverseerGrantTargetKind.REALM, "ghost_realm"),
        )
        for target_kind, target_id in unknown_targets:
            with pytest.raises(ValueError, match=f"{target_kind.value}:{target_id}"):
                await grants.upsert_grant(
                    owner_user_id="usr_1",
                    overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
                    target_kind=target_kind,
                    target_id=target_id,
                    grant_kind=OverseerGrantKind.READ,
                )
            assert await grants.get_grant(
                owner_user_id="usr_1",
                overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
                target_kind=target_kind,
                target_id=target_id,
                grant_kind=OverseerGrantKind.READ,
            ) is None

        timestamp = clock.now().isoformat()
        await connection.execute(
            """
            INSERT INTO overseer_grants(
                owner_user_id,
                overseer_mind_id,
                target_kind,
                target_id,
                grant_kind,
                visibility,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (
                'usr_1',
                ?,
                'mind',
                'ghost_mind',
                'read',
                'attributed',
                '{}',
                ?,
                ?
            )
            """,
            (DEFAULT_OVERSEER_MIND_ID, timestamp, timestamp),
        )
        await connection.commit()

        plan = _plan(
            active_mind_id=DEFAULT_OVERSEER_MIND_ID,
            mind_topology=MindTopology.OJOCENTAURI,
        )
        assert {row["id"] for row in await CandidateSearch(connection, clock).search(plan, "usr_1")} == {
            "mem_overseer"
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ojocentauri_ignores_malformed_grant_from_non_overseer_mind() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        minds = MindRepository(connection, clock)
        await minds.resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            kind=MindKind.OWNED_AI,
            display_name="Alpha",
            source_kind="test",
            source_id="mind_alpha",
        )
        await minds.resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_beta",
            kind=MindKind.OWNED_AI,
            display_name="Beta",
            source_kind="test",
            source_id="mind_beta",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_alpha",
            text="alpha owned-ai memory",
            memory_owner_id="mind_alpha",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_beta",
            text="alpha beta memory",
            memory_owner_id="mind_beta",
        )

        timestamp = clock.now().isoformat()
        await connection.execute(
            """
            INSERT INTO overseer_grants(
                owner_user_id,
                overseer_mind_id,
                target_kind,
                target_id,
                grant_kind,
                visibility,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (
                'usr_1',
                'mind_alpha',
                'mind',
                'mind_beta',
                'read',
                'attributed',
                '{}',
                ?,
                ?
            )
            """,
            (timestamp, timestamp),
        )
        await connection.commit()

        plan = _plan(
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.OJOCENTAURI,
        )
        rows = await CandidateSearch(connection, clock).search(plan, "usr_1")
        assert {row["id"] for row in rows} == {"mem_alpha"}
        assert all(row.get("mind_relation") != "granted" for row in rows)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ojocentauri_candidate_search_requires_active_grant() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        minds = MindRepository(connection, clock)
        grants = OverseerGrantRepository(connection, clock)
        await minds.resolve_active_mind(
            owner_user_id="usr_1",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.OJOCENTAURI,
        )
        for mind_id in ("mind_alpha", "mind_beta"):
            await minds.resolve_mind(
                owner_user_id="usr_1",
                mind_id=mind_id,
                kind=MindKind.OWNED_AI,
                display_name=mind_id,
                source_kind="test",
                source_id=mind_id,
            )
        await _create_user_memory(
            memories,
            memory_id="mem_overseer",
            text="alpha overseer coordination note",
            memory_owner_id=DEFAULT_OVERSEER_MIND_ID,
        )
        await _create_user_memory(
            memories,
            memory_id="mem_alpha",
            text="alpha memory owned by mind alpha",
            memory_owner_id="mind_alpha",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_beta",
            text="alpha memory owned by mind beta",
            memory_owner_id="mind_beta",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_legacy",
            text="alpha legacy unowned memory",
            memory_owner_id=None,
        )

        search = CandidateSearch(connection, clock)
        plan = _plan(
            active_mind_id=DEFAULT_OVERSEER_MIND_ID,
            mind_topology=MindTopology.OJOCENTAURI,
        )
        assert {row["id"] for row in await search.search(plan, "usr_1")} == {
            "mem_overseer"
        }

        await grants.upsert_grant(
            owner_user_id="usr_1",
            overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
            target_kind=OverseerGrantTargetKind.MIND,
            target_id="mind_alpha",
            grant_kind=OverseerGrantKind.READ,
        )
        await grants.upsert_grant(
            owner_user_id="usr_1",
            overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
            target_kind=OverseerGrantTargetKind.MIND,
            target_id="mind_beta",
            grant_kind=OverseerGrantKind.READ,
            expires_at="2000-01-01T00:00:00+00:00",
        )
        granted_rows = await search.search(plan, "usr_1")
        granted_by_id = {row["id"]: row for row in granted_rows}
        assert set(granted_by_id) == {"mem_overseer", "mem_alpha"}
        assert granted_by_id["mem_alpha"]["mind_relation"] == "granted"
        assert granted_by_id["mem_alpha"]["mind_grant_kind"] == "read"

        local_plan = _plan(
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert {row["id"] for row in await search.search(local_plan, "usr_1")} == {
            "mem_alpha"
        }

        await grants.revoke_grant(
            owner_user_id="usr_1",
            overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
            target_kind=OverseerGrantTargetKind.MIND,
            target_id="mind_alpha",
            grant_kind=OverseerGrantKind.READ,
        )
        assert {row["id"] for row in await search.search(plan, "usr_1")} == {
            "mem_overseer"
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ojocentauri_small_corpus_and_composer_preserve_grant_attribution() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        minds = MindRepository(connection, clock)
        grants = OverseerGrantRepository(connection, clock)
        await minds.resolve_active_mind(
            owner_user_id="usr_1",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.OJOCENTAURI,
        )
        await minds.resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            kind=MindKind.OWNED_AI,
            display_name="Alpha",
            source_kind="test",
            source_id="mind_alpha",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_alpha",
            text="alpha small corpus from mind alpha",
            memory_owner_id="mind_alpha",
        )
        await grants.upsert_grant(
            owner_user_id="usr_1",
            overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
            target_kind=OverseerGrantTargetKind.MIND,
            target_id="mind_alpha",
            grant_kind=OverseerGrantKind.READ,
        )

        rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_mind_id=DEFAULT_OVERSEER_MIND_ID,
            mind_topology=MindTopology.OJOCENTAURI,
        )
        assert len(rows) == 1
        assert rows[0]["mind_relation"] == "granted"
        assert rows[0]["mind_grant_target_id"] == "mind_alpha"

        context = ContextComposer(clock).compose(
            [
                ScoredCandidate(
                    memory_id=rows[0]["id"],
                    memory_object=dict(rows[0]),
                    llm_applicability=0.8,
                    retrieval_score=0.7,
                    vitality_boost=0.1,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=0.9,
                )
            ],
            current_contract={},
            user_state=None,
            resolved_policy=_resolved_policy(),
            conversation_messages=[],
        )
        assert (
            "mind: owned by mind_alpha [granted: read; target=mind:mind_alpha]"
            in context.memory_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_filters_by_active_mind_boundary() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_alpha",
            text="alpha memory owned by mind alpha",
            memory_owner_id="mind_alpha",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_beta",
            text="alpha memory owned by mind beta",
            memory_owner_id="mind_beta",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_legacy",
            text="alpha legacy unowned memory",
            memory_owner_id=None,
        )

        search = CandidateSearch(connection, clock)

        multi_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_mind_id="mind_alpha",
                    mind_topology=MindTopology.MULTI_MIND,
                ),
                "usr_1",
            )
        }
        assert multi_ids == {"mem_alpha"}

        unimind_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_mind_id="mind_alpha",
                    mind_topology=MindTopology.UNIMIND,
                ),
                "usr_1",
            )
        }
        assert unimind_ids == {"mem_alpha", "mem_legacy"}

        outside_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_mind_id=None, mind_topology=None),
                "usr_1",
            )
        }
        assert outside_ids == {"mem_legacy"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase3_cross_mind_visibility_fails_closed_without_grants() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_main",
            text="alpha main assistant local memory",
            memory_owner_id="mind_main",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_npc",
            text="alpha npc local observation",
            memory_owner_id="mind_npc",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_legacy",
            text="alpha legacy unowned note",
            memory_owner_id=None,
        )

        search = CandidateSearch(connection, clock)

        main_multi_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_mind_id="mind_main",
                    mind_topology=MindTopology.MULTI_MIND,
                ),
                "usr_1",
            )
        }
        assert main_multi_ids == {"mem_main"}

        main_unimind_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_mind_id="mind_main",
                    mind_topology=MindTopology.UNIMIND,
                ),
                "usr_1",
            )
        }
        assert main_unimind_ids == {"mem_main", "mem_legacy"}
        assert "mem_npc" not in main_unimind_ids

        npc_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_mind_id="mind_npc",
                    mind_topology=MindTopology.MULTI_MIND,
                ),
                "usr_1",
            )
        }
        assert npc_ids == {"mem_npc"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_small_corpus_and_state_snapshots_filter_by_active_mind() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_alpha",
            text="alpha corpus memory",
            memory_owner_id="mind_alpha",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_beta",
            text="alpha corpus other mind memory",
            memory_owner_id="mind_beta",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_legacy",
            text="alpha corpus legacy memory",
            memory_owner_id=None,
        )
        await _create_user_memory(
            memories,
            memory_id="state_alpha",
            text="alpha state",
            memory_owner_id="mind_alpha",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "alpha mind state"},
        )
        await _create_user_memory(
            memories,
            memory_id="state_beta",
            text="alpha state beta",
            memory_owner_id="mind_beta",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "beta mind state"},
        )

        multi_rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert {row["id"] for row in multi_rows} == {"mem_alpha", "state_alpha"}

        unimind_rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.UNIMIND,
        )
        assert {row["id"] for row in unimind_rows} == {
            "mem_alpha",
            "mem_legacy",
            "state_alpha",
        }

        state = await memories.get_state_snapshot(
            "usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert state == {"current_user_state": "alpha mind state"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_projection_is_scoped_by_memory_owner_mind() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        contracts = ContractDimensionRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="contract_alpha",
            text="alpha contract source",
            memory_owner_id="mind_alpha",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "alpha"}},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_beta",
            text="alpha contract source beta",
            memory_owner_id="mind_beta",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "beta"}},
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "alpha"},
            confidence=0.8,
            source_memory_id="contract_alpha",
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "beta"},
            confidence=0.8,
            source_memory_id="contract_beta",
        )

        alpha_rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert [(row["memory_owner_id"], row["value_json"]) for row in alpha_rows] == [
            ("mind_alpha", {"tone": "alpha"})
        ]

        beta_rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_mind_id="mind_beta",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert [(row["memory_owner_id"], row["value_json"]) for row in beta_rows] == [
            ("mind_beta", {"tone": "beta"})
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pins_filter_by_active_mind_in_crud_search_and_candidates() -> None:
    connection, clock = await _setup_connection()
    try:
        pins = VerbatimPinRepository(connection, clock)
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_alpha",
            pin_id="pin_alpha",
            canonical_text="alpha exact pin for mind alpha",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            memory_owner_id="mind_alpha",
            source_mind_id="mind_alpha",
        )
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_beta",
            pin_id="pin_beta",
            canonical_text="alpha exact pin for mind beta",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            memory_owner_id="mind_beta",
            source_mind_id="mind_beta",
        )

        assert await pins.get_verbatim_pin(
            "pin_beta",
            "usr_1",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        ) is None
        inside = await pins.get_verbatim_pin(
            "pin_alpha",
            "usr_1",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert inside is not None
        assert inside["memory_owner_id"] == "mind_alpha"

        search_rows = await pins.search_active_verbatim_pins(
            user_id="usr_1",
            query="alpha",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            limit=10,
            platform_id="default",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert {row["id"] for row in search_rows} == {"pin_alpha"}

        candidate_rows = await CandidateSearch(connection, clock).search(
            _plan(
                active_mind_id="mind_alpha",
                mind_topology=MindTopology.MULTI_MIND,
                raw_context_access_mode="verbatim",
            ),
            "usr_1",
        )
        assert {row["id"] for row in candidate_rows} == {"pin_alpha"}
        assert candidate_rows[0]["memory_owner_id"] == "mind_alpha"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunk_search_filters_and_carries_mind_fields() -> None:
    connection, clock = await _setup_connection()
    try:
        conversations = ConversationRepository(connection, clock)
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Chat",
            platform_id="default",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        artifacts = ArtifactRepository(connection, clock)
        await artifacts.create_artifact(
            artifact_id="art_alpha",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="upload",
            source_ref="alpha.txt",
            title="Alpha",
            status="ready",
            privacy_level=0,
            skip_raw_by_default=False,
            requires_explicit_request=False,
            summary_text="alpha artifact summary",
            index_text="alpha artifact",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
            memory_owner_id="mind_alpha",
            source_mind_id="mind_alpha",
        )
        await artifacts.create_artifact(
            artifact_id="art_beta",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="upload",
            source_ref="beta.txt",
            title="Beta",
            status="ready",
            privacy_level=0,
            skip_raw_by_default=False,
            requires_explicit_request=False,
            summary_text="alpha artifact beta",
            index_text="alpha artifact",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
            memory_owner_id="mind_beta",
            source_mind_id="mind_beta",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_alpha",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for mind alpha",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_alpha",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_beta",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for mind beta",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_beta",
        )

        rows = await artifacts.search_artifact_chunks(
            user_id="usr_1",
            query="alpha",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.CHAT],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            limit=10,
            platform_id="default",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        assert {row["artifact_id"] for row in rows} == {"art_alpha"}
        assert rows[0]["artifact_memory_owner_id"] == "mind_alpha"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_candidates_carry_active_mind_fields() -> None:
    connection, clock = await _setup_connection()
    try:
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Chat",
            platform_id="default",
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
        )
        await messages.create_message(
            "msg_alpha",
            "cnv_1",
            "user",
            1,
            "alpha verbatim transcript detail",
            5,
            {},
            active_mind_id="mind_alpha",
            source_mind_id="mind_alpha",
        )

        plan = _plan(
            active_mind_id="mind_alpha",
            mind_topology=MindTopology.MULTI_MIND,
            raw_context_access_mode="verbatim",
        ).model_copy(update={"scope_filter": [MemoryScope.CHAT]})
        rows = await CandidateSearch(connection, clock).search(
            plan,
            "usr_1",
        )
        evidence_windows = [
            row for row in rows if row.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows
        assert evidence_windows[0]["memory_owner_id"] == "mind_alpha"
        assert evidence_windows[0]["source_mind_id"] == "mind_alpha"
    finally:
        await connection.close()


def test_context_cache_key_partitions_by_active_mind() -> None:
    base = {
        "user_id": "usr_1",
        "assistant_mode_id": "coding_debug",
        "conversation_id": "cnv_1",
        "workspace_id": None,
        "operational_profile_token": "default",
        "active_presence_id": "presence_1",
        "active_space_id": None,
    }
    alpha_key = ContextCacheService.build_cache_key(
        **base,
        active_mind_id="mind_alpha",
        mind_topology="multi_mind",
    )
    beta_key = ContextCacheService.build_cache_key(
        **base,
        active_mind_id="mind_beta",
        mind_topology="multi_mind",
    )
    unimind_key = ContextCacheService.build_cache_key(
        **base,
        active_mind_id="mind_alpha",
        mind_topology="unimind",
    )

    assert alpha_key != beta_key
    assert alpha_key != unimind_key
