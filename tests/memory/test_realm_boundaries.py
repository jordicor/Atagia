"""Tests for Realm world/domain boundary behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.realm_repository import RealmRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    CrossRealmMode,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    ScoredCandidate,
    VerbatimPinTargetKind,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.embeddings import EmbeddingIndex, EmbeddingMatch

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))


def _plan(
    *,
    active_realm_id: str | None,
    raw_context_access_mode: str = "normal",
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="alpha",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        platform_id="default",
        active_realm_id=active_realm_id,
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


class _FakeEmbeddingIndex(EmbeddingIndex):
    def __init__(self, matches: list[EmbeddingMatch]) -> None:
        self._matches = matches

    @property
    def vector_limit(self) -> int:
        return 10

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        return None

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        return self._matches[:top_k]

    async def delete(self, memory_id: str) -> None:
        return None


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


async def _create_user_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    text: str,
    realm_id: str | None,
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
        realm_id=realm_id,
    )


async def _create_realms(connection, clock) -> RealmRepository:
    realms = RealmRepository(connection, clock)
    await realms.resolve_realm(
        owner_user_id="usr_1",
        realm_id="realm_real",
        cross_realm_mode=CrossRealmMode.NONE,
        display_name="Real world",
        source_kind="explicit",
        source_id="realm_real",
    )
    await realms.resolve_realm(
        owner_user_id="usr_1",
        realm_id="realm_aincrad",
        cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        display_name="Aincrad",
        source_kind="explicit",
        source_id="realm_aincrad",
    )
    return realms


@pytest.mark.asyncio
async def test_candidate_search_filters_by_active_realm_and_explicit_bridge() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_real",
            text="alpha real-world memory",
            realm_id="realm_real",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_aincrad",
            text="alpha fiction memory from aincrad",
            realm_id="realm_aincrad",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_unscoped",
            text="alpha unscoped legacy memory",
            realm_id=None,
        )

        search = CandidateSearch(connection, clock)
        real_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_realm_id="realm_real"),
                "usr_1",
            )
        }
        assert real_ids == {"mem_real", "mem_unscoped"}

        outside_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_realm_id=None),
                "usr_1",
            )
        }
        assert outside_ids == {"mem_unscoped"}

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )
        bridged_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_realm_id="realm_real"),
                "usr_1",
            )
        }
        assert bridged_ids == {"mem_real", "mem_aincrad", "mem_unscoped"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_embedding_candidates_require_explicit_realm_bridge() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_aincrad",
            text="embedding-only aincrad memory",
            realm_id="realm_aincrad",
            payload={
                "realm": {
                    "active_realm_id": "realm_aincrad",
                    "cross_realm_mode": "attributed",
                }
            },
        )
        plan = _plan(active_realm_id="realm_real").model_copy(
            update={"vector_limit": 5}
        )
        search = CandidateSearch(
            connection,
            clock,
            embedding_index=_FakeEmbeddingIndex(
                [
                    EmbeddingMatch(
                        memory_id="mem_aincrad",
                        score=0.96,
                        position_rank=1,
                    )
                ]
            ),
        )

        no_bridge_rows = await search.search(plan, "usr_1")
        assert no_bridge_rows == []

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )
        attributed_rows = await search.search(plan, "usr_1")
        assert {row["id"] for row in attributed_rows} == {"mem_aincrad"}
        assert attributed_rows[0]["realm_relation"] == "cross"
        assert attributed_rows[0]["realm_bridge_mode"] == "attributed"

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.APPLICABLE,
        )
        applicable_rows = await search.search(plan, "usr_1")
        assert {row["id"] for row in applicable_rows} == {"mem_aincrad"}
        assert applicable_rows[0]["realm_relation"] == "cross"
        assert applicable_rows[0]["realm_bridge_mode"] == "applicable"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_small_corpus_state_and_visible_memory_filter_by_realm() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_real",
            text="alpha corpus real memory",
            realm_id="realm_real",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_aincrad",
            text="alpha corpus aincrad memory",
            realm_id="realm_aincrad",
        )
        await _create_user_memory(
            memories,
            memory_id="state_real",
            text="alpha real state",
            realm_id="realm_real",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "real-world state"},
        )

        rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_realm_id="realm_real",
        )
        assert {row["id"] for row in rows} == {"mem_real", "state_real"}

        outside_rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_realm_id=None,
        )
        assert outside_rows == []

        token_length = await memories.sum_canonical_text_length_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_realm_id="realm_real",
        )
        assert token_length == len("alpha corpus real memory") + len("alpha real state")

        state = await memories.get_state_snapshot(
            "usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_realm_id="realm_real",
        )
        assert state == {"current_user_state": "real-world state"}

        assert await memories.get_visible_memory_object(
            "mem_aincrad",
            "usr_1",
            conversation_id="cnv_1",
            user_persona_id=None,
            platform_id="default",
            character_id=None,
            incognito=False,
            remember_across_chats=True,
            remember_across_devices=True,
            active_realm_id="realm_real",
        ) is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_small_corpus_list_eligible_annotates_realm_bridge_for_composer() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_aincrad",
            text="alpha bridged small corpus memory",
            realm_id="realm_aincrad",
            payload={
                "realm": {
                    "active_realm_id": "realm_aincrad",
                    "cross_realm_mode": "attributed",
                }
            },
        )

        no_bridge_rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_realm_id="realm_real",
        )
        assert no_bridge_rows == []

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )
        rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_realm_id="realm_real",
        )
        assert len(rows) == 1
        assert rows[0]["realm_relation"] == "cross"
        assert rows[0]["realm_bridge_mode"] == "attributed"

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
            active_realm_id="realm_real",
        )
        assert "cross_realm: attributed" in context.memory_block
        assert "cross_realm: unknown" not in context.memory_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_projection_is_scoped_by_realm() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        contracts = ContractDimensionRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="contract_real",
            text="alpha real contract source",
            realm_id="realm_real",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "real"}},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_aincrad",
            text="alpha aincrad contract source",
            realm_id="realm_aincrad",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "raid_tone", "value_json": {"tone": "fiction"}},
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "real"},
            confidence=0.8,
            source_memory_id="contract_real",
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="raid_tone",
            value_json={"tone": "fiction"},
            confidence=0.8,
            source_memory_id="contract_aincrad",
        )

        rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_realm_id="realm_real",
        )
        assert [(row["realm_id"], row["value_json"]) for row in rows] == [
            ("realm_real", {"tone": "real"})
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_attributed_realm_bridge_does_not_apply_state_or_contract() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        contracts = ContractDimensionRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="state_real",
            text="alpha real state",
            realm_id="realm_real",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "real-world state"},
        )
        await _create_user_memory(
            memories,
            memory_id="state_aincrad",
            text="alpha aincrad state",
            realm_id="realm_aincrad",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "aincrad state"},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_real",
            text="alpha real contract source",
            realm_id="realm_real",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "real"}},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_aincrad",
            text="alpha aincrad contract source",
            realm_id="realm_aincrad",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "raid_tone", "value_json": {"tone": "fiction"}},
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "real"},
            confidence=0.8,
            source_memory_id="contract_real",
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="raid_tone",
            value_json={"tone": "fiction"},
            confidence=0.8,
            source_memory_id="contract_aincrad",
        )
        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )

        state = await memories.get_state_snapshot(
            "usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_realm_id="realm_real",
        )
        rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_realm_id="realm_real",
        )

        assert state == {"current_user_state": "real-world state"}
        assert [(row["realm_id"], row["value_json"]) for row in rows] == [
            ("realm_real", {"tone": "real"})
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_applicable_realm_bridge_can_apply_state_and_contract() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        contracts = ContractDimensionRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="state_real",
            text="alpha real state",
            realm_id="realm_real",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "real-world state"},
        )
        await _create_user_memory(
            memories,
            memory_id="state_aincrad",
            text="alpha aincrad state",
            realm_id="realm_aincrad",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "aincrad state"},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_real",
            text="alpha real contract source",
            realm_id="realm_real",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "real"}},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_aincrad",
            text="alpha aincrad contract source",
            realm_id="realm_aincrad",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "raid_tone", "value_json": {"tone": "fiction"}},
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "real"},
            confidence=0.8,
            source_memory_id="contract_real",
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="raid_tone",
            value_json={"tone": "fiction"},
            confidence=0.8,
            source_memory_id="contract_aincrad",
        )
        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.APPLICABLE,
        )

        state = await memories.get_state_snapshot(
            "usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_realm_id="realm_real",
        )
        rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_realm_id="realm_real",
        )
        await connection.execute(
            """
            UPDATE assistant_modes
            SET memory_policy_json = ?
            WHERE id = 'coding_debug'
            """,
            ((MANIFESTS_DIR / "coding_debug.json").read_text(),),
        )
        await connection.commit()
        current_contract = await ContractProjector(
            llm_client=object(),
            clock=clock,
            message_repository=MessageRepository(connection, clock),
            memory_repository=memories,
            contract_repository=contracts,
        ).get_current_contract(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_realm_id="realm_real",
        )

        assert state == {
            "current_user_state": {
                "value": "aincrad state",
                "realm": {
                    "active_realm_id": "realm_aincrad",
                    "active_request_realm_id": "realm_real",
                    "cross_realm_mode": "applicable",
                },
            }
        }
        assert {row["realm_id"] for row in rows} == {"realm_real", "realm_aincrad"}
        assert current_contract["raid_tone"]["realm"] == {
            "active_realm_id": "realm_aincrad",
            "active_request_realm_id": "realm_real",
            "cross_realm_mode": "applicable",
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pins_filter_by_active_realm_in_crud_search_and_candidates() -> None:
    connection, clock = await _setup_connection()
    try:
        pins = VerbatimPinRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_real",
            pin_id="pin_real",
            canonical_text="alpha exact pin for real world",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            realm_id="realm_real",
        )
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_aincrad",
            pin_id="pin_aincrad",
            canonical_text="alpha exact pin for aincrad",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            realm_id="realm_aincrad",
        )

        assert await pins.get_verbatim_pin(
            "pin_aincrad",
            "usr_1",
            active_realm_id="realm_real",
        ) is None
        assert await pins.get_verbatim_pin(
            "pin_real",
            "usr_1",
            active_realm_id="realm_real",
        ) is not None

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
            active_realm_id="realm_real",
        )
        assert {row["id"] for row in search_rows} == {"pin_real"}

        candidate_rows = await CandidateSearch(connection, clock).search(
            _plan(
                active_realm_id="realm_real",
                raw_context_access_mode="verbatim",
            ),
            "usr_1",
        )
        assert {row["id"] for row in candidate_rows} == {"pin_real"}
        assert candidate_rows[0]["realm_id"] == "realm_real"

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )
        bridged_candidate_rows = await CandidateSearch(connection, clock).search(
            _plan(
                active_realm_id="realm_real",
                raw_context_access_mode="verbatim",
            ),
            "usr_1",
        )
        bridged_by_id = {row["id"]: row for row in bridged_candidate_rows}
        assert set(bridged_by_id) == {"pin_real", "pin_aincrad"}
        assert bridged_by_id["pin_aincrad"]["realm_bridge_mode"] == "attributed"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunks_and_verbatim_evidence_carry_realm() -> None:
    connection, clock = await _setup_connection()
    try:
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        realms = await _create_realms(connection, clock)
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Chat",
            platform_id="default",
            active_realm_id="realm_real",
        )
        await messages.create_message(
            "msg_real",
            "cnv_1",
            "user",
            1,
            "alpha verbatim transcript detail",
            5,
            {},
            active_realm_id="realm_real",
        )

        artifacts = ArtifactRepository(connection, clock)
        await artifacts.create_artifact(
            artifact_id="art_real",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id="msg_real",
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
            realm_id="realm_real",
        )
        await artifacts.create_artifact(
            artifact_id="art_aincrad",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="upload",
            source_ref="aincrad.txt",
            title="Aincrad",
            status="ready",
            privacy_level=0,
            skip_raw_by_default=False,
            requires_explicit_request=False,
            summary_text="alpha artifact aincrad",
            index_text="alpha artifact",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
            realm_id="realm_aincrad",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_real",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for real world",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_real",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_aincrad",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for aincrad",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_aincrad",
        )

        artifact_rows = await artifacts.search_artifact_chunks(
            user_id="usr_1",
            query="alpha",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.CHAT],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            limit=10,
            platform_id="default",
            active_realm_id="realm_real",
        )
        assert {row["artifact_id"] for row in artifact_rows} == {"art_real"}
        assert artifact_rows[0]["artifact_realm_id"] == "realm_real"

        await realms.upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_real",
            target_realm_id="realm_aincrad",
            cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        )
        artifact_plan = _plan(
            active_realm_id="realm_real",
            raw_context_access_mode="artifact",
        ).model_copy(update={"scope_filter": [MemoryScope.CONVERSATION]})
        bridged_artifact_rows = [
            row
            for row in await CandidateSearch(connection, clock).search(
                artifact_plan,
                "usr_1",
            )
            if row.get("is_artifact_chunk")
        ]
        bridged_artifact_by_id = {
            row["artifact_id"]: row for row in bridged_artifact_rows
        }
        assert set(bridged_artifact_by_id) == {"art_real", "art_aincrad"}
        assert (
            bridged_artifact_by_id["art_aincrad"]["realm_bridge_mode"]
            == "attributed"
        )

        plan = _plan(
            active_realm_id="realm_real",
            raw_context_access_mode="verbatim",
        ).model_copy(update={"scope_filter": [MemoryScope.CHAT]})
        evidence_rows = [
            row for row in await CandidateSearch(connection, clock).search(plan, "usr_1")
            if row.get("is_verbatim_evidence_window")
        ]
        assert evidence_rows
        assert evidence_rows[0]["realm_id"] == "realm_real"
    finally:
        await connection.close()


def test_context_cache_key_partitions_by_active_realm() -> None:
    base = {
        "user_id": "usr_1",
        "assistant_mode_id": "coding_debug",
        "conversation_id": "cnv_1",
        "workspace_id": None,
        "operational_profile_token": "default",
        "active_presence_id": "presence_1",
        "active_space_id": None,
        "active_mind_id": None,
        "mind_topology": "unimind",
        "active_embodiment_id": "body_real",
    }
    real_key = ContextCacheService.build_cache_key(
        **base,
        active_realm_id="realm_real",
    )
    aincrad_key = ContextCacheService.build_cache_key(
        **base,
        active_realm_id="realm_aincrad",
    )
    outside_key = ContextCacheService.build_cache_key(
        **base,
        active_realm_id=None,
    )

    assert real_key != aincrad_key
    assert real_key != outside_key
