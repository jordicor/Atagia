"""Tests for Embodiment body/device boundary behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    VerbatimPinTargetKind,
)
from atagia.services.context_cache_service import ContextCacheService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))


def _plan(
    *,
    active_embodiment_id: str | None,
    raw_context_access_mode: str = "normal",
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="alpha",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        platform_id="default",
        active_embodiment_id=active_embodiment_id,
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
    embodiment_id: str | None,
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
        embodiment_id=embodiment_id,
    )


@pytest.mark.asyncio
async def test_candidate_search_filters_by_active_embodiment() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_drone",
            text="alpha body memory for drone",
            embodiment_id="body_drone",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_desktop",
            text="alpha body memory for desktop",
            embodiment_id="body_desktop",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_legacy",
            text="alpha legacy unbodied memory",
            embodiment_id=None,
        )

        search = CandidateSearch(connection, clock)
        drone_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_embodiment_id="body_drone"),
                "usr_1",
            )
        }
        assert drone_ids == {"mem_drone", "mem_legacy"}

        outside_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_embodiment_id=None),
                "usr_1",
            )
        }
        assert outside_ids == {"mem_legacy"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_small_corpus_state_and_visible_memory_filter_by_embodiment() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="mem_drone",
            text="alpha corpus drone memory",
            embodiment_id="body_drone",
        )
        await _create_user_memory(
            memories,
            memory_id="mem_desktop",
            text="alpha corpus desktop memory",
            embodiment_id="body_desktop",
        )
        await _create_user_memory(
            memories,
            memory_id="state_drone",
            text="alpha drone state",
            embodiment_id="body_drone",
            object_type=MemoryObjectType.STATE_SNAPSHOT,
            payload={"current_user_state": "drone body state"},
        )

        rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_embodiment_id="body_drone",
        )
        assert {row["id"] for row in rows} == {"mem_drone", "state_drone"}

        outside_rows = await memories.list_eligible_for_context(
            "usr_1",
            [MemoryScope.USER],
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            privacy_ceiling=3,
            active_embodiment_id=None,
        )
        assert outside_rows == []

        state = await memories.get_state_snapshot(
            "usr_1",
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_1",
            active_embodiment_id="body_drone",
        )
        assert state == {"current_user_state": "drone body state"}

        assert await memories.get_visible_memory_object(
            "mem_desktop",
            "usr_1",
            conversation_id="cnv_1",
            user_persona_id=None,
            platform_id="default",
            character_id=None,
            incognito=False,
            remember_across_chats=True,
            remember_across_devices=True,
            active_embodiment_id="body_drone",
        ) is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_contract_projection_is_scoped_by_embodiment() -> None:
    connection, clock = await _setup_connection()
    try:
        memories = MemoryObjectRepository(connection, clock)
        contracts = ContractDimensionRepository(connection, clock)
        await _create_user_memory(
            memories,
            memory_id="contract_drone",
            text="alpha drone contract source",
            embodiment_id="body_drone",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "drone"}},
        )
        await _create_user_memory(
            memories,
            memory_id="contract_desktop",
            text="alpha desktop contract source",
            embodiment_id="body_desktop",
            object_type=MemoryObjectType.INTERACTION_CONTRACT,
            payload={"dimension_name": "tone", "value_json": {"tone": "desktop"}},
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "drone"},
            confidence=0.8,
            source_memory_id="contract_drone",
        )
        await contracts.upsert_projection(
            user_id="usr_1",
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
            scope=MemoryScope.USER,
            dimension_name="tone",
            value_json={"tone": "desktop"},
            confidence=0.8,
            source_memory_id="contract_desktop",
        )

        rows = await contracts.list_for_context(
            "usr_1",
            "coding_debug",
            None,
            "cnv_1",
            active_embodiment_id="body_drone",
        )
        assert [(row["embodiment_id"], row["value_json"]) for row in rows] == [
            ("body_drone", {"tone": "drone"})
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pins_filter_by_active_embodiment_in_crud_search_and_candidates() -> None:
    connection, clock = await _setup_connection()
    try:
        pins = VerbatimPinRepository(connection, clock)
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_drone",
            pin_id="pin_drone",
            canonical_text="alpha exact pin for drone body",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            embodiment_id="body_drone",
        )
        await pins.create_verbatim_pin(
            user_id="usr_1",
            scope=MemoryScope.USER,
            target_kind=VerbatimPinTargetKind.TEXT_SPAN,
            target_id="target_desktop",
            pin_id="pin_desktop",
            canonical_text="alpha exact pin for desktop body",
            index_text="alpha exact pin",
            privacy_level=0,
            created_by="usr_1",
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
            embodiment_id="body_desktop",
        )

        assert await pins.get_verbatim_pin(
            "pin_desktop",
            "usr_1",
            active_embodiment_id="body_drone",
        ) is None
        assert await pins.get_verbatim_pin(
            "pin_drone",
            "usr_1",
            active_embodiment_id="body_drone",
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
            active_embodiment_id="body_drone",
        )
        assert {row["id"] for row in search_rows} == {"pin_drone"}

        candidate_rows = await CandidateSearch(connection, clock).search(
            _plan(
                active_embodiment_id="body_drone",
                raw_context_access_mode="verbatim",
            ),
            "usr_1",
        )
        assert {row["id"] for row in candidate_rows} == {"pin_drone"}
        assert candidate_rows[0]["embodiment_id"] == "body_drone"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunks_and_verbatim_evidence_carry_embodiment() -> None:
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
            active_embodiment_id="body_drone",
        )
        await messages.create_message(
            "msg_drone",
            "cnv_1",
            "user",
            1,
            "alpha verbatim transcript detail",
            5,
            {},
            active_embodiment_id="body_drone",
        )

        artifacts = ArtifactRepository(connection, clock)
        await artifacts.create_artifact(
            artifact_id="art_drone",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id="msg_drone",
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
            embodiment_id="body_drone",
        )
        await artifacts.create_artifact(
            artifact_id="art_desktop",
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="upload",
            source_ref="desktop.txt",
            title="Desktop",
            status="ready",
            privacy_level=0,
            skip_raw_by_default=False,
            requires_explicit_request=False,
            summary_text="alpha artifact desktop",
            index_text="alpha artifact",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
            embodiment_id="body_desktop",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_drone",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for drone body",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_drone",
        )
        await artifacts.create_artifact_chunk(
            artifact_id="art_desktop",
            user_id="usr_1",
            chunk_index=0,
            text="alpha artifact chunk for desktop body",
            token_count=6,
            kind="extracted",
            chunk_id="chunk_desktop",
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
            active_embodiment_id="body_drone",
        )
        assert {row["artifact_id"] for row in artifact_rows} == {"art_drone"}
        assert artifact_rows[0]["artifact_embodiment_id"] == "body_drone"

        plan = _plan(
            active_embodiment_id="body_drone",
            raw_context_access_mode="verbatim",
        ).model_copy(update={"scope_filter": [MemoryScope.CHAT]})
        evidence_rows = [
            row for row in await CandidateSearch(connection, clock).search(plan, "usr_1")
            if row.get("is_verbatim_evidence_window")
        ]
        assert evidence_rows
        assert evidence_rows[0]["embodiment_id"] == "body_drone"
    finally:
        await connection.close()


def test_context_cache_key_partitions_by_active_embodiment() -> None:
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
    }
    drone_key = ContextCacheService.build_cache_key(
        **base,
        active_embodiment_id="body_drone",
    )
    desktop_key = ContextCacheService.build_cache_key(
        **base,
        active_embodiment_id="body_desktop",
    )
    outside_key = ContextCacheService.build_cache_key(
        **base,
        active_embodiment_id=None,
    )

    assert drone_key != desktop_key
    assert drone_key != outside_key
