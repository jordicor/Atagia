"""Tests for the belief repository."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    beliefs = BeliefRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "One")
    await conversations.create_conversation("cnv_2", "usr_1", None, "coding_debug", "Two")
    await conversations.create_conversation("cnv_3", "usr_2", None, "coding_debug", "Other")
    return connection, memories, beliefs


async def _seed_belief(
    memories: MemoryObjectRepository,
    beliefs: BeliefRepository,
    *,
    memory_id: str,
    user_id: str,
    conversation_id: str | None,
    claim_key: str,
    claim_value: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> dict[str, object]:
    created = await memories.create_memory_object(
        user_id=user_id,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.BELIEF,
        scope=MemoryScope.CONVERSATION if conversation_id is not None else MemoryScope.ASSISTANT_MODE,
        canonical_text=f"{claim_key}:{claim_value}",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.8,
        stability=0.6,
        vitality=0.25,
        maya_score=1.0,
        privacy_level=1,
        status=status,
        payload={
            "claim_key": claim_key,
            "claim_value": claim_value,
            "source_message_ids": [f"msg_{memory_id}"],
        },
        memory_id=memory_id,
    )
    await beliefs.create_first_version(
        belief_id=str(created["id"]),
        claim_key=claim_key,
        claim_value=claim_value,
        created_at=str(created["created_at"]),
    )
    return created


@pytest.mark.asyncio
async def test_create_first_version_creates_version_one_as_current() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        created = await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.CONVERSATION,
            canonical_text="response_style.debugging:terse",
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.8,
            privacy_level=1,
            payload={"claim_key": "response_style.debugging", "claim_value": "terse"},
            memory_id="mem_belief_1",
        )

        version = await beliefs.create_first_version(
            belief_id=str(created["id"]),
            claim_key="response_style.debugging",
            claim_value="terse",
            created_at=str(created["created_at"]),
        )

        assert version["version"] == 1
        assert version["is_current"] == 1
        assert version["claim_key"] == "response_style.debugging"
        assert version["claim_value_json"] == "terse"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_find_active_beliefs_by_claim_key_filters_user_and_status() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_active",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_superseded",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="verbose",
            status=MemoryStatus.SUPERSEDED,
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_archived",
            user_id="usr_1",
            conversation_id="cnv_2",
            claim_key="response_style.debugging",
            claim_value="verbose",
            status=MemoryStatus.ARCHIVED,
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_other_user",
            user_id="usr_2",
            conversation_id="cnv_3",
            claim_key="response_style.debugging",
            claim_value="terse",
        )

        rows = await beliefs.find_active_beliefs_by_claim_key("usr_1", "response_style.debugging")

        assert [row["belief_id"] for row in rows] == ["mem_active"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_new_version_updates_current_version_and_history() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        created = await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_belief_versions",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )

        await beliefs.create_new_version(
            belief_id=str(created["id"]),
            user_id="usr_1",
            version=2,
            claim_key="response_style.debugging",
            claim_value="concise",
            condition={},
            support_count=2,
            contradict_count=0,
            supersedes_version=1,
            created_at="2026-04-01T12:05:00+00:00",
        )

        current = await beliefs.get_current_version(str(created["id"]), "usr_1")
        history = await beliefs.get_version_history(str(created["id"]), "usr_1")

        assert current is not None
        assert current["version"] == 2
        assert current["is_current"] == 1
        assert current["claim_value_json"] == "concise"
        assert [item["version"] for item in history] == [1, 2]
        assert history[0]["is_current"] == 0
        assert history[1]["is_current"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_find_active_belief_candidates_returns_token_similar_keys() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_candidate",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="debugging.response_style",
            claim_value="terse",
        )

        rows = await beliefs.find_active_belief_candidates_by_claim_key(
            "usr_1",
            "response_style.debugging",
        )

        assert [row["belief_id"] for row in rows] == ["mem_candidate"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_count_supporting_evidence_counts_distinct_conversations() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_belief_conv1",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_belief_conv2",
            user_id="usr_1",
            conversation_id="cnv_2",
            claim_key="response_style.debugging",
            claim_value="terse",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The user asked for terse debugging answers.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=1,
            payload={"claim_key": "response_style.debugging"},
            memory_id="mem_evidence_support",
        )

        stats = await beliefs.count_supporting_evidence("usr_1", "response_style.debugging")

        assert stats["total_evidence"] == 3
        assert stats["distinct_conversations"] == 2
        assert stats["distinct_sessions"] == 1
        assert stats["oldest_at"] is not None
        assert stats["newest_at"] is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_memory_link_persists_valid_relation() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Evidence row",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            payload={},
            memory_id="mem_source",
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_target",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )

        link = await beliefs.create_memory_link("mem_source", "mem_target", "supports", 0.9)

        cursor = await connection.execute(
            "SELECT * FROM memory_links WHERE id = ?",
            (link["id"],),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["relation_type"] == "supports"
        assert row["src_memory_id"] == "mem_source"
        assert row["dst_memory_id"] == "mem_target"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_tension_methods_update_and_reset_belief_tension() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_tension",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )

        increased = await beliefs.increment_tension("mem_tension", 0.15, user_id="usr_1")
        decreased = await beliefs.decrement_tension("mem_tension", 0.05, user_id="usr_1")
        reset = await beliefs.reset_tension("mem_tension", user_id="usr_1")

        assert increased == pytest.approx(0.15)
        assert decreased == pytest.approx(0.10)
        assert reset == pytest.approx(0.0)
        assert await beliefs.get_tension("mem_tension", user_id="usr_1") == pytest.approx(0.0)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_beliefs_above_tension_threshold_filters_by_user_and_type() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_hot",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_other_user",
            user_id="usr_2",
            conversation_id="cnv_3",
            claim_key="response_style.debugging",
            claim_value="terse",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Evidence should never appear in belief tension reads.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_evidence",
        )

        await beliefs.increment_tension("mem_hot", 0.55, user_id="usr_1")
        await beliefs.increment_tension("mem_other_user", 0.90, user_id="usr_2")

        rows = await beliefs.get_beliefs_above_tension_threshold("usr_1", 0.5)

        assert [row["id"] for row in rows] == ["mem_hot"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_tension_evidence_buffer_round_trips_unique_ids() -> None:
    connection, memories, beliefs = await _build_runtime()
    try:
        await _seed_belief(
            memories,
            beliefs,
            memory_id="mem_buffer",
            user_id="usr_1",
            conversation_id="cnv_1",
            claim_key="response_style.debugging",
            claim_value="terse",
        )

        merged = await beliefs.add_tension_evidence_ids(
            "mem_buffer",
            ["mem_e1", "mem_e2", "mem_e1"],
            user_id="usr_1",
        )
        popped = await beliefs.pop_tension_evidence_ids("mem_buffer", user_id="usr_1")
        row = await memories.get_memory_object("mem_buffer", "usr_1")

        assert merged == ["mem_e1", "mem_e2"]
        assert popped == ["mem_e1", "mem_e2"]
        assert row is not None
        assert row["payload_json"].get("tension_evidence_memory_ids") is None
    finally:
        await connection.close()
