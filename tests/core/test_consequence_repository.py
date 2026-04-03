"""Tests for consequence chain persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.consequence_repository import ConsequenceRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    chains = ConsequenceRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "One")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Two")
    return connection, memories, chains


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    user_id: str = "usr_1",
    workspace_id: str | None = "wrk_1",
    conversation_id: str | None = "cnv_1",
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    canonical_text: str,
) -> dict[str, object]:
    return await memories.create_memory_object(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=1,
        status=MemoryStatus.ACTIVE,
        payload={"source_message_ids": ["msg_1"]},
        memory_id=memory_id,
    )


@pytest.mark.asyncio
async def test_create_chain_creates_row_and_returns_id() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        outcome = await _seed_memory(memories, memory_id="mem_outcome", canonical_text="Regressions followed.")

        chain_id = await chains.create_chain(
            {
                "id": "chn_1",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome["id"]),
                "tendency_belief_id": None,
                "confidence": 0.72,
                "status": "active",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )

        cursor = await connection.execute("SELECT * FROM consequence_chains WHERE id = ?", (chain_id,))
        row = await cursor.fetchone()
        assert chain_id == "chn_1"
        assert row["action_memory_id"] == str(action["id"])
        assert row["outcome_memory_id"] == str(outcome["id"])
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_chain_validates_memory_objects_belong_to_user() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        outcome = await _seed_memory(
            memories,
            memory_id="mem_foreign_outcome",
            user_id="usr_2",
            workspace_id=None,
            conversation_id="cnv_2",
            canonical_text="Foreign outcome.",
        )

        with pytest.raises(ValueError, match="does not exist for user usr_1"):
            await chains.create_chain(
                {
                    "id": "chn_bad",
                    "user_id": "usr_1",
                    "workspace_id": "wrk_1",
                    "conversation_id": "cnv_1",
                    "assistant_mode_id": "coding_debug",
                    "action_memory_id": str(action["id"]),
                    "outcome_memory_id": str(outcome["id"]),
                    "tendency_belief_id": None,
                    "confidence": 0.6,
                    "status": "active",
                    "created_at": "2026-04-02T12:00:00+00:00",
                    "updated_at": "2026-04-02T12:00:00+00:00",
                }
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_find_chains_for_action_returns_active_matches() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        outcome_one = await _seed_memory(memories, memory_id="mem_outcome_1", canonical_text="First outcome.")
        outcome_two = await _seed_memory(memories, memory_id="mem_outcome_2", canonical_text="Second outcome.")
        await chains.create_chain(
            {
                "id": "chn_1",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome_one["id"]),
                "tendency_belief_id": None,
                "confidence": 0.9,
                "status": "active",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )
        await chains.create_chain(
            {
                "id": "chn_2",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome_two["id"]),
                "tendency_belief_id": None,
                "confidence": 0.4,
                "status": "archived",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )

        found = await chains.find_chains_for_action("usr_1", str(action["id"]))

        assert [row["id"] for row in found] == ["chn_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_find_chains_for_workspace_filters_and_respects_limit() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        for index in range(3):
            outcome = await _seed_memory(
                memories,
                memory_id=f"mem_outcome_{index}",
                canonical_text=f"Outcome {index}",
            )
            await chains.create_chain(
                {
                    "id": f"chn_{index}",
                    "user_id": "usr_1",
                    "workspace_id": "wrk_1",
                    "conversation_id": "cnv_1",
                    "assistant_mode_id": "coding_debug",
                    "action_memory_id": str(action["id"]),
                    "outcome_memory_id": str(outcome["id"]),
                    "tendency_belief_id": None,
                    "confidence": 0.5 + (index * 0.1),
                    "status": "active",
                    "created_at": f"2026-04-02T12:00:0{index}+00:00",
                    "updated_at": f"2026-04-02T12:00:0{index}+00:00",
                }
            )

        found = await chains.find_chains_for_workspace("usr_1", "wrk_1", limit=2)

        assert [row["id"] for row in found] == ["chn_2", "chn_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_find_chains_by_user_returns_active_rows_only() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        outcome = await _seed_memory(memories, memory_id="mem_outcome", canonical_text="Outcome.")
        await chains.create_chain(
            {
                "id": "chn_active",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome["id"]),
                "tendency_belief_id": None,
                "confidence": 0.7,
                "status": "active",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )
        await chains.create_chain(
            {
                "id": "chn_archived",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome["id"]),
                "tendency_belief_id": None,
                "confidence": 0.6,
                "status": "archived",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )

        found = await chains.find_chains_by_user("usr_1")

        assert [row["id"] for row in found] == ["chn_active"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_update_chain_confidence_and_archive_filter_by_user_id() -> None:
    connection, memories, chains = await _build_runtime()
    try:
        action = await _seed_memory(memories, memory_id="mem_action", canonical_text="Suggested a refactor.")
        outcome = await _seed_memory(memories, memory_id="mem_outcome", canonical_text="Outcome.")
        await chains.create_chain(
            {
                "id": "chn_1",
                "user_id": "usr_1",
                "workspace_id": "wrk_1",
                "conversation_id": "cnv_1",
                "assistant_mode_id": "coding_debug",
                "action_memory_id": str(action["id"]),
                "outcome_memory_id": str(outcome["id"]),
                "tendency_belief_id": None,
                "confidence": 0.7,
                "status": "active",
                "created_at": "2026-04-02T12:00:00+00:00",
                "updated_at": "2026-04-02T12:00:00+00:00",
            }
        )

        await chains.update_chain_confidence("chn_1", "usr_2", 0.1)
        await chains.archive_chain("chn_1", "usr_2")
        cursor = await connection.execute("SELECT confidence, status FROM consequence_chains WHERE id = ?", ("chn_1",))
        unchanged = await cursor.fetchone()
        assert unchanged["confidence"] == 0.7
        assert unchanged["status"] == "active"

        await chains.update_chain_confidence("chn_1", "usr_1", 0.9)
        await chains.archive_chain("chn_1", "usr_1")
        cursor = await connection.execute("SELECT confidence, status FROM consequence_chains WHERE id = ?", ("chn_1",))
        updated = await cursor.fetchone()
        assert updated["confidence"] == 0.9
        assert updated["status"] == "archived"
    finally:
        await connection.close()
