"""Integration tests for admin inspection helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository
from atagia.memory.inspector import MemoryInspector
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
    clock = FrozenClock(datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    audits = AdminAuditRepository(connection, clock)
    inspector = MemoryInspector(connection, clock)

    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Debug")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other")
    await messages.create_message("msg_1", "cnv_1", "user", 1, "Need help with retries", 5, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Let's inspect the queue", 6, {})
    await messages.create_message("msg_3", "cnv_2", "user", 1, "Other user prompt", 4, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.BELIEF,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User prefers patch-style fixes.",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.9,
        privacy_level=2,
        intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
        intimacy_boundary_confidence=0.86,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_belief",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User is debugging retry behavior.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_evidence",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.INTERACTION_CONTRACT,
        scope=MemoryScope.ASSISTANT_MODE,
        canonical_text="User prefers concise direct answers.",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.75,
        privacy_level=1,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_contract",
    )
    await memories.create_memory_object(
        user_id="usr_2",
        conversation_id="cnv_2",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Other user's memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_other",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Pending sensitive memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=3,
        status=MemoryStatus.PENDING_USER_CONFIRMATION,
        memory_id="mem_pending",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Declined sensitive memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=3,
        status=MemoryStatus.DECLINED,
        memory_id="mem_declined",
    )
    await connection.execute(
        """
        INSERT INTO belief_versions(
            belief_id,
            version,
            claim_key,
            claim_value_json,
            condition_json,
            support_count,
            contradict_count,
            supersedes_version,
            is_current,
            created_at
        )
        VALUES
            (?, 1, 'response_style.debugging', '{"label":"patch_first"}', '{}', 1, 0, NULL, 0, ?),
            (?, 2, 'response_style.debugging', '{"label":"patch_first"}', '{}', 2, 0, 1, 1, ?)
        """,
        (
            "mem_belief",
            "2026-03-31T00:00:00+00:00",
            "mem_belief",
            "2026-03-31T00:05:00+00:00",
        ),
    )
    await connection.commit()
    event = await events.create_event(
        {
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry queue"], "skip_retrieval": False},
            "selected_memory_ids_json": ["mem_belief", "mem_evidence"],
            "context_view_json": {
                "selected_memory_ids": ["mem_belief", "mem_evidence"],
                "total_tokens_estimate": 72,
                "items_included": 2,
                "items_dropped": 1,
            },
            "outcome_json": {"zero_candidates": False},
        }
    )
    return connection, inspector, audits, event


@pytest.mark.asyncio
async def test_inspect_memory_returns_full_object_and_logs_audit() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        memory = await inspector.inspect_memory("mem_belief", "usr_1", admin_user_id="adm_1")

        assert memory is not None
        assert memory["canonical_text"] == "User prefers patch-style fixes."
        assert memory["intimacy_boundary"] == "romantic_private"
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "memory_object"
        assert audit_rows[-1]["target_id"] == "mem_belief"
        assert audit_rows[-1]["metadata_json"]["user_id"] == "usr_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_retrieval_event_returns_event_with_plan_and_logs_audit() -> None:
    connection, inspector, audits, event = await _build_runtime()
    try:
        inspected = await inspector.inspect_retrieval_event(event["id"], "usr_1", admin_user_id="adm_1")

        assert inspected is not None
        assert inspected["retrieval_plan_json"]["fts_queries"] == ["retry queue"]
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "retrieval_event"
        assert audit_rows[-1]["target_id"] == event["id"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_user_memories_filters_by_type_scope_and_status() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        filtered = await inspector.inspect_user_memories(
            "usr_1",
            admin_user_id="adm_1",
            object_type="belief",
            scope="conversation",
            status="active",
            intimacy_boundary="romantic_private",
            limit=10,
        )

        assert [item["id"] for item in filtered] == ["mem_belief"]
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "user_memory_collection"
        assert audit_rows[-1]["metadata_json"]["result_count"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_user_memories_shows_pending_and_declined_rows() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        memories = await inspector.inspect_user_memories(
            "usr_1",
            admin_user_id="adm_1",
            limit=10,
        )

        statuses = {item["id"]: item["status"] for item in memories}
        assert statuses["mem_pending"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert statuses["mem_declined"] == MemoryStatus.DECLINED.value
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["metadata_json"]["result_count"] == len(memories)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_belief_history_returns_all_versions_and_logs_audit() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        history = await inspector.inspect_belief_history(
            "mem_belief",
            "usr_1",
            admin_user_id="adm_1",
        )

        assert [item["version"] for item in history] == [1, 2]
        assert {item["parent_intimacy_boundary"] for item in history} == {"romantic_private"}
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "belief_history"
        assert audit_rows[-1]["target_id"] == "mem_belief"
        assert audit_rows[-1]["metadata_json"]["result_count"] == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspector_respects_user_isolation() -> None:
    connection, inspector, audits, event = await _build_runtime()
    try:
        assert await inspector.inspect_memory("mem_other", "usr_1", admin_user_id="adm_1") is None
        assert await inspector.inspect_retrieval_event(event["id"], "usr_2", admin_user_id="adm_1") is None
        assert await inspector.inspect_belief_history("mem_belief", "usr_2", admin_user_id="adm_1") == []

        audit_rows = await audits.list_entries("adm_1")
        assert len(audit_rows) == 3
        assert [row["metadata_json"]["user_id"] for row in audit_rows[:2]] == ["usr_1", "usr_2"]
        assert audit_rows[2]["metadata_json"]["user_id"] == "usr_2"
    finally:
        await connection.close()
