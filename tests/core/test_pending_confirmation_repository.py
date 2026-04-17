"""Integration tests for pending memory confirmation markers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.consent_repository import PendingMemoryConfirmationRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@pytest.mark.asyncio
async def test_pending_confirmation_markers_are_user_scoped_and_support_batch_operations() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 10, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        repository = PendingMemoryConfirmationRepository(connection, clock)

        await users.create_user("usr_1")
        await users.create_user("usr_2")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('personal_assistant', 'Personal Assistant', 'hash_1', '{}', ?, ?)
            """,
            (clock.now().isoformat(), clock.now().isoformat()),
        )
        await connection.commit()
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "personal_assistant",
            "Chat 1",
        )
        await conversations.create_conversation(
            "cnv_2",
            "usr_2",
            None,
            "personal_assistant",
            "Chat 2",
        )

        for memory_id, user_id, conversation_id, category in (
            ("mem_a", "usr_1", "cnv_1", MemoryCategory.PIN_OR_PASSWORD),
            ("mem_b", "usr_1", "cnv_1", MemoryCategory.PIN_OR_PASSWORD),
            ("mem_c", "usr_1", "cnv_1", MemoryCategory.FINANCIAL),
            ("mem_d", "usr_2", "cnv_2", MemoryCategory.PIN_OR_PASSWORD),
        ):
            await memories.create_memory_object(
                memory_id=memory_id,
                user_id=user_id,
                conversation_id=conversation_id,
                assistant_mode_id="personal_assistant",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text=f"Pending {memory_id}",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=3,
                status=MemoryStatus.PENDING_USER_CONFIRMATION,
                commit=False,
            )
        await connection.commit()

        first = await repository.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id="mem_a",
            category=MemoryCategory.PIN_OR_PASSWORD,
            created_at="2026-04-06T10:00:00+00:00",
        )
        second = await repository.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id="mem_b",
            category=MemoryCategory.PIN_OR_PASSWORD,
            created_at="2026-04-06T10:01:00+00:00",
        )
        third = await repository.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id="mem_c",
            category=MemoryCategory.FINANCIAL,
            created_at="2026-04-06T10:02:00+00:00",
        )
        await repository.create_marker(
            user_id="usr_2",
            conversation_id="cnv_2",
            memory_id="mem_d",
            category=MemoryCategory.PIN_OR_PASSWORD,
            created_at="2026-04-06T10:03:00+00:00",
        )

        assert first["memory_id"] == "mem_a"
        assert second["memory_id"] == "mem_b"
        assert third["memory_id"] == "mem_c"
        assert (await repository.get_oldest_unasked_marker("usr_1", "cnv_1"))["memory_id"] == "mem_a"
        assert await repository.get_marker_for_memory("usr_1", "mem_d") is None

        pin_batch = await repository.list_markers_for_category(
            "usr_1",
            "cnv_1",
            MemoryCategory.PIN_OR_PASSWORD,
            asked=False,
        )
        assert [row["memory_id"] for row in pin_batch] == ["mem_a", "mem_b"]

        updated = await repository.mark_markers_asked(
            "usr_1",
            ["mem_a", "mem_b"],
            asked_at="2026-04-06T10:05:00+00:00",
        )
        assert updated == 2
        assert (await repository.get_oldest_asked_marker("usr_1", "cnv_1"))["memory_id"] == "mem_a"

        asked_batch = await repository.list_markers_for_category(
            "usr_1",
            "cnv_1",
            MemoryCategory.PIN_OR_PASSWORD,
            asked=True,
        )
        assert all(row["asked_at"] == "2026-04-06T10:05:00+00:00" for row in asked_batch)

        reset = await repository.reset_after_ambiguous("usr_1", ["mem_a", "mem_b"])
        assert reset == 2
        reset_batch = await repository.list_markers_for_category(
            "usr_1",
            "cnv_1",
            MemoryCategory.PIN_OR_PASSWORD,
        )
        assert all(row["asked_at"] is None for row in reset_batch)
        assert all(row["confirmation_asked_once"] == 1 for row in reset_batch)

        cleared = await repository.clear_markers("usr_1", ["mem_a", "mem_b"])
        assert cleared == 2
        assert await repository.get_marker_for_memory("usr_1", "mem_a") is None
        assert await repository.get_marker_for_memory("usr_2", "mem_d") is not None
    finally:
        await connection.close()
