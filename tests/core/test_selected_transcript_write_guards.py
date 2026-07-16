"""SQLite guards for user-wide selected-transcript rebuild isolation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)
CLOCK = FrozenClock(datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc))
BLOCKED_USER = "usr_transcript_blocked"
OTHER_USER = "usr_transcript_other"
THIRD_USER = "usr_transcript_third"
SELECTED_CONVERSATION = "cnv_transcript_selected"
BLOCKED_USER_OTHER_CONVERSATION = "cnv_transcript_blocked_other"
OTHER_SOURCE_CONVERSATION = "cnv_transcript_other_source"
OTHER_TARGET_CONVERSATION = "cnv_transcript_other_target"


async def _seed(connection: aiosqlite.Connection) -> None:
    await sync_assistant_modes(
        connection,
        ManifestLoader(MANIFESTS_DIR).load_all(),
        CLOCK,
    )
    users = UserRepository(connection, CLOCK)
    for user_id in (BLOCKED_USER, OTHER_USER, THIRD_USER):
        await users.create_user(user_id)

    conversations = ConversationRepository(connection, CLOCK)
    for conversation_id, user_id in (
        (SELECTED_CONVERSATION, BLOCKED_USER),
        (BLOCKED_USER_OTHER_CONVERSATION, BLOCKED_USER),
        (OTHER_SOURCE_CONVERSATION, OTHER_USER),
        (OTHER_TARGET_CONVERSATION, OTHER_USER),
    ):
        await conversations.create_conversation(
            conversation_id,
            user_id,
            None,
            "coding_debug",
            "Transcript guard",
        )

    messages = MessageRepository(connection, CLOCK)
    await messages.create_message(
        "msg_blocked_user_other_conversation",
        BLOCKED_USER_OTHER_CONVERSATION,
        "user",
        1,
        "Blocked user's other conversation",
    )
    await messages.create_message(
        "msg_other_source",
        OTHER_SOURCE_CONVERSATION,
        "user",
        1,
        "Other user's source message",
    )

    timestamp = CLOCK.now().isoformat()
    await connection.execute(
        """
        INSERT INTO transcript_rebuild_workflows(
            id, operation_id, user_id, conversation_id, selection_epoch,
            transcript_hash, mutation_kind, selected_message_ids_json,
            abandoned_message_ids_json, supporting_message_ids_json,
            affected_memory_ids_json, affected_summary_ids_json,
            orchestrator_job_id, stage, start_derivation_revision,
            created_at, updated_at
        ) VALUES (
            'trb_write_guard', 'op_write_guard', ?, ?, 1,
            'transcript-hash', 'replace', '[]', '[]', '[]', '[]', '[]',
            'job_write_guard', 'sources', 0, ?, ?
        )
        """,
        (BLOCKED_USER, SELECTED_CONVERSATION, timestamp, timestamp),
    )
    await connection.execute(
        """
        INSERT INTO conversation_transcript_selections(
            user_id, conversation_id, selection_epoch, transcript_hash,
            current_workflow_id, state, updated_at
        ) VALUES (?, ?, 1, 'transcript-hash', 'trb_write_guard', 'rebuilding', ?)
        """,
        (BLOCKED_USER, SELECTED_CONVERSATION, timestamp),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_message_update_checks_old_and_new_user_scopes(
    tmp_path: Path,
) -> None:
    connection = await initialize_database(
        str(tmp_path / "selected-transcript-message-guards.db"),
        MIGRATIONS_DIR,
    )
    try:
        await _seed(connection)

        with pytest.raises(
            aiosqlite.IntegrityError,
            match="selected transcript rebuild blocks user message writes",
        ):
            await connection.execute(
                "UPDATE messages SET conversation_id = ? WHERE id = ?",
                (BLOCKED_USER_OTHER_CONVERSATION, "msg_other_source"),
            )
        await connection.rollback()

        with pytest.raises(
            aiosqlite.IntegrityError,
            match="selected transcript rebuild blocks user message writes",
        ):
            await connection.execute(
                "UPDATE messages SET conversation_id = ? WHERE id = ?",
                (OTHER_TARGET_CONVERSATION, "msg_blocked_user_other_conversation"),
            )
        await connection.rollback()

        await connection.execute(
            "UPDATE messages SET conversation_id = ? WHERE id = ?",
            (OTHER_TARGET_CONVERSATION, "msg_other_source"),
        )
        await connection.commit()
        cursor = await connection.execute(
            "SELECT conversation_id FROM messages WHERE id = 'msg_other_source'"
        )
        assert (await cursor.fetchone())["conversation_id"] == OTHER_TARGET_CONVERSATION
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_owner_update_checks_old_and_new_user_scopes(
    tmp_path: Path,
) -> None:
    connection = await initialize_database(
        str(tmp_path / "selected-transcript-owner-guards.db"),
        MIGRATIONS_DIR,
    )
    try:
        await _seed(connection)

        with pytest.raises(
            aiosqlite.IntegrityError,
            match="selected transcript rebuild blocks conversation ownership moves",
        ):
            await connection.execute(
                "UPDATE conversations SET user_id = ? WHERE id = ?",
                (BLOCKED_USER, OTHER_SOURCE_CONVERSATION),
            )
        await connection.rollback()

        with pytest.raises(
            aiosqlite.IntegrityError,
            match="selected transcript rebuild blocks conversation ownership moves",
        ):
            await connection.execute(
                "UPDATE conversations SET user_id = ? WHERE id = ?",
                (OTHER_USER, BLOCKED_USER_OTHER_CONVERSATION),
            )
        await connection.rollback()

        await connection.execute(
            "UPDATE conversations SET user_id = ? WHERE id = ?",
            (THIRD_USER, OTHER_SOURCE_CONVERSATION),
        )
        await connection.commit()
        cursor = await connection.execute(
            "SELECT user_id FROM conversations WHERE id = ?",
            (OTHER_SOURCE_CONVERSATION,),
        )
        assert (await cursor.fetchone())["user_id"] == THIRD_USER
    finally:
        await connection.close()
