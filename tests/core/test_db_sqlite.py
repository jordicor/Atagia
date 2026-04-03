"""Tests for SQLite initialization and schema behavior."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from atagia.core.db_sqlite import MigrationManager, initialize_database

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _fetch_one_value(connection: aiosqlite.Connection, query: str) -> object:
    cursor = await connection.execute(query)
    row = await cursor.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_initialize_database_applies_schema_and_pragmas() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        cursor = await connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            """
        )
        names = {row["name"] for row in await cursor.fetchall()}

        assert {
            "admin_audit_log",
            "assistant_modes",
            "belief_versions",
            "contract_dimensions_current",
            "conversations",
            "evaluation_metrics",
            "memory_feedback_events",
            "memory_embedding_metadata",
            "memory_links",
            "memory_objects",
            "memory_objects_fts",
            "messages",
            "messages_fts",
            "retrieval_events",
            "schema_migrations",
            "summary_views",
            "users",
            "workspaces",
        }.issubset(names)
        assert await _fetch_one_value(connection, "PRAGMA foreign_keys;") == 1
        assert await _fetch_one_value(connection, "PRAGMA journal_mode;") != "wal"
        assert await _fetch_one_value(connection, "SELECT COUNT(*) FROM schema_migrations;") == 8
        feedback_columns_cursor = await connection.execute("PRAGMA table_info(memory_feedback_events);")
        feedback_columns = {row["name"] for row in await feedback_columns_cursor.fetchall()}
        assert "user_id" in feedback_columns
        message_columns_cursor = await connection.execute("PRAGMA table_info(messages);")
        message_columns = {row["name"] for row in await message_columns_cursor.fetchall()}
        assert "occurred_at" in message_columns
        memory_columns_cursor = await connection.execute("PRAGMA table_info(memory_objects);")
        memory_columns = {row["name"] for row in await memory_columns_cursor.fetchall()}
        assert "extraction_hash" in memory_columns
        summary_columns_cursor = await connection.execute("PRAGMA table_info(summary_views);")
        summary_columns = {row["name"] for row in await summary_columns_cursor.fetchall()}
        assert "model" in summary_columns
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_migrations_are_idempotent() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        manager = MigrationManager(MIGRATIONS_DIR)
        applied = await manager.apply_all(connection)
        assert applied == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rowid_fts_and_review_required_status_work() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES ('usr_1', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL)
            """
        )
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO conversations(id, user_id, workspace_id, assistant_mode_id, title, status, metadata_json, created_at, updated_at)
            VALUES ('cnv_1', 'usr_1', NULL, 'coding_debug', 'Conversation', 'active', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO messages(id, conversation_id, role, seq, text, token_count, metadata_json, created_at)
            VALUES ('msg_1', 'cnv_1', 'user', 1, 'Hello FTS world', 3, '{}', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                canonical_text, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', NULL, 'cnv_1', 'coding_debug', 'evidence', 'conversation',
                'User prefers fast answers', '{}', 'extracted', 0.6, 0.5, 0.0,
                0.0, 1, NULL, NULL, 'review_required', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.commit()

        message_rowid = await _fetch_one_value(connection, "SELECT _rowid FROM messages WHERE id = 'msg_1';")
        memory_rowid = await _fetch_one_value(connection, "SELECT _rowid FROM memory_objects WHERE id = 'mem_1';")
        assert message_rowid == 1
        assert memory_rowid == 1

        message_cursor = await connection.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'hello';"
        )
        memory_cursor = await connection.execute(
            "SELECT rowid FROM memory_objects_fts WHERE memory_objects_fts MATCH 'fast';"
        )
        assert (await message_cursor.fetchone())[0] == 1
        assert (await memory_cursor.fetchone())[0] == 1
        assert await _fetch_one_value(
            connection,
            "SELECT status FROM memory_objects WHERE id = 'mem_1';",
        ) == "review_required"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_view_orphaning_and_contract_uniqueness() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES ('usr_1', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL)
            """
        )
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('general_qa', 'General QA', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO workspaces(id, user_id, name, metadata_json, created_at, updated_at)
            VALUES ('wrk_1', 'usr_1', 'Workspace', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                canonical_text, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', 'wrk_1', NULL, 'general_qa', 'interaction_contract', 'workspace',
                'User prefers concise answers', '{}', 'extracted', 0.8, 0.5, 0.0,
                0.0, 0, NULL, NULL, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO summary_views(
                id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, created_at
            )
            VALUES (
                'sum_1', NULL, 'wrk_1', 0, 0, 'workspace_rollup',
                'Workspace summary', '[]', 1.5, '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO contract_dimensions_current(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, scope,
                dimension_name, value_json, confidence, source_memory_id, updated_at
            )
            VALUES (
                'ctr_1', 'usr_1', 'wrk_1', NULL, 'general_qa', 'workspace',
                'depth', '{"level":"low"}', 0.8, 'mem_1', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.commit()

        with pytest.raises(aiosqlite.IntegrityError):
            await connection.execute(
                """
                INSERT INTO contract_dimensions_current(
                    id, user_id, workspace_id, conversation_id, assistant_mode_id, scope,
                    dimension_name, value_json, confidence, source_memory_id, updated_at
                )
                VALUES (
                    'ctr_2', 'usr_1', 'wrk_1', NULL, 'general_qa', 'workspace',
                    'depth', '{"level":"medium"}', 0.9, 'mem_1', '2026-03-30T00:00:00+00:00'
                )
                """
            )

        await connection.execute("DELETE FROM workspaces WHERE id = 'wrk_1';")
        await connection.commit()

        cursor = await connection.execute(
            "SELECT conversation_id, workspace_id FROM summary_views WHERE id = 'sum_1';"
        )
        summary_row = await cursor.fetchone()
        assert summary_row["conversation_id"] is None
        assert summary_row["workspace_id"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_extraction_hash_is_unique_per_user() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES
                ('usr_1', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL),
                ('usr_2', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL)
            """
        )
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                canonical_text, extraction_hash, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', NULL, NULL, 'coding_debug', 'evidence', 'assistant_mode',
                'Same memory', 'hash_1', '{}', 'extracted', 0.9, 0.5, 0.0,
                0.0, 0, NULL, NULL, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.commit()

        with pytest.raises(aiosqlite.IntegrityError):
            await connection.execute(
                """
                INSERT INTO memory_objects(
                    id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                    canonical_text, extraction_hash, payload_json, source_kind, confidence, stability, vitality,
                    maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
                )
                VALUES (
                    'mem_2', 'usr_1', NULL, NULL, 'coding_debug', 'evidence', 'assistant_mode',
                    'Duplicate hash', 'hash_1', '{}', 'extracted', 0.8, 0.5, 0.0,
                    0.0, 0, NULL, NULL, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                )
                """
            )

        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope,
                canonical_text, extraction_hash, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES (
                'mem_3', 'usr_2', NULL, NULL, 'coding_debug', 'evidence', 'assistant_mode',
                'Other user same hash', 'hash_1', '{}', 'extracted', 0.8, 0.5, 0.0,
                0.0, 0, NULL, NULL, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.commit()
    finally:
        await connection.close()
