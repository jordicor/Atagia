"""Tests for SQLite initialization and schema behavior."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

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
            "artifact_blobs",
            "artifact_chunks",
            "artifact_chunks_fts",
            "artifact_links",
            "artifacts",
            "belief_versions",
            "conversation_activity_stats",
            "conversation_topic_events",
            "conversation_topic_sources",
            "conversation_topics",
            "contract_dimensions_current",
            "conversations",
            "evaluation_metrics",
            "memory_feedback_events",
            "memory_embedding_metadata",
            "memory_links",
            "memory_consent_profile",
            "memory_objects",
            "memory_objects_fts",
            "messages",
            "messages_fts",
            "pending_memory_confirmations",
            "retrieval_events",
            "schema_migrations",
            "summary_views",
            "verbatim_pins",
            "verbatim_pins_fts",
            "users",
            "workspaces",
        }.issubset(names)
        assert await _fetch_one_value(connection, "PRAGMA foreign_keys;") == 1
        assert await _fetch_one_value(connection, "PRAGMA journal_mode;") != "wal"
        assert await _fetch_one_value(
            connection,
            "SELECT COUNT(*) FROM schema_migrations;",
        ) == len(MigrationManager(MIGRATIONS_DIR).discover())
        assistant_modes_columns_cursor = await connection.execute(
            "PRAGMA table_info(assistant_modes);"
        )
        assistant_modes_columns = {
            row["name"] for row in await assistant_modes_columns_cursor.fetchall()
        }
        assert "privacy_ceiling" in assistant_modes_columns
        feedback_columns_cursor = await connection.execute("PRAGMA table_info(memory_feedback_events);")
        feedback_columns = {row["name"] for row in await feedback_columns_cursor.fetchall()}
        assert "user_id" in feedback_columns
        message_columns_cursor = await connection.execute("PRAGMA table_info(messages);")
        message_columns = {row["name"] for row in await message_columns_cursor.fetchall()}
        assert {
            "occurred_at",
            "content_kind",
            "include_raw",
            "skip_by_default",
            "heavy_content",
            "artifact_backed",
            "verbatim_required",
            "requires_explicit_request",
            "context_placeholder",
            "policy_reason",
        }.issubset(message_columns)
        memory_columns_cursor = await connection.execute("PRAGMA table_info(memory_objects);")
        memory_columns = {row["name"] for row in await memory_columns_cursor.fetchall()}
        assert "extraction_hash" in memory_columns
        assert "index_text" in memory_columns
        assert "memory_category" in memory_columns
        assert "preserve_verbatim" in memory_columns
        assert "temporal_type" in memory_columns
        assert "tension_score" in memory_columns
        assert "tension_updated_at" in memory_columns
        consent_columns_cursor = await connection.execute("PRAGMA table_info(memory_consent_profile);")
        consent_columns = {row["name"] for row in await consent_columns_cursor.fetchall()}
        assert {"user_id", "category", "confirmed_count", "declined_count"}.issubset(consent_columns)
        confirmation_columns_cursor = await connection.execute(
            "PRAGMA table_info(pending_memory_confirmations);"
        )
        confirmation_columns = {row["name"] for row in await confirmation_columns_cursor.fetchall()}
        assert {
            "user_id",
            "conversation_id",
            "memory_id",
            "memory_category",
            "asked_at",
            "confirmation_asked_once",
        }.issubset(confirmation_columns)
        memory_fts_columns_cursor = await connection.execute("PRAGMA table_info(memory_objects_fts);")
        memory_fts_columns = {row["name"] for row in await memory_fts_columns_cursor.fetchall()}
        assert "canonical_text" in memory_fts_columns
        assert "index_text" in memory_fts_columns
        pin_columns_cursor = await connection.execute("PRAGMA table_info(verbatim_pins);")
        pin_columns = {row["name"] for row in await pin_columns_cursor.fetchall()}
        assert {
            "user_id",
            "workspace_id",
            "conversation_id",
            "assistant_mode_id",
            "scope",
            "target_kind",
            "target_id",
            "canonical_text",
            "index_text",
            "privacy_level",
            "status",
            "created_by",
            "created_at",
            "updated_at",
            "payload_json",
        }.issubset(pin_columns)
        pin_fts_columns_cursor = await connection.execute("PRAGMA table_info(verbatim_pins_fts);")
        pin_fts_columns = {row["name"] for row in await pin_fts_columns_cursor.fetchall()}
        assert "index_text" in pin_fts_columns
        artifact_columns_cursor = await connection.execute("PRAGMA table_info(artifacts);")
        artifact_columns = {row["name"] for row in await artifact_columns_cursor.fetchall()}
        assert {
            "user_id",
            "workspace_id",
            "conversation_id",
            "message_id",
            "artifact_type",
            "source_kind",
            "source_ref",
            "mime_type",
            "filename",
            "title",
            "content_hash",
            "size_bytes",
            "page_count",
            "status",
            "privacy_level",
            "preserve_verbatim",
            "skip_raw_by_default",
            "requires_explicit_request",
            "metadata_json",
            "summary_text",
            "index_text",
            "created_at",
            "updated_at",
            "deleted_at",
        }.issubset(artifact_columns)
        artifact_chunks_columns_cursor = await connection.execute("PRAGMA table_info(artifact_chunks);")
        artifact_chunks_columns = {row["name"] for row in await artifact_chunks_columns_cursor.fetchall()}
        assert {
            "artifact_id",
            "user_id",
            "chunk_index",
            "source_start_offset",
            "source_end_offset",
            "text",
            "token_count",
            "kind",
            "created_at",
            "updated_at",
        }.issubset(artifact_chunks_columns)
        artifact_links_columns_cursor = await connection.execute("PRAGMA table_info(artifact_links);")
        artifact_links_columns = {row["name"] for row in await artifact_links_columns_cursor.fetchall()}
        assert {
            "user_id",
            "message_id",
            "artifact_id",
            "relation_kind",
            "ordinal",
            "created_at",
        }.issubset(artifact_links_columns)
        summary_columns_cursor = await connection.execute("PRAGMA table_info(summary_views);")
        summary_columns = {row["name"] for row in await summary_columns_cursor.fetchall()}
        assert "model" in summary_columns
        assert "user_id" in summary_columns
        assert "hierarchy_level" in summary_columns
        activity_columns_cursor = await connection.execute(
            "PRAGMA table_info(conversation_activity_stats);"
        )
        activity_columns = {row["name"] for row in await activity_columns_cursor.fetchall()}
        assert {
            "user_id",
            "conversation_id",
            "workspace_id",
            "assistant_mode_id",
            "timezone",
            "recent_1d_message_count",
            "return_interval_histogram_json",
            "main_thread_score",
            "likely_soon_score",
            "schedule_pattern_kind",
        }.issubset(activity_columns)
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
                canonical_text, index_text, payload_json, source_kind, confidence, stability, vitality,
                maya_score, privacy_level, valid_from, valid_to, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', NULL, 'cnv_1', 'coding_debug', 'evidence', 'conversation',
                'User prefers fast answers', 'websocket latency incident', '{}', 'extracted', 0.6, 0.5, 0.0,
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
        index_text_cursor = await connection.execute(
            "SELECT rowid FROM memory_objects_fts WHERE memory_objects_fts MATCH 'latency';"
        )
        assert (await message_cursor.fetchone())[0] == 1
        assert (await memory_cursor.fetchone())[0] == 1
        assert (await index_text_cursor.fetchone())[0] == 1
        await connection.execute(
            "UPDATE memory_objects SET index_text = 'throughput regression incident' WHERE id = 'mem_1';"
        )
        await connection.commit()
        removed_cursor = await connection.execute(
            "SELECT COUNT(*) FROM memory_objects_fts WHERE memory_objects_fts MATCH 'latency';"
        )
        updated_cursor = await connection.execute(
            "SELECT rowid FROM memory_objects_fts WHERE memory_objects_fts MATCH 'throughput';"
        )
        assert (await removed_cursor.fetchone())[0] == 0
        assert (await updated_cursor.fetchone())[0] == 1
        assert await _fetch_one_value(
            connection,
            "SELECT status FROM memory_objects WHERE id = 'mem_1';",
        ) == "review_required"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_objects_accept_pending_and_declined_statuses() -> None:
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
            VALUES ('personal_assistant', 'Personal Assistant', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO memory_objects(
                id, user_id, assistant_mode_id, object_type, scope, canonical_text, payload_json,
                source_kind, confidence, stability, vitality, maya_score, privacy_level,
                memory_category, preserve_verbatim, valid_from, valid_to, temporal_type,
                status, created_at, updated_at
            )
            VALUES
                (
                    'mem_pending', 'usr_1', 'personal_assistant', 'evidence', 'conversation',
                    'Banking card PIN: 4512', '{}', 'extracted', 0.9, 0.5, 0.0, 0.0, 3,
                    'pin_or_password', 1, NULL, NULL, 'unknown',
                    'pending_user_confirmation', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                ),
                (
                    'mem_declined', 'usr_1', 'personal_assistant', 'evidence', 'conversation',
                    'Company card PIN: 7000', '{}', 'extracted', 0.9, 0.5, 0.0, 0.0, 3,
                    'pin_or_password', 1, NULL, NULL, 'unknown',
                    'declined', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                )
            """
        )
        await connection.commit()

        cursor = await connection.execute(
            """
            SELECT id, status, memory_category, preserve_verbatim
            FROM memory_objects
            ORDER BY id ASC
            """
        )
        rows = await cursor.fetchall()

        assert [row["id"] for row in rows] == ["mem_declined", "mem_pending"]
        assert [row["status"] for row in rows] == ["declined", "pending_user_confirmation"]
        assert all(row["memory_category"] == "pin_or_password" for row in rows)
        assert all(row["preserve_verbatim"] == 1 for row in rows)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_pins_fts_uses_safe_index_text_and_honors_rowid() -> None:
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
            INSERT INTO verbatim_pins(
                id, user_id, scope, target_kind, target_id, canonical_text, index_text,
                privacy_level, status, reason, created_by, created_at, updated_at,
                expires_at, deleted_at, payload_json
            )
            VALUES (
                'vbp_1', 'usr_1', 'conversation', 'message', 'msg_1',
                'Bank card PIN: 4512', 'bank card PIN',
                3, 'active', 'banking', 'usr_1',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00',
                NULL, NULL, '{}'
            )
            """
        )
        await connection.commit()

        rowid = await _fetch_one_value(connection, "SELECT _rowid FROM verbatim_pins WHERE id = 'vbp_1';")
        assert rowid == 1

        safe_match = await _fetch_one_value(
            connection,
            "SELECT rowid FROM verbatim_pins_fts WHERE verbatim_pins_fts MATCH 'bank';",
        )
        assert safe_match == 1

        sensitive_match = await _fetch_one_value(
            connection,
            "SELECT COUNT(*) FROM verbatim_pins_fts WHERE verbatim_pins_fts MATCH '4512';",
        )
        assert sensitive_match == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunks_fts_uses_safe_text_and_excludes_base64_payloads() -> None:
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
            INSERT INTO messages(id, conversation_id, role, seq, text, token_count, metadata_json, occurred_at, created_at)
            VALUES ('msg_1', 'cnv_1', 'user', 1, 'Hello attachment world', 3, '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO artifacts(
                id, user_id, workspace_id, conversation_id, message_id, artifact_type, source_kind,
                source_ref, mime_type, filename, title, content_hash, size_bytes, page_count, status,
                privacy_level, preserve_verbatim, skip_raw_by_default, requires_explicit_request,
                metadata_json, summary_text, index_text, created_at, updated_at, deleted_at
            )
            VALUES (
                'art_1', 'usr_1', NULL, 'cnv_1', 'msg_1', 'base64', 'base64',
                'upload.bin', 'application/octet-stream', 'upload.bin', 'Upload',
                'abc123', 3, NULL, 'ready',
                0, 0, 1, 1,
                '{}', 'base64 attachment; artifact_id=art_1', 'Upload base64 attachment',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO artifact_blobs(
                artifact_id, storage_kind, blob_bytes, storage_uri, byte_size, sha256, created_at, updated_at
            )
            VALUES (
                'art_1', 'sqlite_blob', X'48656C6C6F', NULL, 5, 'abc123',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO artifact_chunks(
                id, artifact_id, user_id, chunk_index, source_start_offset, source_end_offset,
                text, token_count, kind, created_at, updated_at
            )
            VALUES (
                'arc_1', 'art_1', 'usr_1', 0, NULL, NULL,
                'base64 attachment; artifact_id=art_1', 6, 'summary',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            )
            """
        )
        await connection.commit()

        rowid = await _fetch_one_value(connection, "SELECT _rowid FROM artifact_chunks WHERE id = 'arc_1';")
        assert rowid == 1

        safe_match = await _fetch_one_value(
            connection,
            "SELECT rowid FROM artifact_chunks_fts WHERE artifact_chunks_fts MATCH 'attachment';",
        )
        assert safe_match == 1

        base64_match = await _fetch_one_value(
            connection,
            "SELECT COUNT(*) FROM artifact_chunks_fts WHERE artifact_chunks_fts MATCH '48656C6C6F';",
        )
        assert base64_match == 0
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
                id, user_id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, created_at
            )
            VALUES (
                'sum_1', 'usr_1', NULL, 'wrk_1', NULL, NULL, 'workspace_rollup',
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


@pytest.mark.asyncio
async def test_migration_0012_backfills_summary_view_user_ids_and_drops_orphans(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy-migrations"
    legacy_dir.mkdir()
    manager = MigrationManager(MIGRATIONS_DIR)
    for migration in manager.discover():
        if migration.version >= 12:
            continue
        copy2(migration.path, legacy_dir / migration.path.name)

    connection = await initialize_database(":memory:", legacy_dir)
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
            INSERT INTO workspaces(id, user_id, name, metadata_json, created_at, updated_at)
            VALUES ('wrk_1', 'usr_1', 'Workspace', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO conversations(id, user_id, workspace_id, assistant_mode_id, title, status, metadata_json, created_at, updated_at)
            VALUES ('cnv_1', 'usr_1', 'wrk_1', 'coding_debug', 'Conversation', 'active', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO summary_views(
                id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, model, created_at
            )
            VALUES
                ('sum_conv', 'cnv_1', 'wrk_1', 1, 2, 'conversation_chunk', 'Conversation summary', '[]', 1.5, 'model-a', '2026-03-30T00:00:00+00:00'),
                ('sum_rollup', NULL, 'wrk_1', 0, 0, 'workspace_rollup', 'Workspace summary', '[]', 1.5, 'model-b', '2026-03-30T00:01:00+00:00'),
                ('sum_orphan', NULL, NULL, 0, 0, 'context_view', 'Orphan summary', '[]', 1.5, 'model-c', '2026-03-30T00:02:00+00:00')
            """
        )
        await connection.commit()

        applied = await manager.apply_all(connection)

        rows_cursor = await connection.execute(
            """
            SELECT id, user_id, hierarchy_level, source_message_start_seq, source_message_end_seq
            FROM summary_views
            ORDER BY id ASC
            """
        )
        rows = await rows_cursor.fetchall()

        assert [migration.version for migration in applied] == [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        assert [row["id"] for row in rows] == ["sum_conv", "sum_rollup"]
        assert all(row["user_id"] == "usr_1" for row in rows)
        assert all(row["hierarchy_level"] == 0 for row in rows)
        assert rows[0]["source_message_start_seq"] == 1
        assert rows[1]["source_message_start_seq"] == 0
    finally:
        await connection.close()
