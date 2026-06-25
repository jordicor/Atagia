"""Tests for SQLite initialization and schema behavior."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

import aiosqlite
import pytest

from atagia.core.db_sqlite import SQLITE_BUSY_TIMEOUT_MS, MigrationManager, initialize_database

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
            "artifact_payload_blobs",
            "artifacts",
            "belief_versions",
            "conversation_activity_stats",
            "conversation_topic_events",
            "conversation_topic_sources",
            "conversation_topics",
            "contract_dimensions_current",
            "conversations",
            "embodiments",
            "evaluation_metrics",
            "graph_entities",
            "graph_entity_aliases",
            "graph_entity_mentions",
            "graph_projection_runs",
            "graph_relationship_sources",
            "graph_relationships",
            "initial_context_packages",
            "memory_feedback_events",
            "memory_fact_facets",
            "memory_retrieval_surfaces",
            "memory_retrieval_surfaces_fts",
            "memory_embedding_metadata",
            "memory_links",
            "memory_consent_profile",
            "memory_edit_history",
            "memory_evidence_spans",
            "memory_objects",
            "memory_objects_fts",
            "memory_object_subjects",
            "memory_support_edges",
            "messages",
            "messages_fts",
            "minds",
            "overseer_grants",
            "deletion_tombstones",
            "pending_file_deletions",
            "pending_memory_confirmations",
            "presences",
            "realm_bridges",
            "realms",
            "retrieval_events",
            "schema_migrations",
            "spaces",
            "summary_views",
            "verbatim_pins",
            "verbatim_pins_fts",
            "users",
            "workspaces",
            "worker_job_runs",
        }.issubset(names)
        assert await _fetch_one_value(connection, "PRAGMA foreign_keys;") == 1
        assert await _fetch_one_value(connection, "PRAGMA journal_mode;") != "wal"
        assert await _fetch_one_value(connection, "PRAGMA busy_timeout;") == SQLITE_BUSY_TIMEOUT_MS
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
            "active_presence_id",
            "source_presence_id",
            "space_id",
            "active_mind_id",
            "source_mind_id",
            "active_embodiment_id",
            "active_realm_id",
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
        assert "archived_by_conversation_id" in memory_columns
        assert "active_presence_id" in memory_columns
        assert "source_presence_id" in memory_columns
        assert "presence_cluster_id" in memory_columns
        assert "space_id" in memory_columns
        assert "space_boundary_mode" in memory_columns
        assert "memory_owner_id" in memory_columns
        assert "source_mind_id" in memory_columns
        assert "embodiment_id" in memory_columns
        assert "realm_id" in memory_columns
        surface_columns_cursor = await connection.execute(
            "PRAGMA table_info(memory_retrieval_surfaces);"
        )
        surface_columns = {row["name"] for row in await surface_columns_cursor.fetchall()}
        assert {
            "_rowid",
            "id",
            "user_id",
            "memory_id",
            "surface_type",
            "anchor_type",
            "alias_kind",
            "language_code",
            "surface_text",
            "surface_key",
            "preserve_verbatim",
            "non_evidential",
            "confidence",
            "derivation_kind",
            "derivation_model",
            "derivation_prompt_version",
            "derivation_json",
            "status",
            "created_at",
            "updated_at",
        }.issubset(surface_columns)
        fact_facet_columns_cursor = await connection.execute(
            "PRAGMA table_info(memory_fact_facets);"
        )
        fact_facet_columns = {
            row["name"] for row in await fact_facet_columns_cursor.fetchall()
        }
        assert {
            "_rowid",
            "id",
            "user_id",
            "conversation_id",
            "memory_id",
            "source_message_id",
            "source_span_id",
            "source_hash",
            "subject_surface",
            "subject_cluster_id",
            "surface_class",
            "facet_label",
            "value_text",
            "value_norm_key",
            "value_type",
            "assertion_kind",
            "list_group_key",
            "support_kind",
            "observed_at",
            "valid_from",
            "valid_to",
            "current_state",
            "resolved_interval_start",
            "resolved_interval_end",
            "temporal_confidence",
            "language_code",
            "confidence",
            "schema_version",
            "created_at",
        }.issubset(fact_facet_columns)
        conversation_columns_cursor = await connection.execute("PRAGMA table_info(conversations);")
        conversation_columns = {row["name"] for row in await conversation_columns_cursor.fetchall()}
        assert {
            "temporary",
            "temporary_ttl_seconds",
            "purge_on_close",
            "isolated_mode",
            "last_activity_at",
            "closed_at",
            "active_presence_id",
            "active_space_id",
            "active_mind_id",
            "mind_topology",
            "active_embodiment_id",
            "active_realm_id",
        }.issubset(conversation_columns)
        await connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES ('usr_schema', NULL, '2026-05-12T00:00:00+00:00', '2026-05-12T00:00:00+00:00', NULL)
            """
        )
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('schema_mode', 'Schema Mode', 'hash_schema', '{}', '2026-05-12T00:00:00+00:00', '2026-05-12T00:00:00+00:00')
            """
        )
        await connection.execute(
            """
            INSERT INTO conversations(
                id,
                user_id,
                assistant_mode_id,
                status,
                metadata_json,
                created_at,
                updated_at,
                mind_topology
                )
            VALUES (
                'cnv_ojocentauri',
                'usr_schema',
                'schema_mode',
                'active',
                '{}',
                '2026-05-12T00:00:00+00:00',
                '2026-05-12T00:00:00+00:00',
                'ojocentauri'
            )
            """
        )
        worker_job_columns_cursor = await connection.execute("PRAGMA table_info(worker_job_runs);")
        worker_job_columns = {row["name"] for row in await worker_job_columns_cursor.fetchall()}
        assert {
            "job_id",
            "stream_name",
            "job_type",
            "user_id",
            "conversation_id",
            "source_message_ids_json",
            "status",
            "attempt_count",
            "source_token_estimate",
            "size_bucket",
            "queued_at",
            "started_at",
            "finished_at",
            "last_heartbeat_at",
            "duration_ms",
            "deferred_until",
            "transient_defer_count",
            "first_deferred_at",
            "last_deferred_at",
            "error_class",
            "error_message",
            "metadata_json",
        }.issubset(worker_job_columns)
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
        graph_mentions_columns_cursor = await connection.execute("PRAGMA table_info(graph_entity_mentions);")
        graph_mentions_columns = {
            row["name"] for row in await graph_mentions_columns_cursor.fetchall()
        }
        assert "source_occurrence_key" in graph_mentions_columns
        graph_source_columns_cursor = await connection.execute("PRAGMA table_info(graph_relationship_sources);")
        graph_source_columns = {
            row["name"] for row in await graph_source_columns_cursor.fetchall()
        }
        assert "source_occurrence_key" in graph_source_columns
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
            "intimacy_boundary",
            "intimacy_boundary_confidence",
            "status",
            "created_by",
            "created_at",
            "updated_at",
            "payload_json",
            "space_id",
            "space_boundary_mode",
            "memory_owner_id",
            "source_mind_id",
            "embodiment_id",
            "realm_id",
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
            "intimacy_boundary",
            "intimacy_boundary_confidence",
            "preserve_verbatim",
            "skip_raw_by_default",
            "requires_explicit_request",
            "metadata_json",
            "summary_text",
            "index_text",
            "payload_blob_id",
            "created_at",
            "updated_at",
            "deleted_at",
            "memory_owner_id",
            "source_mind_id",
            "embodiment_id",
            "realm_id",
        }.issubset(artifact_columns)
        artifact_payload_columns_cursor = await connection.execute("PRAGMA table_info(artifact_payload_blobs);")
        artifact_payload_columns = {row["name"] for row in await artifact_payload_columns_cursor.fetchall()}
        assert {
            "id",
            "user_id",
            "storage_kind",
            "identity_kind",
            "content_sha256",
            "byte_size",
            "blob_bytes",
            "storage_key",
            "external_uri",
            "status",
            "created_at",
            "updated_at",
        }.issubset(artifact_payload_columns)
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
            "intimacy_boundary",
            "intimacy_boundary_confidence",
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
        graph_entity_columns_cursor = await connection.execute("PRAGMA table_info(graph_entities);")
        graph_entity_columns = {row["name"] for row in await graph_entity_columns_cursor.fetchall()}
        assert {
            "_rowid",
            "id",
            "user_id",
            "entity_type",
            "display_name",
            "confidence",
            "status",
            "privacy_level",
            "intimacy_boundary",
            "created_at",
            "updated_at",
        }.issubset(graph_entity_columns)
        graph_relationship_columns_cursor = await connection.execute("PRAGMA table_info(graph_relationships);")
        graph_relationship_columns = {
            row["name"] for row in await graph_relationship_columns_cursor.fetchall()
        }
        assert {
            "_rowid",
            "id",
            "user_id",
            "source_entity_id",
            "target_entity_id",
            "target_value_json",
            "predicate",
            "scope",
            "status",
            "dedupe_key",
        }.issubset(graph_relationship_columns)
        initial_context_package_columns_cursor = await connection.execute(
            "PRAGMA table_info(initial_context_packages);"
        )
        initial_context_package_columns = {
            row["name"] for row in await initial_context_package_columns_cursor.fetchall()
        }
        assert {
            "_rowid",
            "id",
            "package_key_hash",
            "package_kind",
            "version",
            "user_id",
            "conversation_id",
            "retrieval_profile_id",
            "key_json",
            "policy_signature_json",
            "coordinate_signature_json",
            "source_fingerprint_json",
            "blocks_json",
            "source_refs_json",
            "diagnostics_json",
            "build_status",
            "created_at",
            "updated_at",
            "valid_until",
        }.issubset(initial_context_package_columns)
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
async def test_memory_retrieval_surfaces_fts_owner_guard_and_cascade() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await connection.executescript(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES
                ('usr_1', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL),
                ('usr_2', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL);

            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO conversations(id, user_id, workspace_id, assistant_mode_id, title, status, metadata_json, created_at, updated_at)
            VALUES ('cnv_1', 'usr_1', NULL, 'coding_debug', 'Conversation', 'active', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                object_type, scope, canonical_text, index_text, payload_json,
                source_kind, confidence, stability, vitality, maya_score,
                privacy_level, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', NULL, 'cnv_1', 'coding_debug',
                'evidence', 'chat', 'Ben moved to 44 Pine Lane', NULL, '{}',
                'extracted', 0.8, 0.5, 0.0, 0.0,
                0, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            );

            INSERT INTO memory_retrieval_surfaces(
                id, user_id, memory_id, surface_type, language_code,
                surface_text, surface_key, non_evidential, derivation_kind,
                created_at, updated_at
            )
            VALUES (
                'mrs_1', 'usr_1', 'mem_1', 'alias', 'es',
                'nuevo apartamento', 'nuevo apartamento', 1, 'manual_fixture',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            );
            """
        )
        await connection.commit()

        cursor = await connection.execute(
            """
            SELECT mrs.id
            FROM memory_retrieval_surfaces_fts
            JOIN memory_retrieval_surfaces AS mrs
              ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
            WHERE memory_retrieval_surfaces_fts MATCH 'apartamento'
            """
        )
        assert [row["id"] for row in await cursor.fetchall()] == ["mrs_1"]

        await connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET surface_text = ?,
                surface_key = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                "new apartment",
                "new apartment",
                "2026-03-30T00:05:00+00:00",
                "mrs_1",
            ),
        )
        await connection.commit()

        cursor = await connection.execute(
            "SELECT rowid FROM memory_retrieval_surfaces_fts WHERE memory_retrieval_surfaces_fts MATCH 'apartamento'"
        )
        assert await cursor.fetchall() == []
        cursor = await connection.execute(
            "SELECT rowid FROM memory_retrieval_surfaces_fts WHERE memory_retrieval_surfaces_fts MATCH 'apartment'"
        )
        assert len(await cursor.fetchall()) == 1

        with pytest.raises(aiosqlite.IntegrityError, match="user_id must match"):
            await connection.execute(
                """
                INSERT INTO memory_retrieval_surfaces(
                    id, user_id, memory_id, surface_type, surface_text,
                    surface_key, derivation_kind, created_at, updated_at
                )
                VALUES (
                    'mrs_wrong_user', 'usr_2', 'mem_1', 'alias',
                    'wrong owner', 'wrong owner', 'manual_fixture',
                    '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                )
                """
            )

        await connection.execute("DELETE FROM memory_objects WHERE id = 'mem_1'")
        await connection.commit()
        cursor = await connection.execute("SELECT COUNT(*) AS count FROM memory_retrieval_surfaces")
        row = await cursor.fetchone()
        assert row["count"] == 0
        cursor = await connection.execute(
            "SELECT rowid FROM memory_retrieval_surfaces_fts WHERE memory_retrieval_surfaces_fts MATCH 'apartment'"
        )
        assert await cursor.fetchall() == []

        await connection.executescript(
            """
            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                object_type, scope, canonical_text, index_text, payload_json,
                source_kind, confidence, stability, vitality, maya_score,
                privacy_level, status, created_at, updated_at
            )
            VALUES (
                'mem_erase', 'usr_1', NULL, 'cnv_1', 'coding_debug',
                'evidence', 'chat', 'User erasure surface base', NULL, '{}',
                'extracted', 0.8, 0.5, 0.0, 0.0,
                0, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            );

            INSERT INTO memory_retrieval_surfaces(
                id, user_id, memory_id, surface_type, surface_text,
                surface_key, non_evidential, derivation_kind, created_at, updated_at
            )
            VALUES (
                'mrs_erase', 'usr_1', 'mem_erase', 'alias',
                'erase surface', 'erase surface', 1, 'manual_fixture',
                '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
            );
            """
        )
        await connection.commit()
        await connection.execute("DELETE FROM memory_objects WHERE user_id = 'usr_1'")
        await connection.commit()
        cursor = await connection.execute("SELECT COUNT(*) AS count FROM memory_retrieval_surfaces")
        row = await cursor.fetchone()
        assert row["count"] == 0
        cursor = await connection.execute(
            "SELECT rowid FROM memory_retrieval_surfaces_fts WHERE memory_retrieval_surfaces_fts MATCH 'erase'"
        )
        assert await cursor.fetchall() == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_fact_facets_supersedes_owner_guard() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await connection.executescript(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES
                ('usr_1', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL),
                ('usr_2', NULL, '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00', NULL);

            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO conversations(id, user_id, workspace_id, assistant_mode_id, title, status, metadata_json, created_at, updated_at)
            VALUES
                ('cnv_1', 'usr_1', NULL, 'coding_debug', 'One', 'active', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'),
                ('cnv_2', 'usr_2', NULL, 'coding_debug', 'Two', 'active', '{}', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO messages(id, conversation_id, role, seq, text, token_count, metadata_json, created_at)
            VALUES
                ('msg_1', 'cnv_1', 'user', 1, 'Paris.', 1, '{}', '2026-03-30T00:00:00+00:00'),
                ('msg_2', 'cnv_2', 'user', 1, 'Rome.', 1, '{}', '2026-03-30T00:00:00+00:00');

            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                object_type, scope, canonical_text, index_text, payload_json,
                source_kind, confidence, stability, vitality, maya_score,
                privacy_level, status, created_at, updated_at
            )
            VALUES
                (
                    'mem_1', 'usr_1', NULL, 'cnv_1', 'coding_debug',
                    'evidence', 'chat', 'User mentioned Paris.', NULL, '{}',
                    'extracted', 0.8, 0.5, 0.0, 0.0,
                    0, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                ),
                (
                    'mem_2', 'usr_2', NULL, 'cnv_2', 'coding_debug',
                    'evidence', 'chat', 'User mentioned Rome.', NULL, '{}',
                    'extracted', 0.8, 0.5, 0.0, 0.0,
                    0, 'active', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                );

            INSERT INTO memory_support_edges(id, user_id, memory_id, created_at, updated_at)
            VALUES
                ('edge_1', 'usr_1', 'mem_1', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'),
                ('edge_2', 'usr_2', 'mem_2', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO memory_evidence_spans(
                id, user_id, support_edge_id, memory_id, conversation_id, message_id,
                span_role, quote_text, created_at, updated_at
            )
            VALUES
                ('span_1', 'usr_1', 'edge_1', 'mem_1', 'cnv_1', 'msg_1', 'source', 'Paris.', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'),
                ('span_2', 'usr_2', 'edge_2', 'mem_2', 'cnv_2', 'msg_2', 'source', 'Rome.', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00');

            INSERT INTO memory_fact_facets(
                id, user_id, conversation_id, memory_id, source_message_id,
                source_span_id, source_hash, subject_surface, surface_class,
                facet_label, value_text, value_norm_key, assertion_kind,
                support_kind, observed_at, created_at
            )
            VALUES
                (
                    'mff_1', 'usr_1', 'cnv_1', 'mem_1', 'msg_1',
                    'span_1', 'hash_1', 'user', 'structured',
                    'city', 'Paris', 'paris', 'evidence',
                    'direct', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                ),
                (
                    'mff_2', 'usr_2', 'cnv_2', 'mem_2', 'msg_2',
                    'span_2', 'hash_2', 'user', 'structured',
                    'city', 'Rome', 'rome', 'evidence',
                    'direct', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                );
            """
        )

        with pytest.raises(aiosqlite.IntegrityError, match="supersedes_fact_id"):
            await connection.execute(
                """
                INSERT INTO memory_fact_facets(
                    id, user_id, conversation_id, memory_id, source_message_id,
                    source_span_id, source_hash, subject_surface, surface_class,
                    facet_label, value_text, value_norm_key, assertion_kind,
                    support_kind, observed_at, current_state, supersedes_fact_id,
                    created_at
                )
                VALUES (
                    'mff_cross_user', 'usr_2', 'cnv_2', 'mem_2', 'msg_2',
                    'span_2', 'hash_3', 'user', 'structured',
                    'city', 'Milan', 'milan', 'evidence',
                    'direct', '2026-03-30T00:00:00+00:00', 1, 'mff_1',
                    '2026-03-30T00:00:00+00:00'
                )
                """
            )

        with pytest.raises(aiosqlite.IntegrityError, match="supersedes_fact_id"):
            await connection.execute(
                """
                UPDATE memory_fact_facets
                SET supersedes_fact_id = 'mff_1'
                WHERE id = 'mff_2'
                """
            )

        with pytest.raises(aiosqlite.IntegrityError, match="supersedes_fact_id"):
            await connection.execute(
                """
                UPDATE memory_fact_facets
                SET supersedes_fact_id = 'mff_nonexistent'
                WHERE id = 'mff_1'
                """
            )

        await connection.execute(
            """
            UPDATE memory_fact_facets
            SET supersedes_fact_id = 'mff_2'
            WHERE id = 'mff_2'
            """
        )
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
                'mem_1', 'usr_1', NULL, 'cnv_1', 'coding_debug', 'evidence', 'chat',
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
                    'mem_pending', 'usr_1', 'personal_assistant', 'evidence', 'chat',
                    'Banking card PIN: 4512', '{}', 'extracted', 0.9, 0.5, 0.0, 0.0, 3,
                    'pin_or_password', 1, NULL, NULL, 'unknown',
                    'pending_user_confirmation', '2026-03-30T00:00:00+00:00', '2026-03-30T00:00:00+00:00'
                ),
                (
                    'mem_declined', 'usr_1', 'personal_assistant', 'evidence', 'chat',
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
                'vbp_1', 'usr_1', 'chat', 'message', 'msg_1',
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
                'mem_1', 'usr_1', 'wrk_1', NULL, 'general_qa', 'interaction_contract', 'character',
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
                'ctr_1', 'usr_1', 'wrk_1', NULL, 'general_qa', 'character',
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
                    'ctr_2', 'usr_1', 'wrk_1', NULL, 'general_qa', 'character',
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
                'mem_1', 'usr_1', NULL, NULL, 'coding_debug', 'evidence', 'user',
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
                    'mem_2', 'usr_1', NULL, NULL, 'coding_debug', 'evidence', 'user',
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
                'mem_3', 'usr_2', NULL, NULL, 'coding_debug', 'evidence', 'user',
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

        expected_versions = [
            migration.version
            for migration in manager.discover()
            if migration.version >= 12
        ]
        assert [migration.version for migration in applied] == expected_versions
        assert [row["id"] for row in rows] == ["sum_conv", "sum_rollup"]
        assert all(row["user_id"] == "usr_1" for row in rows)
        assert all(row["hierarchy_level"] == 0 for row in rows)
        assert rows[0]["source_message_start_seq"] == 1
        assert rows[1]["source_message_start_seq"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_migration_0032_converts_workspace_rollups_to_character_rollups(tmp_path: Path) -> None:
    db_path = tmp_path / "phase8.sqlite"
    bootstrap_migrations = tmp_path / "bootstrap_migrations_0032"
    bootstrap_migrations.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name.startswith("0032"):
            break
        copy2(path, bootstrap_migrations / path.name)

    connection = await initialize_database(str(db_path), bootstrap_migrations)
    try:
        await connection.executescript(
            """
            INSERT INTO users(id, created_at, updated_at)
            VALUES ('usr_1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');

            INSERT INTO assistant_modes(id, display_name, memory_policy_json, created_at, updated_at, privacy_ceiling)
            VALUES ('mode_1', 'Mode', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 3);

            INSERT INTO workspaces(id, user_id, name, created_at, updated_at)
            VALUES ('ws_1', 'usr_1', 'WS', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');

            INSERT INTO summary_views(
                id, user_id, workspace_id, summary_kind, summary_text,
                source_object_ids_json, maya_score, model, created_at
            )
            VALUES (
                'sum_rollup', 'usr_1', 'ws_1', 'workspace_rollup', 'Workspace summary',
                '[]', 1.5, 'model-a', '2026-01-01T00:00:00+00:00'
            );

            INSERT INTO memory_objects(
                id, user_id, workspace_id, assistant_mode_id, object_type, scope,
                scope_canonical, canonical_text, payload_json, source_kind,
                confidence, privacy_level, status, created_at, updated_at
            )
            VALUES (
                'sum_mem_sum_rollup', 'usr_1', 'ws_1', 'mode_1', 'summary_view', 'workspace',
                'legacy_workspace', 'Workspace summary',
                '{"summary_kind":"workspace_rollup","summary_view_id":"sum_rollup","hierarchy_level":0,"source_object_ids":[]}',
                'summarized', 0.72, 0, 'active',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );
            """
        )
        await connection.commit()
    finally:
        await connection.close()

    upgraded = await initialize_database(str(db_path), MIGRATIONS_DIR)
    try:
        cursor = await upgraded.execute(
            """
            SELECT summary_kind, character_id, platform_id, sensitivity, scope_canonical
            FROM summary_views
            WHERE id = 'sum_rollup'
            """
        )
        summary = await cursor.fetchone()
        assert summary["summary_kind"] == "character_rollup"
        assert summary["character_id"] == "ws_1"
        assert summary["platform_id"] == "default"
        assert summary["sensitivity"] == "public"
        assert summary["scope_canonical"] == "character"

        cursor = await upgraded.execute(
            """
            SELECT payload_json, character_id, platform_id, sensitivity, scope_canonical
            FROM memory_objects
            WHERE id = 'sum_mem_sum_rollup'
            """
        )
        mirror = await cursor.fetchone()
        assert mirror["character_id"] == "ws_1"
        assert mirror["platform_id"] == "default"
        assert mirror["sensitivity"] == "public"
        assert mirror["scope_canonical"] == "character"
        assert '"summary_kind":"character_rollup"' in mirror["payload_json"]
    finally:
        await upgraded.close()


@pytest.mark.asyncio
async def test_migration_0033_upgrades_existing_secondary_surfaces(tmp_path: Path) -> None:
    db_path = tmp_path / "phase10.sqlite"
    bootstrap_migrations = tmp_path / "bootstrap_migrations_0033"
    bootstrap_migrations.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name.startswith("0033"):
            break
        copy2(path, bootstrap_migrations / path.name)

    connection = await initialize_database(str(db_path), bootstrap_migrations)
    try:
        await connection.executescript(
            """
            INSERT INTO users(
                id, created_at, updated_at, remember_across_chats, remember_across_devices
            )
            VALUES ('usr_1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1, 0);

            INSERT INTO assistant_modes(id, display_name, memory_policy_json, created_at, updated_at, privacy_ceiling)
            VALUES ('mode_1', 'Mode', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 3);

            INSERT INTO workspaces(id, user_id, name, created_at, updated_at)
            VALUES ('ws_1', 'usr_1', 'WS', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');

            INSERT INTO conversations(
                id, user_id, workspace_id, assistant_mode_id, status,
                user_persona_id, platform_id, character_id, incognito,
                created_at, updated_at
            )
            VALUES (
                'cnv_1', 'usr_1', 'ws_1', 'mode_1', 'active',
                'persona_1', 'web', 'char_1', 0,
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );

            INSERT INTO artifacts(
                id, user_id, workspace_id, conversation_id, artifact_type,
                source_kind, status, privacy_level, created_at, updated_at
            )
            VALUES (
                'art_1', 'usr_1', 'ws_1', 'cnv_1', 'pasted_text',
                'pasted_text', 'ready', 2,
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );

            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                object_type, scope, canonical_text, payload_json, source_kind,
                confidence, privacy_level, status, created_at, updated_at
            )
            VALUES (
                'mem_1', 'usr_1', 'ws_1', NULL, 'mode_1',
                'interaction_contract', 'conversation', 'Keep answers concise',
                '{}', 'extracted', 0.8, 0, 'active',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );

            INSERT INTO contract_dimensions_current(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                scope, dimension_name, value_json, confidence,
                source_memory_id, updated_at
            )
            VALUES (
                'ctr_1', 'usr_1', 'ws_1', NULL, 'mode_1',
                'conversation', 'tone', '{"style":"brief"}', 0.8,
                'mem_1', '2026-01-01T00:00:00+00:00'
            );
            """
        )
        await connection.commit()
    finally:
        await connection.close()

    upgraded = await initialize_database(str(db_path), MIGRATIONS_DIR)
    try:
        cursor = await upgraded.execute("PRAGMA table_xinfo(contract_dimensions_current)")
        contract_columns = {row["name"] for row in await cursor.fetchall()}
        assert "scope_canonical_key" in contract_columns

        cursor = await upgraded.execute(
            """
            SELECT scope_canonical, scope_canonical_key
            FROM contract_dimensions_current
            WHERE id = 'ctr_1'
            """
        )
        contract = await cursor.fetchone()
        assert contract["scope_canonical"] == "chat"
        assert contract["scope_canonical_key"] == "chat"

        await upgraded.execute(
            """
            INSERT INTO contract_dimensions_current(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                scope, dimension_name, value_json, confidence,
                source_memory_id, user_persona_id, character_id,
                scope_canonical, updated_at
            )
            VALUES (
                'ctr_2', 'usr_1', 'ws_1', NULL, 'mode_1',
                'character', 'tone', '{"style":"warm"}', 0.9,
                'mem_1', 'persona_1', 'char_1',
                'character', '2026-01-01T00:00:00+00:00'
            )
            """
        )

        cursor = await upgraded.execute(
            """
            SELECT
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                platform_locked,
                platform_id_lock,
                scope_canonical,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot
            FROM artifacts
            WHERE id = 'art_1'
            """
        )
        artifact = await cursor.fetchone()
        assert dict(artifact) == {
            "user_persona_id": "persona_1",
            "platform_id": "web",
            "character_id": "char_1",
            "sensitivity": "private",
            "platform_locked": 1,
            "platform_id_lock": "web",
            "scope_canonical": "chat",
            "incognito_snapshot": 0,
            "remember_across_chats_snapshot": 1,
            "remember_across_devices_snapshot": 0,
        }
    finally:
        await upgraded.close()


@pytest.mark.asyncio
async def test_migration_0034_finalizes_canonical_scope_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "phase11.sqlite"
    bootstrap_migrations = tmp_path / "bootstrap_migrations_0034"
    bootstrap_migrations.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name.startswith("0034"):
            break
        copy2(path, bootstrap_migrations / path.name)

    connection = await initialize_database(str(db_path), bootstrap_migrations)
    try:
        await connection.executescript(
            """
            INSERT INTO users(id, created_at, updated_at, remember_across_chats, remember_across_devices)
            VALUES ('usr_1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1, 1);

            INSERT INTO assistant_modes(id, display_name, memory_policy_json, created_at, updated_at, privacy_ceiling)
            VALUES ('mode_1', 'Mode', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 3);

            INSERT INTO workspaces(id, user_id, name, created_at, updated_at)
            VALUES ('ws_1', 'usr_1', 'WS', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');

            INSERT INTO conversations(
                id, user_id, workspace_id, assistant_mode_id, status,
                user_persona_id, platform_id, character_id, mode, incognito,
                created_at, updated_at
            )
            VALUES (
                'cnv_1', 'usr_1', 'ws_1', 'mode_1', 'active',
                NULL, 'default', 'char_1', 'mode_1', 0,
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );

            INSERT INTO memory_objects(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                object_type, scope, scope_canonical, canonical_text, index_text,
                payload_json, source_kind, confidence, privacy_level,
                status, created_at, updated_at
            )
            VALUES
                (
                    'mem_chat', 'usr_1', 'ws_1', 'cnv_1', 'mode_1',
                    'evidence', 'conversation', 'chat', 'chat fact', 'chat fact',
                    '{}', 'extracted', 0.9, 0, 'active',
                    '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                ),
                (
                    'mem_user', 'usr_1', NULL, NULL, NULL,
                    'belief', 'global_user', 'user', 'user fact', 'user fact',
                    '{}', 'extracted', 0.8, 0, 'active',
                    '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                ),
                (
                    'mem_legacy_workspace', 'usr_1', 'ws_1', NULL, 'mode_1',
                    'belief', 'workspace', 'legacy_workspace', 'legacy workspace fact', 'legacy workspace fact',
                    '{}', 'extracted', 0.7, 0, 'active',
                    '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                );

            INSERT INTO memory_embedding_metadata(memory_id, user_id, object_type, scope, created_at, scope_canonical)
            VALUES
                ('mem_chat', 'usr_1', 'evidence', 'conversation', '2026-01-01T00:00:00+00:00', 'chat'),
                ('mem_legacy_workspace', 'usr_1', 'belief', 'workspace', '2026-01-01T00:00:00+00:00', 'legacy_workspace');

            INSERT INTO memory_links(
                id, user_id, src_memory_id, dst_memory_id, relation_type, weight, metadata_json, created_at
            )
            VALUES
                ('lnk_keep', 'usr_1', 'mem_chat', 'mem_chat', 'supports', 1.0, '{}', '2026-01-01T00:00:00+00:00'),
                ('lnk_drop', 'usr_1', 'mem_chat', 'mem_legacy_workspace', 'belongs_to_workspace', 1.0, '{}', '2026-01-01T00:00:00+00:00');

            INSERT INTO verbatim_pins(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                scope, target_kind, target_id, canonical_text, index_text,
                privacy_level, created_by, created_at, updated_at,
                sensitivity, scope_canonical
            )
            VALUES (
                'pin_1', 'usr_1', 'ws_1', 'cnv_1', 'mode_1',
                'conversation', 'message', 'msg_1', 'exact quote', 'exact quote',
                0, 'usr_1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                'public', 'chat'
            );

            INSERT INTO contract_dimensions_current(
                id, user_id, workspace_id, conversation_id, assistant_mode_id,
                scope, scope_canonical, dimension_name, value_json, confidence,
                source_memory_id, updated_at
            )
            VALUES (
                'ctr_1', 'usr_1', 'ws_1', 'cnv_1', 'mode_1',
                'conversation', 'chat', 'tone', '{"style":"brief"}', 0.8,
                'mem_chat', '2026-01-01T00:00:00+00:00'
            );
            """
        )
        await connection.commit()
    finally:
        await connection.close()

    upgraded = await initialize_database(str(db_path), MIGRATIONS_DIR)
    try:
        async def create_sql(table: str) -> str:
            cursor = await upgraded.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            )
            row = await cursor.fetchone()
            return str(row["sql"])

        memory_sql = await create_sql("memory_objects")
        pin_sql = await create_sql("verbatim_pins")
        assert "scope IN ('chat', 'character', 'user')" in memory_sql
        assert "global_user" not in memory_sql
        assert "scope IN ('chat', 'character', 'user')" in pin_sql
        assert "ephemeral_session" not in pin_sql

        with pytest.raises(aiosqlite.IntegrityError):
            await upgraded.execute(
                """
                INSERT INTO memory_objects(
                    id, user_id, object_type, scope, canonical_text,
                    source_kind, confidence, privacy_level, created_at, updated_at
                )
                VALUES (
                    'mem_bad_scope', 'usr_1', 'evidence', 'conversation', 'bad',
                    'extracted', 0.5, 0, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                )
                """
            )

        cursor = await upgraded.execute("SELECT id, scope, scope_canonical FROM memory_objects ORDER BY id")
        memories = {row["id"]: dict(row) for row in await cursor.fetchall()}
        assert memories == {
            "mem_chat": {"id": "mem_chat", "scope": "chat", "scope_canonical": "chat"},
            "mem_user": {"id": "mem_user", "scope": "user", "scope_canonical": "user"},
        }

        cursor = await upgraded.execute(
            "SELECT affected_count, sample_ids_json FROM memory_redesign_phase11_report "
            "WHERE table_name = 'memory_objects'"
        )
        report = await cursor.fetchone()
        assert report["affected_count"] == 1
        assert "mem_legacy_workspace" in report["sample_ids_json"]

        cursor = await upgraded.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_objects_fts
            JOIN memory_objects ON memory_objects._rowid = memory_objects_fts.rowid
            """
        )
        fts_memory_count = await cursor.fetchone()
        assert fts_memory_count["count"] == len(memories)

        cursor = await upgraded.execute("SELECT scope, scope_canonical FROM verbatim_pins WHERE id = 'pin_1'")
        pin = await cursor.fetchone()
        assert dict(pin) == {"scope": "chat", "scope_canonical": "chat"}

        cursor = await upgraded.execute(
            """
            SELECT COUNT(*) AS count
            FROM verbatim_pins_fts
            JOIN verbatim_pins ON verbatim_pins._rowid = verbatim_pins_fts.rowid
            """
        )
        fts_pin_count = await cursor.fetchone()
        assert fts_pin_count["count"] == 1

        cursor = await upgraded.execute("SELECT scope, scope_canonical FROM contract_dimensions_current")
        contract = await cursor.fetchone()
        assert dict(contract) == {"scope": "chat", "scope_canonical": "chat"}

        cursor = await upgraded.execute("SELECT id, relation_type FROM memory_links ORDER BY id")
        links = [dict(row) for row in await cursor.fetchall()]
        assert links == [{"id": "lnk_keep", "relation_type": "supports"}]

        cursor = await upgraded.execute("PRAGMA table_info(conversations)")
        conversation_columns = {row["name"]: row for row in await cursor.fetchall()}
        assert conversation_columns["assistant_mode_id"]["notnull"] == 0

        await upgraded.execute(
            """
            INSERT INTO memory_consent_profile(user_id, category, confirmed_count, declined_count, updated_at)
            VALUES ('usr_1', 'financial', 0, 0, '2026-01-01T00:00:00+00:00')
            """
        )
        await upgraded.execute(
            """
            INSERT INTO memory_consent_profile(
                user_id, category, confirmed_count, declined_count, updated_at, user_persona_id
            )
            VALUES ('usr_1', 'financial', 0, 0, '2026-01-01T00:00:00+00:00', 'persona_1')
            """
        )
        await upgraded.commit()
    finally:
        await upgraded.close()


@pytest.mark.asyncio
async def test_migration_0031_adds_redesign_identity_columns() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        async def column_names(table: str) -> set[str]:
            # `table_xinfo` is the variant that surfaces VIRTUAL generated
            # columns alongside ordinary columns; `table_info` hides them.
            cursor = await connection.execute(f"PRAGMA table_xinfo({table})")
            return {row["name"] for row in await cursor.fetchall()}

        users_columns = await column_names("users")
        assert {
            "remember_across_chats",
            "remember_across_devices",
            "memory_privacy_mode",
        }.issubset(users_columns)

        conversations_columns = await column_names("conversations")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "mode",
            "incognito",
            "active_mind_id",
            "mind_topology",
        }.issubset(conversations_columns)

        memory_columns = await column_names("memory_objects")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "sensitivity",
            "themes_json",
            "auto_expires",
            "platform_locked",
            "platform_id_lock",
            "scope_canonical",
            "memory_owner_id",
            "source_mind_id",
        }.issubset(memory_columns)

        message_columns = await column_names("messages")
        assert {
            "user_persona_id_snapshot",
            "platform_id_snapshot",
            "character_id_snapshot",
            "mode_snapshot",
            "incognito_snapshot",
            "remember_across_chats_snapshot",
            "remember_across_devices_snapshot",
            "temporary_snapshot",
            "purge_on_close_snapshot",
            "valid_to_snapshot",
            "policy_snapshot_json",
            "sensitivity",
            "themes_json",
            "platform_locked",
            "platform_id_lock",
            "active_mind_id",
            "source_mind_id",
            "active_embodiment_id",
        }.issubset(message_columns)

        consent_columns = await column_names("memory_consent_profile")
        assert {"user_persona_id", "user_persona_key"}.issubset(consent_columns)

        contract_columns = await column_names("contract_dimensions_current")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "user_persona_key",
            "character_key",
            "conversation_key",
            "scope_canonical_key",
            "space_id",
            "space_boundary_mode",
            "space_key",
            "memory_owner_id",
            "source_mind_id",
            "memory_owner_key",
            "embodiment_id",
            "embodiment_key",
            "policy_snapshot_json",
        }.issubset(contract_columns)

        artifact_columns = await column_names("artifacts")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "sensitivity",
            "platform_locked",
            "platform_id_lock",
            "scope_canonical",
            "policy_snapshot_json",
            "memory_owner_id",
            "source_mind_id",
            "embodiment_id",
        }.issubset(artifact_columns)

        pending_columns = await column_names("pending_memory_confirmations")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "policy_proven",
            "intended_scope",
            "intended_sensitivity",
        }.issubset(pending_columns)

        worker_columns = await column_names("worker_job_runs")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "incognito_snapshot",
            "policy_snapshot_json",
            "deferred_until",
            "transient_defer_count",
            "first_deferred_at",
            "last_deferred_at",
        }.issubset(worker_columns)

        embedding_columns = await column_names("memory_embedding_metadata")
        assert {
            "user_persona_id",
            "platform_id",
            "character_id",
            "sensitivity",
            "platform_locked",
            "platform_id_lock",
            "scope_canonical",
        }.issubset(embedding_columns)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_migration_0031_backfills_legacy_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "phase1.sqlite"
    pre_connection = await initialize_database(str(db_path), MIGRATIONS_DIR)
    try:
        await pre_connection.execute("ROLLBACK")
    except Exception:
        pass
    await pre_connection.close()

    # Reset and apply only migrations 0001..0030 to simulate a pre-Phase-1 DB.
    if db_path.exists():
        db_path.unlink()
    bootstrap_migrations = tmp_path / "bootstrap_migrations"
    bootstrap_migrations.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name.startswith("0031"):
            break
        copy2(path, bootstrap_migrations / path.name)

    connection = await initialize_database(str(db_path), bootstrap_migrations)
    try:
        await connection.executescript(
            """
            INSERT INTO users(id, created_at, updated_at) VALUES
                ('usr_1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            INSERT INTO assistant_modes(id, display_name, memory_policy_json, created_at, updated_at, privacy_ceiling)
                VALUES ('mode_1', 'Mode', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 3);
            INSERT INTO workspaces(id, user_id, name, created_at, updated_at)
                VALUES ('ws_1', 'usr_1', 'WS', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            INSERT INTO conversations(id, user_id, workspace_id, assistant_mode_id, status, created_at, updated_at, isolated_mode)
                VALUES
                ('cnv_normal', 'usr_1', 'ws_1', 'mode_1', 'active', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 0),
                ('cnv_iso', 'usr_1', NULL, 'mode_1', 'active', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1);
            INSERT INTO memory_objects(id, user_id, workspace_id, conversation_id, assistant_mode_id, object_type, scope, canonical_text, source_kind, privacy_level, created_at, updated_at)
                VALUES
                ('m_global', 'usr_1', NULL, NULL, NULL, 'belief', 'global_user', 'g', 'extracted', 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                ('m_workspace', 'usr_1', 'ws_1', NULL, 'mode_1', 'belief', 'workspace', 'w', 'extracted', 2, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                ('m_conv', 'usr_1', NULL, 'cnv_normal', 'mode_1', 'belief', 'conversation', 'c', 'extracted', 0, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                ('m_eph', 'usr_1', NULL, 'cnv_normal', 'mode_1', 'belief', 'ephemeral_session', 'e', 'extracted', 3, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),
                ('m_amode', 'usr_1', NULL, NULL, 'mode_1', 'belief', 'assistant_mode', 'a', 'extracted', 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            INSERT INTO memory_consent_profile(user_id, category, confirmed_count, declined_count, updated_at)
                VALUES ('usr_1', 'medication', 0, 0, '2026-01-01T00:00:00+00:00');
            """
        )
        await connection.commit()
    finally:
        await connection.close()

    # Apply 0031 against the populated DB.
    upgraded = await initialize_database(str(db_path), MIGRATIONS_DIR)
    try:
        cursor = await upgraded.execute(
            "SELECT id, mode, character_id, platform_id, incognito FROM conversations ORDER BY id"
        )
        conversations = {row["id"]: dict(row) for row in await cursor.fetchall()}
        assert conversations["cnv_normal"]["mode"] == "mode_1"
        assert conversations["cnv_normal"]["character_id"] == "ws_1"
        assert conversations["cnv_normal"]["platform_id"] == "default"
        assert conversations["cnv_normal"]["incognito"] == 0
        assert conversations["cnv_iso"]["incognito"] == 1
        assert conversations["cnv_iso"]["character_id"] is None

        cursor = await upgraded.execute(
            "SELECT id, scope, scope_canonical, sensitivity, auto_expires, character_id, platform_id "
            "FROM memory_objects ORDER BY id"
        )
        memories = {row["id"]: dict(row) for row in await cursor.fetchall()}
        assert memories["m_global"]["scope_canonical"] == "user"
        assert memories["m_global"]["sensitivity"] == "public"
        assert memories["m_conv"]["scope_canonical"] == "chat"
        assert memories["m_conv"]["character_id"] == "ws_1"
        assert memories["m_eph"]["scope_canonical"] == "chat"
        assert memories["m_eph"]["sensitivity"] == "secret"
        assert memories["m_eph"]["auto_expires"] == 1
        assert "m_workspace" not in memories
        assert "m_amode" not in memories

        cursor = await upgraded.execute(
            "SELECT affected_count, sample_ids_json FROM memory_redesign_phase11_report "
            "WHERE table_name = 'memory_objects'"
        )
        phase11_report = await cursor.fetchone()
        assert phase11_report["affected_count"] == 2
        assert "m_workspace" in phase11_report["sample_ids_json"]
        assert "m_amode" in phase11_report["sample_ids_json"]

        cursor = await upgraded.execute(
            """
            SELECT remember_across_chats, remember_across_devices, memory_privacy_mode
            FROM users
            WHERE id = 'usr_1'
            """
        )
        user_row = await cursor.fetchone()
        assert user_row["remember_across_chats"] == 1
        assert user_row["remember_across_devices"] == 1
        assert user_row["memory_privacy_mode"] == "balanced"

        # Tagged-key generated columns must encode NULL as 'n' and non-NULL as
        # 'v:<length>:<value>'. Insert a second persona-bound row to confirm.
        await upgraded.execute(
            "INSERT INTO memory_consent_profile(user_id, category, confirmed_count, declined_count, updated_at, user_persona_id)"
            " VALUES ('usr_1', 'financial', 0, 0, '2026-01-01T00:00:00+00:00', 'persona_alter')"
        )
        await upgraded.commit()
        cursor = await upgraded.execute(
            "SELECT category, user_persona_id, user_persona_key FROM memory_consent_profile ORDER BY category"
        )
        keys = {row["category"]: dict(row) for row in await cursor.fetchall()}
        assert keys["medication"]["user_persona_key"] == "n"
        assert keys["financial"]["user_persona_key"] == "v:13:persona_alter"
    finally:
        await upgraded.close()
