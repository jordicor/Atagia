"""Tests for legacy artifact payload migration helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from atagia.artifact_payload_migrate_cli import _migrate, _verify
from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository


MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@pytest.mark.asyncio
async def test_migrate_legacy_artifact_blobs_deduplicates_payload_records(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = ArtifactRepository(connection, clock)
        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-05-06T12:00:00+00:00', '2026-05-06T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")
        payload_bytes = b"duplicate legacy payload"
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        for artifact_id in ("art_a", "art_b"):
            await repository.create_artifact(
                artifact_id=artifact_id,
                user_id="usr_a",
                workspace_id=None,
                conversation_id="cnv_a",
                message_id=None,
                artifact_type="pasted_text",
                source_kind="pasted_text",
                mime_type="text/plain",
                filename=f"{artifact_id}.txt",
                content_hash=payload_sha256,
                size_bytes=len(payload_bytes),
                status="ready",
                metadata_json={},
                summary_text=f"pasted_text attachment; artifact_id={artifact_id}",
                index_text="duplicate legacy payload",
                storage_kind="sqlite_blob",
                blob_bytes=payload_bytes,
                blob_byte_size=len(payload_bytes),
                blob_sha256=payload_sha256,
            )

        result = await _migrate(
            connection,
            storage_path=tmp_path / "artifact_blobs",
            target_storage_kind="sqlite_blob",
            batch_size=10,
        )
        assert result.migrated_artifacts == 2
        assert result.legacy_active_artifacts_without_payload == 0
        assert result.payload_blob_count == 1

        cursor = await connection.execute(
            """
            SELECT COUNT(DISTINCT payload_blob_id) AS distinct_payloads
            FROM artifacts
            WHERE user_id = 'usr_a'
            """
        )
        row = await cursor.fetchone()
        assert row["distinct_payloads"] == 1
        blob = await repository.get_artifact_blob("art_a", "usr_a")
        assert blob is not None
        assert blob["storage_kind"] == "sqlite_blob"
        assert blob["blob_bytes"] == payload_bytes

        verify_result = await _verify(connection, storage_path=tmp_path / "artifact_blobs")
        assert verify_result.legacy_active_artifacts_without_payload == 0
        assert verify_result.hash_mismatches == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_migrate_rejects_legacy_blob_hash_mismatch_even_when_payload_exists(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = ArtifactRepository(connection, clock)
        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-05-06T12:00:00+00:00', '2026-05-06T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")
        good_bytes = b"same-sized payload"
        bad_bytes = b"same-sized PAYLOAD"
        good_sha256 = hashlib.sha256(good_bytes).hexdigest()
        for artifact_id, blob_bytes in (("art_good", good_bytes), ("art_bad", bad_bytes)):
            await repository.create_artifact(
                artifact_id=artifact_id,
                user_id="usr_a",
                workspace_id=None,
                conversation_id="cnv_a",
                message_id=None,
                artifact_type="pasted_text",
                source_kind="pasted_text",
                mime_type="text/plain",
                filename=f"{artifact_id}.txt",
                content_hash=good_sha256,
                size_bytes=len(blob_bytes),
                status="ready",
                metadata_json={},
                summary_text=f"pasted_text attachment; artifact_id={artifact_id}",
                index_text="legacy payload",
                storage_kind="sqlite_blob",
                blob_bytes=blob_bytes,
                blob_byte_size=len(blob_bytes),
                blob_sha256=good_sha256,
            )

        result = await _migrate(
            connection,
            storage_path=tmp_path / "artifact_blobs",
            target_storage_kind="sqlite_blob",
            batch_size=10,
        )
        assert result.migrated_artifacts == 1
        assert result.error_count == 1
        assert result.payload_blob_count == 1
        remaining_cursor = await connection.execute(
            "SELECT id FROM artifacts WHERE payload_blob_id IS NULL ORDER BY id ASC"
        )
        remaining = await remaining_cursor.fetchall()
        assert [row["id"] for row in remaining] == ["art_bad"]
    finally:
        await connection.close()
