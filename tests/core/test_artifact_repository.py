"""Tests for artifact persistence, chunking, and lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.models.schemas_api import AttachmentInput
from atagia.models.schemas_memory import MemoryScope
from atagia.services.artifact_blob_store import ArtifactBlobStore
from atagia.services.artifact_service import ArtifactService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


@pytest.mark.asyncio
async def test_artifact_service_creates_chunks_and_links_and_respects_user_isolation() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        artifact_service = ArtifactService(connection, clock)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await users.create_user("usr_b")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")
        await conversations.create_conversation("cnv_b", "usr_b", None, "coding_debug", "Other Chat")

        bundle = artifact_service.prepare_attachments(
            message_text="Please inspect the attachment.",
            attachments=[
                AttachmentInput(
                    kind="pasted_text",
                    content_text=("Alpha beta gamma delta epsilon " * 16) + "\n\n" + ("Zeta eta theta iota kappa " * 16),
                    title="Long Notes",
                    filename="notes.txt",
                    mime_type="text/plain",
                    privacy_level=1,
                    preserve_verbatim=True,
                    skip_raw_by_default=True,
                    requires_explicit_request=True,
                    metadata={"source": "unit-test"},
                )
            ],
            user_id="usr_a",
            conversation={
                "id": "cnv_a",
                "workspace_id": None,
                "assistant_mode_id": "coding_debug",
            },
        )
        assert "[Attachments omitted]" in bundle.prompt_text
        assert bundle.attachments[0]["artifact_id"]
        assert bundle.attachments[0]["relevance_state"] == "active_work_material"
        assert bundle.attachments[0]["relevance_source"] == "attachment_ingest"

        await connection.execute("BEGIN")
        try:
            user_message = await messages.create_message(
                message_id=None,
                conversation_id="cnv_a",
                role="user",
                seq=None,
                text=bundle.prompt_text,
                token_count=None,
                metadata={
                    "attachments": bundle.attachments,
                    "artifact_backed": True,
                    "skip_by_default": True,
                    "include_raw": False,
                    "requires_explicit_request": True,
                    "content_kind": "artifact",
                    "context_placeholder": bundle.context_placeholder,
                },
                occurred_at="2026-03-30T12:00:00+00:00",
                commit=False,
            )
            await artifact_service.persist_prepared_attachments(
                bundle=bundle,
                message_id=str(user_message["id"]),
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        artifacts_for_a = await repository.list_artifacts("usr_a", conversation_id="cnv_a")
        assert len(artifacts_for_a) == 1
        artifact = artifacts_for_a[0]
        assert artifact["status"] == "ready"
        assert artifact["source_kind"] == "pasted_text"
        assert artifact["preserve_verbatim"] == 1
        assert artifact["skip_raw_by_default"] == 1
        assert artifact["requires_explicit_request"] == 1

        assert await repository.get_artifact(str(artifact["id"]), "usr_b") is None
        assert await repository.list_artifacts("usr_b") == []

        blocked_payload = await artifact_service.fetch_artifact_payload(
            user_id="usr_a",
            artifact_id=str(artifact["id"]),
        )
        assert blocked_payload is not None
        assert blocked_payload.raw_available is True
        assert blocked_payload.raw_returned is False
        assert blocked_payload.raw_block_reason == "explicit_request_required"
        assert blocked_payload.content_bytes is None

        raw_payload = await artifact_service.fetch_artifact_payload(
            user_id="usr_a",
            artifact_id=str(artifact["id"]),
            include_raw=True,
        )
        assert raw_payload is not None
        assert raw_payload.raw_returned is True
        assert raw_payload.storage_kind == "sqlite_blob"
        assert raw_payload.content_bytes is not None
        assert raw_payload.content_bytes.decode("utf-8").startswith("Alpha beta gamma")
        assert await artifact_service.fetch_artifact_payload(
            user_id="usr_b",
            artifact_id=str(artifact["id"]),
            include_raw=True,
        ) is None

        chunks = await repository.list_artifact_chunks(str(artifact["id"]), "usr_a")
        assert len(chunks) >= 2
        assert chunks[0]["kind"] == "summary"
        assert chunks[0]["text"].startswith("pasted_text attachment")

        search_rows = await repository.search_artifact_chunks(
            user_id="usr_a",
            query="theta",
            privacy_ceiling=1,
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            limit=5,
        )
        assert search_rows
        assert search_rows[0]["artifact_id"] == str(artifact["id"])

        link_rows = await connection.execute(
            "SELECT COUNT(*) FROM artifact_links WHERE artifact_id = ? AND message_id = ?",
            (artifact["id"], user_message["id"]),
        )
        assert (await link_rows.fetchone())[0] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_delete_and_purge_exclude_results() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        artifact_service = ArtifactService(connection, clock)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        bundle = artifact_service.prepare_attachments(
            message_text="Here is the pasted note.",
            attachments=[
                AttachmentInput(
                    kind="pasted_text",
                    content_text="Alpha beta gamma delta",
                    title="Short Note",
                    filename="note.txt",
                    mime_type="text/plain",
                )
            ],
            user_id="usr_a",
            conversation={
                "id": "cnv_a",
                "workspace_id": None,
                "assistant_mode_id": "coding_debug",
            },
        )
        await connection.execute("BEGIN")
        try:
            user_message = await messages.create_message(
                message_id=None,
                conversation_id="cnv_a",
                role="user",
                seq=None,
                text=bundle.prompt_text,
                token_count=None,
                metadata={
                    "attachments": bundle.attachments,
                    "artifact_backed": True,
                    "skip_by_default": True,
                    "include_raw": False,
                    "requires_explicit_request": True,
                    "content_kind": "artifact",
                    "context_placeholder": bundle.context_placeholder,
                },
                occurred_at="2026-03-30T12:00:00+00:00",
                commit=False,
            )
            await artifact_service.persist_prepared_attachments(
                bundle=bundle,
                message_id=str(user_message["id"]),
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        artifact = (await repository.list_artifacts("usr_a", conversation_id="cnv_a"))[0]
        artifact_id = str(artifact["id"])

        await repository.delete_artifact(artifact_id, "usr_a", commit=True)
        assert await repository.list_artifacts("usr_a", conversation_id="cnv_a") == []
        assert (
            await repository.search_artifact_chunks(
                user_id="usr_a",
                query="Alpha",
                privacy_ceiling=1,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_a",
                limit=5,
            )
        ) == []

        purged = await repository.delete_artifact(artifact_id, "usr_a", purge=True, commit=True)
        assert purged is not None
        assert purged["status"] == "purged"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_service_can_store_and_fetch_local_file_payloads(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
        artifact_service = ArtifactService(connection, clock, blob_store=blob_store)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        payload_text = "Local artifact payload with integrity checks."
        bundle = artifact_service.prepare_attachments(
            message_text="Please inspect the attachment.",
            attachments=[
                AttachmentInput(
                    kind="pasted_text",
                    content_text=payload_text,
                    title="Local Note",
                    filename="local-note.txt",
                    mime_type="text/plain",
                )
            ],
            user_id="usr_a",
            conversation={
                "id": "cnv_a",
                "workspace_id": None,
                "assistant_mode_id": "coding_debug",
            },
        )
        assert bundle.artifacts[0].blob is not None
        assert bundle.artifacts[0].blob["storage_kind"] == "local_file"
        assert bundle.artifacts[0].blob["storage_uri"] is None

        await connection.execute("BEGIN")
        try:
            user_message = await messages.create_message(
                message_id=None,
                conversation_id="cnv_a",
                role="user",
                seq=None,
                text=bundle.prompt_text,
                token_count=None,
                metadata={
                    "attachments": bundle.attachments,
                    "artifact_backed": True,
                    "skip_by_default": True,
                    "include_raw": False,
                    "requires_explicit_request": True,
                    "content_kind": "artifact",
                    "context_placeholder": bundle.context_placeholder,
                },
                occurred_at="2026-03-30T12:00:00+00:00",
                commit=False,
            )
            await artifact_service.persist_prepared_attachments(
                bundle=bundle,
                message_id=str(user_message["id"]),
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        artifact = (await repository.list_artifacts("usr_a", conversation_id="cnv_a"))[0]
        blob = await repository.get_artifact_blob(str(artifact["id"]), "usr_a")
        assert blob is not None
        assert blob["storage_kind"] == "local_file"
        assert blob["blob_bytes"] is None
        assert blob["byte_size"] == len(payload_text.encode("utf-8"))
        assert blob["sha256"] == hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        assert blob["storage_uri"] is not None
        assert Path(str(blob["storage_uri"])).is_file()
        assert Path(str(blob["storage_uri"])).resolve().relative_to(blob_store.base_dir)

        blocked_payload = await artifact_service.fetch_artifact_payload(
            user_id="usr_a",
            artifact_id=str(artifact["id"]),
        )
        assert blocked_payload is not None
        assert blocked_payload.raw_available is True
        assert blocked_payload.raw_returned is False

        raw_payload = await artifact_service.fetch_artifact_payload(
            user_id="usr_a",
            artifact_id=str(artifact["id"]),
            include_raw=True,
        )
        assert raw_payload is not None
        assert raw_payload.storage_kind == "local_file"
        assert raw_payload.content_bytes == payload_text.encode("utf-8")
        assert raw_payload.storage_uri is None
    finally:
        await connection.close()


def test_artifact_blob_store_hashes_user_path_segments(tmp_path: Path) -> None:
    blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")

    stored = blob_store.store_bytes(
        user_id="../outside",
        content_bytes=b"payload",
    )

    assert stored.storage_uri is not None
    stored_path = Path(stored.storage_uri).resolve()
    assert stored_path.is_file()
    assert stored_path.relative_to(blob_store.base_dir)
    assert ".." not in stored_path.relative_to(blob_store.base_dir).parts


def test_artifact_blob_store_rejects_symlink_storage_escape(tmp_path: Path) -> None:
    blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
    blob_store.base_dir.mkdir(parents=True)
    outside_payload = tmp_path / "outside.bin"
    outside_payload.write_bytes(b"outside")
    escaping_link = blob_store.base_dir / "escaping.bin"
    escaping_link.symlink_to(outside_payload)

    with pytest.raises(ValueError, match="escapes configured storage directory"):
        blob_store.read_bytes(str(escaping_link))


@pytest.mark.asyncio
async def test_artifact_service_rejects_corrupt_local_file_payload(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
        artifact_service = ArtifactService(connection, clock, blob_store=blob_store)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        original_bytes = b"local artifact bytes"
        stored = blob_store.store_bytes(user_id="usr_a", content_bytes=original_bytes)
        artifact = await repository.create_artifact(
            artifact_id="art_local",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=None,
            artifact_type="file",
            source_kind="host_embedded",
            filename="payload.bin",
            content_hash=stored.sha256,
            size_bytes=stored.byte_size,
            status="ready",
            metadata_json={},
            summary_text="file attachment; artifact_id=art_local",
            index_text="file attachment",
            storage_kind=stored.storage_kind,
            storage_uri=stored.storage_uri,
            blob_byte_size=stored.byte_size,
            blob_sha256=stored.sha256,
        )
        assert artifact["id"] == "art_local"
        Path(str(stored.storage_uri)).write_bytes(b"tampered bytes")

        with pytest.raises(ValueError, match="Artifact blob hash mismatch"):
            await artifact_service.fetch_artifact_payload(
                user_id="usr_a",
                artifact_id="art_local",
                include_raw=True,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_local_file_payload_requires_configured_blob_store(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        stored = blob_store.store_bytes(user_id="usr_a", content_bytes=b"local artifact bytes")
        await repository.create_artifact(
            artifact_id="art_local",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=None,
            artifact_type="file",
            source_kind="host_embedded",
            filename="payload.bin",
            content_hash=stored.sha256,
            size_bytes=stored.byte_size,
            status="ready",
            metadata_json={},
            summary_text="file attachment; artifact_id=art_local",
            index_text="file attachment",
            storage_kind=stored.storage_kind,
            storage_uri=stored.storage_uri,
            blob_byte_size=stored.byte_size,
            blob_sha256=stored.sha256,
        )

        payload = await ArtifactService(connection, clock).fetch_artifact_payload(
            user_id="usr_a",
            artifact_id="art_local",
            include_raw=True,
        )
        assert payload is not None
        assert payload.raw_available is True
        assert payload.raw_returned is False
        assert payload.raw_block_reason == "local_file_store_unavailable"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_service_rejects_invalid_base64_payload() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        artifact_service = ArtifactService(connection, clock)
        with pytest.raises(ValueError, match="Invalid base64 attachment payload"):
            artifact_service.prepare_attachments(
                message_text="Please inspect this upload.",
                attachments=[
                    AttachmentInput(
                        kind="base64",
                        content_base64="this-is-not-valid-base64!",
                        filename="invalid.bin",
                    )
                ],
                user_id="usr_a",
                conversation={
                    "id": "cnv_a",
                    "workspace_id": None,
                    "assistant_mode_id": "coding_debug",
                },
            )
    finally:
        await connection.close()
