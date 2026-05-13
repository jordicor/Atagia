"""Tests for artifact persistence, chunking, and lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.artifact_payload_repository import ArtifactPayloadRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.models.schemas_api import AttachmentInput
from atagia.models.schemas_memory import IntimacyBoundary, MemoryScope
from atagia.services.artifact_blob_store import ArtifactBlobStore
from atagia.services.artifact_service import ArtifactService
from atagia.services.lifecycle_service import ConversationLifecycleService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class _ArtifactBlobSettings:
    def __init__(self, artifact_blobs_dir: Path) -> None:
        self._artifact_blobs_dir = artifact_blobs_dir

    def artifact_blobs_dir(self) -> Path:
        return self._artifact_blobs_dir


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
        assert "Alpha beta gamma" not in bundle.prompt_text
        assert bundle.context_placeholder is not None
        assert "Alpha beta gamma" not in bundle.context_placeholder
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
        assert "Alpha beta gamma" not in chunks[0]["text"]
        assert any("Alpha beta gamma" in chunk["text"] for chunk in chunks[1:])

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
        assert "Alpha beta gamma" not in bundle.prompt_text
        assert bundle.context_placeholder is not None
        assert "Alpha beta gamma" not in bundle.context_placeholder
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
async def test_artifact_chunk_search_filters_intimacy_boundary_until_authorized() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        ordinary = await repository.create_artifact(
            artifact_id="art_ordinary",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="pasted_text",
            status="ready",
            privacy_level=0,
            summary_text="ordinary pottery note",
            index_text="ordinary pottery note",
        )
        restricted = await repository.create_artifact(
            artifact_id="art_private",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="pasted_text",
            status="ready",
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
            summary_text="private pottery note",
            index_text="private pottery note",
        )
        await repository.create_artifact_chunk(
            artifact_id=str(ordinary["id"]),
            user_id="usr_a",
            chunk_index=0,
            text="ordinary pottery checklist",
            token_count=3,
            kind="summary",
        )
        await repository.create_artifact_chunk(
            artifact_id=str(restricted["id"]),
            user_id="usr_a",
            chunk_index=0,
            text="private pottery continuity note",
            token_count=4,
            kind="summary",
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
        )

        ordinary_rows = await repository.search_artifact_chunks(
            user_id="usr_a",
            query="pottery",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            limit=10,
        )
        authorized_rows = await repository.search_artifact_chunks(
            user_id="usr_a",
            query="pottery",
            privacy_ceiling=3,
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_a",
            limit=10,
            allow_intimacy_context=True,
        )

        assert {row["artifact_id"] for row in ordinary_rows} == {"art_ordinary"}
        assert {row["artifact_id"] for row in authorized_rows} == {"art_ordinary", "art_private"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunk_search_keeps_incognito_artifacts_chat_local() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation(
            "cnv_incognito",
            "usr_a",
            None,
            "coding_debug",
            "Incognito",
            user_persona_id="persona_a",
            platform_id="default",
            incognito=True,
        )
        await conversations.create_conversation(
            "cnv_visible",
            "usr_a",
            None,
            "coding_debug",
            "Visible",
            user_persona_id="persona_a",
            platform_id="default",
        )

        artifact = await repository.create_artifact(
            artifact_id="art_incognito",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_incognito",
            message_id=None,
            artifact_type="pasted_text",
            source_kind="pasted_text",
            status="ready",
            privacy_level=0,
            summary_text="incognito artifact",
            index_text="incognito artifact",
            user_persona_id="persona_a",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
            incognito_snapshot=True,
        )
        await repository.create_artifact_chunk(
            artifact_id=str(artifact["id"]),
            user_id="usr_a",
            chunk_index=0,
            text="incognito artifact local note",
            token_count=4,
            kind="summary",
        )

        broad_rows = await repository.search_artifact_chunks(
            user_id="usr_a",
            query="incognito",
            privacy_ceiling=1,
            scope_filter=[MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_visible",
            limit=10,
            user_persona_id="persona_a",
            platform_id="default",
        )
        local_rows = await repository.search_artifact_chunks(
            user_id="usr_a",
            query="incognito",
            privacy_ceiling=1,
            scope_filter=[MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id="cnv_incognito",
            limit=10,
            user_persona_id="persona_a",
            platform_id="default",
            incognito=True,
        )

        assert broad_rows == []
        assert [row["artifact_id"] for row in local_rows] == ["art_incognito"]
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
        assert artifact["payload_blob_id"] is not None
        blob = await repository.get_artifact_blob(str(artifact["id"]), "usr_a")
        assert blob is not None
        assert blob["payload_blob_id"] == artifact["payload_blob_id"]
        assert blob["storage_kind"] == "local_file"
        assert blob["blob_bytes"] is None
        assert blob["byte_size"] == len(payload_text.encode("utf-8"))
        assert blob["sha256"] == hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        assert blob["storage_uri"] is not None
        stored_path = blob_store.path_for_storage_uri(str(blob["storage_uri"]))
        assert stored_path.is_file()
        assert stored_path.relative_to(blob_store.base_dir)
        legacy_count_cursor = await connection.execute("SELECT COUNT(*) AS count FROM artifact_blobs")
        legacy_count = await legacy_count_cursor.fetchone()
        assert legacy_count["count"] == 0

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
        assert raw_payload.storage_uri == blob["storage_uri"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_payload_dedupe_keeps_storage_modes_independently_readable(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        local_blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
        local_service = ArtifactService(connection, clock, blob_store=local_blob_store)
        sqlite_service = ArtifactService(connection, clock)
        repository = ArtifactRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_local", "usr_a", None, "coding_debug", "Local Artifact Chat")
        await conversations.create_conversation("cnv_sqlite", "usr_a", None, "coding_debug", "SQLite Artifact Chat")

        async def persist(service: ArtifactService, conversation_id: str) -> dict[str, object]:
            bundle = service.prepare_attachments(
                message_text="Please inspect this attachment.",
                attachments=[
                    AttachmentInput(
                        kind="pasted_text",
                        content_text="Payload that exists across storage modes.",
                        title="Mode Note",
                        filename="mode-note.txt",
                        mime_type="text/plain",
                    )
                ],
                user_id="usr_a",
                conversation={
                    "id": conversation_id,
                    "workspace_id": None,
                    "assistant_mode_id": "coding_debug",
                },
            )
            await connection.execute("BEGIN")
            try:
                user_message = await messages.create_message(
                    message_id=None,
                    conversation_id=conversation_id,
                    role="user",
                    seq=None,
                    text=bundle.prompt_text,
                    token_count=None,
                    metadata=bundle.message_metadata(),
                    occurred_at="2026-03-30T12:00:00+00:00",
                    commit=False,
                )
                await service.persist_prepared_attachments(
                    bundle=bundle,
                    message_id=str(user_message["id"]),
                    commit=False,
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
            return (await repository.list_artifacts("usr_a", conversation_id=conversation_id))[0]

        local_artifact = await persist(local_service, "cnv_local")
        sqlite_artifact = await persist(sqlite_service, "cnv_sqlite")
        assert local_artifact["payload_blob_id"] != sqlite_artifact["payload_blob_id"]

        payload_rows_cursor = await connection.execute(
            """
            SELECT storage_kind, COUNT(*) AS count
            FROM artifact_payload_blobs
            WHERE user_id = 'usr_a'
            GROUP BY storage_kind
            """
        )
        payload_counts = {row["storage_kind"]: row["count"] for row in await payload_rows_cursor.fetchall()}
        assert payload_counts == {"local_file": 1, "sqlite_blob": 1}

        sqlite_payload = await sqlite_service.fetch_artifact_payload(
            user_id="usr_a",
            artifact_id=str(sqlite_artifact["id"]),
            include_raw=True,
        )
        assert sqlite_payload is not None
        assert sqlite_payload.storage_kind == "sqlite_blob"
        assert sqlite_payload.raw_returned is True
        assert sqlite_payload.content_bytes == b"Payload that exists across storage modes."
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_payload_blobs_dedupe_and_lifecycle_keeps_shared_local_files(tmp_path: Path) -> None:
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
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat A")
        await conversations.create_conversation("cnv_b", "usr_a", None, "coding_debug", "Artifact Chat B")

        async def persist_attachment(conversation_id: str) -> dict[str, object]:
            bundle = artifact_service.prepare_attachments(
                message_text="Please inspect this attachment.",
                attachments=[
                    AttachmentInput(
                        kind="pasted_text",
                        content_text="Shared local artifact payload.",
                        title="Shared Note",
                        filename="shared-note.txt",
                        mime_type="text/plain",
                    )
                ],
                user_id="usr_a",
                conversation={
                    "id": conversation_id,
                    "workspace_id": None,
                    "assistant_mode_id": "coding_debug",
                },
            )
            await connection.execute("BEGIN")
            try:
                user_message = await messages.create_message(
                    message_id=None,
                    conversation_id=conversation_id,
                    role="user",
                    seq=None,
                    text=bundle.prompt_text,
                    token_count=None,
                    metadata=bundle.message_metadata(),
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
            return (await repository.list_artifacts("usr_a", conversation_id=conversation_id))[0]

        first_artifact = await persist_attachment("cnv_a")
        second_artifact = await persist_attachment("cnv_b")
        assert first_artifact["payload_blob_id"] == second_artifact["payload_blob_id"]
        payload_count_cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM artifact_payload_blobs WHERE user_id = 'usr_a'"
        )
        payload_count = await payload_count_cursor.fetchone()
        assert payload_count["count"] == 1

        blob = await repository.get_artifact_blob(str(first_artifact["id"]), "usr_a")
        assert blob is not None
        storage_uri = str(blob["storage_uri"])
        blob_path = blob_store.path_for_storage_uri(storage_uri)
        assert blob_path.is_file()

        lifecycle = ConversationLifecycleService(
            SimpleNamespace(
                settings=_ArtifactBlobSettings(blob_store.base_dir),
                clock=clock,
                artifact_blob_store=blob_store,
            )
        )
        await connection.execute("BEGIN IMMEDIATE")
        await lifecycle._queue_file_deletions_for_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=[str(first_artifact["id"])],
            tombstone_id="tmb_first",
            reason="conversation_delete",
            timestamp=clock.now().isoformat(),
        )
        first_pending_cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM pending_file_deletions WHERE tombstone_id = 'tmb_first'"
        )
        first_pending = await first_pending_cursor.fetchone()
        assert first_pending["count"] == 0
        await lifecycle._delete_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=[str(first_artifact["id"])],
        )
        await connection.commit()
        assert blob_path.is_file()

        await connection.execute("BEGIN IMMEDIATE")
        await lifecycle._queue_file_deletions_for_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=[str(second_artifact["id"])],
            tombstone_id="tmb_second",
            reason="conversation_delete",
            timestamp=clock.now().isoformat(),
        )
        await lifecycle._delete_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=[str(second_artifact["id"])],
        )
        await connection.commit()
        second_pending_cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM pending_file_deletions WHERE tombstone_id = 'tmb_second'"
        )
        second_pending = await second_pending_cursor.fetchone()
        assert second_pending["count"] == 1
        assert await lifecycle._process_pending_file_deletions(connection, tombstone_id="tmb_second") == 1
        assert not blob_path.exists()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_lifecycle_queues_quarantined_local_payload_files(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")
        repository = ArtifactRepository(connection, clock)
        payload_repository = ArtifactPayloadRepository(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        stored = blob_store.store_bytes(user_id="usr_a", content_bytes=b"quarantined payload bytes")
        payload = await payload_repository.create_payload_blob(
            user_id="usr_a",
            storage_kind="local_file",
            identity_kind="content_sha256",
            content_sha256=stored.sha256,
            byte_size=stored.byte_size,
            blob_bytes=None,
            storage_key=stored.storage_uri,
            external_uri=None,
            status="quarantined",
        )
        artifact = await repository.create_artifact(
            artifact_id="art_quarantined",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=None,
            artifact_type="file",
            source_kind="host_embedded",
            filename="quarantined.bin",
            content_hash=stored.sha256,
            size_bytes=stored.byte_size,
            status="ready",
            metadata_json={},
            summary_text="file attachment; artifact_id=art_quarantined",
            index_text="file attachment",
            payload_blob_id=str(payload["id"]),
        )
        assert artifact["payload_blob_id"] == payload["id"]
        assert await repository.get_artifact_blob("art_quarantined", "usr_a") is None
        quarantined_payload = await ArtifactService(connection, clock, blob_store=blob_store).fetch_artifact_payload(
            user_id="usr_a",
            artifact_id="art_quarantined",
            include_raw=True,
        )
        assert quarantined_payload is not None
        assert quarantined_payload.raw_available is False
        assert quarantined_payload.raw_returned is False
        assert quarantined_payload.content_bytes is None
        blob_path = blob_store.path_for_storage_uri(str(stored.storage_uri))
        assert blob_path.is_file()

        lifecycle = ConversationLifecycleService(
            SimpleNamespace(
                settings=_ArtifactBlobSettings(blob_store.base_dir),
                clock=clock,
                artifact_blob_store=blob_store,
            )
        )
        await connection.execute("BEGIN IMMEDIATE")
        await lifecycle._queue_file_deletions_for_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=["art_quarantined"],
            tombstone_id="tmb_quarantined",
            reason="user_erasure",
            timestamp=clock.now().isoformat(),
        )
        await lifecycle._delete_artifacts(
            connection,
            user_id="usr_a",
            artifact_ids=["art_quarantined"],
        )
        await connection.commit()

        pending_cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM pending_file_deletions WHERE tombstone_id = 'tmb_quarantined'"
        )
        pending = await pending_cursor.fetchone()
        assert pending["count"] == 1
        assert await lifecycle._process_pending_file_deletions(connection, tombstone_id="tmb_quarantined") == 1
        assert not blob_path.exists()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pending_file_deletion_uses_queued_storage_root(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        old_store = ArtifactBlobStore(tmp_path / "old_artifact_blobs")
        current_store = ArtifactBlobStore(tmp_path / "current_artifact_blobs")
        old_stored = old_store.store_bytes(user_id="usr_a", content_bytes=b"root-sensitive payload")
        current_stored = current_store.store_bytes(user_id="usr_a", content_bytes=b"root-sensitive payload")
        assert old_stored.storage_uri == current_stored.storage_uri
        old_path = old_store.path_for_storage_uri(str(old_stored.storage_uri))
        current_path = current_store.path_for_storage_uri(str(current_stored.storage_uri))
        assert old_path.is_file()
        assert current_path.is_file()

        await connection.execute(
            """
            INSERT INTO pending_file_deletions(
                id,
                storage_uri,
                storage_root,
                sha256,
                reason,
                tombstone_id,
                created_at
            )
            VALUES (
                'pfd_rooted',
                ?,
                ?,
                ?,
                'user_erasure',
                'tmb_rooted',
                '2026-03-30T12:00:00+00:00'
            )
            """,
            (old_stored.storage_uri, str(old_store.base_dir), old_stored.sha256),
        )
        await connection.commit()

        lifecycle = ConversationLifecycleService(
            SimpleNamespace(
                settings=_ArtifactBlobSettings(current_store.base_dir),
                clock=clock,
                artifact_blob_store=current_store,
            )
        )
        assert await lifecycle._process_pending_file_deletions(connection, tombstone_id="tmb_rooted") == 1
        assert not old_path.exists()
        assert current_path.is_file()
        row_cursor = await connection.execute(
            "SELECT deleted_at, last_error FROM pending_file_deletions WHERE id = 'pfd_rooted'"
        )
        row = await row_cursor.fetchone()
        assert row["deleted_at"] is not None
        assert row["last_error"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pending_file_deletion_old_root_not_blocked_by_live_same_relative_key(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        repository = ArtifactRepository(connection, clock)
        payload_repository = ArtifactPayloadRepository(connection, clock)
        old_store = ArtifactBlobStore(tmp_path / "old_artifact_blobs")
        current_store = ArtifactBlobStore(tmp_path / "current_artifact_blobs")

        old_stored = old_store.store_bytes(user_id="usr_a", content_bytes=b"root-sensitive payload")
        current_stored = current_store.store_bytes(user_id="usr_a", content_bytes=b"root-sensitive payload")
        assert old_stored.storage_uri == current_stored.storage_uri
        old_path = old_store.path_for_storage_uri(str(old_stored.storage_uri))
        current_path = current_store.path_for_storage_uri(str(current_stored.storage_uri))

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_current", "usr_a", None, "coding_debug", "Current Artifact Chat")
        payload = await payload_repository.create_payload_blob(
            user_id="usr_a",
            storage_kind="local_file",
            identity_kind="content_sha256",
            content_sha256=current_stored.sha256,
            byte_size=current_stored.byte_size,
            blob_bytes=None,
            storage_key=current_stored.storage_uri,
            external_uri=None,
        )
        await repository.create_artifact(
            artifact_id="art_current",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_current",
            message_id=None,
            artifact_type="file",
            source_kind="host_embedded",
            filename="current.bin",
            content_hash=current_stored.sha256,
            size_bytes=current_stored.byte_size,
            status="ready",
            metadata_json={},
            summary_text="file attachment; artifact_id=art_current",
            index_text="file attachment",
            payload_blob_id=str(payload["id"]),
        )
        await connection.execute(
            """
            INSERT INTO pending_file_deletions(
                id,
                storage_uri,
                storage_root,
                sha256,
                reason,
                tombstone_id,
                created_at
            )
            VALUES (
                'pfd_old_root',
                ?,
                ?,
                ?,
                'user_erasure',
                'tmb_old_root',
                '2026-03-30T12:00:00+00:00'
            )
            """,
            (old_stored.storage_uri, str(old_store.base_dir), old_stored.sha256),
        )
        await connection.commit()

        lifecycle = ConversationLifecycleService(
            SimpleNamespace(
                settings=_ArtifactBlobSettings(current_store.base_dir),
                clock=clock,
                artifact_blob_store=current_store,
            )
        )
        assert await lifecycle._process_pending_file_deletions(connection, tombstone_id="tmb_old_root") == 1
        assert not old_path.exists()
        assert current_path.is_file()
        row_cursor = await connection.execute(
            "SELECT deleted_at, last_error FROM pending_file_deletions WHERE id = 'pfd_old_root'"
        )
        row = await row_cursor.fetchone()
        assert row["deleted_at"] is not None
        assert row["last_error"] is None
    finally:
        await connection.close()


def test_artifact_blob_store_hashes_user_path_segments(tmp_path: Path) -> None:
    blob_store = ArtifactBlobStore(tmp_path / "artifact_blobs")

    stored = blob_store.store_bytes(
        user_id="../outside",
        content_bytes=b"payload",
    )

    assert stored.storage_uri is not None
    stored_path = blob_store.path_for_storage_uri(stored.storage_uri)
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
        assert stored.storage_uri is not None
        blob_store.path_for_storage_uri(stored.storage_uri).write_bytes(b"tampered bytes")

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
