"""Tests for artifact persistence, chunking, and lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.models.schemas_api import AttachmentInput
from atagia.models.schemas_memory import MemoryScope
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
