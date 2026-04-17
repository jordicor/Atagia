"""Tests for artifact chunk retrieval candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.models.schemas_api import AttachmentInput
from atagia.models.schemas_memory import MemoryScope, MemoryStatus, PlannedSubQuery, RetrievalPlan
from atagia.services.artifact_service import ArtifactService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _artifact_plan(
    *,
    conversation_id: str = "cnv_a",
    scope_filter: list[MemoryScope] | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="theta",
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id=conversation_id,
        fts_queries=["theta"],
        sub_query_plans=[PlannedSubQuery(text="theta", fts_queries=["theta"])],
        raw_context_access_mode="artifact",
        query_type="default",
        scope_filter=scope_filter or [MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=5,
        max_context_items=5,
        privacy_ceiling=1,
    )


@pytest.mark.asyncio
async def test_candidate_search_surfaces_artifact_chunks_in_artifact_mode() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        artifacts = ArtifactService(connection, clock)
        search = CandidateSearch(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")

        bundle = artifacts.prepare_attachments(
            message_text="Please inspect this note.",
            attachments=[
                AttachmentInput(
                    kind="pasted_text",
                    content_text="alpha beta theta delta epsilon",
                    title="Artifact Note",
                    filename="note.txt",
                    mime_type="text/plain",
                    preserve_verbatim=True,
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
                    "attachment_count": len(bundle.artifacts),
                    "attachment_artifact_ids": [
                        str(prepared.artifact["id"]) for prepared in bundle.artifacts
                    ],
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
            await artifacts.persist_prepared_attachments(
                bundle=bundle,
                message_id=str(user_message["id"]),
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        candidates = await search.search(_artifact_plan(), "usr_a")
        assert candidates
        top_candidate = candidates[0]
        assert top_candidate["is_artifact_chunk"] is True
        assert "theta" in str(top_candidate["canonical_text"]).lower()
        assert top_candidate["scope"] == MemoryScope.CONVERSATION.value

        repository = ArtifactRepository(connection, clock)
        artifact_id = str(top_candidate["artifact_id"])
        await repository.delete_artifact(artifact_id, "usr_a", purge=True, commit=True)

        remaining_candidates = await search.search(_artifact_plan(), "usr_a")
        assert not any(candidate.get("is_artifact_chunk") for candidate in remaining_candidates)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_chunks_do_not_leak_from_conversation_scope_to_global_scope() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        artifacts = ArtifactService(connection, clock)
        search = CandidateSearch(connection, clock)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")
        await conversations.create_conversation("cnv_b", "usr_a", None, "coding_debug", "Daily Chat")

        bundle = artifacts.prepare_attachments(
            message_text="Please remember this attachment exists.",
            attachments=[
                AttachmentInput(
                    kind="pasted_text",
                    content_text="theta appears only in the cnv_a attachment",
                    title="Conversation Scoped Note",
                    filename="scoped-note.txt",
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
                    "attachment_count": len(bundle.artifacts),
                    "attachment_artifact_ids": [
                        str(prepared.artifact["id"]) for prepared in bundle.artifacts
                    ],
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
            await artifacts.persist_prepared_attachments(
                bundle=bundle,
                message_id=str(user_message["id"]),
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        conversation_candidates = await search.search(_artifact_plan(), "usr_a")
        assert any(candidate.get("is_artifact_chunk") for candidate in conversation_candidates)

        global_candidates = await search.search(
            _artifact_plan(
                conversation_id="cnv_b",
                scope_filter=[MemoryScope.GLOBAL_USER],
            ),
            "usr_a",
        )
        assert not any(candidate.get("is_artifact_chunk") for candidate in global_candidates)
    finally:
        await connection.close()
