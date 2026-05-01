"""Tests for artifact chunk retrieval candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.models.schemas_api import AttachmentInput
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)
from atagia.services.artifact_service import ArtifactService
from atagia.services.embeddings import EmbeddingMatch

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class FakeEmbeddingIndex:
    vector_limit = 1

    def __init__(self, matches: list[EmbeddingMatch]) -> None:
        self.matches = list(matches)

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise AssertionError("upsert() is not used in candidate search tests")

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        return list(self.matches[:top_k])

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used in candidate search tests")


def _artifact_plan(
    *,
    conversation_id: str = "cnv_a",
    scope_filter: list[MemoryScope] | None = None,
    fts_queries: list[str] | None = None,
    exact_recall_mode: bool = False,
    original_query: str = "theta",
    vector_limit: int = 0,
    privacy_ceiling: int = 1,
    allow_intimacy_context: bool = False,
) -> RetrievalPlan:
    resolved_fts_queries = fts_queries or ["theta"]
    return RetrievalPlan(
        original_query=original_query,
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id=conversation_id,
        fts_queries=resolved_fts_queries,
        sub_query_plans=[PlannedSubQuery(text="theta", fts_queries=resolved_fts_queries)],
        raw_context_access_mode="artifact",
        query_type="default",
        scope_filter=scope_filter or [MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=vector_limit,
        max_candidates=5,
        max_context_items=5,
        privacy_ceiling=privacy_ceiling,
        allow_intimacy_context=allow_intimacy_context,
        exact_recall_mode=exact_recall_mode,
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
async def test_candidate_search_filters_intimacy_bound_artifacts_until_authorized() -> None:
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
                    content_text="theta appears in a private continuity note",
                    title="Private Note",
                    mime_type="text/plain",
                    privacy_level=0,
                    intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
                    intimacy_boundary_confidence=0.88,
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
        assert bundle.attachments[0]["privacy_level"] == 2
        assert bundle.attachments[0]["intimacy_boundary"] == "romantic_private"
        assert "theta appears" not in bundle.attachments[0]["index_text"]

        await connection.execute("BEGIN")
        try:
            user_message = await messages.create_message(
                message_id=None,
                conversation_id="cnv_a",
                role="user",
                seq=None,
                text=bundle.prompt_text,
                token_count=None,
                metadata={"attachments": bundle.attachments, "artifact_backed": True},
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

        ordinary_candidates = await search.search(
            _artifact_plan(
                fts_queries=["theta"],
                privacy_ceiling=3,
                allow_intimacy_context=False,
            ),
            "usr_a",
        )
        authorized_candidates = await search.search(
            _artifact_plan(
                fts_queries=["theta"],
                privacy_ceiling=3,
                allow_intimacy_context=True,
            ),
            "usr_a",
        )

        assert not any(candidate.get("is_artifact_chunk") for candidate in ordinary_candidates)
        artifact_candidates = [
            candidate for candidate in authorized_candidates if candidate.get("is_artifact_chunk")
        ]
        assert artifact_candidates
        assert artifact_candidates[0]["intimacy_boundary"] == "romantic_private"
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


@pytest.mark.asyncio
async def test_exact_recall_artifact_search_falls_back_to_broader_fts_queries() -> None:
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
            message_text="The kids made this with clay.",
            attachments=[
                AttachmentInput(
                    kind="image",
                    content_text=(
                        "Visual description of attached image: a cup with a dog face on it\n"
                        "Associated message speaker: Melanie\n"
                        "Associated message text: The kids made this with clay."
                    ),
                    title="Pottery image",
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
                role="assistant",
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

        candidates = await search.search(
            _artifact_plan(
                fts_queries=[
                    "mel clay pot kids",
                    "mel clay pot",
                    "mel OR clay OR pot OR kids",
                ],
                exact_recall_mode=True,
            ),
            "usr_a",
        )

        assert any(
            candidate.get("is_artifact_chunk")
            and "dog face" in str(candidate.get("canonical_text", "")).lower()
            for candidate in candidates
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_search_includes_chunks_linked_to_retrieved_source_messages() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
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
            message_text="We painted something together.",
            attachments=[
                AttachmentInput(
                    kind="image",
                    content_text="Visual description of attached image: a sunset with a palm tree",
                    title="Painting image",
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
                role="assistant",
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

        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Melanie's latest project was an artwork.",
            payload={"source_message_ids": [str(user_message["id"])]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _artifact_plan(
                fts_queries=["latest project"],
                exact_recall_mode=True,
            ),
            "usr_a",
        )

        assert any(
            candidate.get("is_artifact_chunk")
            and "palm tree" in str(candidate.get("canonical_text", "")).lower()
            for candidate in candidates
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_search_includes_chunks_linked_to_embedding_candidates() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    try:
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        artifacts = ArtifactRepository(connection, clock)
        embedding_index = FakeEmbeddingIndex([EmbeddingMatch(memory_id="mem_semantic", score=0.92)])
        search = CandidateSearch(connection, clock, embedding_index=embedding_index)

        await users.create_user("usr_a")
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES ('coding_debug', 'Coding Debug', 'hash_1', '{}', '2026-03-30T12:00:00+00:00', '2026-03-30T12:00:00+00:00')
            """
        )
        await conversations.create_conversation("cnv_a", "usr_a", None, "coding_debug", "Artifact Chat")
        user_message = await messages.create_message(
            message_id="msg_semantic",
            conversation_id="cnv_a",
            role="user",
            seq=None,
            text="Here is the promotion clip.",
            token_count=None,
            metadata={
                "artifact_backed": True,
                "skip_by_default": True,
                "include_raw": False,
                "requires_explicit_request": True,
                "content_kind": "artifact",
            },
            occurred_at="2026-03-30T12:00:00+00:00",
        )
        artifact = await artifacts.create_artifact(
            artifact_id="art_semantic",
            user_id="usr_a",
            workspace_id=None,
            conversation_id="cnv_a",
            message_id=str(user_message["id"]),
            artifact_type="image",
            source_kind="url",
            title="Promotion clip frame",
            privacy_level=0,
            preserve_verbatim=True,
            summary_text="Image attachment",
            index_text="image attachment",
        )
        await artifacts.create_artifact_chunk(
            artifact_id=str(artifact["id"]),
            user_id="usr_a",
            chunk_index=0,
            text="Visual description: a red hoodie appears in a short video frame.",
            token_count=12,
            kind="ocr",
            chunk_id="arc_semantic",
        )
        await memories.create_memory_object(
            user_id="usr_a",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_a",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="A promotion clip was shared.",
            payload={"source_message_ids": [str(user_message["id"])]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_semantic",
        )

        candidates = await search.search(
            _artifact_plan(
                fts_queries=["unrelated lexical query"],
                exact_recall_mode=True,
                vector_limit=1,
            ),
            "usr_a",
        )

        assert any(candidate["id"] == "mem_semantic" for candidate in candidates)
        assert any(
            candidate.get("is_artifact_chunk")
            and "red hoodie" in str(candidate.get("canonical_text", "")).lower()
            for candidate in candidates
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_artifact_search_tries_original_query_rewrites_after_sparse_queries() -> None:
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
            message_text="We painted something together.",
            attachments=[
                AttachmentInput(
                    kind="image",
                    content_text=(
                        "Visual description of attached image: a sunset with a palm tree\n"
                        "Associated message text: Here's our latest work from last weekend."
                    ),
                    title="Painting image",
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
                role="assistant",
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

        candidates = await search.search(
            _artifact_plan(
                fts_queries=["mel kids paint project"],
                exact_recall_mode=True,
                original_query="latest work",
            ),
            "usr_a",
        )

        assert any(
            candidate.get("is_artifact_chunk")
            and "palm tree" in str(candidate.get("canonical_text", "")).lower()
            for candidate in candidates
        )
    finally:
        await connection.close()
