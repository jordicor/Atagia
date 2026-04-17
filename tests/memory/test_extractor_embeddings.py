"""Tests for embedding integration in the memory extractor."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import ExtractionConversationContext, MemoryStatus
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class ExtractionProvider(LLMProvider):
    name = "extractor-embeddings"

    def __init__(self, payload: dict[str, object]) -> None:
        normalized_payload = dict(payload)
        normalized_payload["evidences"] = [
            (
                {
                    **item,
                    "language_codes": ["en"],
                }
                if isinstance(item, dict) and "canonical_text" in item and "language_codes" not in item
                else item
            )
            for item in normalized_payload.get("evidences", [])
        ]
        self.payload = normalized_payload

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"is_explicit": True, "reasoning": "ok"}),
            )
        if request.metadata.get("purpose") == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"equivalent": True}),
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Extractor embedding tests use a fake embedding backend")


class TrackingEmbeddingIndex:
    vector_limit = 1

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        self.calls.append({"memory_id": memory_id, "text": text, "metadata": dict(metadata)})

    async def search(self, query: str, user_id: str, top_k: int):
        raise AssertionError("search() is not used in extractor embedding tests")

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used in extractor embedding tests")


class TrackingNoneBackend(NoneBackend):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        self.called = True


class FailingEmbeddingIndex(TrackingEmbeddingIndex):
    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        raise RuntimeError("embedding failed")


class VisibilityEmbeddingIndex(TrackingEmbeddingIndex):
    def __init__(self, observer_connection) -> None:
        super().__init__()
        self._observer_connection = observer_connection

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        cursor = await self._observer_connection.execute(
            "SELECT id FROM memory_objects WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        self.calls.append(
            {
                "memory_id": memory_id,
                "text": text,
                "metadata": dict(metadata),
                "visible": row is not None,
            }
        )


async def _build_runtime(embedding_index: object, *, database_path: str = ":memory:"):
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 10, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    provider = ExtractionProvider(
        {
            "evidences": [
                {
                    "canonical_text": "I prefer concise debugging answers",
                    "index_text": "This preference was stated during incident debugging about websocket retries.",
                    "scope": "assistant_mode",
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 1,
                    "payload": {"kind": "preference"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    extractor = MemoryExtractor(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=InProcessBackend(),
        embedding_index=embedding_index,
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    return connection, messages, memories, extractor, resolved_policy


def _context(message_id: str) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id=message_id,
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


@pytest.mark.asyncio
async def test_extraction_with_embedding_index_upserts_after_persist() -> None:
    embedding_index = TrackingEmbeddingIndex()
    connection, messages, memories, extractor, resolved_policy = await _build_runtime(embedding_index)
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "I prefer concise debugging answers.",
            6,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=resolved_policy,
        )

        stored = await memories.list_for_user("usr_1")
        assert len(stored) == 1
        assert len(embedding_index.calls) == 1
        assert embedding_index.calls[0]["memory_id"] == stored[0]["id"]
        assert embedding_index.calls[0]["metadata"]["index_text"] == (
            "This preference was stated during incident debugging about websocket retries."
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_extraction_with_none_backend_skips_upsert_calls() -> None:
    embedding_index = TrackingNoneBackend()
    connection, messages, memories, extractor, resolved_policy = await _build_runtime(embedding_index)
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "I prefer concise debugging answers.",
            6,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=resolved_policy,
        )

        assert len(await memories.list_for_user("usr_1")) == 1
        assert embedding_index.called is False
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_embedding_upsert_runs_after_canonical_commit(tmp_path: Path) -> None:
    database_path = str(tmp_path / "atagia-extractor-embeddings.db")
    observer_connection = await open_connection(database_path)
    embedding_index = VisibilityEmbeddingIndex(observer_connection)
    connection, messages, _memories, extractor, resolved_policy = await _build_runtime(
        embedding_index,
        database_path=database_path,
    )
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "I prefer concise debugging answers.",
            6,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="user",
            conversation_context=_context(str(message["id"])),
            resolved_policy=resolved_policy,
        )

        assert len(embedding_index.calls) == 1
        assert embedding_index.calls[0]["visible"] is True
    finally:
        await connection.close()
        await observer_connection.close()


@pytest.mark.asyncio
async def test_embedding_upsert_failure_still_persists_memory(caplog: pytest.LogCaptureFixture) -> None:
    connection, messages, memories, extractor, resolved_policy = await _build_runtime(FailingEmbeddingIndex())
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "I prefer concise debugging answers.",
            6,
            {},
        )

        with caplog.at_level(logging.WARNING):
            await extractor.extract(
                message_text=str(message["text"]),
                role="user",
                conversation_context=_context(str(message["id"])),
                resolved_policy=resolved_policy,
            )

        assert len(await memories.list_for_user("usr_1")) == 1
        assert "Embedding upsert failed" in caplog.text
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_protected_verbatim_memories_embed_only_index_text() -> None:
    embedding_index = TrackingEmbeddingIndex()
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 10, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "personal_assistant", "Chat")
    provider = ExtractionProvider(
        {
            "evidences": [
                {
                    "canonical_text": "Banking card PIN: 4512",
                    "index_text": "User's banking card credential",
                    "scope": "global_user",
                    "confidence": 0.95,
                    "source_kind": "extracted",
                    "privacy_level": 3,
                    "memory_category": "pin_or_password",
                    "preserve_verbatim": True,
                    "payload": {},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    extractor = MemoryExtractor(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=InProcessBackend(),
        embedding_index=embedding_index,
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["personal_assistant"]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "My banking card PIN is 4512.",
            6,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="user",
            conversation_context=ExtractionConversationContext(
                user_id="usr_1",
                conversation_id="cnv_1",
                source_message_id="msg_1",
                workspace_id=None,
                assistant_mode_id="personal_assistant",
                recent_messages=[],
            ),
            resolved_policy=resolved_policy,
        )

        persisted = await memories.list_for_user("usr_1", statuses=None)
        assert len(persisted) == 1
        assert persisted[0]["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert len(embedding_index.calls) == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_review_required_memories_do_not_create_embeddings() -> None:
    embedding_index = TrackingEmbeddingIndex()
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 10, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    provider = ExtractionProvider(
        {
            "evidences": [
                {
                    "canonical_text": "Maybe concise replies would help.",
                    "index_text": "Sensitive preference candidate for debugging.",
                    "scope": "assistant_mode",
                    "confidence": 0.9,
                    "source_kind": "extracted",
                    "privacy_level": 2,
                    "payload": {},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    extractor = MemoryExtractor(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=InProcessBackend(),
        embedding_index=embedding_index,
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "assistant",
            1,
            "Maybe concise replies would help.",
            6,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="assistant",
            conversation_context=_context(str(message["id"])),
            resolved_policy=resolved_policy,
        )

        persisted = await memories.list_for_user("usr_1", statuses=None)
        assert len(persisted) == 1
        assert persisted[0]["status"] == MemoryStatus.REVIEW_REQUIRED.value
        assert embedding_index.calls == []
    finally:
        await connection.close()
