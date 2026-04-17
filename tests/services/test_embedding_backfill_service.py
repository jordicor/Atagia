"""Tests for the embedding backfill service."""

from __future__ import annotations

from importlib.util import find_spec
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus
from atagia.services.embedding_backfill_service import EmbeddingBackfillService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMEmbeddingVector,
    LLMProvider,
)
from atagia.services.sqlite_vec_backend import SQLiteVecBackend

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
HAS_SQLITE_VEC = find_spec("sqlite_vec") is not None


class RecordingEmbeddingProvider(LLMProvider):
    name = "embedding-backfill-tests"
    supports_embedding_dimensions = True

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = dict(vectors_by_text)
        self.requests: list[LLMEmbeddingRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError("Completions are not used in embedding backfill tests")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        self.requests.append(request)
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[
                LLMEmbeddingVector(index=index, values=self._vectors_by_text[text])
                for index, text in enumerate(request.input_texts)
            ],
        )


def _settings(database_path: str) -> Settings:
    return Settings(
        sqlite_path=database_path,
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        embedding_backend="sqlite_vec",
        embedding_model="text-embedding-3-small",
        embedding_dimension=2,
    )


async def _create_memory(
    connection,
    clock: FrozenClock,
    *,
    memory_id: str,
    user_id: str,
    canonical_text: str,
    status: MemoryStatus,
    index_text: str | None = None,
    privacy_level: int = 0,
    preserve_verbatim: bool = False,
) -> None:
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    conversation_id = f"cnv_{user_id}"
    timestamp = clock.now().isoformat()
    if await users.get_user(user_id) is None:
        await users.create_user(user_id)
    await connection.execute(
        """
        INSERT OR IGNORE INTO assistant_modes(
            id,
            display_name,
            prompt_hash,
            memory_policy_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "coding_debug",
            "Coding Debug",
            "stub",
            "{}",
            timestamp,
            timestamp,
        ),
    )
    await connection.commit()
    if await conversations.get_conversation(conversation_id, user_id) is None:
        await conversations.create_conversation(
            conversation_id,
            user_id,
            None,
            "coding_debug",
            "Chat",
        )
    await memories.create_memory_object(
        user_id=user_id,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=canonical_text,
        index_text=index_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=privacy_level,
        preserve_verbatim=preserve_verbatim,
        status=status,
        memory_id=memory_id,
    )


async def _build_service(tmp_path: Path, vectors_by_text: dict[str, list[float]]):
    database_path = str(tmp_path / "embedding-backfill.db")
    scan_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    embedding_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    provider = RecordingEmbeddingProvider(vectors_by_text)
    backend = SQLiteVecBackend(
        embedding_connection,
        LLMClient(provider_name=provider.name, providers=[provider]),
        _settings(database_path),
    )
    await backend.initialize()
    service = EmbeddingBackfillService(
        connection=scan_connection,
        embedding_index=backend,
    )
    return scan_connection, embedding_connection, provider, service


pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec is not installed")


@pytest.mark.asyncio
async def test_backfill_embeds_missing_row_and_writes_metadata(tmp_path: Path) -> None:
    scan_connection, embedding_connection, provider, service = await _build_service(
        tmp_path,
        {"Backfill memory": [1.0, 0.0]},
    )
    clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))
    try:
        await _create_memory(
            scan_connection,
            clock,
            memory_id="mem_backfill",
            user_id="usr_1",
            canonical_text="Backfill memory",
            status=MemoryStatus.ACTIVE,
        )

        result = await service.run(batch_size=10, delay_ms=0)

        metadata_cursor = await scan_connection.execute(
            "SELECT user_id FROM memory_embedding_metadata WHERE memory_id = ?",
            ("mem_backfill",),
        )
        vec_cursor = await embedding_connection.execute(
            "SELECT memory_id FROM vec_memory_embeddings WHERE memory_id = ?",
            ("mem_backfill",),
        )

        assert result.model_dump() == {
            "examined": 1,
            "embedded": 1,
            "skipped": 0,
            "failed": 0,
            "batch_size": 10,
            "delay_ms": 0,
            "user_id": None,
        }
        assert provider.requests[0].input_texts == ["Backfill memory"]
        assert (await metadata_cursor.fetchone())["user_id"] == "usr_1"
        assert (await vec_cursor.fetchone())["memory_id"] == "mem_backfill"
    finally:
        await scan_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_backfill_protected_verbatim_rows_embed_only_safe_index_text(tmp_path: Path) -> None:
    scan_connection, embedding_connection, provider, service = await _build_service(
        tmp_path,
        {"Lives near the old lighthouse.": [1.0, 0.0]},
    )
    clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))
    try:
        await _create_memory(
            scan_connection,
            clock,
            memory_id="mem_secret",
            user_id="usr_1",
            canonical_text="Home alarm code is 4812.",
            index_text="Lives near the old lighthouse.",
            privacy_level=2,
            preserve_verbatim=True,
            status=MemoryStatus.ACTIVE,
        )

        result = await service.run(batch_size=10, delay_ms=0)

        assert result.embedded == 1
        assert provider.requests[0].input_texts == ["Lives near the old lighthouse."]
    finally:
        await scan_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_backfill_skips_ineligible_statuses(tmp_path: Path) -> None:
    scan_connection, embedding_connection, _provider, service = await _build_service(
        tmp_path,
        {},
    )
    clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))
    try:
        for memory_id, status in (
            ("mem_archived", MemoryStatus.ARCHIVED),
            ("mem_review", MemoryStatus.REVIEW_REQUIRED),
            ("mem_pending", MemoryStatus.PENDING_USER_CONFIRMATION),
            ("mem_declined", MemoryStatus.DECLINED),
        ):
            await _create_memory(
                scan_connection,
                clock,
                memory_id=memory_id,
                user_id="usr_1",
                canonical_text=f"Memory {memory_id}",
                status=status,
            )

        result = await service.run(batch_size=2, delay_ms=0)

        metadata_cursor = await scan_connection.execute(
            "SELECT COUNT(*) AS count FROM memory_embedding_metadata"
        )

        assert result.examined == 4
        assert result.embedded == 0
        assert result.skipped == 4
        assert result.failed == 0
        assert (await metadata_cursor.fetchone())["count"] == 0
    finally:
        await scan_connection.close()
        await embedding_connection.close()


@pytest.mark.asyncio
async def test_backfill_rerun_is_no_op(tmp_path: Path) -> None:
    scan_connection, embedding_connection, _provider, service = await _build_service(
        tmp_path,
        {"Backfill memory": [1.0, 0.0]},
    )
    clock = FrozenClock(datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc))
    try:
        await _create_memory(
            scan_connection,
            clock,
            memory_id="mem_backfill",
            user_id="usr_1",
            canonical_text="Backfill memory",
            status=MemoryStatus.ACTIVE,
        )

        first = await service.run(batch_size=10, delay_ms=0)
        second = await service.run(batch_size=10, delay_ms=0)

        assert first.embedded == 1
        assert second.examined == 0
        assert second.embedded == 0
        assert second.skipped == 0
        assert second.failed == 0
    finally:
        await scan_connection.close()
        await embedding_connection.close()
