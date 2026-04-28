"""Tests for the sqlite-vec embedding backend."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from pathlib import Path

import pytest

from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.services.llm_client import (
    ConfigurationError,
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


class EmbeddingProvider(LLMProvider):
    name = "sqlite-vec-tests"
    supports_embedding_dimensions = True

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = dict(vectors_by_text)
        self.requests: list[LLMEmbeddingRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError("Completions are not used in sqlite-vec backend tests")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        self.requests.append(request)
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[
                LLMEmbeddingVector(index=index, values=self.vectors_by_text[text])
                for index, text in enumerate(request.input_texts)
            ],
        )


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
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
        embedding_model="openai/embed-test-model",
        embedding_dimension=2,
    )


pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec is not installed")


_TS = "2026-04-04T00:00:00+00:00"


async def _insert_stub_memory(connection, memory_id: str, user_id: str) -> None:
    """Insert minimal parent rows so FK constraints are satisfied."""
    await connection.execute(
        "INSERT OR IGNORE INTO users(id, created_at, updated_at) VALUES (?, ?, ?)",
        (user_id, _TS, _TS),
    )
    await connection.execute(
        """INSERT OR IGNORE INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
           VALUES ('general_qa', 'General QA', 'stub', '{}', ?, ?)""",
        (_TS, _TS),
    )
    await connection.execute(
        """INSERT OR IGNORE INTO conversations(id, user_id, assistant_mode_id, created_at, updated_at)
           VALUES ('conv_stub', ?, 'general_qa', ?, ?)""",
        (user_id, _TS, _TS),
    )
    await connection.execute(
        """INSERT OR IGNORE INTO memory_objects(
               id, user_id, conversation_id, assistant_mode_id, object_type, scope,
               canonical_text, payload_json, source_kind, confidence, stability, vitality,
               maya_score, privacy_level, status, created_at, updated_at
           ) VALUES (?, ?, 'conv_stub', 'general_qa', 'evidence', 'conversation',
               'stub', '{}', 'extracted', 0.5, 0.5, 1.0, 0.0, 0, 'active', ?, ?)""",
        (memory_id, user_id, _TS, _TS),
    )
    await connection.commit()


async def _build_backend(database_path: str = ":memory:"):
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    provider = EmbeddingProvider(
        {
            "memory one": [1.0, 0.0],
            "memory one\nretry policies for websocket incidents": [1.0, 0.0],
            "memory two": [0.0, 1.0],
            "query one": [1.0, 0.0],
            "query none": [-1.0, 0.0],
            "updated memory one": [0.8, 0.2],
        }
    )
    backend = SQLiteVecBackend(
        connection,
        LLMClient(provider_name=provider.name, providers=[provider]),
        _settings(),
    )
    await backend.initialize()
    return connection, backend, provider


@pytest.mark.asyncio
async def test_upsert_stores_embedding_and_metadata() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )

        metadata_cursor = await connection.execute("SELECT user_id FROM memory_embedding_metadata WHERE memory_id = 'mem_1'")
        vec_cursor = await connection.execute("SELECT memory_id FROM vec_memory_embeddings WHERE memory_id = 'mem_1'")

        assert (await metadata_cursor.fetchone())["user_id"] == "usr_1"
        assert (await vec_cursor.fetchone())["memory_id"] == "mem_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_returns_nearest_neighbors_filtered_by_user_id() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await _insert_stub_memory(connection, "mem_2", "usr_2")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )
        await backend.upsert(
            "mem_2",
            "memory two",
            {"user_id": "usr_2", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:01:00+00:00"},
        )

        matches = await backend.search("query one", "usr_1", top_k=5)

        assert [match.memory_id for match in matches] == ["mem_1"]
        assert matches[0].score > 0.9
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_prefilters_user_partition_before_knn_limit() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_target", "usr_target")
        await backend.upsert(
            "mem_target",
            "memory two",
            {
                "user_id": "usr_target",
                "object_type": "evidence",
                "scope": "conversation",
                "created_at": _TS,
            },
        )
        for index in range(120):
            memory_id = f"mem_other_{index}"
            await _insert_stub_memory(connection, memory_id, "usr_other")
            await backend.upsert(
                memory_id,
                "memory one",
                {
                    "user_id": "usr_other",
                    "object_type": "evidence",
                    "scope": "conversation",
                    "created_at": _TS,
                },
            )

        matches = await backend.search("query one", "usr_target", top_k=1)

        assert [match.memory_id for match in matches] == ["mem_target"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_matches_exist() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        matches = await backend.search("query none", "usr_1", top_k=5)

        assert matches == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delete_removes_embedding_and_metadata() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )

        await backend.delete("mem_1")

        metadata_cursor = await connection.execute("SELECT COUNT(*) AS count FROM memory_embedding_metadata WHERE memory_id = 'mem_1'")
        vec_cursor = await connection.execute("SELECT COUNT(*) AS count FROM vec_memory_embeddings WHERE memory_id = 'mem_1'")

        assert (await metadata_cursor.fetchone())["count"] == 0
        assert (await vec_cursor.fetchone())["count"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_upsert_twice_updates_existing_embedding_idempotently() -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )
        await backend.upsert(
            "mem_1",
            "updated memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:05:00+00:00"},
        )

        cursor = await connection.execute("SELECT COUNT(*) AS count FROM memory_embedding_metadata WHERE memory_id = 'mem_1'")
        matches = await backend.search("query one", "usr_1", top_k=5)

        assert (await cursor.fetchone())["count"] == 1
        assert matches[0].memory_id == "mem_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_upsert_passes_dimensions_to_openai_compatible_provider() -> None:
    connection, backend, provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")

        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )

        assert provider.requests[0].dimensions == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_upsert_rejects_dimension_mismatch() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    provider = EmbeddingProvider({"memory one": [1.0, 0.0, 0.5]})
    backend = SQLiteVecBackend(
        connection,
        LLMClient(provider_name=provider.name, providers=[provider]),
        _settings(),
    )
    await backend.initialize()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")

        with pytest.raises(
            ConfigurationError,
            match="Embedding dimension mismatch: expected 2, received 3",
        ):
            await backend.upsert(
                "mem_1",
                "memory one",
                {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_initialize_cleans_up_archived_embeddings(tmp_path: Path) -> None:
    database_path = str(tmp_path / "atagia-cleanup.db")
    connection, backend, _provider = await _build_backend(database_path)
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )
        await connection.execute(
            "UPDATE memory_objects SET status = 'archived' WHERE id = 'mem_1'"
        )
        await connection.commit()
    finally:
        await connection.close()

    reopened = await initialize_database(database_path, MIGRATIONS_DIR)
    backend = SQLiteVecBackend(
        reopened,
        LLMClient(provider_name=EmbeddingProvider.name, providers=[EmbeddingProvider({"memory one": [1.0, 0.0]})]),
        _settings(),
    )
    await backend.initialize()
    try:
        metadata_cursor = await reopened.execute(
            "SELECT COUNT(*) AS count FROM memory_embedding_metadata WHERE memory_id = 'mem_1'"
        )
        vec_cursor = await reopened.execute(
            "SELECT COUNT(*) AS count FROM vec_memory_embeddings WHERE memory_id = 'mem_1'"
        )

        assert (await metadata_cursor.fetchone())["count"] == 0
        assert (await vec_cursor.fetchone())["count"] == 0
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_upsert_commits_metadata_durably(tmp_path: Path) -> None:
    database_path = str(tmp_path / "atagia-sqlite-vec.db")
    connection, backend, _provider = await _build_backend(database_path)
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )
    finally:
        await connection.close()

    reopened = await initialize_database(database_path, MIGRATIONS_DIR)
    try:
        cursor = await reopened.execute(
            "SELECT user_id FROM memory_embedding_metadata WHERE memory_id = 'mem_1'"
        )
        assert (await cursor.fetchone())["user_id"] == "usr_1"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_delete_commits_metadata_removal_durably(tmp_path: Path) -> None:
    database_path = str(tmp_path / "atagia-sqlite-vec-delete.db")
    connection, backend, _provider = await _build_backend(database_path)
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {"user_id": "usr_1", "object_type": "evidence", "scope": "conversation", "created_at": "2026-04-04T12:00:00+00:00"},
        )
        await backend.delete("mem_1")
    finally:
        await connection.close()

    reopened = await initialize_database(database_path, MIGRATIONS_DIR)
    try:
        cursor = await reopened.execute(
            "SELECT COUNT(*) AS count FROM memory_embedding_metadata WHERE memory_id = 'mem_1'"
        )
        assert (await cursor.fetchone())["count"] == 0
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_search_returns_available_user_rows_without_postfilter_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection, backend, _provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_target", "usr_target")
        await _insert_stub_memory(connection, "mem_other_1", "usr_other")
        await _insert_stub_memory(connection, "mem_other_2", "usr_other")
        await backend.upsert(
            "mem_target",
            "memory one",
            {"user_id": "usr_target", "object_type": "evidence", "scope": "conversation", "created_at": _TS},
        )
        await backend.upsert(
            "mem_other_1",
            "memory one",
            {"user_id": "usr_other", "object_type": "evidence", "scope": "conversation", "created_at": _TS},
        )
        await backend.upsert(
            "mem_other_2",
            "memory two",
            {"user_id": "usr_other", "object_type": "evidence", "scope": "conversation", "created_at": _TS},
        )

        with caplog.at_level("WARNING", logger="atagia.services.sqlite_vec_backend"):
            matches = await backend.search("query one", "usr_target", top_k=5)

        assert len(matches) == 1
        assert matches[0].memory_id == "mem_target"
        assert not caplog.records
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_upsert_embeds_canonical_and_index_text_together() -> None:
    connection, backend, provider = await _build_backend()
    try:
        await _insert_stub_memory(connection, "mem_1", "usr_1")
        await backend.upsert(
            "mem_1",
            "memory one",
            {
                "user_id": "usr_1",
                "object_type": "evidence",
                "scope": "conversation",
                "created_at": "2026-04-04T12:00:00+00:00",
                "index_text": "retry policies for websocket incidents",
            },
        )

        assert provider.requests[0].input_texts == [
            "memory one\nretry policies for websocket incidents"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_initialize_rebuilds_legacy_unpartitioned_vector_table() -> None:
    sqlite_vec = importlib.import_module("sqlite_vec")
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    provider = EmbeddingProvider(
        {
            "memory one": [1.0, 0.0],
            "query one": [1.0, 0.0],
        }
    )
    try:
        await connection.enable_load_extension(True)
        try:
            await connection._execute(sqlite_vec.load, connection._conn)  # noqa: SLF001
        finally:
            await connection.enable_load_extension(False)
        await connection.execute(
            """
            CREATE VIRTUAL TABLE vec_memory_embeddings USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding float[2]
            )
            """
        )
        await _insert_stub_memory(connection, "mem_legacy", "usr_1")
        await connection.execute(
            """
            INSERT INTO vec_memory_embeddings(memory_id, embedding)
            VALUES (?, ?)
            """,
            ("mem_legacy", sqlite_vec.serialize_float32([1.0, 0.0])),
        )
        await connection.execute(
            """
            INSERT INTO memory_embedding_metadata(memory_id, user_id, object_type, scope, created_at)
            VALUES ('mem_legacy', 'usr_1', 'evidence', 'conversation', ?)
            """,
            (_TS,),
        )
        await connection.commit()

        backend = SQLiteVecBackend(
            connection,
            LLMClient(provider_name=provider.name, providers=[provider]),
            _settings(),
        )
        await backend.initialize()
        matches = await backend.search("query one", "usr_1", top_k=1)
        schema_cursor = await connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_memory_embeddings'"
        )
        schema = str((await schema_cursor.fetchone())["sql"])

        assert "user_id text partition key" in schema.lower()
        assert [match.memory_id for match in matches] == ["mem_legacy"]
    finally:
        await connection.close()
