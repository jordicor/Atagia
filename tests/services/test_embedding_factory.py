"""Tests for embedding index factory helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.services.embeddings import NoneBackend, create_embedding_index
from atagia.services.llm_client import ConfigurationError, LLMClient, LLMEmbeddingRequest, LLMEmbeddingResponse, LLMEmbeddingVector, LLMProvider
from atagia.services.sqlite_vec_backend import SQLiteVecBackend

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class StubProvider(LLMProvider):
    name = "embedding-factory-tests"

    async def complete(self, request):
        raise AssertionError("Completions are not used in embedding factory tests")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            provider=self.name,
            model=request.model,
            vectors=[LLMEmbeddingVector(index=0, values=[0.1, 0.2])],
        )


def _settings(*, backend: str, model: str | None = None, dimension: int = 2) -> Settings:
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
        embedding_backend=backend,
        embedding_model=model,
        embedding_dimension=dimension,
    )


@pytest.mark.asyncio
async def test_create_embedding_index_with_none_returns_none_backend() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        index = await create_embedding_index(
            _settings(backend="none"),
            connection,
            LLMClient(provider_name="embedding-factory-tests", providers=[StubProvider()]),
        )

        assert isinstance(index, NoneBackend)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_embedding_index_with_sqlite_vec_returns_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_op_initialize(self) -> None:
        return None

    monkeypatch.setattr(SQLiteVecBackend, "initialize", _no_op_initialize)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        index = await create_embedding_index(
            _settings(backend="sqlite_vec", model="embed-test-model"),
            connection,
            LLMClient(provider_name="embedding-factory-tests", providers=[StubProvider()]),
        )

        assert isinstance(index, SQLiteVecBackend)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_embedding_index_with_unknown_backend_raises() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        with pytest.raises(ConfigurationError, match="Unknown embedding backend"):
            await create_embedding_index(
                _settings(backend="mystery"),
                connection,
                LLMClient(provider_name="embedding-factory-tests", providers=[StubProvider()]),
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_embedding_index_requires_model_for_sqlite_vec() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        with pytest.raises(ConfigurationError, match="ATAGIA_EMBEDDING_MODEL is required when backend is sqlite_vec"):
            await create_embedding_index(
                _settings(backend="sqlite_vec", model=None),
                connection,
                LLMClient(provider_name="embedding-factory-tests", providers=[StubProvider()]),
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sqlite_vec_backend_validates_dimension_bounds() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        with pytest.raises(ConfigurationError, match="embedding_dimension must be between 1 and 8192"):
            SQLiteVecBackend(
                connection,
                LLMClient(provider_name="embedding-factory-tests", providers=[StubProvider()]),
                _settings(backend="sqlite_vec", model="embed-test-model", dimension=0),
            )
    finally:
        await connection.close()
