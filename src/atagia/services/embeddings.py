"""Embedding index abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.core.config import Settings
from atagia.services.llm_client import ConfigurationError, LLMClient


class EmbeddingMatch(BaseModel):
    """Single embedding search result."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    score: float
    position_rank: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingIndex(ABC):
    """Abstract embedding index interface."""

    @property
    @abstractmethod
    def vector_limit(self) -> int:
        """Return 0 when vector recall is disabled, or a positive candidate limit."""

    @abstractmethod
    async def upsert(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Insert or update an embedding."""

    @abstractmethod
    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        """Search the embedding index."""

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Delete an embedding."""


class NoneBackend(EmbeddingIndex):
    """No-op embedding backend for Phase 1."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    @property
    def vector_limit(self) -> int:
        return 0

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        return None

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        return []

    async def delete(self, memory_id: str) -> None:
        return None


async def create_embedding_index(
    settings: Settings,
    connection: Any,
    llm_client: LLMClient[Any],
) -> EmbeddingIndex:
    if settings.embedding_backend == "none":
        return NoneBackend()
    if settings.embedding_backend == "sqlite_vec":
        if not settings.embedding_model:
            raise ConfigurationError(
                "ATAGIA_EMBEDDING_MODEL is required when backend is sqlite_vec"
            )
        from atagia.services.sqlite_vec_backend import SQLiteVecBackend

        backend = SQLiteVecBackend(connection, llm_client, settings)
        await backend.initialize()
        return backend
    raise ConfigurationError(f"Unknown embedding backend: {settings.embedding_backend}")
