"""sqlite-vec embedding backend."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import aiosqlite

from atagia.core.config import Settings
from atagia.services.embeddings import EmbeddingIndex, EmbeddingMatch
from atagia.services.llm_client import ConfigurationError, LLMClient, LLMEmbeddingRequest

logger = logging.getLogger(__name__)


class SQLiteVecBackend(EmbeddingIndex):
    """Embedding index backed by sqlite-vec virtual tables."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        settings: Settings,
    ) -> None:
        if not settings.embedding_model:
            raise ConfigurationError(
                "ATAGIA_EMBEDDING_MODEL is required when ATAGIA_EMBEDDING_BACKEND=sqlite_vec"
            )
        self._connection = connection
        self._llm_client = llm_client
        self._settings = settings
        self._embedding_model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        if not (1 <= self._dimension <= 8192):
            raise ConfigurationError("embedding_dimension must be between 1 and 8192")
        self._sqlite_vec: Any | None = None

    @property
    def vector_limit(self) -> int:
        return 1

    async def initialize(self) -> None:
        try:
            sqlite_vec = importlib.import_module("sqlite_vec")
        except ImportError as exc:
            raise ConfigurationError(
                "sqlite-vec extension not found. Install with: pip install sqlite-vec"
            ) from exc

        self._sqlite_vec = sqlite_vec
        await self._connection.enable_load_extension(True)
        try:
            await self._connection._execute(sqlite_vec.load, self._connection._conn)  # noqa: SLF001
        except Exception as exc:
            raise ConfigurationError(
                "sqlite-vec extension not found. Install with: pip install sqlite-vec"
            ) from exc
        finally:
            await self._connection.enable_load_extension(False)

        await self._connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory_embeddings USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding float[{dimension}]
            )
            """.format(dimension=self._dimension)
        )
        await self._connection.commit()

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        embedding = await self._embed_texts(
            [text],
            metadata={"purpose": "embedding_upsert", "memory_id": memory_id, **metadata},
        )
        if not embedding:
            return
        try:
            await self._connection.execute(
                "DELETE FROM vec_memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )
            await self._connection.execute(
                """
                INSERT INTO vec_memory_embeddings(memory_id, embedding)
                VALUES (?, ?)
                """,
                (
                    memory_id,
                    self._serialize_vector(embedding),
                ),
            )
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO memory_embedding_metadata(
                    memory_id,
                    user_id,
                    object_type,
                    scope,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    str(metadata["user_id"]),
                    str(metadata["object_type"]),
                    str(metadata["scope"]),
                    str(metadata.get("created_at", "")),
                ),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def search(self, query: str, user_id: str, top_k: int) -> list[EmbeddingMatch]:
        """Return semantic matches for a user-scoped query.

        Note: sqlite-vec vec0 returns k nearest neighbors BEFORE the user_id JOIN
        filter. In multi-tenant scenarios with skewed data, this may return fewer
        results than top_k. The 5x over-fetch mitigates this. A per-user vec0 table
        would solve it completely but is deferred to a future phase.
        """
        if top_k <= 0:
            return []
        try:
            query_embedding = await self._embed_texts(
                [query],
                metadata={"purpose": "embedding_search", "user_id": user_id},
            )
        except Exception:
            logger.warning("Embedding search failed for user_id=%s", user_id, exc_info=True)
            return []
        if not query_embedding:
            return []

        fetch_k = max(50, top_k * 5)
        cursor = await self._connection.execute(
            """
            SELECT
                v.memory_id,
                v.distance,
                m.object_type,
                m.scope,
                m.created_at
            FROM vec_memory_embeddings AS v
            JOIN memory_embedding_metadata AS m ON m.memory_id = v.memory_id
            WHERE v.embedding MATCH ?
              AND k = ?
              AND m.user_id = ?
            ORDER BY v.distance ASC
            LIMIT ?
            """,
            (
                self._serialize_vector(query_embedding),
                fetch_k,
                user_id,
                fetch_k,
            ),
        )
        rows = await cursor.fetchall()
        return [
            EmbeddingMatch(
                memory_id=str(row["memory_id"]),
                score=1.0 / (1.0 + max(0.0, float(row["distance"]))),
                metadata={
                    "distance": float(row["distance"]),
                    "object_type": str(row["object_type"]),
                    "scope": str(row["scope"]),
                    "created_at": str(row["created_at"]),
                },
            )
            for row in rows
        ]

    async def delete(self, memory_id: str) -> None:
        try:
            await self._connection.execute(
                "DELETE FROM vec_memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )
            await self._connection.execute(
                "DELETE FROM memory_embedding_metadata WHERE memory_id = ?",
                (memory_id,),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def _embed_texts(self, texts: list[str], metadata: dict[str, Any]) -> list[float]:
        response = await self._llm_client.embed(
            LLMEmbeddingRequest(
                model=self._embedding_model,
                input_texts=texts,
                metadata=metadata,
            )
        )
        if not response.vectors:
            return []
        return list(response.vectors[0].values)

    def _serialize_vector(self, values: list[float]) -> Any:
        if self._sqlite_vec is None:
            raise RuntimeError("sqlite-vec backend used before initialization")
        return self._sqlite_vec.serialize_float32(values)
