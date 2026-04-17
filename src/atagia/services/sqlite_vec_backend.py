"""sqlite-vec embedding backend."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import aiosqlite

from atagia.core.config import Settings
from atagia.models.schemas_memory import MemoryStatus
from atagia.services.embeddings import EmbeddingIndex, EmbeddingMatch
from atagia.services.llm_client import ConfigurationError, LLMClient, LLMEmbeddingRequest

logger = logging.getLogger(__name__)


def compose_embedding_text(canonical_text: str, index_text: Any | None) -> str:
    """Combine canonical and retrieval-oriented context for embedding."""
    normalized_index_text = ""
    if index_text is not None:
        normalized_index_text = str(index_text).strip()
    if not normalized_index_text:
        return canonical_text
    return f"{canonical_text}\n{normalized_index_text}"


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
        await self._cleanup_ineligible_embeddings()
        await self._connection.commit()

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        embedding_text = compose_embedding_text(text, metadata.get("index_text"))
        embedding = await self._embed_texts(
            [embedding_text],
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
        results than top_k. The current over-fetch mitigates this. A per-user vec0 table
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

        # TODO: add per-user partitioning when sqlite-vec multi-tenant scale matters.
        fetch_k = max(100, top_k * 8)
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
        if len(rows) < top_k:
            logger.warning(
                "sqlite-vec search returned fewer rows than requested after user filtering: "
                "user_id=%s requested_top_k=%s fetched_k=%s returned_count=%s",
                user_id,
                top_k,
                fetch_k,
                len(rows),
            )
        return [
            EmbeddingMatch(
                memory_id=str(row["memory_id"]),
                score=1.0 / (1.0 + max(0.0, float(row["distance"]))),
                position_rank=index,
                metadata={
                    "distance": float(row["distance"]),
                    "object_type": str(row["object_type"]),
                    "scope": str(row["scope"]),
                    "created_at": str(row["created_at"]),
                },
            )
            for index, row in enumerate(rows, start=1)
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
                dimensions=(
                    self._dimension
                    if self._llm_client.embedding_provider.supports_embedding_dimensions
                    else None
                ),
                metadata=metadata,
            )
        )
        if not response.vectors:
            return []
        vector = list(response.vectors[0].values)
        if len(vector) != self._dimension:
            raise ConfigurationError(
                "Embedding dimension mismatch: expected "
                f"{self._dimension}, received {len(vector)} from "
                f"{response.provider}/{response.model}"
            )
        return vector

    def _serialize_vector(self, values: list[float]) -> Any:
        if self._sqlite_vec is None:
            raise RuntimeError("sqlite-vec backend used before initialization")
        return self._sqlite_vec.serialize_float32(values)

    async def _cleanup_ineligible_embeddings(self) -> None:
        cursor = await self._connection.execute(
            """
            SELECT v.memory_id
            FROM vec_memory_embeddings AS v
            LEFT JOIN memory_objects AS mo ON mo.id = v.memory_id
            WHERE mo.id IS NULL
               OR mo.status NOT IN (?, ?)
            """,
            (MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value),
        )
        rows = await cursor.fetchall()
        stale_memory_ids = [str(row["memory_id"]) for row in rows]
        metadata_cursor = await self._connection.execute(
            """
            SELECT mem.memory_id
            FROM memory_embedding_metadata AS mem
            LEFT JOIN memory_objects AS mo ON mo.id = mem.memory_id
            LEFT JOIN vec_memory_embeddings AS v ON v.memory_id = mem.memory_id
            WHERE mo.id IS NULL
               OR mo.status NOT IN (?, ?)
               OR v.memory_id IS NULL
            """,
            (MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value),
        )
        metadata_rows = await metadata_cursor.fetchall()
        stale_metadata_ids = [str(row["memory_id"]) for row in metadata_rows]
        if stale_memory_ids:
            placeholders = ", ".join("?" for _ in stale_memory_ids)
            await self._connection.execute(
                f"DELETE FROM vec_memory_embeddings WHERE memory_id IN ({placeholders})",
                tuple(stale_memory_ids),
            )
        stale_metadata_only = [
            memory_id for memory_id in stale_metadata_ids if memory_id not in set(stale_memory_ids)
        ]
        if stale_memory_ids or stale_metadata_only:
            metadata_ids = stale_memory_ids + stale_metadata_only
            placeholders = ", ".join("?" for _ in metadata_ids)
            await self._connection.execute(
                f"DELETE FROM memory_embedding_metadata WHERE memory_id IN ({placeholders})",
                tuple(metadata_ids),
            )
