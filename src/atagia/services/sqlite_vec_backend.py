"""sqlite-vec embedding backend."""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

import aiosqlite

from atagia.core.config import Settings
from atagia.core.clock import Clock
from atagia.models.schemas_memory import MemoryStatus
from atagia.services.embeddings import EmbeddingIndex, EmbeddingMatch
from atagia.services.llm_client import (
    ConfigurationError,
    LLMClient,
    LLMEmbeddingRequest,
)
from atagia.services.model_resolution import DEFAULT_EMBEDDING_MODEL
from atagia.services.job_execution_context import current_job_claim

logger = logging.getLogger(__name__)


class StaleEmbeddingLifecycleError(RuntimeError):
    """An embedding write no longer owns an active user/job lifecycle."""


def compose_embedding_text(canonical_text: str, index_text: Any | None) -> str:
    """Combine canonical and retrieval-oriented context for embedding."""
    normalized_index_text = ""
    if index_text is not None:
        normalized_index_text = str(index_text).strip()
    if not normalized_index_text:
        return canonical_text
    return f"{canonical_text}\n{normalized_index_text}"


def _canonical_metadata_scope(scope: object) -> str:
    value = str(scope or "").strip()
    if value in {"conversation", "ephemeral_session", "chat"}:
        return "chat"
    if value in {"workspace", "character"}:
        return "character"
    if value in {"global_user", "assistant_mode", "user"}:
        return "user"
    return value


class SQLiteVecBackend(EmbeddingIndex):
    """Embedding index backed by sqlite-vec virtual tables."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        self._connection = connection
        self._llm_client = llm_client
        self._settings = settings
        self._clock = clock
        self._write_lock = asyncio.Lock()
        self._embedding_model = settings.embedding_model or DEFAULT_EMBEDDING_MODEL
        self._dimension = settings.embedding_dimension
        if not (1 <= self._dimension <= 8192):
            raise ConfigurationError("embedding_dimension must be between 1 and 8192")
        self._sqlite_vec: Any | None = None

    @property
    def vector_limit(self) -> int:
        return self._settings.embedding_vector_limit_cap

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

        await self._ensure_vector_table()
        await self._cleanup_ineligible_embeddings()
        await self._connection.commit()

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, Any]) -> None:
        embedding_text = compose_embedding_text(text, metadata.get("index_text"))
        embedding = await self._embed_texts(
            [embedding_text],
            metadata={
                "purpose": "embedding_upsert",
                "memory_id": memory_id,
                **metadata,
            },
        )
        if not embedding:
            return
        user_id = str(metadata["user_id"])
        async with self._write_lock:
            try:
                await self._begin_lifecycle_fenced_write(user_id)
                await self._connection.execute(
                    "DELETE FROM vec_memory_embeddings WHERE memory_id = ?",
                    (memory_id,),
                )
                await self._connection.execute(
                    """
                    INSERT INTO vec_memory_embeddings(memory_id, user_id, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (
                        memory_id,
                        user_id,
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
                        user_id,
                        str(metadata["object_type"]),
                        _canonical_metadata_scope(metadata["scope"]),
                        str(metadata.get("created_at", "")),
                    ),
                )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise

    async def search(
        self, query: str, user_id: str, top_k: int
    ) -> list[EmbeddingMatch]:
        """Return semantic matches for a user-scoped query."""
        if top_k <= 0:
            return []
        try:
            query_embedding = await self._embed_texts(
                [query],
                metadata={"purpose": "embedding_search", "user_id": user_id},
            )
        except Exception:
            logger.warning(
                "Embedding search failed for user_id=%s", user_id, exc_info=True
            )
            return []
        if not query_embedding:
            return []

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
              AND v.user_id = ?
              AND k = ?
              AND m.user_id = ?
            ORDER BY v.distance ASC
            LIMIT ?
            """,
            (
                self._serialize_vector(query_embedding),
                user_id,
                top_k,
                user_id,
                top_k,
            ),
        )
        rows = await cursor.fetchall()
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
        async with self._write_lock:
            try:
                claim = current_job_claim()
                if claim is not None:
                    await self._begin_lifecycle_fenced_write(claim.envelope.user_id)
                    cursor = await self._connection.execute(
                        """
                        SELECT user_id
                        FROM (
                            SELECT user_id, 0 AS source_order
                            FROM memory_embedding_metadata
                            WHERE memory_id = ?
                            UNION ALL
                            SELECT user_id, 1 AS source_order
                            FROM vec_memory_embeddings
                            WHERE memory_id = ?
                        )
                        ORDER BY source_order ASC
                        LIMIT 1
                        """,
                        (memory_id, memory_id),
                    )
                    owner = await cursor.fetchone()
                    if owner is not None and str(owner["user_id"]) != (
                        claim.envelope.user_id
                    ):
                        raise StaleEmbeddingLifecycleError(
                            "Embedding delete does not belong to the current durable job"
                        )
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

    async def _begin_lifecycle_fenced_write(self, user_id: str) -> None:
        """Acquire SQLite ownership and validate lifecycle/fence before writing."""

        await self._connection.execute("BEGIN IMMEDIATE")
        claim = current_job_claim()
        if claim is None:
            cursor = await self._connection.execute(
                """
                SELECT 1
                FROM users AS user
                JOIN user_lifecycles AS lifecycle
                  ON lifecycle.user_id = user.id
                WHERE user.id = ?
                  AND user.deleted_at IS NULL
                  AND lifecycle.state = 'active'
                  AND lifecycle.erasure_cleanup_id IS NULL
                LIMIT 1
                """,
                (user_id,),
            )
        else:
            if claim.envelope.user_id != user_id or self._clock is None:
                raise StaleEmbeddingLifecycleError(
                    "Embedding write lacks the current durable job lifecycle"
                )
            cursor = await self._connection.execute(
                """
                SELECT 1
                FROM worker_job_runs AS job
                JOIN user_lifecycles AS lifecycle
                  ON lifecycle.user_id = job.user_id
                 AND lifecycle.lifecycle_epoch = job.lifecycle_epoch
                JOIN users AS user ON user.id = job.user_id
                WHERE job.job_id = ?
                  AND job.user_id = ?
                  AND job.status = 'running'
                  AND job.execution_owner = ?
                  AND job.execution_fence = ?
                  AND job.lifecycle_epoch = ?
                  AND job.derivation_revision = ?
                  AND job.execution_lease_expires_at > ?
                  AND lifecycle.derivation_revision = job.derivation_revision
                  AND lifecycle.state = 'active'
                  AND lifecycle.erasure_cleanup_id IS NULL
                  AND user.deleted_at IS NULL
                LIMIT 1
                """,
                (
                    claim.envelope.job_id,
                    user_id,
                    claim.owner_id,
                    claim.execution_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    self._clock.now().isoformat(),
                ),
            )
        if await cursor.fetchone() is None:
            raise StaleEmbeddingLifecycleError(
                f"Embedding lifecycle is no longer active for user {user_id}"
            )

    async def _embed_texts(
        self, texts: list[str], metadata: dict[str, Any]
    ) -> list[float]:
        response = await self._llm_client.embed(
            LLMEmbeddingRequest(
                model=self._embedding_model,
                input_texts=texts,
                dimensions=(
                    self._dimension
                    if self._llm_client.supports_embedding_dimensions(
                        self._embedding_model
                    )
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

    async def _ensure_vector_table(self) -> None:
        if await self._vector_table_has_user_partition():
            return
        if await self._vector_table_exists():
            await self._rebuild_vector_table_with_user_partition()
            return
        await self._create_vector_table()

    async def _vector_table_exists(self) -> bool:
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'vec_memory_embeddings'
            """
        )
        return await cursor.fetchone() is not None

    async def _vector_table_has_user_partition(self) -> bool:
        if not await self._vector_table_exists():
            return False
        cursor = await self._connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_memory_embeddings'"
        )
        row = await cursor.fetchone()
        sql = str(row["sql"] if row is not None else "").lower()
        return "user_id" in sql and "partition key" in sql

    async def _create_vector_table(self) -> None:
        await self._connection.execute(
            """
            CREATE VIRTUAL TABLE vec_memory_embeddings USING vec0(
                memory_id TEXT PRIMARY KEY,
                user_id TEXT partition key,
                embedding float[{dimension}]
            )
            """.format(dimension=self._dimension)
        )

    async def _rebuild_vector_table_with_user_partition(self) -> None:
        await self._connection.execute(
            """
            CREATE TEMP TABLE atagia_vec_memory_embeddings_rebuild AS
            SELECT v.memory_id, m.user_id, v.embedding
            FROM vec_memory_embeddings AS v
            JOIN memory_embedding_metadata AS m ON m.memory_id = v.memory_id
            WHERE m.user_id IS NOT NULL
            """
        )
        await self._connection.execute("DROP TABLE vec_memory_embeddings")
        await self._create_vector_table()
        await self._connection.execute(
            """
            INSERT INTO vec_memory_embeddings(memory_id, user_id, embedding)
            SELECT memory_id, user_id, embedding
            FROM atagia_vec_memory_embeddings_rebuild
            """
        )
        await self._connection.execute(
            "DROP TABLE atagia_vec_memory_embeddings_rebuild"
        )

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
            memory_id
            for memory_id in stale_metadata_ids
            if memory_id not in set(stale_memory_ids)
        ]
        if stale_memory_ids or stale_metadata_only:
            metadata_ids = stale_memory_ids + stale_metadata_only
            placeholders = ", ".join("?" for _ in metadata_ids)
            await self._connection.execute(
                f"DELETE FROM memory_embedding_metadata WHERE memory_id IN ({placeholders})",
                tuple(metadata_ids),
            )
