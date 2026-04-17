"""Backfill sqlite-vec embeddings for existing memory rows."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from atagia.models.schemas_memory import MemoryStatus
from atagia.services.embedding_payloads import build_embedding_upsert_payload
from atagia.services.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)

_EMBEDDABLE_STATUSES = frozenset({MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED})
ProgressCallback = Callable[["EmbeddingBackfillResult"], Awaitable[None] | None]


class EmbeddingBackfillResult(BaseModel):
    """Counters and parameters for one backfill run."""

    model_config = ConfigDict(extra="forbid")

    examined: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    batch_size: int = Field(ge=1)
    delay_ms: int = Field(ge=0)
    user_id: str | None = None


class EmbeddingBackfillService:
    """Scan for rows missing embedding metadata and backfill them."""

    def __init__(
        self,
        *,
        connection: aiosqlite.Connection,
        embedding_index: EmbeddingIndex,
        progress_callback: ProgressCallback | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._connection = connection
        self._embedding_index = embedding_index
        self._progress_callback = progress_callback
        self._sleep = sleep or asyncio.sleep

    async def run(
        self,
        *,
        batch_size: int,
        delay_ms: int,
        user_id: str | None = None,
    ) -> EmbeddingBackfillResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if self._embedding_index.vector_limit == 0:
            raise ValueError("Embedding backfill requires an active embedding backend")

        result = EmbeddingBackfillResult(
            batch_size=batch_size,
            delay_ms=delay_ms,
            user_id=user_id,
        )
        cursor = await self._connection.execute(*self._scan_query(user_id))
        try:
            rows = await cursor.fetchmany(batch_size)
            while rows:
                for row in rows:
                    result.examined += 1
                    await self._process_row(row, result)
                await self._emit_progress(result)
                rows = await cursor.fetchmany(batch_size)
                if rows and delay_ms > 0:
                    await self._sleep(delay_ms / 1000.0)
        finally:
            await cursor.close()
        return result

    def _scan_query(self, user_id: str | None) -> tuple[str, tuple[Any, ...]]:
        clauses = ["mem.memory_id IS NULL"]
        parameters: list[Any] = []
        if user_id is not None:
            clauses.append("mo.user_id = ?")
            parameters.append(user_id)
        return (
            """
            SELECT
                mo.id,
                mo.user_id,
                mo.object_type,
                mo.scope,
                mo.canonical_text,
                mo.index_text,
                mo.privacy_level,
                mo.preserve_verbatim,
                mo.status,
                mo.created_at
            FROM memory_objects AS mo
            LEFT JOIN memory_embedding_metadata AS mem ON mem.memory_id = mo.id
            WHERE {where_clause}
            ORDER BY mo.created_at ASC, mo.id ASC
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def _process_row(
        self,
        row: aiosqlite.Row,
        result: EmbeddingBackfillResult,
    ) -> None:
        status = MemoryStatus(str(row["status"]))
        if status not in _EMBEDDABLE_STATUSES:
            result.skipped += 1
            return

        try:
            payload = build_embedding_upsert_payload(
                canonical_text=str(row["canonical_text"]),
                index_text=str(row["index_text"]) if row["index_text"] is not None else None,
                privacy_level=int(row["privacy_level"]),
                preserve_verbatim=bool(int(row["preserve_verbatim"])),
            )
            await self._embedding_index.upsert(
                memory_id=str(row["id"]),
                text=payload.text,
                metadata={
                    "user_id": str(row["user_id"]),
                    "object_type": str(row["object_type"]),
                    "scope": str(row["scope"]),
                    "created_at": str(row["created_at"]),
                    "index_text": payload.index_text,
                },
            )
            result.embedded += 1
        except Exception:
            result.failed += 1
            logger.warning(
                "Embedding backfill failed for memory_id=%s",
                row["id"],
                exc_info=True,
            )

    async def _emit_progress(self, result: EmbeddingBackfillResult) -> None:
        if self._progress_callback is None:
            return
        maybe_awaitable = self._progress_callback(result.model_copy())
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
