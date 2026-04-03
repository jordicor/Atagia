"""Synchronous admin rebuild service for conversations and users."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MessageRepository
from atagia.core.storage_backend import InProcessBackend, StorageBackend
from atagia.core.timestamps import resolve_message_occurred_at
from atagia.memory.policy_manifest import ManifestLoader
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CompactionJobKind,
    CompactionJobPayload,
    JobEnvelope,
    JobType,
    MessageJobPayload,
    REVISE_STREAM_NAME,
    WORKER_GROUP_NAME,
)
from atagia.services.embeddings import EmbeddingIndex, NoneBackend
from atagia.services.llm_client import LLMClient
from atagia.workers.compaction_worker import CompactionWorker
from atagia.workers.contract_worker import ContractWorker
from atagia.workers.ingest_worker import IngestWorker
from atagia.workers.revision_worker import RevisionWorker

logger = logging.getLogger(__name__)
RECENT_CONTEXT_MESSAGES = 6


class RebuildResult(BaseModel):
    """Concrete rebuild outcome returned by admin endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str = "rebuilt"
    user_id: str
    conversation_ids: list[str] = Field(default_factory=list)
    workspace_ids: list[str] = Field(default_factory=list)
    processed_messages: int = 0
    extract_jobs_processed: int = 0
    contract_jobs_processed: int = 0
    revision_jobs_processed: int = 0
    conversation_compaction_jobs_processed: int = 0
    workspace_rollup_jobs_processed: int = 0


class AdminRebuildService:
    """Replay durable rebuild flows end-to-end without orphan queue names."""

    def __init__(
        self,
        *,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        embedding_index: EmbeddingIndex | None,
        clock: Clock,
        manifest_loader: ManifestLoader,
        settings: Settings | None = None,
        storage_backend: StorageBackend | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._llm_client = llm_client
        self._embedding_index = embedding_index or NoneBackend()
        self._manifest_loader = manifest_loader
        self._settings = settings or Settings.from_env()
        self._cache_storage_backend = storage_backend
        self._conversation_repository = ConversationRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)

    async def rebuild_conversation(self, user_id: str, conversation_id: str) -> RebuildResult:
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ValueError(f"Unknown conversation_id: {conversation_id}")

        workspace_id = str(conversation["workspace_id"]) if conversation.get("workspace_id") else None
        await self._invalidate_user_cache(user_id)
        try:
            await self._purge_conversation_state(user_id, conversation_id)

            result = RebuildResult(
                user_id=user_id,
                conversation_ids=[conversation_id],
                workspace_ids=[workspace_id] if workspace_id is not None else [],
            )
            await self._rebuild_conversations([conversation], result)
            return result
        finally:
            await self._invalidate_user_cache(user_id)

    async def rebuild_user(self, user_id: str) -> RebuildResult:
        conversations = await self._list_user_conversations(user_id)
        workspace_ids = await self._list_user_workspaces(user_id)
        await self._invalidate_user_cache(user_id)
        try:
            await self._purge_user_state(user_id)

            result = RebuildResult(
                user_id=user_id,
                conversation_ids=[str(row["id"]) for row in conversations],
                workspace_ids=workspace_ids,
            )
            await self._rebuild_conversations(conversations, result)
            return result
        finally:
            await self._invalidate_user_cache(user_id)

    async def _rebuild_conversations(
        self,
        conversations: list[dict[str, Any]],
        result: RebuildResult,
    ) -> None:
        rebuild_backend = InProcessBackend()
        try:
            ingest_worker = IngestWorker(
                storage_backend=rebuild_backend,
                connection=self._connection,
                llm_client=self._llm_client,
                clock=self._clock,
                manifest_loader=self._manifest_loader,
                embedding_index=self._embedding_index,
                settings=self._settings,
            )
            contract_worker = ContractWorker(
                storage_backend=rebuild_backend,
                connection=self._connection,
                llm_client=self._llm_client,
                clock=self._clock,
                manifest_loader=self._manifest_loader,
                settings=self._settings,
            )
            revision_worker = RevisionWorker(
                storage_backend=rebuild_backend,
                connection=self._connection,
                llm_client=self._llm_client,
                clock=self._clock,
                settings=self._settings,
            )
            compaction_worker = CompactionWorker(
                storage_backend=rebuild_backend,
                connection=self._connection,
                llm_client=self._llm_client,
                clock=self._clock,
                settings=self._settings,
            )

            for conversation in conversations:
                messages = await self._message_repository.get_messages(
                    str(conversation["id"]),
                    str(conversation["user_id"]),
                    limit=5000,
                    offset=0,
                )
                for index, message in enumerate(messages):
                    message_role = str(message["role"])
                    recent_messages = [
                        {
                            "role": str(item["role"]),
                            "content": str(item["text"]),
                        }
                        for item in messages[max(0, index - RECENT_CONTEXT_MESSAGES) : index]
                    ]
                    payload = MessageJobPayload(
                        message_id=str(message["id"]),
                        message_text=str(message["text"]),
                        message_occurred_at=resolve_message_occurred_at(message),
                        role=message_role,
                        assistant_mode_id=str(conversation["assistant_mode_id"]),
                        workspace_id=(
                            str(conversation["workspace_id"])
                            if conversation.get("workspace_id") is not None
                            else None
                        ),
                        recent_messages=recent_messages,
                    )
                    await ingest_worker.process_job(
                        self._message_job(
                            user_id=str(conversation["user_id"]),
                            conversation_id=str(conversation["id"]),
                            payload=payload,
                            job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
                        )
                    )
                    result.processed_messages += 1
                    result.extract_jobs_processed += 1
                    if message_role == "user":
                        await contract_worker.process_job(
                            self._message_job(
                                user_id=str(conversation["user_id"]),
                                conversation_id=str(conversation["id"]),
                                payload=payload,
                                job_type=JobType.PROJECT_CONTRACT,
                            )
                        )
                        result.contract_jobs_processed += 1

            revision_results = await self._drain_stream(
                rebuild_backend,
                REVISE_STREAM_NAME,
                revision_worker.process_job,
            )
            result.revision_jobs_processed += len(revision_results)

            compaction_results = await self._drain_stream(
                rebuild_backend,
                COMPACT_STREAM_NAME,
                compaction_worker.process_job,
            )
            result.conversation_compaction_jobs_processed += len(compaction_results)

            for workspace_id in result.workspace_ids:
                await compaction_worker.process_job(
                    self._workspace_rollup_job(
                        user_id=result.user_id,
                        workspace_id=workspace_id,
                    )
                )
                result.workspace_rollup_jobs_processed += 1
        finally:
            await rebuild_backend.close()

    async def _invalidate_user_cache(self, user_id: str) -> None:
        if self._cache_storage_backend is None:
            return
        await self._cache_storage_backend.delete_context_views_for_user(user_id)

    @staticmethod
    def _message_job(
        *,
        user_id: str,
        conversation_id: str,
        payload: MessageJobPayload,
        job_type: JobType,
    ) -> dict[str, Any]:
        return JobEnvelope(
            job_id=f"job_rebuild_{job_type.value}_{payload.message_id}",
            job_type=job_type,
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=[payload.message_id],
            payload=payload.model_dump(mode="json"),
            created_at=None,
        ).model_dump(mode="json")

    @staticmethod
    def _workspace_rollup_job(*, user_id: str, workspace_id: str) -> dict[str, Any]:
        return JobEnvelope(
            job_id=f"job_rebuild_workspace_{workspace_id}",
            job_type=JobType.COMPACT_SUMMARIES,
            user_id=user_id,
            payload=CompactionJobPayload(
                user_id=user_id,
                workspace_id=workspace_id,
                conversation_id=None,
                job_kind=CompactionJobKind.WORKSPACE_ROLLUP,
            ).model_dump(mode="json"),
            created_at=None,
        ).model_dump(mode="json")

    async def _drain_stream(
        self,
        storage_backend: InProcessBackend,
        stream_name: str,
        handler: Callable[[dict[str, object]], Awaitable[dict[str, Any] | None]],
    ) -> list[dict[str, Any] | None]:
        await storage_backend.stream_ensure_group(stream_name, WORKER_GROUP_NAME)
        results: list[dict[str, Any] | None] = []
        while True:
            messages = await storage_backend.stream_read(
                stream_name,
                WORKER_GROUP_NAME,
                consumer_name="admin-rebuild",
                count=1,
                block_ms=0,
            )
            if not messages:
                return results
            for message in messages:
                result = await handler(message.payload)
                await storage_backend.stream_ack(
                    stream_name,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                results.append(result)

    async def _purge_conversation_state(self, user_id: str, conversation_id: str) -> None:
        memory_ids = await self._memory_ids_for_conversation(user_id, conversation_id)
        try:
            await self._connection.execute("BEGIN")
            await self._connection.execute(
                """
                DELETE FROM consequence_chains
                WHERE user_id = ?
                  AND conversation_id = ?
                """,
                (user_id, conversation_id),
            )
            await self._connection.execute(
                """
                DELETE FROM summary_views
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            await self._delete_memory_ids(user_id, memory_ids)
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        await self._delete_embeddings(memory_ids)

    async def _purge_user_state(self, user_id: str) -> None:
        memory_ids = await self._memory_ids_for_user(user_id)
        try:
            await self._connection.execute("BEGIN")
            await self._connection.execute(
                """
                DELETE FROM consequence_chains
                WHERE user_id = ?
                """,
                (user_id,),
            )
            await self._connection.execute(
                """
                DELETE FROM summary_views
                WHERE conversation_id IN (
                    SELECT id
                    FROM conversations
                    WHERE user_id = ?
                )
                   OR workspace_id IN (
                    SELECT id
                    FROM workspaces
                    WHERE user_id = ?
                )
                """,
                (user_id, user_id),
            )
            await self._delete_memory_ids(user_id, memory_ids)
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        await self._delete_embeddings(memory_ids)

    async def _delete_memory_ids(self, user_id: str, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        for chunk_start in range(0, len(memory_ids), 200):
            chunk = memory_ids[chunk_start : chunk_start + 200]
            placeholders = ", ".join("?" for _ in chunk)
            await self._connection.execute(
                """
                DELETE FROM memory_objects
                WHERE user_id = ?
                  AND id IN ({placeholders})
                """.format(placeholders=placeholders),
                (user_id, *chunk),
            )

    async def _delete_embeddings(self, memory_ids: list[str]) -> None:
        for memory_id in memory_ids:
            try:
                await self._embedding_index.delete(memory_id)
            except Exception:
                logger.warning("Embedding cleanup failed for memory_id=%s", memory_id, exc_info=True)

    async def _memory_ids_for_conversation(self, user_id: str, conversation_id: str) -> list[str]:
        cursor = await self._connection.execute(
            """
            SELECT DISTINCT mo.id
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND (
                  mo.conversation_id = ?
                  OR EXISTS (
                      SELECT 1
                      FROM json_each(mo.payload_json, '$.source_message_ids') AS source_ids
                      JOIN messages AS m ON m.id = CAST(source_ids.value AS TEXT)
                      JOIN conversations AS c ON c.id = m.conversation_id
                      WHERE c.id = ?
                        AND c.user_id = ?
                  )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM json_each(mo.payload_json, '$.source_message_ids') AS source_ids
                  JOIN messages AS m ON m.id = CAST(source_ids.value AS TEXT)
                  JOIN conversations AS c ON c.id = m.conversation_id
                  WHERE c.user_id = ?
                    AND c.id != ?
              )
            ORDER BY mo.id ASC
            """,
            (user_id, conversation_id, conversation_id, user_id, user_id, conversation_id),
        )
        rows = await cursor.fetchall()
        return [str(row["id"]) for row in rows]

    async def _memory_ids_for_user(self, user_id: str) -> list[str]:
        cursor = await self._connection.execute(
            """
            SELECT id
            FROM memory_objects
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["id"]) for row in rows]

    async def _list_user_conversations(self, user_id: str) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM conversations
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _list_user_workspaces(self, user_id: str) -> list[str]:
        cursor = await self._connection.execute(
            """
            SELECT id
            FROM workspaces
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [str(row["id"]) for row in rows]
