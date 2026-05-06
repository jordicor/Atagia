"""Summary compaction worker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.ids import new_job_id
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.storage_backend import StorageBackend
from atagia.memory.compactor import Compactor
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CompactionJobKind,
    CompactionJobPayload,
    JobEnvelope,
    JobType,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import ConversationStatus
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import LLMClient, StructuredOutputError, TransientLLMError
from atagia.services.worker_control_service import WorkerControlService, wait_if_worker_claims_paused

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3


class CompactionWorker:
    """Consumes compaction jobs from the configured stream backend."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        clock: Clock,
        embedding_index: EmbeddingIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._connection = connection
        self._clock = clock
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._worker_control = WorkerControlService(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._job_tracking = JobTrackingService(
            connection,
            clock,
            workers_enabled=resolved_settings.workers_enabled,
            settings=resolved_settings,
        )
        self._compactor = Compactor(
            connection=connection,
            llm_client=llm_client,
            clock=clock,
            embedding_index=embedding_index,
            settings=resolved_settings,
        )

    async def run(self, consumer_name: str = "compact-1") -> None:
        await self._storage_backend.stream_ensure_group(COMPACT_STREAM_NAME, WORKER_GROUP_NAME)
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in compaction worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "compact-1",
        block_ms: int | None = 0,
    ) -> WorkerIterationResult:
        if await wait_if_worker_claims_paused(self._worker_control, block_ms=block_ms):
            return WorkerIterationResult()
        messages = await self._next_messages(consumer_name=consumer_name, block_ms=block_ms)
        if not messages:
            return WorkerIterationResult()

        acked = 0
        failed = 0
        dead_lettered = 0
        for message in messages:
            try:
                await self._job_tracking.mark_running(message)
                await self.process_job(message.payload)
                await self._job_tracking.mark_succeeded(message)
                await self._storage_backend.stream_ack(
                    COMPACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                self._log_job_failure(message, exc)
                if await self._dead_letter_if_exhausted(message, exc):
                    dead_lettered += 1
                else:
                    await self._job_tracking.mark_retrying(message, exc)
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            dead_lettered=dead_lettered,
        )

    @staticmethod
    def _log_job_failure(message: StreamMessage, exc: Exception) -> None:
        if isinstance(exc, StructuredOutputError):
            details = "; ".join(exc.details) if exc.details else str(exc)
            logger.warning(
                "Failed to process compaction job %s due to structured output: %s",
                message.message_id,
                details,
            )
            return
        if isinstance(exc, TransientLLMError):
            logger.warning(
                "Failed to process compaction job %s due to transient provider error: %s",
                message.message_id,
                exc,
            )
            return
        logger.exception("Failed to process compaction job %s", message.message_id)

    async def process_job(self, payload: dict[str, object]) -> dict[str, Any] | None:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.COMPACT_SUMMARIES:
            raise ValueError(f"Unsupported compaction job type: {envelope.job_type}")
        job_payload = CompactionJobPayload.model_validate(envelope.payload)
        if job_payload.job_kind is CompactionJobKind.CONVERSATION_CHUNK:
            if job_payload.conversation_id is None:
                raise ValueError("conversation_chunk jobs require conversation_id")
            if (
                job_payload.temporary
                or job_payload.purge_on_close
                or not job_payload.remember_across_devices
            ):
                return {"job_kind": job_payload.job_kind.value, "summary_ids": []}
            if not await self._conversation_allows_local_chunk(
                user_id=job_payload.user_id,
                conversation_id=job_payload.conversation_id,
            ):
                return {"job_kind": job_payload.job_kind.value, "summary_ids": []}
            summary_ids = await self._compactor.generate_conversation_chunks(
                user_id=job_payload.user_id,
                conversation_id=job_payload.conversation_id,
            )
            if (
                not await self._broad_hierarchy_blocked(job_payload)
                and await self._conversation_allows_hierarchy(
                    user_id=job_payload.user_id,
                    conversation_id=job_payload.conversation_id,
                )
            ):
                await self._enqueue_hierarchy_job(
                    user_id=job_payload.user_id,
                    job_kind=CompactionJobKind.EPISODE,
                    parent=envelope,
                )
            return {"job_kind": job_payload.job_kind.value, "summary_ids": summary_ids}
        if job_payload.job_kind is CompactionJobKind.WORKSPACE_ROLLUP:
            character_id = job_payload.character_id or job_payload.workspace_id
            if character_id is None:
                raise ValueError("workspace_rollup jobs require workspace_id or character_id")
            if await self._broad_hierarchy_blocked(job_payload):
                return {"job_kind": job_payload.job_kind.value, "summary_id": None}
            summary_id = await self._compactor.generate_character_rollup(
                user_id=job_payload.user_id,
                character_id=character_id,
                workspace_id=job_payload.workspace_id,
            )
            return {"job_kind": job_payload.job_kind.value, "summary_id": summary_id}
        if job_payload.job_kind is CompactionJobKind.EPISODE:
            if await self._broad_hierarchy_blocked(job_payload):
                return {"job_kind": job_payload.job_kind.value, "summary_ids": []}
            summary_ids = await self._compactor.generate_episodes(job_payload.user_id)
            await self._enqueue_hierarchy_job(
                user_id=job_payload.user_id,
                job_kind=CompactionJobKind.THEMATIC_PROFILE,
                parent=envelope,
            )
            return {"job_kind": job_payload.job_kind.value, "summary_ids": summary_ids}
        if job_payload.job_kind is CompactionJobKind.THEMATIC_PROFILE:
            if await self._broad_hierarchy_blocked(job_payload):
                return {"job_kind": job_payload.job_kind.value, "summary_ids": []}
            summary_ids = await self._compactor.generate_thematic_profiles(job_payload.user_id)
            return {"job_kind": job_payload.job_kind.value, "summary_ids": summary_ids}
        raise ValueError(f"Unsupported compaction job_kind: {job_payload.job_kind}")

    async def _enqueue_hierarchy_job(
        self,
        *,
        user_id: str,
        job_kind: CompactionJobKind,
        parent: JobEnvelope,
    ) -> None:
        job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.COMPACT_SUMMARIES,
            user_id=user_id,
            payload=CompactionJobPayload(
                **{
                    **CompactionJobPayload.model_validate(parent.payload).model_dump(mode="json"),
                    "user_id": user_id,
                    "job_kind": job_kind.value,
                }
            ).model_dump(mode="json"),
            created_at=None,
            operational_profile=parent.operational_profile,
        )
        await self._job_tracking.create_queued_job(COMPACT_STREAM_NAME, job)
        try:
            await self._storage_backend.stream_add(
                COMPACT_STREAM_NAME,
                job.model_dump(mode="json"),
            )
        except Exception as exc:
            await self._job_tracking.mark_enqueue_failed(job, exc)
            raise

    async def _broad_hierarchy_blocked(self, payload: CompactionJobPayload) -> bool:
        active_user = await self._user_repository.get_active_user(payload.user_id)
        if active_user is None:
            return True
        return (
            payload.temporary
            or payload.purge_on_close
            or payload.incognito
            or not payload.remember_across_chats
            or not payload.remember_across_devices
            or not bool(active_user["remember_across_chats"])
            or not bool(active_user["remember_across_devices"])
        )

    async def _conversation_allows_local_chunk(self, *, user_id: str, conversation_id: str) -> bool:
        active_user = await self._user_repository.get_active_user(user_id)
        if active_user is None or not bool(active_user["remember_across_devices"]):
            return False
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            return False
        if bool(conversation.get("temporary")) or bool(conversation.get("purge_on_close")):
            return False
        return str(conversation.get("status")) == ConversationStatus.ACTIVE.value

    async def _conversation_allows_hierarchy(self, *, user_id: str, conversation_id: str) -> bool:
        active_user = await self._user_repository.get_active_user(user_id)
        if (
            active_user is None
            or not bool(active_user["remember_across_chats"])
            or not bool(active_user["remember_across_devices"])
        ):
            return False
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            return False
        if (
            bool(conversation.get("temporary"))
            or bool(conversation.get("purge_on_close"))
            or bool(conversation.get("incognito"))
            or bool(conversation.get("isolated_mode"))
        ):
            return False
        return str(conversation.get("status")) == ConversationStatus.ACTIVE.value

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            COMPACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            COMPACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            count=1,
            block_ms=block_ms,
        )

    async def _dead_letter_if_exhausted(
        self,
        message: StreamMessage,
        exc: Exception,
    ) -> bool:
        if message.delivery_count < MAX_STREAM_DELIVERIES:
            return False
        await self._storage_backend.enqueue_job(
            f"dead_letter:{COMPACT_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
                "error_details": (
                    list(exc.details)
                    if isinstance(exc, StructuredOutputError)
                    else []
                ),
            },
        )
        await self._storage_backend.stream_ack(
            COMPACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        await self._job_tracking.mark_dead_lettered(message, exc)
        return True
