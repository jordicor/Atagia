"""Contract projection worker."""

from __future__ import annotations

import asyncio
import hashlib
import logging

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.storage_backend import StorageBackend
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    JobEnvelope,
    StreamMessage,
    JobType,
    MessageJobPayload,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.llm_client import LLMClient

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
CONTRACT_DEDUPE_TTL_SECONDS = 60 * 60 * 24
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3


class ContractWorker:
    """Consumes contract projection jobs from the configured stream backend."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[object],
        clock: Clock,
        manifest_loader: ManifestLoader,
        settings: Settings | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._manifest_loader = manifest_loader
        self._policy_resolver = PolicyResolver()
        self._projector = ContractProjector(
            llm_client=llm_client,
            clock=clock,
            message_repository=MessageRepository(connection, clock),
            memory_repository=MemoryObjectRepository(connection, clock),
            contract_repository=ContractDimensionRepository(connection, clock),
            settings=settings,
        )

    async def run(self, consumer_name: str = "contract-1") -> None:
        await self._storage_backend.stream_ensure_group(CONTRACT_STREAM_NAME, WORKER_GROUP_NAME)
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in contract worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "contract-1",
        block_ms: int | None = 0,
    ) -> WorkerIterationResult:
        messages = await self._next_messages(
            consumer_name=consumer_name,
            block_ms=block_ms,
        )
        if not messages:
            return WorkerIterationResult()

        acked = 0
        failed = 0
        dead_lettered = 0
        for message in messages:
            try:
                await self.process_job(message.payload)
                await self._storage_backend.stream_ack(
                    CONTRACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                logger.exception("Failed to process contract job %s", message.message_id)
                if await self._dead_letter_if_exhausted(message, exc):
                    dead_lettered += 1
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            dead_lettered=dead_lettered,
        )

    async def process_job(self, payload: dict[str, object]) -> None:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.PROJECT_CONTRACT:
            raise ValueError(f"Unsupported contract job type: {envelope.job_type}")
        if envelope.conversation_id is None:
            raise ValueError("Contract projection jobs require conversation_id")
        job_payload = MessageJobPayload.model_validate(envelope.payload)
        dedupe_key = self._contract_dedupe_key(
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            message_id=job_payload.message_id,
        )
        lock_token = await self._storage_backend.acquire_lock(f"{dedupe_key}:lock", ttl_seconds=60)
        if lock_token is None:
            return
        try:
            if await self._storage_backend.has_dedupe(dedupe_key):
                return

            manifest = self._manifest_loader.get(job_payload.assistant_mode_id)
            resolved_policy = self._policy_resolver.resolve(manifest, None, None)
            context = ExtractionConversationContext(
                user_id=envelope.user_id,
                conversation_id=envelope.conversation_id,
                source_message_id=job_payload.message_id,
                workspace_id=job_payload.workspace_id,
                assistant_mode_id=job_payload.assistant_mode_id,
                recent_messages=[
                    ExtractionContextMessage.model_validate(item)
                    for item in job_payload.recent_messages
                ],
            )
            await self._projector.project(
                message_text=job_payload.message_text,
                role=job_payload.role,
                conversation_context=context,
                resolved_policy=resolved_policy,
                user_id=envelope.user_id,
                occurred_at=job_payload.message_occurred_at,
            )
            await self._storage_backend.remember_dedupe(
                dedupe_key,
                CONTRACT_DEDUPE_TTL_SECONDS,
            )
        finally:
            await self._storage_backend.release_lock(f"{dedupe_key}:lock", lock_token)

    @staticmethod
    def _contract_dedupe_key(*, user_id: str, conversation_id: str, message_id: str) -> str:
        raw_key = f"{user_id}:{conversation_id}:{message_id}:contract"
        return f"contract:{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}"

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            CONTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            CONTRACT_STREAM_NAME,
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
            f"dead_letter:{CONTRACT_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
            },
        )
        await self._storage_backend.stream_ack(
            CONTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        return True
