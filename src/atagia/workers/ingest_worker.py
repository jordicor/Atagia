"""Ingest worker for durable extraction jobs."""

from __future__ import annotations

import asyncio
import json
import logging

import aiosqlite

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.ids import new_job_id
from atagia.core.storage_backend import StorageBackend
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.consequence_builder import ConsequenceChainBuilder
from atagia.memory.consequence_detector import ConsequenceDetector
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.intent_classifier import are_claim_keys_equivalent
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, ResolvedPolicy
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CompactionJobKind,
    CompactionJobPayload,
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    JobType,
    MessageJobPayload,
    REVISE_STREAM_NAME,
    RevisionJobPayload,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
    ExtractionResult,
    MemoryObjectType,
)
from atagia.services.llm_client import LLMClient, StructuredOutputError
from atagia.services.embeddings import EmbeddingIndex

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3
CONSEQUENCE_CONFIDENCE_THRESHOLD = 0.5
CONVERSATION_CHUNK_TRIGGER_MESSAGES = 10


class IngestWorker:
    """Consumes extraction jobs from the configured stream backend."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[object],
        clock: Clock,
        manifest_loader: ManifestLoader,
        embedding_index: EmbeddingIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._storage_backend = storage_backend
        self._manifest_loader = manifest_loader
        self._policy_resolver = PolicyResolver()
        self._llm_client = llm_client
        self._message_repository = MessageRepository(connection, clock)
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._belief_repository = BeliefRepository(connection, clock)
        self._summary_repository = SummaryRepository(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._settings = resolved_settings
        self._classifier_model = (
            resolved_settings.llm_classifier_model
            or resolved_settings.llm_scoring_model
            or resolved_settings.llm_extraction_model
            or "claude-sonnet-4-6"
        )
        self._extractor = MemoryExtractor(
            llm_client=llm_client,
            clock=clock,
            message_repository=self._message_repository,
            memory_repository=self._memory_repository,
            storage_backend=storage_backend,
            embedding_index=embedding_index,
            settings=resolved_settings,
        )
        self._consequence_detector = ConsequenceDetector(
            llm_client=llm_client,
            clock=clock,
            settings=resolved_settings,
        )
        self._consequence_builder = ConsequenceChainBuilder(
            connection=connection,
            llm_client=llm_client,
            clock=clock,
            settings=resolved_settings,
        )

    async def run(self, consumer_name: str = "ingest-1") -> None:
        await self._storage_backend.stream_ensure_group(EXTRACT_STREAM_NAME, WORKER_GROUP_NAME)
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in ingest worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "ingest-1",
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
                    EXTRACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                logger.exception("Failed to process extraction job %s", message.message_id)
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
        if envelope.job_type is not JobType.EXTRACT_MEMORY_CANDIDATES:
            raise ValueError(f"Unsupported ingest job type: {envelope.job_type}")
        if envelope.conversation_id is None:
            raise ValueError("Extraction jobs require conversation_id")
        job_payload = MessageJobPayload.model_validate(envelope.payload)
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
        try:
            result, persisted = await self._extractor.extract_with_persistence_details(
                message_text=job_payload.message_text,
                role=job_payload.role,
                conversation_context=context,
                resolved_policy=resolved_policy,
                occurred_at=job_payload.message_occurred_at,
            )
        except StructuredOutputError as exc:
            if "after schema fallback" not in str(exc):
                raise
            logger.warning(
                "Skipping extraction job %s after schema fallback returned non-JSON output",
                job_payload.message_id,
                exc_info=True,
            )
            result = ExtractionResult(nothing_durable=True)
            persisted = []
        if not result.nothing_durable and not self._settings.skip_belief_revision:
            await self._emit_revision_jobs(
                envelope=envelope,
                job_payload=job_payload,
                persisted=persisted,
            )
        if job_payload.role == "user":
            await self._process_consequence_detection(
                job_payload=job_payload,
                context=context,
                resolved_policy=resolved_policy,
                user_id=envelope.user_id,
            )
        await self._maybe_enqueue_conversation_compaction(
            envelope=envelope,
            job_payload=job_payload,
        )

    async def _emit_revision_jobs(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: MessageJobPayload,
        persisted: list[dict[str, object]],
    ) -> None:
        evidence_rows = [
            row for row in persisted if row.get("object_type") == MemoryObjectType.EVIDENCE.value
        ]
        belief_rows = [
            row for row in persisted if row.get("object_type") == MemoryObjectType.BELIEF.value
        ]
        evidence_ids = [str(row["id"]) for row in evidence_rows]

        for belief_row in belief_rows:
            payload_json = belief_row.get("payload_json")
            if not isinstance(payload_json, dict):
                continue
            claim_key = payload_json.get("claim_key")
            if not isinstance(claim_key, str) or not claim_key:
                continue
            existing_belief_id = await self._find_existing_belief_id(
                claim_key=claim_key,
                current_belief_id=str(belief_row["id"]),
                source_message_id=job_payload.message_id,
                assistant_mode_id=job_payload.assistant_mode_id,
                workspace_id=job_payload.workspace_id,
                conversation_id=envelope.conversation_id,
                scope=str(belief_row["scope"]),
                user_id=envelope.user_id,
            )
            await self._enqueue_revision_job(
                envelope=envelope,
                payload=RevisionJobPayload(
                    belief_id=existing_belief_id or "",
                    claim_key=claim_key,
                    claim_value=json.dumps(
                        payload_json.get("claim_value"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    evidence_memory_ids=evidence_ids,
                    source_message_id=job_payload.message_id,
                    user_id=envelope.user_id,
                    assistant_mode_id=job_payload.assistant_mode_id,
                    workspace_id=job_payload.workspace_id,
                    conversation_id=envelope.conversation_id,
                    scope=str(belief_row["scope"]),
                ),
            )

        for evidence_row in evidence_rows:
            payload_json = evidence_row.get("payload_json")
            if not isinstance(payload_json, dict):
                continue
            claim_key = payload_json.get("claim_key")
            if not isinstance(claim_key, str) or not claim_key:
                continue
            await self._enqueue_revision_job(
                envelope=envelope,
                payload=RevisionJobPayload(
                    belief_id="",
                    claim_key=claim_key,
                    claim_value=json.dumps(
                        payload_json.get("claim_value"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    evidence_memory_ids=[str(evidence_row["id"])],
                    source_message_id=job_payload.message_id,
                    user_id=envelope.user_id,
                    assistant_mode_id=job_payload.assistant_mode_id,
                    workspace_id=job_payload.workspace_id,
                    conversation_id=envelope.conversation_id,
                    scope=str(evidence_row["scope"]),
                ),
            )

    async def _enqueue_revision_job(
        self,
        *,
        envelope: JobEnvelope,
        payload: RevisionJobPayload,
    ) -> None:
        revision_job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.REVISE_BELIEFS,
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            message_ids=[payload.source_message_id],
            payload=payload.model_dump(mode="json"),
            created_at=envelope.created_at,
        )
        await self._storage_backend.stream_add(
            REVISE_STREAM_NAME,
            revision_job.model_dump(mode="json"),
        )

    async def _maybe_enqueue_conversation_compaction(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: MessageJobPayload,
    ) -> None:
        if self._settings.skip_compaction:
            return
        if envelope.conversation_id is None:
            return

        message_count = await self._conversation_message_count(
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
        )
        latest_chunk = await self._summary_repository.get_latest_conversation_chunk(
            envelope.user_id,
            envelope.conversation_id,
        )
        last_chunk_end_seq = 0 if latest_chunk is None else int(latest_chunk["source_message_end_seq"])
        if message_count - last_chunk_end_seq < CONVERSATION_CHUNK_TRIGGER_MESSAGES:
            return

        compaction_job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.COMPACT_SUMMARIES,
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            message_ids=[job_payload.message_id],
            payload=CompactionJobPayload(
                user_id=envelope.user_id,
                workspace_id=job_payload.workspace_id,
                conversation_id=envelope.conversation_id,
                job_kind=CompactionJobKind.CONVERSATION_CHUNK,
            ).model_dump(mode="json"),
            created_at=envelope.created_at,
        )
        await self._storage_backend.stream_add(
            COMPACT_STREAM_NAME,
            compaction_job.model_dump(mode="json"),
        )

    async def _process_consequence_detection(
        self,
        *,
        job_payload: MessageJobPayload,
        context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        user_id: str,
    ) -> None:
        recent_assistant_messages = await self._recent_assistant_messages(
            user_id=user_id,
            conversation_id=context.conversation_id,
            source_message_id=context.source_message_id,
        )
        signal = await self._consequence_detector.detect(
            message_text=job_payload.message_text,
            role=job_payload.role,
            conversation_context=context,
            recent_assistant_messages=recent_assistant_messages,
        )
        if signal is None or signal.confidence < CONSEQUENCE_CONFIDENCE_THRESHOLD:
            return
        chain = await self._consequence_builder.build_chain(
            signal=signal,
            user_id=user_id,
            conversation_context=context,
            resolved_policy=resolved_policy,
        )
        if chain is not None:
            logger.info(
                "Built consequence chain %s for source message %s",
                chain.chain_id,
                context.source_message_id,
            )

    async def _recent_assistant_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
        source_message_id: str,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        source_message = await self._message_repository.get_message(source_message_id, user_id)
        if source_message is None:
            return []
        cursor = await self._connection.execute(
            """
            SELECT m.id, m.text, m.seq
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.role = 'assistant'
              AND m.seq < ?
            ORDER BY m.seq DESC
            LIMIT ?
            """,
            (
                conversation_id,
                user_id,
                source_message["seq"],
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return [
            {"id": row["id"], "text": row["text"], "seq": row["seq"]}
            for row in reversed(rows)
        ]

    async def _find_existing_belief_id(
        self,
        *,
        claim_key: str,
        current_belief_id: str,
        source_message_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        scope: str,
        user_id: str,
    ) -> str | None:
        candidates = await self._belief_repository.find_active_belief_candidates_by_claim_key(
            user_id,
            claim_key,
        )
        ranked: list[tuple[int, str]] = []
        for candidate in candidates:
            if not await are_claim_keys_equivalent(
                self._llm_client,
                self._classifier_model,
                claim_key,
                str(candidate["claim_key"]),
            ):
                continue
            belief_id = str(candidate["belief_id"])
            if belief_id == current_belief_id:
                continue
            payload_json = candidate.get("payload_json")
            source_ids = payload_json.get("source_message_ids", []) if isinstance(payload_json, dict) else []
            if source_message_id in source_ids:
                continue
            score = 0
            if candidate.get("scope") == scope:
                score += 4
            if conversation_id is not None and candidate.get("conversation_id") == conversation_id:
                score += 3
            if workspace_id is not None and candidate.get("workspace_id") == workspace_id:
                score += 2
            if candidate.get("assistant_mode_id") == assistant_mode_id:
                score += 1
            ranked.append((score, belief_id))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1]

    async def _conversation_message_count(self, *, user_id: str, conversation_id: str) -> int:
        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
            """,
            (conversation_id, user_id),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            EXTRACT_STREAM_NAME,
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
            f"dead_letter:{EXTRACT_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
                # Populated only for StructuredOutputError; other exceptions collapse to [].
                "error_details": list(exc.details) if isinstance(exc, StructuredOutputError) else [],
            },
        )
        await self._storage_backend.stream_ack(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        return True
