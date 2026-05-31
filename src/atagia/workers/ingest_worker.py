"""Ingest worker for durable extraction jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import Clock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.ids import new_job_id
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.storage_backend import StorageBackend
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.consequence_builder import ConsequenceChainBuilder
from atagia.memory.consequence_detector import ConsequenceDetector
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.intent_classifier import are_claim_keys_equivalent
from atagia.memory.language_profile import UserCommunicationProfileService
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, ResolvedRetrievalPolicy
from atagia.memory.retrieval_surface_dry_run import (
    RetrievalSurfaceDryRunGenerator,
    RetrievalSurfaceWriter,
)
from atagia.models.schemas_jobs import (
    COMPACT_STREAM_NAME,
    CompactionJobKind,
    CompactionJobPayload,
    EXTRACT_STREAM_NAME,
    GRAPH_STREAM_NAME,
    GraphProjectionChunkPayload,
    GraphProjectionJobPayload,
    InitialContextPackageRefreshReason,
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
    ConversationStatus,
    MemoryObjectType,
)
from atagia.memory.text_chunker import ChunkingPlan, TextChunk
from atagia.services.chat_support import apply_conversation_policy_overlay
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import LLMClient, LLMError, StructuredOutputError, TransientLLMError
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.initial_context_package_refresh_service import (
    InitialContextPackageRefreshEnqueuer,
)
from atagia.services.model_resolution import resolve_component_model
from atagia.services.worker_control_service import WorkerControlService, wait_if_worker_claims_paused
from atagia.services.topic_working_set_service import TopicWorkingSetRefreshService

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3
CONSEQUENCE_CONFIDENCE_THRESHOLD = 0.5
CONVERSATION_CHUNK_TRIGGER_MESSAGES = 10
COMPACTION_ENQUEUE_DEDUPE_TTL_SECONDS = 3600


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
        self._clock = clock
        self._message_repository = MessageRepository(connection, clock)
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._belief_repository = BeliefRepository(connection, clock)
        self._summary_repository = SummaryRepository(connection, clock)
        self._communication_profile_repository = CommunicationProfileRepository(
            connection,
            clock,
        )
        self._worker_control = WorkerControlService(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._job_tracking = JobTrackingService(
            connection,
            clock,
            workers_enabled=resolved_settings.workers_enabled,
            settings=resolved_settings,
        )
        self._initial_context_package_refresh = InitialContextPackageRefreshEnqueuer(
            storage_backend=storage_backend,
            clock=clock,
            job_tracking_service=self._job_tracking,
            package_repository=InitialContextPackageRepository(connection, clock),
            refresh_enabled=resolved_settings.initial_context_package_refresh_enabled,
        )
        self._settings = resolved_settings
        self._classifier_model = resolve_component_model(
            resolved_settings,
            "intent_classifier",
        )
        retrieval_packet_dry_run_generator = None
        retrieval_packet_surface_writer = None
        if resolved_settings.retrieval_packets_dry_run_enabled:
            retrieval_packet_dry_run_generator = RetrievalSurfaceDryRunGenerator(
                llm_client,
                clock,
                resolved_settings,
            )
            if resolved_settings.retrieval_packets_write_enabled:
                retrieval_packet_surface_writer = RetrievalSurfaceWriter(
                    MemoryRetrievalSurfaceRepository(connection, clock),
                    clock,
                )
        self._extractor = MemoryExtractor(
            llm_client=llm_client,
            clock=clock,
            message_repository=self._message_repository,
            memory_repository=self._memory_repository,
            storage_backend=storage_backend,
            embedding_index=embedding_index,
            settings=resolved_settings,
            retrieval_packet_dry_run_generator=retrieval_packet_dry_run_generator,
            enable_retrieval_packet_dry_run=(
                resolved_settings.retrieval_packets_dry_run_enabled
            ),
            retrieval_packet_surface_writer=retrieval_packet_surface_writer,
            enable_retrieval_packet_surface_write=(
                resolved_settings.retrieval_packets_dry_run_enabled
                and resolved_settings.retrieval_packets_write_enabled
            ),
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
        self._user_language_profile_service = UserCommunicationProfileService(
            llm_client=llm_client,
            clock=clock,
            profile_repository=self._communication_profile_repository,
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
        if await wait_if_worker_claims_paused(self._worker_control, block_ms=block_ms):
            return WorkerIterationResult()
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
                await self._job_tracking.mark_running(message)
                await self.process_job(message.payload)
                await self._job_tracking.mark_succeeded(message)
                await self._storage_backend.stream_ack(
                    EXTRACT_STREAM_NAME,
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
        if isinstance(exc, (StructuredOutputError, TransientLLMError)):
            logger.warning(
                "Failed to process extraction job %s: %s",
                message.message_id,
                exc,
            )
            return
        logger.exception("Failed to process extraction job %s", message.message_id)

    async def process_job(self, payload: dict[str, object]) -> None:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.EXTRACT_MEMORY_CANDIDATES:
            raise ValueError(f"Unsupported ingest job type: {envelope.job_type}")
        if envelope.conversation_id is None:
            raise ValueError("Extraction jobs require conversation_id")
        job_payload = MessageJobPayload.model_validate(envelope.payload)
        active_user = await self._user_repository.get_active_user(envelope.user_id)
        if active_user is None:
            return
        conversation = await self._conversation_repository.get_conversation(
            envelope.conversation_id,
            envelope.user_id,
        )
        if conversation is None or str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
            return
        manifest = self._manifest_loader.get(job_payload.assistant_mode_id)
        resolved_policy = self._policy_resolver.resolve(manifest, None, None)
        resolved_policy = apply_conversation_policy_overlay(resolved_policy, conversation)
        job_payload = job_payload.model_copy(
            update={
                "temporary": bool(conversation.get("temporary")) or bool(job_payload.temporary),
                "temporary_ttl_seconds": self._strictest_ttl(
                    job_payload.temporary_ttl_seconds,
                    conversation.get("temporary_ttl_seconds"),
                ),
                "purge_on_close": bool(conversation.get("purge_on_close")) or bool(job_payload.purge_on_close),
                "isolated_mode": bool(conversation.get("isolated_mode")) or bool(job_payload.isolated_mode),
                "incognito": bool(conversation.get("incognito")) or bool(job_payload.incognito),
                "active_embodiment_id": (
                    job_payload.active_embodiment_id
                    or conversation.get("active_embodiment_id")
                ),
                "cross_embodiment_mode": (
                    job_payload.cross_embodiment_mode
                    or conversation.get("cross_embodiment_mode")
                    or "direct_if_same_body"
                ),
                "active_realm_id": (
                    job_payload.active_realm_id
                    or conversation.get("active_realm_id")
                ),
                "cross_realm_mode": (
                    job_payload.cross_realm_mode
                    or conversation.get("cross_realm_mode")
                    or "none"
                ),
                "remember_across_chats": (
                    bool(job_payload.remember_across_chats)
                    and bool(active_user["remember_across_chats"])
                ),
                "remember_across_devices": (
                    bool(job_payload.remember_across_devices)
                    and bool(active_user["remember_across_devices"])
                ),
            }
        )
        character_id = job_payload.character_id
        if character_id is None:
            character_id = conversation.get("character_id") or conversation.get("workspace_id")
        context = ExtractionConversationContext(
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            source_message_id=job_payload.message_id,
            workspace_id=job_payload.workspace_id,
            assistant_mode_id=job_payload.assistant_mode_id,
            user_persona_id=(
                job_payload.user_persona_id
                if job_payload.user_persona_id is not None
                else conversation.get("user_persona_id")
            ),
            platform_id=str(job_payload.platform_id or conversation.get("platform_id") or "default"),
            character_id=character_id,
            active_presence_id=(
                job_payload.active_presence_id or conversation.get("active_presence_id")
            ),
            active_presence_kind=job_payload.active_presence_kind,
            active_presence_display_name=job_payload.active_presence_display_name,
            source_presence_id=job_payload.source_presence_id,
            source_presence_kind=job_payload.source_presence_kind,
            source_presence_display_name=job_payload.source_presence_display_name,
            active_space_id=(
                job_payload.active_space_id or conversation.get("active_space_id")
            ),
            active_space_boundary_mode=(
                job_payload.active_space_boundary_mode
                or conversation.get("active_space_boundary_mode")
                or "focus"
            ),
            active_space_display_name=(
                job_payload.active_space_display_name
                or conversation.get("active_space_display_name")
            ),
            active_mind_id=(
                job_payload.active_mind_id or conversation.get("active_mind_id")
            ),
            source_mind_id=(
                job_payload.source_mind_id
                or job_payload.active_mind_id
                or conversation.get("active_mind_id")
            ),
            active_mind_display_name=job_payload.active_mind_display_name,
            mind_topology=(
                job_payload.mind_topology
                or conversation.get("mind_topology")
                or "unimind"
            ),
            active_embodiment_id=(
                job_payload.active_embodiment_id
                or conversation.get("active_embodiment_id")
            ),
            active_embodiment_display_name=job_payload.active_embodiment_display_name,
            cross_embodiment_mode=(
                job_payload.cross_embodiment_mode
                or conversation.get("cross_embodiment_mode")
                or "direct_if_same_body"
            ),
            active_realm_id=(
                job_payload.active_realm_id
                or conversation.get("active_realm_id")
            ),
            active_realm_display_name=job_payload.active_realm_display_name,
            cross_realm_mode=(
                job_payload.cross_realm_mode
                or conversation.get("cross_realm_mode")
                or "none"
            ),
            mode=str(job_payload.mode or conversation.get("mode") or job_payload.assistant_mode_id),
            recent_messages=[
                ExtractionContextMessage.model_validate(item)
                for item in job_payload.recent_messages
            ],
            temporary=job_payload.temporary,
            temporary_ttl_seconds=job_payload.temporary_ttl_seconds,
            purge_on_close=job_payload.purge_on_close,
            isolated_mode=job_payload.isolated_mode,
            incognito=job_payload.incognito or job_payload.isolated_mode,
            remember_across_chats=job_payload.remember_across_chats,
            remember_across_devices=job_payload.remember_across_devices,
            ingest_origin=job_payload.ingest_origin,
            confirmation_strategy=job_payload.confirmation_strategy,
            memory_privacy_mode=job_payload.memory_privacy_mode,
            privacy_enforcement=job_payload.privacy_enforcement,
            authenticated_user_privilege_level=(
                job_payload.authenticated_user_privilege_level
            ),
            authenticated_user_is_atagia_master=(
                job_payload.authenticated_user_is_atagia_master
            ),
        )
        try:
            extraction_details = await self._extractor.extract_with_persistence_and_chunk_plan(
                message_text=job_payload.message_text,
                role=job_payload.role,
                conversation_context=context,
                resolved_policy=resolved_policy,
                occurred_at=job_payload.message_occurred_at,
            )
            result = extraction_details.result
            persisted = extraction_details.persisted
            chunk_plan = extraction_details.chunk_plan
        except StructuredOutputError as exc:
            if "after schema fallback" not in str(exc):
                raise
            details = "; ".join(exc.details) if exc.details else str(exc)
            logger.warning(
                "Skipping extraction job %s after schema fallback returned non-JSON output: %s",
                job_payload.message_id,
                details,
            )
            result = ExtractionResult(nothing_durable=True)
            persisted = []
            chunk_plan = None
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
        await self._maybe_enqueue_graph_projection(
            envelope=envelope,
            job_payload=job_payload,
            persisted=persisted,
            chunk_plan=chunk_plan,
        )
        await self._maybe_enqueue_conversation_compaction(
            envelope=envelope,
            job_payload=job_payload,
        )
        await self._maybe_refresh_topic_working_set(
            envelope=envelope,
            job_payload=job_payload,
        )
        if job_payload.role == "user":
            await self._update_user_communication_profile(
                job_payload=job_payload,
                context=context,
            )
        await self._enqueue_initial_context_package_refresh(
            envelope=envelope,
            job_payload=job_payload,
        )

    async def _update_user_communication_profile(
        self,
        *,
        job_payload: MessageJobPayload,
        context: ExtractionConversationContext,
    ) -> None:
        try:
            await self._user_language_profile_service.update_from_message(
                message_text=job_payload.message_text,
                role=job_payload.role,
                conversation_context=context,
                occurred_at=job_payload.message_occurred_at,
            )
        except Exception as exc:
            logger.warning(
                "user_language_profile_update_failed_skipping",
                extra={
                    "user_id": context.user_id,
                    "conversation_id": context.conversation_id,
                    "source_message_id": context.source_message_id,
                    "error": str(exc),
                },
            )

    async def _enqueue_initial_context_package_refresh(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: MessageJobPayload,
    ) -> None:
        if envelope.conversation_id is None:
            return
        await self._initial_context_package_refresh.enqueue_refresh(
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            retrieval_profile_id=job_payload.assistant_mode_id,
            reason=InitialContextPackageRefreshReason.MEMORY_EXTRACTION,
            source_message_ids=[job_payload.message_id],
            privacy_enforcement=job_payload.privacy_enforcement,
            operational_profile=envelope.operational_profile,
            fail_open=True,
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
                user_persona_id=(
                    belief_row.get("user_persona_id")
                    if isinstance(belief_row.get("user_persona_id"), str)
                    else job_payload.user_persona_id
                ),
                platform_id=str(belief_row.get("platform_id") or job_payload.platform_id or "default"),
                character_id=(
                    str(belief_row["character_id"])
                    if belief_row.get("character_id") is not None
                    else job_payload.character_id
                ),
                incognito=bool(job_payload.incognito),
                remember_across_chats=bool(job_payload.remember_across_chats),
                remember_across_devices=bool(job_payload.remember_across_devices),
                isolated_mode=bool(job_payload.isolated_mode),
                active_mind_id=job_payload.active_mind_id,
                mind_topology=job_payload.mind_topology,
                active_embodiment_id=job_payload.active_embodiment_id,
                active_realm_id=job_payload.active_realm_id,
            )
            await self._enqueue_revision_job(
                envelope=envelope,
                payload=RevisionJobPayload(
                    belief_id=existing_belief_id or "",
                    claim_key=claim_key,
                    claim_value=json_utils.dumps(
                        payload_json.get("claim_value"),
                        sort_keys=True,
                    ),
                    evidence_memory_ids=evidence_ids,
                    source_message_id=job_payload.message_id,
                    user_id=envelope.user_id,
                    assistant_mode_id=job_payload.assistant_mode_id,
                    workspace_id=job_payload.workspace_id,
                    conversation_id=envelope.conversation_id,
                    user_persona_id=belief_row.get("user_persona_id") if isinstance(belief_row.get("user_persona_id"), str) else job_payload.user_persona_id,
                    platform_id=str(belief_row.get("platform_id") or job_payload.platform_id or "default"),
                    character_id=(
                        str(belief_row["character_id"])
                        if belief_row.get("character_id") is not None
                        else job_payload.character_id
                    ),
                    active_mind_id=(
                        str(belief_row["memory_owner_id"])
                        if belief_row.get("memory_owner_id") is not None
                        else job_payload.active_mind_id
                    ),
                    source_mind_id=(
                        str(belief_row["source_mind_id"])
                        if belief_row.get("source_mind_id") is not None
                        else job_payload.source_mind_id
                    ),
                    mind_topology=job_payload.mind_topology,
                    active_embodiment_id=(
                        str(belief_row["embodiment_id"])
                        if belief_row.get("embodiment_id") is not None
                        else job_payload.active_embodiment_id
                    ),
                    cross_embodiment_mode=job_payload.cross_embodiment_mode,
                    active_realm_id=(
                        str(belief_row["realm_id"])
                        if belief_row.get("realm_id") is not None
                        else job_payload.active_realm_id
                    ),
                    cross_realm_mode=job_payload.cross_realm_mode,
                    mode=job_payload.mode,
                    incognito=bool(job_payload.incognito),
                    remember_across_chats=bool(job_payload.remember_across_chats),
                    remember_across_devices=bool(job_payload.remember_across_devices),
                    temporary=bool(job_payload.temporary),
                    temporary_ttl_seconds=job_payload.temporary_ttl_seconds,
                    purge_on_close=bool(job_payload.purge_on_close),
                    valid_to=str(belief_row["valid_to"]) if belief_row.get("valid_to") is not None else job_payload.valid_to,
                    sensitivity=str(belief_row.get("sensitivity") or "unknown"),
                    platform_locked=bool(belief_row.get("platform_locked")),
                    platform_id_lock=(
                        str(belief_row["platform_id_lock"])
                        if belief_row.get("platform_id_lock") is not None
                        else None
                    ),
                    scope_canonical=str(belief_row.get("scope_canonical") or belief_row.get("scope") or ""),
                    scope=str(belief_row["scope"]),
                    isolated_mode=bool(job_payload.isolated_mode),
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
                    claim_value=json_utils.dumps(
                        payload_json.get("claim_value"),
                        sort_keys=True,
                    ),
                    evidence_memory_ids=[str(evidence_row["id"])],
                    source_message_id=job_payload.message_id,
                    user_id=envelope.user_id,
                    assistant_mode_id=job_payload.assistant_mode_id,
                    workspace_id=job_payload.workspace_id,
                    conversation_id=envelope.conversation_id,
                    user_persona_id=(
                        evidence_row.get("user_persona_id")
                        if isinstance(evidence_row.get("user_persona_id"), str)
                        else job_payload.user_persona_id
                    ),
                    platform_id=str(evidence_row.get("platform_id") or job_payload.platform_id or "default"),
                    character_id=(
                        str(evidence_row["character_id"])
                        if evidence_row.get("character_id") is not None
                        else job_payload.character_id
                    ),
                    active_mind_id=(
                        str(evidence_row["memory_owner_id"])
                        if evidence_row.get("memory_owner_id") is not None
                        else job_payload.active_mind_id
                    ),
                    source_mind_id=(
                        str(evidence_row["source_mind_id"])
                        if evidence_row.get("source_mind_id") is not None
                        else job_payload.source_mind_id
                    ),
                    mind_topology=job_payload.mind_topology,
                    active_embodiment_id=(
                        str(evidence_row["embodiment_id"])
                        if evidence_row.get("embodiment_id") is not None
                        else job_payload.active_embodiment_id
                    ),
                    cross_embodiment_mode=job_payload.cross_embodiment_mode,
                    active_realm_id=(
                        str(evidence_row["realm_id"])
                        if evidence_row.get("realm_id") is not None
                        else job_payload.active_realm_id
                    ),
                    cross_realm_mode=job_payload.cross_realm_mode,
                    mode=job_payload.mode,
                    incognito=bool(job_payload.incognito),
                    remember_across_chats=bool(job_payload.remember_across_chats),
                    remember_across_devices=bool(job_payload.remember_across_devices),
                    temporary=bool(job_payload.temporary),
                    temporary_ttl_seconds=job_payload.temporary_ttl_seconds,
                    purge_on_close=bool(job_payload.purge_on_close),
                    valid_to=str(evidence_row["valid_to"]) if evidence_row.get("valid_to") is not None else job_payload.valid_to,
                    sensitivity=str(evidence_row.get("sensitivity") or "unknown"),
                    platform_locked=bool(evidence_row.get("platform_locked")),
                    platform_id_lock=(
                        str(evidence_row["platform_id_lock"])
                        if evidence_row.get("platform_id_lock") is not None
                        else None
                    ),
                    scope_canonical=str(evidence_row.get("scope_canonical") or evidence_row.get("scope") or ""),
                    scope=str(evidence_row["scope"]),
                    isolated_mode=bool(job_payload.isolated_mode),
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
            operational_profile=envelope.operational_profile,
        )
        await self._enqueue_tracked_job(REVISE_STREAM_NAME, revision_job)

    @staticmethod
    def _strictest_ttl(source_ttl: int | None, current_ttl: object) -> int | None:
        values = [
            int(value)
            for value in (source_ttl, current_ttl)
            if value is not None and int(value) > 0
        ]
        if not values:
            return source_ttl
        return min(values)

    async def _maybe_enqueue_graph_projection(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: MessageJobPayload,
        persisted: list[dict[str, object]],
        chunk_plan: ChunkingPlan | None,
    ) -> None:
        if not self._settings.graph_projection_enabled:
            return
        if envelope.conversation_id is None:
            return
        if chunk_plan is None:
            return
        graph_payload = GraphProjectionJobPayload(
            **job_payload.model_dump(mode="json"),
            source_memory_ids=[
                str(row["id"])
                for row in persisted
                if isinstance(row.get("id"), str)
            ],
            chunks=self._graph_projection_chunks(
                chunk_plan=chunk_plan,
                persisted=persisted,
            ),
        )
        graph_job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.SYNC_GRAPH,
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            message_ids=[job_payload.message_id],
            payload=graph_payload.model_dump(mode="json"),
            created_at=envelope.created_at,
            operational_profile=envelope.operational_profile,
        )
        await self._enqueue_tracked_job(GRAPH_STREAM_NAME, graph_job)

    @staticmethod
    def _graph_projection_chunks(
        *,
        chunk_plan: ChunkingPlan,
        persisted: list[dict[str, object]],
    ) -> list[GraphProjectionChunkPayload]:
        all_memory_ids = [
            str(row["id"])
            for row in persisted
            if isinstance(row.get("id"), str)
        ]
        chunks: list[GraphProjectionChunkPayload] = []
        for chunk in chunk_plan.chunks:
            chunk_source_memory_ids = all_memory_ids
            if chunk_plan.chunked:
                chunk_source_memory_ids = [
                    str(row["id"])
                    for row in persisted
                    if isinstance(row.get("id"), str)
                    and IngestWorker._persisted_row_matches_chunk(row, chunk)
                ]
            chunks.append(
                GraphProjectionChunkPayload(
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    chunk_count=chunk.chunk_count,
                    chunking_strategy=chunk.chunking_strategy,
                    level1_failure_reason=chunk.level1_failure_reason,
                    level1_attempts=chunk.level1_attempts,
                    source_memory_ids=chunk_source_memory_ids,
                )
            )
        return chunks

    @staticmethod
    def _persisted_row_matches_chunk(row: dict[str, object], chunk: TextChunk) -> bool:
        payload = row.get("payload_json")
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("chunk_index") == chunk.chunk_index
            and payload.get("chunk_count") == chunk.chunk_count
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
        if job_payload.temporary or job_payload.purge_on_close:
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
        if not await self._storage_backend.remember_dedupe(
            self._compaction_dedupe_key(
                user_id=envelope.user_id,
                conversation_id=envelope.conversation_id,
                last_chunk_end_seq=last_chunk_end_seq,
            ),
            COMPACTION_ENQUEUE_DEDUPE_TTL_SECONDS,
        ):
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
                user_persona_id=job_payload.user_persona_id,
                platform_id=job_payload.platform_id,
                character_id=job_payload.character_id,
                mode=job_payload.mode,
                incognito=job_payload.incognito,
                remember_across_chats=job_payload.remember_across_chats,
                remember_across_devices=job_payload.remember_across_devices,
                temporary=job_payload.temporary,
                temporary_ttl_seconds=job_payload.temporary_ttl_seconds,
                purge_on_close=job_payload.purge_on_close,
                valid_to=job_payload.valid_to,
                privacy_enforcement=job_payload.privacy_enforcement,
                job_kind=CompactionJobKind.CONVERSATION_CHUNK,
            ).model_dump(mode="json"),
            created_at=envelope.created_at,
            operational_profile=envelope.operational_profile,
        )
        await self._enqueue_tracked_job(COMPACT_STREAM_NAME, compaction_job)

    @staticmethod
    def _compaction_dedupe_key(
        *,
        user_id: str,
        conversation_id: str,
        last_chunk_end_seq: int,
    ) -> str:
        return f"compaction:conversation_chunk:{user_id}:{conversation_id}:{last_chunk_end_seq}"

    async def _maybe_refresh_topic_working_set(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: MessageJobPayload,
    ) -> None:
        if envelope.conversation_id is None or not self._settings.topic_working_set_enabled:
            return
        try:
            await TopicWorkingSetRefreshService(
                connection=self._connection,
                llm_client=self._llm_client,
                clock=self._clock,
                settings=self._settings,
            ).maybe_refresh_after_message(
                user_id=envelope.user_id,
                conversation_id=envelope.conversation_id,
                message_id=job_payload.message_id,
            )
        except LLMError as exc:
            await self._rollback_topic_refresh_failure()
            logger.warning(
                "Skipping Topic Working Set refresh for message %s: %s",
                job_payload.message_id,
                exc,
            )
        except Exception:
            await self._rollback_topic_refresh_failure()
            logger.exception(
                "Failed to refresh Topic Working Set for message %s",
                job_payload.message_id,
            )

    async def _rollback_topic_refresh_failure(self) -> None:
        try:
            await self._connection.rollback()
        except Exception:
            logger.warning("Failed to rollback Topic Working Set refresh failure", exc_info=True)

    async def _process_consequence_detection(
        self,
        *,
        job_payload: MessageJobPayload,
        context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
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
        user_persona_id: str | None,
        platform_id: str,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        isolated_mode: bool = False,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> str | None:
        candidates = await self._belief_repository.find_active_belief_candidates_by_claim_key(
            user_id,
            claim_key,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            incognito=incognito or isolated_mode,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
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
            if isolated_mode and candidate.get("conversation_id") != conversation_id:
                continue
            if not self._candidate_matches_active_mind(
                candidate,
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
            ):
                continue
            if not self._candidate_matches_active_embodiment(
                candidate,
                active_embodiment_id=active_embodiment_id,
            ):
                continue
            if not self._candidate_matches_active_realm(
                candidate,
                active_realm_id=active_realm_id,
            ):
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

    @staticmethod
    def _candidate_matches_active_mind(
        candidate: dict[str, Any],
        *,
        active_mind_id: str | None,
        mind_topology: str | None,
    ) -> bool:
        candidate_owner = candidate.get("memory_owner_id")
        candidate_owner_id = None if candidate_owner is None else str(candidate_owner)
        if active_mind_id is None:
            return candidate_owner_id is None
        if str(mind_topology or "unimind") == "unimind":
            return candidate_owner_id is None or candidate_owner_id == active_mind_id
        return candidate_owner_id == active_mind_id

    @staticmethod
    def _candidate_matches_active_embodiment(
        candidate: dict[str, Any],
        *,
        active_embodiment_id: str | None,
    ) -> bool:
        candidate_embodiment = candidate.get("embodiment_id")
        candidate_embodiment_id = (
            None if candidate_embodiment is None else str(candidate_embodiment)
        )
        if active_embodiment_id is None:
            return candidate_embodiment_id is None
        return (
            candidate_embodiment_id is None
            or candidate_embodiment_id == active_embodiment_id
        )

    @staticmethod
    def _candidate_matches_active_realm(
        candidate: dict[str, Any],
        *,
        active_realm_id: str | None,
    ) -> bool:
        candidate_realm = candidate.get("realm_id")
        candidate_realm_id = None if candidate_realm is None else str(candidate_realm)
        if active_realm_id is None:
            return candidate_realm_id is None
        return candidate_realm_id is None or candidate_realm_id == active_realm_id

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
        await self._job_tracking.mark_dead_lettered(message, exc)
        return True

    async def _enqueue_tracked_job(self, stream_name: str, job: JobEnvelope) -> None:
        await self._job_tracking.create_queued_job(stream_name, job)
        try:
            await self._storage_backend.stream_add(
                stream_name,
                job.model_dump(mode="json"),
            )
        except Exception as exc:
            await self._job_tracking.mark_enqueue_failed(job, exc)
            raise
