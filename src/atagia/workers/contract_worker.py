"""Contract projection worker."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import hashlib
import logging

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.storage_backend import StorageBackend
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.contract_projection import ContractProjector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    StreamMessage,
    JobType,
    MessageJobPayload,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import (
    ConversationStatus,
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.chat_support import apply_conversation_policy_overlay
from atagia.services.initial_context_package_refresh_service import (
    InitialContextPackageRefreshEnqueuer,
)
from atagia.services.job_execution_context import (
    TransientJobLockUnavailable,
    acquire_current_job_lock,
    current_derivation_dedupe_scope,
    release_current_job_lock,
    require_current_job_lock_scope,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import LLMClient, StructuredOutputError
from atagia.services.worker_control_service import (
    WorkerControlService,
    wait_if_worker_claims_paused,
)
from atagia.services.worker_effect_fence import WorkerEffectFence
from atagia.services.worker_job_lease import JobLeaseLostError, WorkerJobLease

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
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
        job_connection: aiosqlite.Connection | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._manifest_loader = manifest_loader
        self._policy_resolver = PolicyResolver()
        self._connection = connection
        self._clock = clock
        self._worker_control = WorkerControlService(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._settings = resolved_settings
        self._stream_reclaim_idle_ms = int(
            resolved_settings.worker_stream_reclaim_idle_seconds * 1000
        )
        self._effect_fence = WorkerEffectFence(connection, clock)
        self._job_tracking = JobTrackingService(
            job_connection or connection,
            clock,
            workers_enabled=resolved_settings.workers_enabled,
            settings=resolved_settings,
            child_job_connection=connection,
        )
        self._initial_context_package_refresh = InitialContextPackageRefreshEnqueuer(
            storage_backend=storage_backend,
            clock=clock,
            job_tracking_service=self._job_tracking,
            package_repository=InitialContextPackageRepository(connection, clock),
            refresh_enabled=resolved_settings.initial_context_package_refresh_enabled,
        )
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._projector = ContractProjector(
            llm_client=llm_client,
            clock=clock,
            message_repository=MessageRepository(connection, clock),
            memory_repository=MemoryObjectRepository(connection, clock),
            contract_repository=ContractDimensionRepository(connection, clock),
            settings=resolved_settings,
        )

    async def run(self, consumer_name: str = "contract-1") -> None:
        await self._storage_backend.stream_ensure_group(
            CONTRACT_STREAM_NAME, WORKER_GROUP_NAME
        )
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
        deferred = 0
        dead_lettered = 0
        for message in messages:
            claim = await self._job_tracking.claim_notification(
                message,
                owner_id=consumer_name,
            )
            if claim is None:
                await self._storage_backend.stream_ack(
                    CONTRACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
                continue
            lease = WorkerJobLease(
                self._job_tracking,
                claim,
                effect_fence=self._effect_fence,
            )
            try:
                lock_deferred = False
                async with lease:
                    try:
                        await self.process_job(claim.envelope.model_dump(mode="json"))
                    except TransientJobLockUnavailable as exc:
                        await lease.defer(
                            exc,
                            deferred_until=(
                                self._clock.now()
                                + timedelta(
                                    seconds=self._settings.worker_transient_defer_seconds
                                )
                            ),
                        )
                        lock_deferred = True
                    else:
                        await lease.succeed()
                await self._storage_backend.stream_ack(
                    CONTRACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                if lock_deferred:
                    deferred += 1
                else:
                    acked += 1
            except JobLeaseLostError:
                await self._storage_backend.stream_ack(
                    CONTRACT_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
            except Exception as exc:
                if lock_deferred:
                    raise
                failed += 1
                if isinstance(exc, StructuredOutputError):
                    details = "; ".join(exc.details) if exc.details else str(exc)
                    logger.warning(
                        "Failed to process contract job %s due to structured output: %s",
                        message.message_id,
                        details,
                    )
                else:
                    logger.exception(
                        "Failed to process contract job %s", message.message_id
                    )
                if claim.attempt_count >= MAX_STREAM_DELIVERIES:
                    finalized = await lease.dead_letter(
                        self._storage_backend,
                        stream_name=CONTRACT_STREAM_NAME,
                        group_name=WORKER_GROUP_NAME,
                        message=message,
                        exc=exc,
                    )
                    if not finalized:
                        continue
                    dead_lettered += 1
                else:
                    await lease.retry(exc)
                    await self._storage_backend.stream_ack(
                        CONTRACT_STREAM_NAME,
                        WORKER_GROUP_NAME,
                        message.message_id,
                    )
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            deferred=deferred,
            dead_lettered=dead_lettered,
        )

    async def process_job(self, payload: dict[str, object]) -> None:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.PROJECT_CONTRACT:
            raise ValueError(f"Unsupported contract job type: {envelope.job_type}")
        if envelope.conversation_id is None:
            raise ValueError("Contract projection jobs require conversation_id")
        job_payload = MessageJobPayload.model_validate(envelope.payload)
        projection_key = self._contract_projection_key(
            user_id=envelope.user_id,
            conversation_id=envelope.conversation_id,
            message_id=job_payload.message_id,
            derivation_scope=current_derivation_dedupe_scope(
                user_id=envelope.user_id,
                job_id=envelope.job_id,
            ),
        )
        lock_scope = require_current_job_lock_scope(
            user_id=envelope.user_id,
            job_id=envelope.job_id,
        )
        lock_key = f"{projection_key}:lock"
        lock_token = await acquire_current_job_lock(
            self._storage_backend,
            lock_key,
            60,
            scope=lock_scope,
        )
        try:
            active_user = await self._user_repository.get_active_user(envelope.user_id)
            if active_user is None:
                return
            conversation = await self._conversation_repository.get_conversation(
                envelope.conversation_id,
                envelope.user_id,
            )
            if (
                conversation is None
                or str(conversation.get("status")) != ConversationStatus.ACTIVE.value
            ):
                return

            manifest = self._manifest_loader.get(job_payload.assistant_mode_id)
            resolved_policy = self._policy_resolver.resolve(manifest, None, None)
            resolved_policy = apply_conversation_policy_overlay(
                resolved_policy, conversation
            )
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
                platform_id=str(
                    job_payload.platform_id
                    or conversation.get("platform_id")
                    or "default"
                ),
                character_id=(
                    job_payload.character_id
                    if job_payload.character_id is not None
                    else conversation.get("character_id")
                    or conversation.get("workspace_id")
                ),
                active_presence_id=(
                    job_payload.active_presence_id
                    or conversation.get("active_presence_id")
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
                    job_payload.active_realm_id or conversation.get("active_realm_id")
                ),
                active_realm_display_name=job_payload.active_realm_display_name,
                cross_realm_mode=(
                    job_payload.cross_realm_mode
                    or conversation.get("cross_realm_mode")
                    or "none"
                ),
                mode=str(
                    job_payload.mode
                    or conversation.get("mode")
                    or job_payload.assistant_mode_id
                ),
                recent_messages=[
                    ExtractionContextMessage.model_validate(item)
                    for item in job_payload.recent_messages
                ],
                temporary=bool(conversation.get("temporary"))
                or bool(job_payload.temporary),
                temporary_ttl_seconds=self._strictest_ttl(
                    job_payload.temporary_ttl_seconds,
                    conversation.get("temporary_ttl_seconds"),
                ),
                purge_on_close=bool(conversation.get("purge_on_close"))
                or bool(job_payload.purge_on_close),
                isolated_mode=bool(conversation.get("isolated_mode"))
                or bool(job_payload.isolated_mode),
                incognito=bool(conversation.get("incognito"))
                or bool(job_payload.incognito),
                remember_across_chats=(
                    bool(job_payload.remember_across_chats)
                    and bool(active_user["remember_across_chats"])
                ),
                remember_across_devices=(
                    bool(job_payload.remember_across_devices)
                    and bool(active_user["remember_across_devices"])
                ),
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
            await self._projector.project(
                message_text=job_payload.message_text,
                role=job_payload.role,
                conversation_context=context,
                resolved_policy=resolved_policy,
                user_id=envelope.user_id,
                occurred_at=job_payload.message_occurred_at,
            )
            await self._initial_context_package_refresh.enqueue_refresh(
                user_id=envelope.user_id,
                conversation_id=envelope.conversation_id,
                retrieval_profile_id=job_payload.assistant_mode_id,
                reason=InitialContextPackageRefreshReason.CONTRACT_PROJECTION,
                source_message_ids=[job_payload.message_id],
                privacy_enforcement=job_payload.privacy_enforcement,
                operational_profile=envelope.operational_profile,
                parent_job_id=envelope.job_id,
                fail_open=True,
            )
        finally:
            await release_current_job_lock(
                self._storage_backend,
                lock_key,
                lock_token,
                scope=lock_scope,
            )

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

    @staticmethod
    def _contract_projection_key(
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        derivation_scope: str = "unclaimed",
    ) -> str:
        raw_key = (
            f"{user_id}:{conversation_id}:{message_id}:{derivation_scope}:contract"
        )
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
            min_idle_ms=self._stream_reclaim_idle_ms,
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
