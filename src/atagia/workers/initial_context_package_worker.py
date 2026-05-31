"""Prepared initial-context package refresh worker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.storage_backend import StorageBackend
from atagia.memory.operational_profile import (
    OperationalProfileLoader,
    UnknownOperationalProfileError,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_jobs import (
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    InitialContextPackageRefreshJobPayload,
    JobEnvelope,
    JobType,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import ConversationStatus
from atagia.services.chat_support import (
    apply_conversation_policy_overlay,
    resolve_operational_profile,
)
from atagia.services.initial_context_package_builder import (
    InitialContextPackageBuildBudget,
    InitialContextPackageBuilder,
)
from atagia.services.initial_context_package_curator import InitialContextPackageCurator
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import LLMClient
from atagia.services.worker_control_service import WorkerControlService, wait_if_worker_claims_paused

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3


class InitialContextPackageWorker:
    """Consumes package refresh jobs and materializes prompt-ready packages."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        clock: Clock,
        manifest_loader: ManifestLoader,
        settings: Settings | None = None,
        operational_profile_loader: OperationalProfileLoader | None = None,
        llm_client: LLMClient[Any] | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._connection = connection
        self._clock = clock
        self._manifest_loader = manifest_loader
        self._policy_resolver = PolicyResolver()
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._package_repository = InitialContextPackageRepository(connection, clock)
        self._worker_control = WorkerControlService(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._settings = resolved_settings
        curator = (
            InitialContextPackageCurator(
                llm_client=llm_client,
                settings=resolved_settings,
            )
            if llm_client is not None
            and resolved_settings.initial_context_package_curation_enabled
            else None
        )
        self._builder = InitialContextPackageBuilder(
            connection,
            clock,
            budget=InitialContextPackageBuildBudget(
                package_budget_tokens=(
                    resolved_settings.initial_context_package_total_max_tokens
                ),
                profile_block_budget_tokens=(
                    resolved_settings.initial_context_package_profile_max_tokens
                ),
                curated_block_budget_tokens=(
                    resolved_settings.initial_context_package_curated_block_max_tokens
                ),
                max_curated_items=(
                    resolved_settings.initial_context_package_curated_max_items
                ),
            ),
            curator=curator,
            curate_recent_verbatim_seed=(
                not resolved_settings.benchmark_disable_raw_recent_transcript
            ),
        )
        self._operational_profile_loader = (
            operational_profile_loader
            or OperationalProfileLoader(resolved_settings.operational_profiles_dir())
        )
        self._job_tracking = JobTrackingService(
            connection,
            clock,
            workers_enabled=resolved_settings.workers_enabled,
            settings=resolved_settings,
        )

    async def run(self, consumer_name: str = "initial-context-package-1") -> None:
        await self._storage_backend.stream_ensure_group(
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            WORKER_GROUP_NAME,
        )
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in initial context package worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "initial-context-package-1",
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
                result = await self.process_job(message.payload)
                await self._job_tracking.mark_succeeded(message, metadata=result)
                await self._storage_backend.stream_ack(
                    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                logger.exception(
                    "Failed to process initial context package job %s",
                    message.message_id,
                )
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

    async def process_job(self, payload: dict[str, object]) -> dict[str, Any]:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.REFRESH_INITIAL_CONTEXT_PACKAGE:
            raise ValueError(f"Unsupported initial context package job type: {envelope.job_type}")
        job_payload = InitialContextPackageRefreshJobPayload.model_validate(
            envelope.payload
        )
        if job_payload.user_id != envelope.user_id:
            raise ValueError("Refresh payload user_id must match envelope user_id")
        active_user = await self._user_repository.get_active_user(envelope.user_id)
        if active_user is None:
            return {"status": "skipped", "reason": "user_not_active"}
        if not self._settings.initial_context_package_refresh_enabled:
            stale = await self._mark_refresh_job_family_stale(job_payload)
            return {
                "status": "skipped",
                "reason": "refresh_disabled",
                "stale_packages": stale,
            }

        if job_payload.conversation_id is not None:
            return await self._process_conversation_refresh(
                envelope=envelope,
                job_payload=job_payload,
            )
        return await self._process_user_refresh(envelope, job_payload)

    async def _mark_refresh_job_family_stale(
        self,
        job_payload: InitialContextPackageRefreshJobPayload,
    ) -> int:
        stale = 0
        if self._builds_conversation(job_payload.package_kind):
            stale += await self._package_repository.mark_stale_for_key_family(
                user_id=job_payload.user_id,
                package_kind=InitialContextPackageKind.CONVERSATION,
                retrieval_profile_id=job_payload.retrieval_profile_id,
                conversation_id=job_payload.conversation_id,
            )
        if self._builds_baseline(job_payload.package_kind):
            stale += await self._package_repository.mark_stale_for_key_family(
                user_id=job_payload.user_id,
                package_kind=InitialContextPackageKind.BASELINE,
                retrieval_profile_id=job_payload.retrieval_profile_id,
            )
        return stale

    async def _process_conversation_refresh(
        self,
        *,
        envelope: JobEnvelope,
        job_payload: InitialContextPackageRefreshJobPayload,
    ) -> dict[str, Any]:
        conversation_id = str(job_payload.conversation_id)
        lock_key = self._lock_key(
            user_id=envelope.user_id,
            conversation_id=conversation_id,
            package_kind=job_payload.package_kind,
            retrieval_profile_id=job_payload.retrieval_profile_id,
            privacy_enforcement=job_payload.privacy_enforcement,
            operational_profile_token=self._operational_profile_token(
                envelope.operational_profile
            ),
        )
        lock_token = await self._storage_backend.acquire_lock(lock_key, ttl_seconds=60)
        if lock_token is None:
            return {"status": "skipped", "reason": "refresh_locked"}
        try:
            conversation = await self._conversation_repository.get_conversation(
                conversation_id,
                envelope.user_id,
            )
            if (
                conversation is None
                or str(conversation.get("status")) != ConversationStatus.ACTIVE.value
            ):
                deleted = await self._package_repository.delete_for_conversation(
                    user_id=envelope.user_id,
                    conversation_id=conversation_id,
                )
                return {
                    "status": "skipped",
                    "reason": "conversation_not_active",
                    "deleted_packages": deleted,
                }
            built, stale = await self._build_for_conversation(
                conversation,
                package_kind=job_payload.package_kind,
                retrieval_profile_id=job_payload.retrieval_profile_id,
                privacy_enforcement=job_payload.privacy_enforcement,
                operational_profile=envelope.operational_profile,
            )
            return {
                "status": "refreshed",
                "built_packages": built,
                "stale_packages": stale,
            }
        finally:
            await self._storage_backend.release_lock(lock_key, lock_token)

    async def _process_user_refresh(
        self,
        envelope: JobEnvelope,
        job_payload: InitialContextPackageRefreshJobPayload,
    ) -> dict[str, Any]:
        lock_key = self._lock_key(
            user_id=job_payload.user_id,
            conversation_id=None,
            package_kind=job_payload.package_kind,
            retrieval_profile_id=job_payload.retrieval_profile_id,
            privacy_enforcement=job_payload.privacy_enforcement,
            operational_profile_token=self._operational_profile_token(
                envelope.operational_profile
            ),
        )
        lock_token = await self._storage_backend.acquire_lock(lock_key, ttl_seconds=60)
        if lock_token is None:
            return {"status": "skipped", "reason": "refresh_locked"}
        try:
            conversations = await self._conversation_repository.list_conversations(
                job_payload.user_id,
                assistant_mode_id=job_payload.retrieval_profile_id,
                include_temporary=True,
            )
            built_total: list[str] = []
            stale_total = 0
            for conversation in conversations:
                built, stale = await self._build_for_conversation(
                    conversation,
                    package_kind=job_payload.package_kind,
                    retrieval_profile_id=job_payload.retrieval_profile_id,
                    privacy_enforcement=job_payload.privacy_enforcement,
                    operational_profile=envelope.operational_profile,
                )
                built_total.extend(built)
                stale_total += stale
            return {
                "status": "refreshed",
                "built_packages": built_total,
                "stale_packages": stale_total,
            }
        finally:
            await self._storage_backend.release_lock(lock_key, lock_token)

    async def _build_for_conversation(
        self,
        conversation: dict[str, Any],
        *,
        package_kind: str,
        retrieval_profile_id: str | None,
        privacy_enforcement: str,
        operational_profile: Any,
    ) -> tuple[list[str], int]:
        user_id = str(conversation["user_id"])
        conversation_id = str(conversation["id"])
        profile_id = retrieval_profile_id or str(conversation["assistant_mode_id"])
        operational_profile_token = self._operational_profile_token(operational_profile)
        manifest = self._manifest_loader.get(profile_id)
        resolved_operational_profile = None
        if operational_profile is not None:
            try:
                resolved_operational_profile = resolve_operational_profile(
                    loader=self._operational_profile_loader,
                    settings=self._settings,
                    operational_profile=getattr(operational_profile, "profile_id", None),
                    operational_signals=getattr(operational_profile, "signals", None),
                )
            except UnknownOperationalProfileError:
                logger.warning(
                    "Unknown operational profile snapshot %s for initial context package refresh; "
                    "building without operational policy override",
                    getattr(operational_profile, "profile_id", None),
                )
        resolved_policy = self._policy_resolver.resolve(
            manifest,
            None,
            None,
            (
                resolved_operational_profile.policy_override
                if resolved_operational_profile is not None
                else None
            ),
        )
        resolved_policy = apply_conversation_policy_overlay(
            resolved_policy,
            conversation,
        )
        built: list[str] = []
        stale = 0
        if self._builds_baseline(package_kind):
            subject = self._baseline_subject(conversation, profile_id)
            baseline = await self._builder.build_baseline_package(
                user_id=user_id,
                resolved_policy=resolved_policy,
                workspace_id=subject.get("workspace_id"),
                assistant_mode_id=profile_id,
                user_persona_id=subject.get("user_persona_id"),
                platform_id=subject.get("platform_id"),
                character_id=subject.get("character_id"),
                active_presence_id=self._optional_text(
                    conversation.get("active_presence_id")
                ),
                active_space_id=self._optional_text(
                    conversation.get("active_space_id")
                ),
                active_space_boundary_mode=self._optional_text(
                    conversation.get("active_space_boundary_mode")
                ),
                active_mind_id=self._optional_text(
                    conversation.get("active_mind_id")
                ),
                mind_topology=self._optional_text(conversation.get("mind_topology")),
                active_embodiment_id=self._optional_text(
                    conversation.get("active_embodiment_id")
                ),
                active_realm_id=self._optional_text(
                    conversation.get("active_realm_id")
                ),
                incognito=(
                    bool(conversation.get("incognito"))
                    or bool(conversation.get("isolated_mode"))
                ),
                privacy_enforcement=privacy_enforcement,
                operational_profile=operational_profile,
            )
            built.append(baseline.id)
            stale += await self._package_repository.mark_stale_for_baseline_subject(
                user_id=user_id,
                retrieval_profile_id=profile_id,
                subject_json=subject,
                privacy_enforcement=privacy_enforcement,
                operational_profile_token=operational_profile_token,
                exclude_package_key_hashes=[baseline.package_key_hash],
            )
        if self._builds_conversation(package_kind):
            conversation_package = await self._builder.build_conversation_package(
                user_id=user_id,
                conversation_id=conversation_id,
                conversation=conversation,
                resolved_policy=resolved_policy,
                privacy_enforcement=privacy_enforcement,
                operational_profile=operational_profile,
            )
            built.append(conversation_package.id)
            stale += await self._package_repository.mark_stale_for_key_family(
                user_id=user_id,
                package_kind=InitialContextPackageKind.CONVERSATION,
                retrieval_profile_id=profile_id,
                conversation_id=conversation_id,
                privacy_enforcement=privacy_enforcement,
                operational_profile_token=operational_profile_token,
                exclude_package_key_hashes=[conversation_package.package_key_hash],
            )
        return built, stale

    @staticmethod
    def _baseline_subject(
        conversation: dict[str, Any],
        profile_id: str,
    ) -> dict[str, str | None]:
        workspace_id = InitialContextPackageWorker._optional_text(
            conversation.get("workspace_id")
        )
        character_id = InitialContextPackageWorker._optional_text(
            conversation.get("character_id")
        )
        if character_id is None:
            character_id = workspace_id
        return {
            "user_persona_id": InitialContextPackageWorker._optional_text(
                conversation.get("user_persona_id")
            ),
            "platform_id": InitialContextPackageWorker._optional_text(
                conversation.get("platform_id")
            ),
            "character_id": character_id,
            "workspace_id": workspace_id,
            "assistant_mode_id": profile_id,
            "mode": InitialContextPackageWorker._optional_text(
                conversation.get("mode")
            )
            or profile_id,
        }

    @staticmethod
    def _builds_baseline(package_kind: str) -> bool:
        return package_kind in {"all", InitialContextPackageKind.BASELINE.value}

    @staticmethod
    def _builds_conversation(package_kind: str) -> bool:
        return package_kind in {"all", InitialContextPackageKind.CONVERSATION.value}

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _operational_profile_token(operational_profile: Any) -> str | None:
        if operational_profile is None:
            return None
        token = getattr(operational_profile, "token", None)
        if token is not None:
            return str(token)
        if isinstance(operational_profile, dict) and operational_profile.get("token") is not None:
            return str(operational_profile["token"])
        return None

    @staticmethod
    def _lock_key(
        *,
        user_id: str,
        conversation_id: str | None,
        package_kind: str,
        retrieval_profile_id: str | None,
        privacy_enforcement: str,
        operational_profile_token: str | None,
    ) -> str:
        return ":".join(
            [
                "initial_context_package_refresh",
                user_id,
                conversation_id or "all",
                package_kind,
                retrieval_profile_id or "default",
                privacy_enforcement,
                operational_profile_token or "no-operational-profile",
            ]
        )

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
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
            f"dead_letter:{INITIAL_CONTEXT_PACKAGE_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
                "error_details": [],
            },
        )
        await self._storage_backend.stream_ack(
            INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        await self._job_tracking.mark_dead_lettered(message, exc)
        return True
