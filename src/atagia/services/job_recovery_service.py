"""Recover durable job tracking rows into transient in-process streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.storage_backend import StorageBackend
from atagia.core.timestamps import parse_optional_datetime
from atagia.memory.operational_profile import OperationalProfileLoader
from atagia.models.schemas_jobs import (
    CONTRACT_STREAM_NAME,
    EXTRACT_STREAM_NAME,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    InitialContextPackageRefreshJobPayload,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobType,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import ConversationStatus, OperationalProfileSnapshot
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    build_message_jobs,
    default_operational_profile_snapshot,
    resolve_operational_profile,
)
from atagia.core.config import Settings

logger = logging.getLogger(__name__)

_RECOVERABLE_STREAMS: dict[str, str] = {
    JobType.EXTRACT_MEMORY_CANDIDATES.value: EXTRACT_STREAM_NAME,
    JobType.PROJECT_CONTRACT.value: CONTRACT_STREAM_NAME,
    JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
}


@dataclass(slots=True)
class JobRecoveryResult:
    """Summary of startup recovery from durable tracking to transient streams."""

    recovered_jobs: int = 0
    skipped_jobs: int = 0
    failed_jobs: int = 0
    recovered_by_type: dict[str, int] = field(default_factory=dict)


class JobRecoveryService:
    """Requeue recoverable nonterminal jobs after an in-process restart."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        settings: Settings,
        storage_backend: StorageBackend,
        operational_profile_loader: OperationalProfileLoader,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._settings = settings
        self._storage_backend = storage_backend
        self._operational_profile_loader = operational_profile_loader
        self._jobs = JobRunRepository(connection, clock)
        self._users = UserRepository(connection, clock)
        self._conversations = ConversationRepository(connection, clock)
        self._messages = MessageRepository(connection, clock)

    async def recover_inprocess_stream_jobs(self) -> JobRecoveryResult:
        """Requeue durable nonterminal message jobs into an empty in-process stream."""
        result = JobRecoveryResult()
        for row in await self._jobs.nonterminal_root_jobs_for_recovery():
            recovered = await self._recover_job(row)
            if recovered:
                result.recovered_jobs += 1
                job_type = str(row["job_type"])
                result.recovered_by_type[job_type] = (
                    result.recovered_by_type.get(job_type, 0) + 1
                )
            else:
                result.skipped_jobs += 1
        if result.recovered_jobs:
            logger.info(
                "Recovered %s Atagia in-process worker jobs: %s",
                result.recovered_jobs,
                result.recovered_by_type,
            )
        return result

    async def _recover_job(self, row: dict[str, Any]) -> bool:
        job_id = str(row["job_id"])
        job_type = str(row["job_type"])
        stream_name = _RECOVERABLE_STREAMS.get(job_type)
        if stream_name is None:
            return False
        if job_type == JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value:
            return await self._recover_initial_context_package_job(row, stream_name)
        message_ids = row.get("source_message_ids_json")
        if not isinstance(message_ids, list) or len(message_ids) != 1:
            await self._mark_unrecoverable(job_id, "unsupported_source_message_shape")
            return False
        message_id = str(message_ids[0])
        user_id = str(row["user_id"])
        conversation_id = str(row["conversation_id"])
        active_user = await self._users.get_active_user(user_id)
        if active_user is None:
            await self._mark_unrecoverable(job_id, "user_not_active")
            return False
        conversation = await self._conversations.get_conversation(conversation_id, user_id)
        if (
            conversation is None
            or str(conversation.get("status")) != ConversationStatus.ACTIVE.value
        ):
            await self._mark_unrecoverable(job_id, "conversation_not_active")
            return False
        message = await self._messages.get_message(message_id, user_id)
        if message is None or str(message.get("conversation_id")) != conversation_id:
            await self._mark_unrecoverable(job_id, "source_message_not_found")
            return False

        prior_messages = await self._messages.get_recent_messages_before_seq(
            conversation_id,
            user_id,
            before_seq=int(message["seq"]),
            limit=RECENT_FETCH_LIMIT,
        )
        preferences = await self._users.get_memory_preferences(user_id)
        operational_profile = self._operational_profile_snapshot(row)
        include_contract_projection = job_type == JobType.PROJECT_CONTRACT.value
        built_jobs = build_message_jobs(
            clock=self._clock,
            conversation=conversation,
            message_id=message_id,
            prior_messages=prior_messages,
            message_text=str(message["text"]),
            occurred_at=message.get("occurred_at"),
            role=str(message["role"]),
            include_contract_projection=include_contract_projection,
            operational_profile=operational_profile,
            memory_preferences=preferences,
            active_presence_id=message.get("active_presence_id"),
            source_presence_id=message.get("source_presence_id"),
            active_space_id=message.get("space_id") or conversation.get("active_space_id"),
            active_mind_id=message.get("active_mind_id") or conversation.get("active_mind_id"),
            source_mind_id=(
                message.get("source_mind_id")
                or message.get("active_mind_id")
                or conversation.get("active_mind_id")
            ),
            mind_topology=conversation.get("mind_topology") or "unimind",
            active_embodiment_id=(
                message.get("active_embodiment_id")
                or conversation.get("active_embodiment_id")
            ),
            active_realm_id=(
                message.get("active_realm_id")
                or conversation.get("active_realm_id")
            ),
        )
        matching_job = next(
            (
                envelope
                for candidate_stream, envelope in built_jobs
                if candidate_stream == stream_name
                and envelope.job_type.value == job_type
            ),
            None,
        )
        if matching_job is None:
            await self._mark_unrecoverable(job_id, "job_type_not_rebuilt")
            return False

        recovered_job = matching_job.model_copy(
            update={
                "job_id": job_id,
                "created_at": parse_optional_datetime(row.get("queued_at")),
            }
        )
        delay_seconds = self._deferred_delay_seconds(row)
        try:
            if delay_seconds is not None:
                await self._storage_backend.stream_defer(
                    stream_name,
                    WORKER_GROUP_NAME,
                    "0-0",
                    recovered_job.model_dump(mode="json"),
                    delay_seconds=delay_seconds,
                )
            else:
                await self._jobs.mark_requeued_for_recovery(
                    job_id,
                    metadata={"inprocess_recovered_at": self._clock.now().isoformat()},
                )
                await self._storage_backend.stream_add(
                    stream_name,
                    recovered_job.model_dump(mode="json"),
                )
        except Exception as exc:
            await self._jobs.mark_failed(
                job_id,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        return True

    async def _recover_initial_context_package_job(
        self,
        row: dict[str, Any],
        stream_name: str,
    ) -> bool:
        job_id = str(row["job_id"])
        user_id = str(row["user_id"])
        conversation_id = (
            str(row["conversation_id"])
            if row.get("conversation_id") is not None
            else None
        )
        active_user = await self._users.get_active_user(user_id)
        if active_user is None:
            await self._mark_unrecoverable(job_id, "user_not_active")
            return False
        if conversation_id is not None:
            conversation = await self._conversations.get_conversation(
                conversation_id,
                user_id,
            )
            if (
                conversation is None
                or str(conversation.get("status")) != ConversationStatus.ACTIVE.value
            ):
                await self._mark_unrecoverable(job_id, "conversation_not_active")
                return False
        metadata = row.get("metadata_json")
        metadata = metadata if isinstance(metadata, dict) else {}
        message_ids = row.get("source_message_ids_json")
        source_message_ids = (
            [str(item) for item in message_ids]
            if isinstance(message_ids, list)
            else []
        )
        payload = InitialContextPackageRefreshJobPayload(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=str(metadata.get("package_kind") or "all"),  # type: ignore[arg-type]
            retrieval_profile_id=(
                str(metadata["retrieval_profile_id"])
                if metadata.get("retrieval_profile_id") is not None
                else None
            ),
            reason=InitialContextPackageRefreshReason(
                str(metadata.get("reason") or InitialContextPackageRefreshReason.BACKFILL.value)
            ),
            source_message_ids=source_message_ids,
            privacy_enforcement=str(metadata.get("privacy_enforcement") or "enforce"),  # type: ignore[arg-type]
        )
        recovered_job = JobEnvelope(
            job_id=job_id,
            job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=source_message_ids,
            payload=payload.model_dump(mode="json"),
            created_at=parse_optional_datetime(row.get("queued_at")),
            operational_profile=self._operational_profile_snapshot(row),
        )
        delay_seconds = self._deferred_delay_seconds(row)
        try:
            if delay_seconds is not None:
                await self._storage_backend.stream_defer(
                    stream_name,
                    WORKER_GROUP_NAME,
                    "0-0",
                    recovered_job.model_dump(mode="json"),
                    delay_seconds=delay_seconds,
                )
            else:
                await self._jobs.mark_requeued_for_recovery(
                    job_id,
                    metadata={"inprocess_recovered_at": self._clock.now().isoformat()},
                )
                await self._storage_backend.stream_add(
                    stream_name,
                    recovered_job.model_dump(mode="json"),
                )
        except Exception as exc:
            await self._jobs.mark_failed(
                job_id,
                error_class=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        return True

    def _operational_profile_snapshot(self, row: dict[str, Any]):
        metadata = row.get("metadata_json")
        profile_id = None
        if isinstance(metadata, dict):
            snapshot = metadata.get("operational_profile_snapshot")
            if isinstance(snapshot, dict):
                try:
                    return OperationalProfileSnapshot.model_validate(snapshot)
                except Exception:
                    logger.warning(
                        "Ignoring invalid operational profile snapshot during job recovery",
                        exc_info=True,
                    )
            raw_profile_id = metadata.get("operational_profile")
            profile_id = str(raw_profile_id).strip() if raw_profile_id else None
        try:
            return resolve_operational_profile(
                loader=self._operational_profile_loader,
                settings=self._settings,
                operational_profile=profile_id,
            ).snapshot
        except Exception:
            logger.warning(
                "Falling back to default operational profile during job recovery",
                exc_info=True,
            )
            return default_operational_profile_snapshot(
                loader=self._operational_profile_loader,
                settings=self._settings,
            )

    async def _mark_unrecoverable(self, job_id: str, reason: str) -> None:
        await self._jobs.mark_failed(
            job_id,
            error_class="JobRecoveryError",
            error_message=reason,
        )

    def _deferred_delay_seconds(self, row: dict[str, Any]) -> float | None:
        deferred_until = parse_optional_datetime(row.get("deferred_until"))
        if deferred_until is None:
            return None
        delay = (deferred_until - self._clock.now()).total_seconds()
        if delay <= 0:
            return None
        return delay
