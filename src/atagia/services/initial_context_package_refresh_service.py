"""Enqueue prepared initial-context package refresh work."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import new_job_id
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.storage_backend import StorageBackend
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_jobs import (
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    InitialContextPackageRefreshJobPayload,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobType,
)
from atagia.models.schemas_memory import ConversationStatus, OperationalProfileSnapshot
from atagia.services.job_tracking_service import JobTrackingService

logger = logging.getLogger(__name__)

INITIAL_CONTEXT_PACKAGE_REFRESH_DEDUPE_TTL_SECONDS = 60


class InitialContextPackageRefreshEnqueuer:
    """Coalesce and enqueue package refresh jobs after canonical state changes."""

    def __init__(
        self,
        *,
        storage_backend: StorageBackend,
        clock: Clock,
        job_tracking_service: JobTrackingService | None = None,
        package_repository: InitialContextPackageRepository | None = None,
        refresh_enabled: bool = True,
    ) -> None:
        self._storage_backend = storage_backend
        self._clock = clock
        self._job_tracking = job_tracking_service
        self._package_repository = package_repository
        self._refresh_enabled = refresh_enabled

    async def enqueue_refresh(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        package_kind: InitialContextPackageKind | str | None = None,
        retrieval_profile_id: str | None = None,
        reason: InitialContextPackageRefreshReason,
        source_message_ids: list[str] | None = None,
        privacy_enforcement: str = "enforce",
        operational_profile: OperationalProfileSnapshot | None = None,
        force: bool = False,
        fail_open: bool = True,
    ) -> str | None:
        """Enqueue one refresh job and return its id when accepted."""

        normalized_source_ids = _stable_strings(source_message_ids or [])
        resolved_kind = _package_kind_value(package_kind)
        payload = InitialContextPackageRefreshJobPayload(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=resolved_kind,
            retrieval_profile_id=retrieval_profile_id,
            reason=reason,
            source_message_ids=normalized_source_ids,
            privacy_enforcement=privacy_enforcement,  # type: ignore[arg-type]
        )
        operational_profile_token = _operational_profile_token(operational_profile)
        dedupe_key = self._dedupe_key(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=payload.package_kind,
            retrieval_profile_id=retrieval_profile_id,
            reason=reason,
            privacy_enforcement=payload.privacy_enforcement,
            operational_profile_token=operational_profile_token,
        )
        try:
            await self._mark_existing_packages_stale(
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=payload.package_kind,
                retrieval_profile_id=retrieval_profile_id,
                privacy_enforcement=payload.privacy_enforcement,
                operational_profile_token=operational_profile_token,
            )
        except Exception as exc:
            if not fail_open:
                raise
            logger.warning(
                "initial_context_package_stale_mark_failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "reason": reason.value,
                    "error": str(exc),
                },
            )
        if not self._refresh_enabled:
            return None
        if not force and not await self._remember_dedupe(dedupe_key):
            return None

        job = JobEnvelope(
            job_id=new_job_id(),
            job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=normalized_source_ids,
            payload=payload.model_dump(mode="json"),
            created_at=self._clock.now(),
            operational_profile=operational_profile,
        )
        if self._job_tracking is not None:
            await self._job_tracking.create_queued_job(
                INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                job,
            )
        try:
            await self._storage_backend.stream_add(
                INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                job.model_dump(mode="json"),
            )
        except Exception as exc:
            if self._job_tracking is not None:
                await self._job_tracking.mark_enqueue_failed(job, exc)
            if not fail_open:
                raise
            logger.warning(
                "initial_context_package_refresh_enqueue_failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "reason": reason.value,
                    "error": str(exc),
                },
            )
            return None
        return job.job_id

    async def _mark_existing_packages_stale(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        package_kind: str,
        retrieval_profile_id: str | None,
        privacy_enforcement: str,
        operational_profile_token: str | None,
    ) -> None:
        if self._package_repository is None:
            return
        if package_kind in {"all", InitialContextPackageKind.CONVERSATION.value}:
            await self._package_repository.mark_stale_for_key_family(
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=InitialContextPackageKind.CONVERSATION,
                retrieval_profile_id=retrieval_profile_id,
            )
        if package_kind in {"all", InitialContextPackageKind.BASELINE.value}:
            await self._package_repository.mark_stale_for_key_family(
                user_id=user_id,
                package_kind=InitialContextPackageKind.BASELINE,
                retrieval_profile_id=retrieval_profile_id,
            )

    async def _remember_dedupe(self, key: str) -> bool:
        try:
            return await self._storage_backend.remember_dedupe(
                key,
                INITIAL_CONTEXT_PACKAGE_REFRESH_DEDUPE_TTL_SECONDS,
            )
        except (AttributeError, NotImplementedError):
            return True

    @staticmethod
    def _dedupe_key(
        *,
        user_id: str,
        conversation_id: str | None,
        package_kind: str,
        retrieval_profile_id: str | None,
        reason: InitialContextPackageRefreshReason,
        privacy_enforcement: str,
        operational_profile_token: str | None,
    ) -> str:
        raw = "\x1f".join(
            [
                user_id,
                conversation_id or "",
                package_kind,
                retrieval_profile_id or "",
                reason.value,
                privacy_enforcement,
                operational_profile_token or "",
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"initial_context_package_refresh:{digest}"


async def enqueue_initial_context_package_backfill(
    *,
    connection: aiosqlite.Connection,
    storage_backend: StorageBackend,
    clock: Clock,
    job_tracking_service: JobTrackingService | None = None,
    user_id: str | None = None,
    limit: int | None = None,
    refresh_enabled: bool = True,
) -> list[str]:
    """Enqueue refresh jobs for active conversations that need prepared packages."""

    clauses = ["u.deleted_at IS NULL", "c.status = ?"]
    parameters: list[Any] = [ConversationStatus.ACTIVE.value]
    if user_id is not None:
        clauses.append("c.user_id = ?")
        parameters.append(user_id)
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        parameters.append(max(0, int(limit)))

    cursor = await connection.execute(
        f"""
        SELECT c.user_id, c.id AS conversation_id, c.assistant_mode_id
        FROM conversations AS c
        JOIN users AS u ON u.id = c.user_id
        WHERE {" AND ".join(clauses)}
        ORDER BY c.updated_at DESC, c.id ASC
        {limit_clause}
        """,
        tuple(parameters),
    )
    rows = await cursor.fetchall()
    enqueuer = InitialContextPackageRefreshEnqueuer(
        storage_backend=storage_backend,
        clock=clock,
        job_tracking_service=job_tracking_service,
        refresh_enabled=refresh_enabled,
    )
    job_ids: list[str] = []
    for row in rows:
        job_id = await enqueuer.enqueue_refresh(
            user_id=str(row["user_id"]),
            conversation_id=str(row["conversation_id"]),
            retrieval_profile_id=str(row["assistant_mode_id"]),
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        if job_id is not None:
            job_ids.append(job_id)
    return job_ids


def _package_kind_value(
    package_kind: InitialContextPackageKind | str | None,
) -> str:
    if package_kind is None:
        return "all"
    value = package_kind.value if isinstance(package_kind, InitialContextPackageKind) else str(package_kind)
    if value not in {"all", "baseline", "conversation"}:
        raise ValueError(f"Unsupported package_kind for refresh: {value}")
    return value


def _stable_strings(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _operational_profile_token(
    operational_profile: OperationalProfileSnapshot | None,
) -> str | None:
    return None if operational_profile is None else operational_profile.token
