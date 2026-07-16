"""Enqueue prepared initial-context package refresh work."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core import json_utils
from atagia.core.ids import derive_child_job_id, new_job_id
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.storage_backend import StorageBackend
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_jobs import (
    ClaimedJob,
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    InitialContextPackageRefreshJobPayload,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobType,
)
from atagia.models.schemas_memory import ConversationStatus, OperationalProfileSnapshot
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.job_execution_context import current_job_claim

logger = logging.getLogger(__name__)



async def prepare_initial_context_package_refresh_payload(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    user_id: str,
    conversation_id: str | None,
    package_kind: InitialContextPackageKind | str | None,
    retrieval_profile_id: str | None,
    reason: InitialContextPackageRefreshReason,
    source_message_ids: list[str] | None = None,
    privacy_enforcement: str = "enforce",
    operational_profile: OperationalProfileSnapshot | None = None,
) -> InitialContextPackageRefreshJobPayload:
    """Reserve and build an ICP payload inside the caller-owned transaction."""

    generation = await UserLifecycleRepository(
        connection,
        clock,
    ).reserve_icp_refresh_generation(
        user_id,
        commit=False,
    )
    if generation is None:
        raise RuntimeError(
            f"Cannot reserve initial-context refresh for inactive user {user_id}"
        )
    resolved_kind = _package_kind_value(package_kind)
    dedupe_key = initial_context_package_refresh_dedupe_key(
        user_id=user_id,
        conversation_id=conversation_id,
        package_kind=resolved_kind,
        retrieval_profile_id=retrieval_profile_id,
        reason=reason,
        privacy_enforcement=privacy_enforcement,
        operational_profile_token=_operational_profile_token(operational_profile),
    )
    return InitialContextPackageRefreshJobPayload(
        user_id=user_id,
        conversation_id=conversation_id,
        package_kind=resolved_kind,
        retrieval_profile_id=retrieval_profile_id,
        reason=reason,
        refresh_generation=generation,
        refresh_dedupe_key=dedupe_key,
        source_message_ids=_stable_strings(source_message_ids or []),
        privacy_enforcement=privacy_enforcement,  # type: ignore[arg-type]
    )


def initial_context_package_refresh_dedupe_key(
    *,
    user_id: str,
    conversation_id: str | None,
    package_kind: str,
    retrieval_profile_id: str | None,
    reason: InitialContextPackageRefreshReason,
    privacy_enforcement: str,
    operational_profile_token: str | None,
) -> str:
    """Return the stable, content-free durable coalescing key."""

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
        parent_job_id: str | None = None,
        force: bool = False,
        fail_open: bool = True,
        commit: bool = True,
        dispatch: bool = True,
        mark_existing_packages_stale: bool = True,
        return_existing: bool = False,
    ) -> str | None:
        """Enqueue one refresh job and return its id when accepted."""

        normalized_source_ids = _stable_strings(source_message_ids or [])
        resolved_kind = _package_kind_value(package_kind)
        operational_profile_token = _operational_profile_token(operational_profile)
        dedupe_key = self._dedupe_key(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=resolved_kind,
            retrieval_profile_id=retrieval_profile_id,
            reason=reason,
            privacy_enforcement=privacy_enforcement,
            operational_profile_token=operational_profile_token,
        )
        if not self._refresh_enabled:
            if mark_existing_packages_stale:
                await self._best_effort_mark_existing_packages_stale(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    package_kind=resolved_kind,
                    retrieval_profile_id=retrieval_profile_id,
                    privacy_enforcement=privacy_enforcement,
                    operational_profile_token=operational_profile_token,
                    fail_open=fail_open,
                    reason=reason,
                    max_refresh_generation=None,
                )
            return None
        if self._job_tracking is None:
            raise RuntimeError(
                "Durable job tracking is required for initial-context refresh"
            )
        active_job_id = await self._job_tracking.active_icp_refresh_job_id(
            user_id=user_id,
            refresh_dedupe_key=dedupe_key,
        )
        if not force and active_job_id is not None:
            return active_job_id if return_existing else None
        refresh_generation = await self._job_tracking.reserve_icp_refresh_generation(
            user_id
        )
        payload = InitialContextPackageRefreshJobPayload(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=resolved_kind,
            retrieval_profile_id=retrieval_profile_id,
            reason=reason,
            refresh_generation=refresh_generation,
            refresh_dedupe_key=dedupe_key,
            source_message_ids=normalized_source_ids,
            privacy_enforcement=privacy_enforcement,  # type: ignore[arg-type]
        )

        parent_claim = current_job_claim()
        child_logical_key = json_utils.dumps(
            {
                "payload": payload.model_dump(mode="json"),
                "operational_profile_token": operational_profile_token,
            },
            sort_keys=True,
        )
        job = JobEnvelope(
            job_id=(
                derive_child_job_id(
                    parent_job_id,
                    JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value,
                    child_logical_key,
                )
                if parent_job_id is not None
                else new_job_id()
            ),
            job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
            user_id=user_id,
            parent_job_id=parent_job_id,
            conversation_id=conversation_id,
            message_ids=normalized_source_ids,
            payload=payload.model_dump(mode="json"),
            created_at=(
                parent_claim.envelope.created_at
                if parent_claim is not None
                and parent_claim.envelope.job_id == parent_job_id
                else self._clock.now()
            ),
            operational_profile=operational_profile,
        )
        try:
            await self._job_tracking.enqueue_job(
                self._storage_backend,
                INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                job,
                commit=commit,
                dispatch=dispatch,
            )
        except Exception as exc:
            if commit:
                await self._job_tracking.rollback_pending_enqueue()
            existing = await self._job_tracking.get_job_run(job.job_id)
            if existing is not None:
                return job.job_id
            coalesced_job_id = await self._job_tracking.active_icp_refresh_job_id(
                user_id=user_id,
                refresh_dedupe_key=dedupe_key,
            )
            if coalesced_job_id is not None:
                return coalesced_job_id if return_existing else None
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
        if mark_existing_packages_stale:
            await self._best_effort_mark_existing_packages_stale(
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=payload.package_kind,
                retrieval_profile_id=retrieval_profile_id,
                privacy_enforcement=payload.privacy_enforcement,
                operational_profile_token=operational_profile_token,
                fail_open=fail_open,
                reason=reason,
                # Delivery is a wake-up and may run the worker immediately.
                # Never let post-dispatch cleanup stale the generation that
                # this enqueue just reserved and may already have activated.
                max_refresh_generation=max(0, refresh_generation - 1),
                commit=commit,
            )
        return job.job_id

    async def replace_source_changed_claim(
        self,
        *,
        claim: ClaimedJob,
        job_payload: InitialContextPackageRefreshJobPayload,
        terminal_metadata: dict[str, Any],
    ) -> str | None:
        """Atomically release the failed generation and persist its successor."""

        if self._job_tracking is None:
            raise RuntimeError(
                "Durable job tracking is required for initial-context refresh"
            )
        operational_profile = claim.envelope.operational_profile
        operational_profile_token = _operational_profile_token(operational_profile)

        def successor_factory(generation: int) -> JobEnvelope:
            base_dedupe_key = initial_context_package_refresh_dedupe_key(
                user_id=job_payload.user_id,
                conversation_id=job_payload.conversation_id,
                package_kind=job_payload.package_kind,
                retrieval_profile_id=job_payload.retrieval_profile_id,
                reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
                privacy_enforcement=job_payload.privacy_enforcement,
                operational_profile_token=operational_profile_token,
            )
            successor_discriminator = hashlib.sha256(
                f"{claim.envelope.job_id}\x1f{generation}".encode("utf-8")
            ).hexdigest()
            payload = InitialContextPackageRefreshJobPayload(
                user_id=job_payload.user_id,
                conversation_id=job_payload.conversation_id,
                package_kind=job_payload.package_kind,
                retrieval_profile_id=job_payload.retrieval_profile_id,
                reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
                refresh_generation=generation,
                refresh_dedupe_key=(
                    f"{base_dedupe_key}:successor:{successor_discriminator}"
                ),
                source_message_ids=job_payload.source_message_ids,
                privacy_enforcement=job_payload.privacy_enforcement,
            )
            return JobEnvelope(
                job_id=new_job_id(),
                job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
                user_id=job_payload.user_id,
                conversation_id=job_payload.conversation_id,
                message_ids=job_payload.source_message_ids,
                payload=payload.model_dump(mode="json"),
                created_at=self._clock.now(),
                operational_profile=operational_profile,
            )

        successor_job_id = (
            await self._job_tracking.finish_claim_with_icp_source_successor(
                claim,
                storage_backend=self._storage_backend,
                stream_name=INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                successor_factory=successor_factory,
                metadata=terminal_metadata,
            )
        )
        return successor_job_id

    async def _best_effort_mark_existing_packages_stale(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        package_kind: str,
        retrieval_profile_id: str | None,
        privacy_enforcement: str,
        operational_profile_token: str | None,
        fail_open: bool,
        reason: InitialContextPackageRefreshReason,
        max_refresh_generation: int | None,
        commit: bool = True,
    ) -> None:
        try:
            await self._mark_existing_packages_stale(
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=package_kind,
                retrieval_profile_id=retrieval_profile_id,
                privacy_enforcement=privacy_enforcement,
                operational_profile_token=operational_profile_token,
                max_refresh_generation=max_refresh_generation,
                commit=commit,
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

    async def _mark_existing_packages_stale(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        package_kind: str,
        retrieval_profile_id: str | None,
        privacy_enforcement: str,
        operational_profile_token: str | None,
        max_refresh_generation: int | None,
        commit: bool,
    ) -> None:
        if self._package_repository is None:
            return
        if package_kind in {"all", InitialContextPackageKind.CONVERSATION.value}:
            await self._package_repository.mark_stale_for_key_family(
                user_id=user_id,
                conversation_id=conversation_id,
                package_kind=InitialContextPackageKind.CONVERSATION,
                retrieval_profile_id=retrieval_profile_id,
                max_refresh_generation=max_refresh_generation,
                commit=commit,
            )
        if package_kind in {"all", InitialContextPackageKind.BASELINE.value}:
            await self._package_repository.mark_stale_for_key_family(
                user_id=user_id,
                package_kind=InitialContextPackageKind.BASELINE,
                retrieval_profile_id=retrieval_profile_id,
                max_refresh_generation=max_refresh_generation,
                commit=commit,
            )

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
        return initial_context_package_refresh_dedupe_key(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=package_kind,
            retrieval_profile_id=retrieval_profile_id,
            reason=reason,
            privacy_enforcement=privacy_enforcement,
            operational_profile_token=operational_profile_token,
        )


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
    value = (
        package_kind.value
        if isinstance(package_kind, InitialContextPackageKind)
        else str(package_kind)
    )
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
