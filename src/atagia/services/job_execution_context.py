"""Task-local identity for child jobs created by a durable worker attempt."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import logging

from atagia.core.storage_backend import StorageBackend
from atagia.models.schemas_jobs import ClaimedJob

logger = logging.getLogger(__name__)

_CURRENT_JOB_CLAIM: ContextVar[ClaimedJob | None] = ContextVar(
    "atagia_current_job_claim",
    default=None,
)


class StaleParentJobFenceError(RuntimeError):
    """A worker tried to create child work without its current durable fence."""


class TransientJobLockUnavailable(RuntimeError):
    """A job-owned transient lock is unavailable and must be retried durably."""


@dataclass(frozen=True, slots=True)
class CurrentJobLockScope:
    """Immutable backend scope captured by the current durable claim."""

    lifecycle_cleanup_key: str
    lifecycle_epoch: str
    job_id: str
    execution_fence: int


def current_job_claim() -> ClaimedJob | None:
    """Return the durable claim owning the current worker task, if any."""

    return _CURRENT_JOB_CLAIM.get()


def _matching_job_claim(*, user_id: str, job_id: str) -> ClaimedJob | None:
    claim = current_job_claim()
    if claim is None:
        return None
    if claim.envelope.user_id != user_id or claim.envelope.job_id != job_id:
        raise StaleParentJobFenceError(
            "Transient state scope does not match the current durable claim"
        )
    return claim


def require_current_job_lock_scope(
    *,
    user_id: str,
    job_id: str,
) -> CurrentJobLockScope:
    """Return lifecycle and execution-fence scope for a job-owned lock."""

    claim = _matching_job_claim(user_id=user_id, job_id=job_id)
    if claim is None:
        raise StaleParentJobFenceError(
            "User-owned transient state requires a current durable job claim"
        )
    return CurrentJobLockScope(
        lifecycle_cleanup_key=claim.lifecycle_cleanup_key,
        lifecycle_epoch=claim.lifecycle_epoch,
        job_id=claim.envelope.job_id,
        execution_fence=claim.execution_fence,
    )


async def acquire_current_job_lock(
    storage_backend: StorageBackend,
    key: str,
    ttl_seconds: int,
    *,
    scope: CurrentJobLockScope,
) -> str:
    """Acquire one attempt-fenced lock or request a durable coordination defer."""

    try:
        token = await storage_backend.acquire_lock(
            key,
            ttl_seconds,
            lifecycle_cleanup_key=scope.lifecycle_cleanup_key,
            lifecycle_epoch=scope.lifecycle_epoch,
            job_id=scope.job_id,
            execution_fence=scope.execution_fence,
        )
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise TransientJobLockUnavailable(
            "Transient lock backend is unavailable"
        ) from exc
    if token is None:
        raise TransientJobLockUnavailable(
            "Transient lock or lifecycle mirror is unavailable"
        )
    return token


async def release_current_job_lock(
    storage_backend: StorageBackend,
    key: str,
    token: str,
    *,
    scope: CurrentJobLockScope,
) -> None:
    """Best-effort release after the durable effect; TTL/revocation remains safe."""

    try:
        await storage_backend.release_lock(
            key,
            token,
            lifecycle_cleanup_key=scope.lifecycle_cleanup_key,
            lifecycle_epoch=scope.lifecycle_epoch,
        )
    except Exception:
        logger.warning(
            "job_lock_release_failed",
            extra={"job_id": scope.job_id, "lock_key": key},
            exc_info=True,
        )


def current_derivation_dedupe_scope(*, user_id: str, job_id: str) -> str:
    """Return the durable lifecycle/revision namespace for external dedupe.

    Direct unit and maintenance calls that do not run under a durable claim use
    a separate namespace. Production worker calls must match the bound claim so
    a stale or unrelated task cannot create a misleading dedupe marker.
    """

    claim = _matching_job_claim(user_id=user_id, job_id=job_id)
    if claim is None:
        return "unclaimed"
    return f"{claim.lifecycle_epoch}:{claim.derivation_revision}"


def bind_job_claim(claim: ClaimedJob) -> Token[ClaimedJob | None]:
    """Bind a claim while its worker attempt is active."""

    return _CURRENT_JOB_CLAIM.set(claim)


def reset_job_claim(token: Token[ClaimedJob | None]) -> None:
    """Restore the previous task-local worker claim."""

    _CURRENT_JOB_CLAIM.reset(token)
