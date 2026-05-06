"""Durable worker-job tracking repository."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from atagia.core import json_utils
from atagia.core.repositories import BaseRepository
from atagia.models.schemas_jobs import JobRunStatus, JobType

NONTERMINAL_JOB_STATUSES: tuple[JobRunStatus, ...] = (
    JobRunStatus.QUEUED,
    JobRunStatus.RUNNING,
    JobRunStatus.RETRYING,
)
TERMINAL_JOB_STATUSES: tuple[JobRunStatus, ...] = (
    JobRunStatus.SUCCEEDED,
    JobRunStatus.SKIPPED,
    JobRunStatus.FAILED,
    JobRunStatus.DEAD_LETTERED,
    JobRunStatus.CANCELLED,
)
ROOT_JOB_TYPES: tuple[JobType, ...] = (
    JobType.EXTRACT_MEMORY_CANDIDATES,
    JobType.PROJECT_CONTRACT,
)


@dataclass(frozen=True, slots=True)
class JobNamespaceFilter:
    """Namespace policy used for non-admin job-status views."""

    user_persona_id: str | None
    platform_id: str
    character_id: str | None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True


class JobRunRepository(BaseRepository):
    """Persistence operations for background worker job runs."""

    async def create_queued_job(
        self,
        *,
        job_id: str,
        stream_name: str,
        job_type: str,
        user_id: str,
        conversation_id: str | None,
        source_message_ids: list[str],
        source_token_estimate: int | None,
        size_bucket: str | None,
        queued_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito_snapshot: bool = False,
        remember_across_chats_snapshot: bool = True,
        remember_across_devices_snapshot: bool = True,
        temporary_snapshot: bool = False,
        purge_on_close_snapshot: bool = False,
        policy_snapshot: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        timestamp = queued_at or self._timestamp()
        await self._connection.execute(
            """
            INSERT OR IGNORE INTO worker_job_runs(
                job_id,
                stream_name,
                job_type,
                user_id,
                conversation_id,
                source_message_ids_json,
                status,
                source_token_estimate,
                size_bucket,
                queued_at,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                temporary_snapshot,
                purge_on_close_snapshot,
                policy_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                stream_name,
                job_type,
                user_id,
                conversation_id,
                json_utils.dumps(source_message_ids, sort_keys=True),
                JobRunStatus.QUEUED.value,
                source_token_estimate,
                size_bucket,
                timestamp,
                json_utils.dumps(metadata or {}, sort_keys=True),
                user_persona_id,
                platform_id,
                character_id,
                1 if incognito_snapshot else 0,
                1 if remember_across_chats_snapshot else 0,
                1 if remember_across_devices_snapshot else 0,
                1 if temporary_snapshot else 0,
                1 if purge_on_close_snapshot else 0,
                json_utils.dumps(policy_snapshot or {}, sort_keys=True),
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Failed to create worker job run {job_id}")
        return row

    async def mark_running(
        self,
        job_id: str,
        *,
        attempt_count: int,
        commit: bool = True,
    ) -> None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                attempt_count = MAX(attempt_count, ?),
                started_at = COALESCE(started_at, ?),
                last_heartbeat_at = ?
            WHERE job_id = ?
            """,
            (
                JobRunStatus.RUNNING.value,
                max(1, int(attempt_count)),
                timestamp,
                timestamp,
                job_id,
            ),
        )
        if commit:
            await self._connection.commit()

    async def mark_retrying(
        self,
        job_id: str,
        *,
        attempt_count: int,
        error_class: str | None = None,
        error_message: str | None = None,
        commit: bool = True,
    ) -> None:
        await self._mark_terminal_or_retrying(
            job_id,
            status=JobRunStatus.RETRYING,
            attempt_count=attempt_count,
            error_class=error_class,
            error_message=error_message,
            finished=False,
            commit=commit,
        )

    async def mark_succeeded(
        self,
        job_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> None:
        await self._mark_completed(
            job_id,
            status=JobRunStatus.SUCCEEDED,
            metadata=metadata,
            commit=commit,
        )

    async def mark_skipped(
        self,
        job_id: str,
        *,
        reason: str | None = None,
        commit: bool = True,
    ) -> None:
        metadata = {"skip_reason": reason} if reason else None
        await self._mark_completed(
            job_id,
            status=JobRunStatus.SKIPPED,
            metadata=metadata,
            commit=commit,
        )

    async def mark_failed(
        self,
        job_id: str,
        *,
        error_class: str | None = None,
        error_message: str | None = None,
        commit: bool = True,
    ) -> None:
        await self._mark_terminal_or_retrying(
            job_id,
            status=JobRunStatus.FAILED,
            error_class=error_class,
            error_message=error_message,
            finished=True,
            commit=commit,
        )

    async def mark_dead_lettered(
        self,
        job_id: str,
        *,
        attempt_count: int,
        error_class: str | None = None,
        error_message: str | None = None,
        commit: bool = True,
    ) -> None:
        await self._mark_terminal_or_retrying(
            job_id,
            status=JobRunStatus.DEAD_LETTERED,
            attempt_count=attempt_count,
            error_class=error_class,
            error_message=error_message,
            finished=True,
            commit=commit,
        )

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM worker_job_runs
            WHERE job_id = ?
            """,
            (job_id,),
        )

    async def oldest_nonterminal_queued_at(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> str | None:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            """
            SELECT MIN(queued_at) AS queued_at
            FROM worker_job_runs
            WHERE {where_clause}
              AND status IN ({placeholders})
            """.format(where_clause=where_clause, placeholders=placeholders),
            (*parameters, *(status.value for status in NONTERMINAL_JOB_STATUSES)),
        )
        row = await cursor.fetchone()
        return None if row is None or row["queued_at"] is None else str(row["queued_at"])

    async def status_counts(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
        window_start: str | None = None,
        nonterminal_only: bool = False,
    ) -> list[dict[str, Any]]:
        where_clause, parameters = self._optional_scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        clauses = [where_clause] if where_clause else []
        if window_start is not None:
            clauses.append("queued_at >= ?")
            parameters.append(window_start)
        if nonterminal_only:
            placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in NONTERMINAL_JOB_STATUSES)
        final_where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return await self._fetch_all(
            """
            SELECT status, job_type, COUNT(*) AS count
            FROM worker_job_runs
            {where_clause}
            GROUP BY status, job_type
            ORDER BY status ASC, job_type ASC
            """.format(where_clause=final_where),
            tuple(parameters),
        )

    async def nonterminal_jobs(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> list[dict[str, Any]]:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        return await self._fetch_all(
            """
            SELECT *
            FROM worker_job_runs
            WHERE {where_clause}
              AND status IN ({placeholders})
            ORDER BY queued_at ASC, job_id ASC
            """.format(where_clause=where_clause, placeholders=placeholders),
            (*parameters, *(status.value for status in NONTERMINAL_JOB_STATUSES)),
        )

    async def source_message_progress(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        namespace_filter: JobNamespaceFilter | None = None,
        window_start: str | None,
    ) -> dict[str, int]:
        if window_start is None:
            return {
                "tracked_source_messages": 0,
                "processed_source_messages": 0,
                "pending_source_messages": 0,
            }
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        root_placeholders = ", ".join("?" for _ in ROOT_JOB_TYPES)
        nonterminal_placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        cursor = await self._connection.execute(
            """
            WITH root_jobs AS (
                SELECT job_id, status, source_message_ids_json
                FROM worker_job_runs
                WHERE {where_clause}
                  AND queued_at >= ?
                  AND job_type IN ({root_placeholders})
            ),
            source_jobs AS (
                SELECT
                    CAST(json_each.value AS TEXT) AS source_message_id,
                    root_jobs.status AS status
                FROM root_jobs, json_each(root_jobs.source_message_ids_json)
            ),
            source_rollup AS (
                SELECT
                    source_message_id,
                    SUM(CASE WHEN status IN ({nonterminal_placeholders}) THEN 1 ELSE 0 END) AS nonterminal_jobs
                FROM source_jobs
                WHERE source_message_id IS NOT NULL
                  AND source_message_id != ''
                GROUP BY source_message_id
            )
            SELECT
                COUNT(*) AS tracked_source_messages,
                COALESCE(SUM(CASE WHEN nonterminal_jobs = 0 THEN 1 ELSE 0 END), 0) AS processed_source_messages,
                COALESCE(SUM(CASE WHEN nonterminal_jobs > 0 THEN 1 ELSE 0 END), 0) AS pending_source_messages
            FROM source_rollup
            """.format(
                where_clause=where_clause,
                root_placeholders=root_placeholders,
                nonterminal_placeholders=nonterminal_placeholders,
            ),
            (
                *parameters,
                window_start,
                *(job_type.value for job_type in ROOT_JOB_TYPES),
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
            ),
        )
        row = await cursor.fetchone()
        return {
            "tracked_source_messages": int(row["tracked_source_messages"] or 0),
            "processed_source_messages": int(row["processed_source_messages"] or 0),
            "pending_source_messages": int(row["pending_source_messages"] or 0),
        }

    async def recent_completed_durations(
        self,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT job_type, size_bucket, duration_ms
            FROM worker_job_runs
            WHERE status = ?
              AND duration_ms IS NOT NULL
            ORDER BY finished_at DESC, job_id DESC
            LIMIT ?
            """,
            (JobRunStatus.SUCCEEDED.value, limit),
        )

    async def newest_job_queued_at(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> str | None:
        where_clause, parameters = self._scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        cursor = await self._connection.execute(
            """
            SELECT MAX(queued_at) AS queued_at
            FROM worker_job_runs
            WHERE {where_clause}
            """.format(where_clause=where_clause),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return None if row is None or row["queued_at"] is None else str(row["queued_at"])

    async def purge_for_user(self, user_id: str, *, commit: bool = True) -> int:
        cursor = await self._connection.execute(
            "DELETE FROM worker_job_runs WHERE user_id = ?",
            (user_id,),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def purge_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM worker_job_runs
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def nonterminal_root_jobs_for_recovery(self) -> list[dict[str, Any]]:
        """Return message-derived nonterminal jobs that can be stream-recovered."""
        root_placeholders = ", ".join("?" for _ in ROOT_JOB_TYPES)
        nonterminal_placeholders = ", ".join("?" for _ in NONTERMINAL_JOB_STATUSES)
        return await self._fetch_all(
            """
            SELECT *
            FROM worker_job_runs
            WHERE job_type IN ({root_placeholders})
              AND status IN ({nonterminal_placeholders})
            ORDER BY queued_at ASC, job_id ASC
            """.format(
                root_placeholders=root_placeholders,
                nonterminal_placeholders=nonterminal_placeholders,
            ),
            (
                *(job_type.value for job_type in ROOT_JOB_TYPES),
                *(status.value for status in NONTERMINAL_JOB_STATUSES),
            ),
        )

    async def mark_requeued_for_recovery(
        self,
        job_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> None:
        """Move a nonterminal durable job back to queued after stream recovery."""
        metadata_json = None
        if metadata:
            existing = await self.get_job(job_id)
            existing_metadata = (
                existing.get("metadata_json")
                if existing and isinstance(existing.get("metadata_json"), dict)
                else {}
            )
            metadata_json = json_utils.dumps(
                {**existing_metadata, **metadata},
                sort_keys=True,
            )
        await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                started_at = NULL,
                finished_at = NULL,
                last_heartbeat_at = NULL,
                duration_ms = NULL,
                error_class = NULL,
                error_message = NULL,
                metadata_json = COALESCE(?, metadata_json)
            WHERE job_id = ?
            """,
            (
                JobRunStatus.QUEUED.value,
                metadata_json,
                job_id,
            ),
        )
        if commit:
            await self._connection.commit()

    async def _mark_completed(
        self,
        job_id: str,
        *,
        status: JobRunStatus,
        metadata: dict[str, Any] | None = None,
        commit: bool,
    ) -> None:
        timestamp = self._timestamp()
        if metadata:
            existing = await self.get_job(job_id)
            existing_metadata = (
                existing.get("metadata_json")
                if existing and isinstance(existing.get("metadata_json"), dict)
                else {}
            )
            metadata_json = json_utils.dumps(
                {**existing_metadata, **metadata},
                sort_keys=True,
            )
        else:
            metadata_json = None
        await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                finished_at = ?,
                last_heartbeat_at = ?,
                error_class = NULL,
                error_message = NULL,
                duration_ms = CASE
                    WHEN started_at IS NOT NULL
                    THEN MAX(0.0, (julianday(?) - julianday(started_at)) * 86400000.0)
                    ELSE duration_ms
                END,
                metadata_json = COALESCE(?, metadata_json)
            WHERE job_id = ?
            """,
            (
                status.value,
                timestamp,
                timestamp,
                timestamp,
                metadata_json,
                job_id,
            ),
        )
        if commit:
            await self._connection.commit()

    async def _mark_terminal_or_retrying(
        self,
        job_id: str,
        *,
        status: JobRunStatus,
        attempt_count: int | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
        finished: bool,
        commit: bool,
    ) -> None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                attempt_count = CASE
                    WHEN ? IS NULL THEN attempt_count
                    ELSE MAX(attempt_count, ?)
                END,
                last_heartbeat_at = ?,
                finished_at = CASE WHEN ? THEN ? ELSE finished_at END,
                duration_ms = CASE
                    WHEN ? AND started_at IS NOT NULL
                    THEN MAX(0.0, (julianday(?) - julianday(started_at)) * 86400000.0)
                    ELSE duration_ms
                END,
                error_class = ?,
                error_message = ?
            WHERE job_id = ?
            """,
            (
                status.value,
                attempt_count,
                attempt_count,
                timestamp,
                int(finished),
                timestamp,
                int(finished),
                timestamp,
                error_class,
                _truncate_error(error_message),
                job_id,
            ),
        )
        if commit:
            await self._connection.commit()

    @staticmethod
    def _scope_where_clause(
        *,
        user_id: str,
        conversation_id: str | None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> tuple[str, list[Any]]:
        where_clause, parameters = JobRunRepository._optional_scope_where_clause(
            user_id=user_id,
            conversation_id=conversation_id,
            namespace_filter=namespace_filter,
        )
        if not where_clause:
            raise ValueError("user_id is required")
        return where_clause, parameters

    @staticmethod
    def _optional_scope_where_clause(
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        namespace_filter: JobNamespaceFilter | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(conversation_id)
        if namespace_filter is not None:
            clauses.append("user_persona_id IS ?")
            parameters.append(namespace_filter.user_persona_id)
            platform_id = namespace_filter.platform_id or "default"
            if namespace_filter.incognito or not namespace_filter.remember_across_chats:
                clauses.append("conversation_id IS NOT NULL")
                clauses.append("incognito_snapshot = ?")
                parameters.append(1 if namespace_filter.incognito else 0)
                if not namespace_filter.remember_across_chats:
                    clauses.append("remember_across_chats_snapshot = 0")
            else:
                clauses.append("incognito_snapshot = 0")
                clauses.append("remember_across_chats_snapshot = 1")
                clauses.append("character_id IS ?")
                parameters.append(namespace_filter.character_id)
            if namespace_filter.remember_across_devices:
                clauses.append("(remember_across_devices_snapshot = 1 OR platform_id = ?)")
                parameters.append(platform_id)
            else:
                clauses.append("platform_id = ?")
                parameters.append(platform_id)
        return " AND ".join(clauses), parameters


def job_status_values(statuses: Iterable[JobRunStatus]) -> tuple[str, ...]:
    """Return enum values as a tuple for callers that need raw SQL parameters."""
    return tuple(status.value for status in statuses)


def _truncate_error(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalized[:limit]
