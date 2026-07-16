"""Exact source-revision coordinates for initial-context-package builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.core.repositories import BaseRepository, _encode_json


@dataclass(frozen=True, slots=True)
class InitialContextPackageSourceCoordinates:
    """Immutable lifecycle epochs and revisions captured before a build."""

    user_lifecycle_epoch: str
    user_revision: int
    conversation_lifecycle_epoch: str | None = None
    conversation_revision: int | None = None

    @property
    def is_conversation_scoped(self) -> bool:
        return self.conversation_lifecycle_epoch is not None


@dataclass(frozen=True, slots=True)
class InitialContextPackageBuildAttempt:
    """Private build-attempt identity kept off the canonical package row."""

    attempt_id: str
    package_key_hash: str
    refresh_generation: int
    expected_package_row_version: int | None
    source_coordinates: InitialContextPackageSourceCoordinates


class InitialContextPackageRevisionRepository(BaseRepository):
    """Capture and compare database-owned package source coordinates."""

    async def capture_active_coordinates(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
    ) -> InitialContextPackageSourceCoordinates | None:
        row = await self._fetch_one(
            """
            SELECT
                lifecycle.lifecycle_epoch AS user_lifecycle_epoch,
                lifecycle.source_revision AS user_revision,
                conversation_lifecycle.lifecycle_epoch
                    AS conversation_lifecycle_epoch,
                conversation_lifecycle.source_revision
                    AS conversation_revision
            FROM user_lifecycles AS lifecycle
            JOIN users
              ON users.id = lifecycle.user_id
             AND users.deleted_at IS NULL
            LEFT JOIN conversation_lifecycles AS conversation_lifecycle
              ON conversation_lifecycle.user_id = lifecycle.user_id
             AND conversation_lifecycle.conversation_id = ?
            LEFT JOIN conversations
              ON conversations.id = conversation_lifecycle.conversation_id
             AND conversations.user_id = conversation_lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.state = 'active'
              AND (
                  ? IS NULL
                  OR (
                      conversations.id IS NOT NULL
                      AND conversation_lifecycle.lifecycle_epoch IS NOT NULL
                  )
              )
            LIMIT 1
            """,
            (conversation_id, user_id, conversation_id),
        )
        if row is None:
            return None
        return InitialContextPackageSourceCoordinates(
            user_lifecycle_epoch=str(row["user_lifecycle_epoch"]),
            user_revision=int(row["user_revision"]),
            conversation_lifecycle_epoch=(
                None
                if row.get("conversation_lifecycle_epoch") is None
                else str(row["conversation_lifecycle_epoch"])
            ),
            conversation_revision=(
                None
                if row.get("conversation_revision") is None
                else int(row["conversation_revision"])
            ),
        )

    async def coordinates_are_current(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        coordinates: InitialContextPackageSourceCoordinates,
    ) -> bool:
        current = await self.capture_active_coordinates(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return current == coordinates

    async def reserve_refresh_generation(
        self,
        *,
        user_id: str,
        expected_user_lifecycle_epoch: str,
        commit: bool = True,
    ) -> int | None:
        cursor = await self._connection.execute(
            """
            UPDATE user_lifecycles
            SET icp_refresh_generation = icp_refresh_generation + 1,
                updated_at = ?
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND state = 'active'
              AND EXISTS (
                  SELECT 1
                  FROM users
                  WHERE users.id = user_lifecycles.user_id
                    AND users.deleted_at IS NULL
              )
            RETURNING icp_refresh_generation
            """,
            (self._timestamp(), user_id, expected_user_lifecycle_epoch),
        )
        row = await cursor.fetchone()
        if commit:
            await self._connection.commit()
        return None if row is None else int(row["icp_refresh_generation"])

    async def begin_build_attempt(
        self,
        *,
        attempt_id: str,
        user_id: str,
        conversation_id: str | None,
        package_key_hash: str,
        refresh_generation: int,
        source_coordinates: InitialContextPackageSourceCoordinates,
        refresh_request_job_id: str | None,
        commit: bool = True,
    ) -> InitialContextPackageBuildAttempt:
        cursor = await self._connection.execute(
            """
            SELECT package_row_version
            FROM initial_context_packages
            WHERE user_id = ?
              AND package_key_hash = ?
            LIMIT 1
            """,
            (user_id, package_key_hash),
        )
        row = await cursor.fetchone()
        expected_row_version = None if row is None else int(row["package_row_version"])
        await self._connection.execute(
            """
            INSERT INTO initial_context_package_build_attempts(
                attempt_id,
                user_id,
                conversation_id,
                package_key_hash,
                refresh_generation,
                expected_package_row_version,
                source_user_lifecycle_epoch,
                source_user_revision,
                source_conversation_lifecycle_epoch,
                source_conversation_revision,
                status,
                refresh_request_job_id,
                created_at,
                diagnostics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'building', ?, ?, '{}')
            """,
            (
                attempt_id,
                user_id,
                conversation_id,
                package_key_hash,
                refresh_generation,
                expected_row_version,
                source_coordinates.user_lifecycle_epoch,
                source_coordinates.user_revision,
                source_coordinates.conversation_lifecycle_epoch,
                source_coordinates.conversation_revision,
                refresh_request_job_id,
                self._timestamp(),
            ),
        )
        if commit:
            await self._connection.commit()
        return InitialContextPackageBuildAttempt(
            attempt_id=attempt_id,
            package_key_hash=package_key_hash,
            refresh_generation=refresh_generation,
            expected_package_row_version=expected_row_version,
            source_coordinates=source_coordinates,
        )

    async def finish_build_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        diagnostics: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> bool:
        if status not in {"activated", "source_changed", "superseded", "failed"}:
            raise ValueError(f"Unsupported terminal build-attempt status: {status}")
        cursor = await self._connection.execute(
            """
            UPDATE initial_context_package_build_attempts
            SET status = ?,
                finished_at = ?,
                diagnostics_json = ?
            WHERE attempt_id = ?
              AND status = 'building'
            """,
            (
                status,
                self._timestamp(),
                _encode_json(diagnostics or {}),
                attempt_id,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0) == 1

    async def fail_building_attempts_for_generation(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        refresh_generation: int,
        error_class: str,
        commit: bool = True,
    ) -> int:
        """Close only attempts owned by one failed refresh generation/scope."""

        cursor = await self._connection.execute(
            """
            UPDATE initial_context_package_build_attempts
            SET status = 'failed',
                finished_at = ?,
                diagnostics_json = ?
            WHERE user_id = ?
              AND conversation_id IS ?
              AND refresh_generation = ?
              AND status = 'building'
            """,
            (
                self._timestamp(),
                _encode_json({"error_class": error_class}),
                user_id,
                conversation_id,
                refresh_generation,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def fail_abandoned_build_attempts(
        self,
        *,
        commit: bool = True,
    ) -> int:
        """Terminalize attempts that no live fenced refresh claim can resume."""

        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE initial_context_package_build_attempts AS attempt
            SET status = 'failed',
                finished_at = ?,
                diagnostics_json = ?
            WHERE attempt.status = 'building'
              AND NOT EXISTS (
                  SELECT 1
                  FROM worker_job_runs AS job
                  JOIN user_lifecycles AS lifecycle
                    ON lifecycle.user_id = job.user_id
                   AND lifecycle.lifecycle_epoch = job.lifecycle_epoch
                  LEFT JOIN users AS active_user ON active_user.id = job.user_id
                  WHERE job.job_id = attempt.refresh_request_job_id
                    AND job.user_id = attempt.user_id
                    AND job.job_type = 'refresh_initial_context_package'
                    AND job.status = 'running'
                    AND job.execution_owner IS NOT NULL
                    AND job.execution_lease_expires_at > ?
                    AND job.lifecycle_epoch = attempt.source_user_lifecycle_epoch
                    AND job.derivation_revision = lifecycle.derivation_revision
                    AND lifecycle.state = 'active'
                    AND lifecycle.erasure_cleanup_id IS NULL
                    AND active_user.id IS NOT NULL
                    AND active_user.deleted_at IS NULL
              )
            """,
            (
                timestamp,
                _encode_json({"reason": "abandoned_attempt_recovery"}),
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)
