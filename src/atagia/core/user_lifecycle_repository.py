"""SQLite-owned user lifecycle identity and revision operations."""

from __future__ import annotations

from dataclasses import dataclass

from atagia.core.repositories import BaseRepository


@dataclass(frozen=True, slots=True)
class UserLifecycleIdentity:
    """Immutable lifecycle identity plus its current cache revision."""

    user_id: str
    lifecycle_epoch: str
    lifecycle_cleanup_key: str
    cache_revision: int
    derivation_revision: int
    source_revision: int
    icp_refresh_generation: int


class UserLifecycleRepository(BaseRepository):
    """Read and advance the canonical lifecycle identity stored in SQLite."""

    async def get_active_identity(self, user_id: str) -> UserLifecycleIdentity | None:
        row = await self._fetch_one(
            """
            SELECT
                lifecycle.user_id AS id,
                lifecycle.lifecycle_epoch,
                lifecycle.lifecycle_cleanup_key,
                lifecycle.cache_revision,
                lifecycle.derivation_revision,
                lifecycle.source_revision,
                lifecycle.icp_refresh_generation
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
            """,
            (user_id,),
        )
        if row is None:
            return None
        return UserLifecycleIdentity(
            user_id=str(row["id"]),
            lifecycle_epoch=str(row["lifecycle_epoch"]),
            lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
            cache_revision=int(row["cache_revision"]),
            derivation_revision=int(row["derivation_revision"]),
            source_revision=int(row["source_revision"]),
            icp_refresh_generation=int(row["icp_refresh_generation"]),
        )

    async def matches_active_cache_identity(
        self,
        user_id: str,
        *,
        lifecycle_epoch: str,
        cache_revision: int,
    ) -> bool:
        """Return whether an exact SQLite-owned cache identity is still active."""

        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM user_lifecycles AS lifecycle
            JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.lifecycle_epoch = ?
              AND lifecycle.cache_revision = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND users.deleted_at IS NULL
            LIMIT 1
            """,
            (user_id, lifecycle_epoch, cache_revision),
        )
        return await cursor.fetchone() is not None

    async def matches_active_context_identity(
        self,
        user_id: str,
        *,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
    ) -> bool:
        """Return whether cache and canonical source coordinates are exact."""

        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM user_lifecycles AS lifecycle
            JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.lifecycle_epoch = ?
              AND lifecycle.cache_revision = ?
              AND lifecycle.derivation_revision = ?
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
              AND users.deleted_at IS NULL
            LIMIT 1
            """,
            (
                user_id,
                lifecycle_epoch,
                cache_revision,
                derivation_revision,
            ),
        )
        return await cursor.fetchone() is not None

    async def matches_active_epoch(self, user_id: str, lifecycle_epoch: str) -> bool:
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM user_lifecycles AS lifecycle
            LEFT JOIN users ON users.id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.lifecycle_epoch = ?
              AND (
                  lifecycle.user_id = 'atagia_system'
                  OR (users.id IS NOT NULL AND users.deleted_at IS NULL)
              )
              AND lifecycle.state = 'active'
              AND lifecycle.erasure_cleanup_id IS NULL
            LIMIT 1
            """,
            (user_id, lifecycle_epoch),
        )
        return await cursor.fetchone() is not None

    async def bump_cache_revision(
        self,
        user_id: str,
        *,
        expected_lifecycle_epoch: str,
        commit: bool = True,
    ) -> int | None:
        """Advance the revision only within the expected active lifecycle."""

        cursor = await self._connection.execute(
            """
            UPDATE user_lifecycles
            SET cache_revision = cache_revision + 1,
                updated_at = ?
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND state = 'active'
              AND erasure_cleanup_id IS NULL
            RETURNING cache_revision
            """,
            (self._timestamp(), user_id, expected_lifecycle_epoch),
        )
        row = await cursor.fetchone()
        if commit:
            await self._connection.commit()
        return None if row is None else int(row["cache_revision"])

    async def bump_derivation_revision(
        self,
        user_id: str,
        *,
        expected_lifecycle_epoch: str,
        commit: bool = True,
    ) -> int | None:
        """Invalidate captured derived-work claims within one active lifecycle."""

        cursor = await self._connection.execute(
            """
            UPDATE user_lifecycles
            SET derivation_revision = derivation_revision + 1,
                updated_at = ?
            WHERE user_id = ?
              AND lifecycle_epoch = ?
              AND state = 'active'
              AND erasure_cleanup_id IS NULL
              AND (
                  user_id = 'atagia_system'
                  OR EXISTS (
                      SELECT 1
                      FROM users
                      WHERE users.id = user_lifecycles.user_id
                        AND users.deleted_at IS NULL
                  )
              )
            RETURNING derivation_revision
            """,
            (self._timestamp(), user_id, expected_lifecycle_epoch),
        )
        row = await cursor.fetchone()
        if commit:
            await self._connection.commit()
        return None if row is None else int(row["derivation_revision"])

    async def reserve_icp_refresh_generation(
        self,
        user_id: str,
        *,
        expected_lifecycle_epoch: str | None = None,
        commit: bool = True,
    ) -> int | None:
        """Reserve one globally monotonic refresh generation for an active user."""

        clauses = [
            "user_id = ?",
            "state = 'active'",
            "erasure_cleanup_id IS NULL",
        ]
        parameters: list[object] = [user_id]
        if expected_lifecycle_epoch is not None:
            clauses.append("lifecycle_epoch = ?")
            parameters.append(expected_lifecycle_epoch)
        cursor = await self._connection.execute(
            f"""
            UPDATE user_lifecycles
            SET icp_refresh_generation = icp_refresh_generation + 1,
                updated_at = ?
            WHERE {" AND ".join(clauses)}
              AND EXISTS (
                  SELECT 1
                  FROM users
                  WHERE users.id = user_lifecycles.user_id
                    AND users.deleted_at IS NULL
              )
            RETURNING icp_refresh_generation
            """,
            (self._timestamp(), *parameters),
        )
        row = await cursor.fetchone()
        if commit:
            await self._connection.commit()
        return None if row is None else int(row["icp_refresh_generation"])
