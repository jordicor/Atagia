"""SQLite repository helpers for natural-memory consent profiles."""

from __future__ import annotations

from typing import Any

from atagia.core.repositories import BaseRepository
from atagia.models.schemas_memory import MemoryCategory


class MemoryConsentProfileRepository(BaseRepository):
    """Persistence operations for memory_consent_profile."""

    async def get_profile(
        self,
        user_id: str,
        category: MemoryCategory,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_consent_profile
            WHERE user_id = ?
              AND category = ?
            """,
            (user_id, category.value),
        )

    async def upsert_profile(
        self,
        *,
        user_id: str,
        category: MemoryCategory,
        confirmed_count: int,
        declined_count: int,
        last_confirmed_at: str | None = None,
        last_declined_at: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            INSERT INTO memory_consent_profile(
                user_id,
                category,
                confirmed_count,
                declined_count,
                last_confirmed_at,
                last_declined_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, category) DO UPDATE SET
                confirmed_count = excluded.confirmed_count,
                declined_count = excluded.declined_count,
                last_confirmed_at = excluded.last_confirmed_at,
                last_declined_at = excluded.last_declined_at,
                updated_at = excluded.updated_at
            RETURNING *
            """,
            (
                user_id,
                category.value,
                confirmed_count,
                declined_count,
                last_confirmed_at,
                last_declined_at,
                timestamp,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if commit:
            await self._connection.commit()
        if row is None:
            raise RuntimeError("Failed to upsert consent profile")
        decoded = self._decode_row(row)
        if decoded is None:
            raise RuntimeError("Failed to decode consent profile")
        return decoded

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)
