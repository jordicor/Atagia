"""SQLite repository helpers for natural-memory consent state."""

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


class PendingMemoryConfirmationRepository(BaseRepository):
    """Persistence operations for pending memory confirmation markers."""

    async def create_marker(
        self,
        *,
        user_id: str,
        conversation_id: str,
        memory_id: str,
        category: MemoryCategory,
        created_at: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        cursor = await self._connection.execute(
            """
            INSERT INTO pending_memory_confirmations(
                user_id,
                conversation_id,
                memory_id,
                memory_category,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, memory_id) DO NOTHING
            RETURNING *
            """,
            (
                user_id,
                conversation_id,
                memory_id,
                category.value,
                created_at,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if commit:
            await self._connection.commit()
        if row is not None:
            decoded = self._decode_row(row)
            if decoded is not None:
                return decoded
        existing = await self.get_marker_for_memory(user_id, memory_id)
        if existing is None:
            raise RuntimeError("Failed to create pending confirmation marker")
        return existing

    async def get_marker_for_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM pending_memory_confirmations
            WHERE user_id = ?
              AND memory_id = ?
            """,
            (user_id, memory_id),
        )

    async def get_oldest_unasked_marker(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM pending_memory_confirmations
            WHERE user_id = ?
              AND conversation_id = ?
              AND asked_at IS NULL
            ORDER BY created_at ASC, _rowid ASC
            LIMIT 1
            """,
            (user_id, conversation_id),
        )

    async def get_oldest_asked_marker(
        self,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM pending_memory_confirmations
            WHERE user_id = ?
              AND conversation_id = ?
              AND asked_at IS NOT NULL
            ORDER BY asked_at ASC, created_at ASC, _rowid ASC
            LIMIT 1
            """,
            (user_id, conversation_id),
        )

    async def list_markers_for_category(
        self,
        user_id: str,
        conversation_id: str,
        category: MemoryCategory,
        *,
        asked: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses = [
            "user_id = ?",
            "conversation_id = ?",
            "memory_category = ?",
        ]
        parameters: list[Any] = [user_id, conversation_id, category.value]
        if asked is True:
            clauses.append("asked_at IS NOT NULL")
        elif asked is False:
            clauses.append("asked_at IS NULL")
        return await self._fetch_all(
            """
            SELECT *
            FROM pending_memory_confirmations
            WHERE {clauses}
            ORDER BY created_at ASC, _rowid ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def mark_markers_asked(
        self,
        user_id: str,
        memory_ids: list[str],
        *,
        asked_at: str | None = None,
        commit: bool = True,
    ) -> int:
        if not memory_ids:
            return 0
        timestamp = asked_at or self._timestamp()
        placeholders = ", ".join("?" for _ in memory_ids)
        cursor = await self._connection.execute(
            f"""
            UPDATE pending_memory_confirmations
            SET asked_at = ?
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
            """,
            (timestamp, user_id, *memory_ids),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def reset_after_ambiguous(
        self,
        user_id: str,
        memory_ids: list[str],
        *,
        commit: bool = True,
    ) -> int:
        if not memory_ids:
            return 0
        placeholders = ", ".join("?" for _ in memory_ids)
        cursor = await self._connection.execute(
            f"""
            UPDATE pending_memory_confirmations
            SET asked_at = NULL,
                confirmation_asked_once = 1
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
            """,
            (user_id, *memory_ids),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def clear_markers(
        self,
        user_id: str,
        memory_ids: list[str],
        *,
        commit: bool = True,
    ) -> int:
        if not memory_ids:
            return 0
        placeholders = ", ".join("?" for _ in memory_ids)
        cursor = await self._connection.execute(
            f"""
            DELETE FROM pending_memory_confirmations
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
            """,
            (user_id, *memory_ids),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)
