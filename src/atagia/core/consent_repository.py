"""SQLite repository helpers for natural-memory consent state."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.repositories import BaseRepository
from atagia.models.schemas_memory import MemoryCategory, MemoryScope, MemorySensitivity


class MemoryConsentProfileRepository(BaseRepository):
    """Persistence operations for memory_consent_profile."""

    async def get_profile(
        self,
        user_id: str,
        category: MemoryCategory,
        user_persona_id: str | None = None,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_consent_profile
            WHERE user_id = ?
              AND category = ?
              AND user_persona_id IS ?
            """,
            (user_id, category.value, user_persona_id),
        )

    async def upsert_profile(
        self,
        *,
        user_id: str,
        category: MemoryCategory,
        confirmed_count: int,
        declined_count: int,
        user_persona_id: str | None = None,
        last_confirmed_at: str | None = None,
        last_declined_at: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        timestamp = self._timestamp()
        existing = await self.get_profile(
            user_id,
            category,
            user_persona_id=user_persona_id,
        )
        if existing is not None:
            cursor = await self._connection.execute(
                """
                UPDATE memory_consent_profile
                SET confirmed_count = ?,
                    declined_count = ?,
                    last_confirmed_at = ?,
                    last_declined_at = ?,
                    updated_at = ?
                WHERE _rowid = ?
                RETURNING *
                """,
                (
                    confirmed_count,
                    declined_count,
                    last_confirmed_at,
                    last_declined_at,
                    timestamp,
                    existing["_rowid"],
                ),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if commit:
                await self._connection.commit()
            if row is None:
                raise RuntimeError("Failed to update consent profile")
            decoded = self._decode_row(row)
            if decoded is None:
                raise RuntimeError("Failed to decode consent profile")
            return decoded
        try:
            cursor = await self._connection.execute(
                """
                INSERT INTO memory_consent_profile(
                    user_id,
                    category,
                    confirmed_count,
                    declined_count,
                    last_confirmed_at,
                    last_declined_at,
                    updated_at,
                    user_persona_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    user_persona_id,
                ),
            )
        except aiosqlite.IntegrityError:
            # The additive migration cannot remove the old UNIQUE(user_id,
            # category) autoindex until the Phase 11 table rebuild. Fail closed:
            # do not overwrite another persona's consent profile.
            return {
                "user_id": user_id,
                "category": category.value,
                "confirmed_count": confirmed_count,
                "declined_count": declined_count,
                "last_confirmed_at": last_confirmed_at,
                "last_declined_at": last_declined_at,
                "updated_at": timestamp,
                "user_persona_id": user_persona_id,
                "not_persisted_reason": "legacy_unique_user_category",
            }
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
        payload = dict(row)
        value = payload.get("policy_snapshot_json")
        if isinstance(value, str):
            payload["policy_snapshot_json"] = json_utils.loads(value)
        return payload


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
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito_snapshot: bool = False,
        remember_across_chats_snapshot: bool = True,
        remember_across_devices_snapshot: bool = True,
        temporary_snapshot: bool = False,
        purge_on_close_snapshot: bool = False,
        valid_to_snapshot: str | None = None,
        intended_scope: MemoryScope | None = None,
        intended_sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        policy_snapshot: dict[str, Any] | None = None,
        policy_proven: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        cursor = await self._connection.execute(
            """
            INSERT INTO pending_memory_confirmations(
                user_id,
                conversation_id,
                memory_id,
                memory_category,
                created_at,
                user_persona_id,
                platform_id,
                character_id,
                mode,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                temporary_snapshot,
                purge_on_close_snapshot,
                valid_to_snapshot,
                intended_scope,
                intended_sensitivity,
                platform_locked,
                platform_id_lock,
                policy_snapshot_json,
                policy_proven
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, memory_id) DO NOTHING
            RETURNING *
            """,
            (
                user_id,
                conversation_id,
                memory_id,
                category.value,
                created_at,
                user_persona_id,
                platform_id,
                character_id,
                mode,
                1 if incognito_snapshot else 0,
                1 if remember_across_chats_snapshot else 0,
                1 if remember_across_devices_snapshot else 0,
                1 if temporary_snapshot else 0,
                1 if purge_on_close_snapshot else 0,
                valid_to_snapshot,
                intended_scope.value if intended_scope is not None else None,
                intended_sensitivity.value,
                1 if platform_locked else 0,
                platform_id_lock,
                json_utils.dumps(policy_snapshot or {}, sort_keys=True),
                1 if policy_proven else 0,
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

    async def list_pending_markers(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        platform_id: str | None = None,
        user_persona_id: str | None = None,
        character_id: str | None = None,
        category: MemoryCategory | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = [
            "pmc.user_id = ?",
            "mo.status = 'pending_user_confirmation'",
        ]
        parameters: list[Any] = [user_id]
        if conversation_id is not None:
            clauses.append("pmc.conversation_id = ?")
            parameters.append(conversation_id)
        if platform_id is not None:
            clauses.append("pmc.platform_id = ?")
            parameters.append(platform_id)
        if user_persona_id is not None:
            clauses.append("pmc.user_persona_id IS ?")
            parameters.append(user_persona_id)
        if character_id is not None:
            clauses.append("pmc.character_id IS ?")
            parameters.append(character_id)
        if category is not None:
            clauses.append("pmc.memory_category = ?")
            parameters.append(category.value)
        return await self._fetch_all(
            """
            SELECT
                pmc.*,
                mo.object_type,
                mo.scope,
                mo.scope_canonical,
                mo.index_text,
                mo.canonical_text,
                mo.privacy_level,
                mo.sensitivity,
                mo.intimacy_boundary,
                mo.status AS memory_status
            FROM pending_memory_confirmations AS pmc
            JOIN memory_objects AS mo ON mo.id = pmc.memory_id
            WHERE {clauses}
            ORDER BY pmc.created_at ASC, pmc._rowid ASC
            LIMIT ?
            OFFSET ?
            """.format(clauses=" AND ".join(clauses)),
            (*parameters, max(1, min(int(limit), 500)), max(0, int(offset))),
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
        payload = dict(row)
        value = payload.get("policy_snapshot_json")
        if isinstance(value, str):
            payload["policy_snapshot_json"] = json_utils.loads(value)
        return payload
