"""SQLite repository for belief versioning and links."""

from __future__ import annotations

import re
from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.models.schemas_memory import MemoryObjectType, MemoryStatus

ALLOWED_MEMORY_LINK_RELATIONS = frozenset(
    {
        "supports",
        "contradicts",
        "depends_on",
        "derived_from",
        "supersedes",
        "exception_to",
        "about_topic",
        "mentions_entity",
        "applies_in_mode",
        "belongs_to_workspace",
        "led_to",
        "reinforces",
        "weakens",
    }
)
_CLAIM_KEY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _claim_key_tokens(claim_key: str) -> set[str]:
    return set(_CLAIM_KEY_TOKEN_PATTERN.findall(claim_key.lower()))


class BeliefRepository(BaseRepository):
    """Focused persistence helpers for belief revision flows."""

    async def create_first_version(
        self,
        belief_id: str,
        claim_key: str,
        claim_value: Any,
        created_at: str,
        *,
        condition: dict[str, Any] | None = None,
        support_count: int = 1,
        contradict_count: int = 0,
        supersedes_version: int | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        await self._connection.execute(
            """
            INSERT INTO belief_versions(
                belief_id,
                version,
                claim_key,
                claim_value_json,
                condition_json,
                support_count,
                contradict_count,
                supersedes_version,
                is_current,
                created_at
            )
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                belief_id,
                claim_key,
                _encode_json(claim_value),
                _encode_json(condition or {}),
                support_count,
                contradict_count,
                supersedes_version,
                created_at,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self._fetch_one(
            """
            SELECT *
            FROM belief_versions
            WHERE belief_id = ?
              AND version = 1
            """,
            (belief_id,),
        )
        if row is None:
            raise RuntimeError(f"Failed to create first version for belief {belief_id}")
        return row

    async def find_active_beliefs_by_claim_key(
        self,
        user_id: str,
        claim_key: str,
    ) -> list[dict[str, Any]]:
        rows = await self._active_belief_rows(user_id)
        normalized = claim_key.strip().lower()
        return [
            row
            for row in rows
            if str(row["claim_key"]).strip().lower() == normalized
        ]

    async def find_active_belief_candidates_by_claim_key(
        self,
        user_id: str,
        claim_key: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = await self._active_belief_rows(user_id)
        normalized = claim_key.strip().lower()
        target_tokens = _claim_key_tokens(normalized)
        ranked: list[tuple[tuple[int, int, str, str], dict[str, Any]]] = []
        for row in rows:
            candidate_key = str(row["claim_key"]).strip().lower()
            candidate_tokens = _claim_key_tokens(candidate_key)
            overlap = len(target_tokens & candidate_tokens)
            exact = int(candidate_key == normalized)
            if not exact and overlap == 0:
                continue
            rank = (
                -exact,
                -overlap,
                str(row["updated_at"]),
                str(row["belief_id"]),
            )
            ranked.append((rank, row))
        ranked.sort(key=lambda item: item[0], reverse=False)
        return [
            row
            for _, row in ranked[:limit]
        ]

    async def _active_belief_rows(self, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT
                mo.id AS belief_id,
                bv.claim_key,
                bv.claim_value_json,
                bv.condition_json,
                mo.scope,
                mo.confidence,
                mo.stability,
                bv.support_count,
                bv.contradict_count,
                mo.assistant_mode_id,
                mo.workspace_id,
                mo.conversation_id,
                mo.payload_json,
                mo.status,
                mo.created_at,
                mo.updated_at
            FROM memory_objects AS mo
            JOIN belief_versions AS bv
              ON bv.belief_id = mo.id
             AND bv.is_current = 1
            WHERE mo.user_id = ?
              AND mo.object_type = ?
              AND mo.status = ?
            ORDER BY mo.updated_at DESC, mo.id ASC
            """,
            (
                user_id,
                MemoryObjectType.BELIEF.value,
                MemoryStatus.ACTIVE.value,
            ),
        )

    async def create_new_version(
        self,
        belief_id: str,
        user_id: str,
        version: int,
        claim_key: str,
        claim_value: Any,
        condition: dict[str, Any] | None,
        support_count: int,
        contradict_count: int,
        supersedes_version: int | None,
        created_at: str,
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        await self._connection.execute(
            """
            UPDATE belief_versions
            SET is_current = 0
            WHERE belief_id IN (
                SELECT id
                FROM memory_objects
                WHERE id = ?
                  AND user_id = ?
            )
            """,
            (belief_id, user_id),
        )
        await self._connection.execute(
            """
            INSERT INTO belief_versions(
                belief_id,
                version,
                claim_key,
                claim_value_json,
                condition_json,
                support_count,
                contradict_count,
                supersedes_version,
                is_current,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                belief_id,
                version,
                claim_key,
                _encode_json(claim_value),
                _encode_json(condition or {}),
                support_count,
                contradict_count,
                supersedes_version,
                created_at,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self._fetch_one(
            """
            SELECT *
            FROM belief_versions
            WHERE belief_id IN (
                SELECT id
                FROM memory_objects
                WHERE id = ?
                  AND user_id = ?
            )
              AND version = ?
            """,
            (belief_id, user_id, version),
        )
        if row is None:
            raise RuntimeError(f"Failed to create version {version} for belief {belief_id}")
        return row

    async def get_current_version(self, belief_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM belief_versions
            WHERE belief_id IN (
                SELECT id
                FROM memory_objects
                WHERE id = ?
                  AND user_id = ?
            )
              AND is_current = 1
            ORDER BY version DESC
            LIMIT 1
            """,
            (belief_id, user_id),
        )

    async def get_version_history(self, belief_id: str, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM belief_versions
            WHERE belief_id IN (
                SELECT id
                FROM memory_objects
                WHERE id = ?
                  AND user_id = ?
            )
            ORDER BY version ASC
            """,
            (belief_id, user_id),
        )

    async def count_supporting_evidence(
        self,
        user_id: str,
        claim_key: str,
        min_conversations: int = 1,
    ) -> dict[str, Any]:
        cursor = await self._connection.execute(
            """
            WITH support_candidates AS (
                SELECT
                    mo.id AS memory_id,
                    mo.conversation_id,
                    mo.assistant_mode_id,
                    mo.created_at
                FROM memory_objects AS mo
                JOIN belief_versions AS bv
                  ON bv.belief_id = mo.id
                 AND bv.is_current = 1
                WHERE mo.user_id = ?
                  AND mo.status = ?
                  AND mo.object_type = ?
                  AND bv.claim_key = ?
                UNION
                SELECT
                    mo.id AS memory_id,
                    mo.conversation_id,
                    mo.assistant_mode_id,
                    mo.created_at
                FROM memory_objects AS mo
                WHERE mo.user_id = ?
                  AND mo.status = ?
                  AND mo.object_type = ?
                  AND json_extract(mo.payload_json, '$.claim_key') = ?
            )
            SELECT
                COUNT(*) AS total_evidence,
                COUNT(DISTINCT conversation_id) AS distinct_conversations,
                COUNT(DISTINCT substr(created_at, 1, 10)) AS distinct_sessions,
                MIN(created_at) AS oldest_at,
                MAX(created_at) AS newest_at
            FROM support_candidates
            """,
            (
                user_id,
                MemoryStatus.ACTIVE.value,
                MemoryObjectType.BELIEF.value,
                claim_key,
                user_id,
                MemoryStatus.ACTIVE.value,
                MemoryObjectType.EVIDENCE.value,
                claim_key,
            ),
        )
        row = await cursor.fetchone()
        stats = {
            "total_evidence": int(row["total_evidence"] or 0),
            "distinct_conversations": int(row["distinct_conversations"] or 0),
            "distinct_sessions": int(row["distinct_sessions"] or 0),
            "oldest_at": row["oldest_at"],
            "newest_at": row["newest_at"],
        }
        if stats["distinct_conversations"] < min_conversations:
            return stats
        return stats

    async def create_memory_link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        *,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if relation_type not in ALLOWED_MEMORY_LINK_RELATIONS:
            raise ValueError(f"Unsupported memory link relation_type: {relation_type}")

        source_row = await self._fetch_one(
            "SELECT id, user_id FROM memory_objects WHERE id = ?",
            (source_id,),
        )
        target_row = await self._fetch_one(
            "SELECT id, user_id FROM memory_objects WHERE id = ?",
            (target_id,),
        )
        if source_row is None or target_row is None:
            raise ValueError("Memory links require existing source and target memory objects")
        if source_row["user_id"] != target_row["user_id"]:
            raise ValueError("Memory links cannot cross user boundaries")

        link_id = generate_prefixed_id("lnk")
        created_at = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO memory_links(
                id,
                user_id,
                src_memory_id,
                dst_memory_id,
                relation_type,
                weight,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                link_id,
                source_row["user_id"],
                source_id,
                target_id,
                relation_type,
                confidence,
                _encode_json(metadata or {}),
                created_at,
            ),
        )
        if commit:
            await self._connection.commit()
        cursor = await self._connection.execute(
            "SELECT * FROM memory_links WHERE id = ?",
            (link_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to create memory link {link_id}")
        return _decode_json_columns(row) or {}
