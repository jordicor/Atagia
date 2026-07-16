"""Durable suppression identities for explicitly retired extracted memories."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from atagia.core.canonical import canonical_json_bytes
from atagia.core.repositories import BaseRepository


def memory_extraction_identity_hash(
    *,
    canonical_text: str,
    object_type: str,
    scope: str,
    user_persona_id: str | None,
    character_id: str | None,
    conversation_id: str | None,
    active_presence_id: str | None,
    source_presence_id: str | None,
    space_id: str | None,
    memory_owner_id: str | None,
    source_mind_id: str | None,
    embodiment_id: str | None,
    realm_id: str | None,
) -> str:
    """Hash the same canonical namespace identity used by extraction merging."""

    payload = {
        "v": 1,
        "canonical_text": " ".join(canonical_text.split()).casefold(),
        "object_type": object_type,
        "scope": scope,
        "user_persona_id": user_persona_id,
        "character_id": character_id,
        "conversation_id": conversation_id,
        "active_presence_id": active_presence_id,
        "source_presence_id": source_presence_id,
        "space_id": space_id,
        "memory_owner_id": memory_owner_id,
        "source_mind_id": source_mind_id,
        "embodiment_id": embodiment_id,
        "realm_id": realm_id,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class MemoryExtractionSuppressionRepository(BaseRepository):
    """Persist source-specific retirements across extraction worker revisions."""

    async def suppress_memories(
        self,
        memories: Iterable[dict[str, Any]],
        *,
        reason: str,
        replacement_memory_id: str | None = None,
        commit: bool = True,
    ) -> int:
        timestamp = self._timestamp()
        suppressed = 0
        for memory in memories:
            if str(memory.get("source_kind") or "") != "extracted":
                continue
            extraction_hash = self._extraction_hash(memory)
            user_id = str(memory["user_id"])
            memory_id = str(memory["id"])
            identity_hash = self.identity_hash_for_memory(memory)
            source_message_ids = await self.source_message_ids_for_memories([memory])
            for source_message_id in source_message_ids:
                await self._connection.execute(
                    """
                    INSERT INTO memory_extraction_suppressions(
                        user_id,
                        source_message_id,
                        identity_hash,
                        extraction_hash,
                        retired_memory_id,
                        replacement_memory_id,
                        reason,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        user_id,
                        source_message_id,
                        retired_memory_id
                    ) DO UPDATE SET
                        identity_hash = excluded.identity_hash,
                        extraction_hash = COALESCE(
                            excluded.extraction_hash,
                            memory_extraction_suppressions.extraction_hash
                        ),
                        replacement_memory_id = excluded.replacement_memory_id,
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        source_message_id,
                        identity_hash,
                        extraction_hash,
                        memory_id,
                        replacement_memory_id,
                        reason,
                        timestamp,
                        timestamp,
                    ),
                )
                suppressed += 1
        if commit:
            await self._connection.commit()
        return suppressed

    async def is_suppressed(
        self,
        *,
        user_id: str,
        source_message_id: str,
    ) -> bool:
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM memory_extraction_suppressions
            WHERE user_id = ?
              AND source_message_id = ?
            LIMIT 1
            """,
            (user_id, source_message_id),
        )
        return await cursor.fetchone() is not None

    async def source_message_ids_for_memories(
        self,
        memories: Iterable[dict[str, Any]],
    ) -> list[str]:
        """Return current, user-owned source messages from all provenance surfaces."""

        memory_rows = list(memories)
        if not memory_rows:
            return []
        user_ids = {str(memory["user_id"]) for memory in memory_rows}
        if len(user_ids) != 1:
            raise ValueError("memories must belong to one user")
        user_id = next(iter(user_ids))
        memory_ids = [str(memory["id"]) for memory in memory_rows]
        source_message_ids: set[str] = set()
        for memory in memory_rows:
            source_message_ids.update(self._payload_source_message_ids(memory))

        placeholders = ", ".join("?" for _ in memory_ids)
        cursor = await self._connection.execute(
            f"""
            SELECT message_id AS source_message_id
            FROM memory_evidence_spans
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
              AND message_id IS NOT NULL
            UNION
            SELECT source_message_id
            FROM memory_fact_facets
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
              AND source_message_id IS NOT NULL
            """,
            (user_id, *memory_ids, user_id, *memory_ids),
        )
        source_message_ids.update(
            str(row["source_message_id"])
            for row in await cursor.fetchall()
            if row["source_message_id"] is not None
        )
        if not source_message_ids:
            return []

        source_placeholders = ", ".join("?" for _ in source_message_ids)
        cursor = await self._connection.execute(
            f"""
            SELECT messages.id
            FROM messages
            JOIN conversations
              ON conversations.id = messages.conversation_id
            WHERE conversations.user_id = ?
              AND messages.id IN ({source_placeholders})
            """,
            (user_id, *sorted(source_message_ids)),
        )
        return sorted(str(row["id"]) for row in await cursor.fetchall())

    @staticmethod
    def identity_hash_for_memory(memory: dict[str, Any]) -> str:
        return memory_extraction_identity_hash(
            canonical_text=str(memory["canonical_text"]),
            object_type=str(memory["object_type"]),
            scope=str(memory.get("scope_canonical") or memory["scope"]),
            user_persona_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("user_persona_id")
            ),
            character_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("character_id")
            ),
            conversation_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("conversation_id")
            ),
            active_presence_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("active_presence_id")
            ),
            source_presence_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("source_presence_id")
            ),
            space_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("space_id")
            ),
            memory_owner_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("memory_owner_id")
            ),
            source_mind_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("source_mind_id")
            ),
            embodiment_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("embodiment_id")
            ),
            realm_id=MemoryExtractionSuppressionRepository._optional_text(
                memory.get("realm_id")
            ),
        )

    @staticmethod
    def _extraction_hash(memory: dict[str, Any]) -> str | None:
        direct = MemoryExtractionSuppressionRepository._optional_text(
            memory.get("extraction_hash")
        )
        if direct is not None:
            return direct
        payload = memory.get("payload_json")
        if isinstance(payload, dict):
            return MemoryExtractionSuppressionRepository._optional_text(
                payload.get("extraction_hash")
            )
        return None

    @staticmethod
    def _payload_source_message_ids(memory: dict[str, Any]) -> list[str]:
        payload = memory.get("payload_json")
        if not isinstance(payload, dict):
            return []
        raw_ids = payload.get("source_message_ids")
        if not isinstance(raw_ids, list):
            return []
        return sorted({str(item).strip() for item in raw_ids if str(item).strip()})

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None
