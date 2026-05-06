"""SQLite repository for belief versioning and links."""

from __future__ import annotations

import re
from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySensitivity, MemoryStatus

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

    async def increment_tension(
        self,
        belief_id: str,
        delta: float,
        *,
        user_id: str,
        commit: bool = True,
    ) -> float:
        return await self._update_tension(
            belief_id,
            user_id=user_id,
            delta=abs(float(delta)),
            commit=commit,
        )

    async def decrement_tension(
        self,
        belief_id: str,
        delta: float,
        *,
        user_id: str,
        commit: bool = True,
    ) -> float:
        return await self._update_tension(
            belief_id,
            user_id=user_id,
            delta=-abs(float(delta)),
            commit=commit,
        )

    async def get_tension(self, belief_id: str, *, user_id: str) -> float:
        row = await self._fetch_one(
            """
            SELECT tension_score
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
              AND object_type = ?
            LIMIT 1
            """,
            (
                belief_id,
                user_id,
                MemoryObjectType.BELIEF.value,
            ),
        )
        if row is None:
            return 0.0
        return max(0.0, float(row.get("tension_score") or 0.0))

    async def reset_tension(
        self,
        belief_id: str,
        *,
        user_id: str,
        commit: bool = True,
    ) -> float:
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET tension_score = 0.0,
                tension_updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND object_type = ?
            """,
            (
                self._timestamp(),
                belief_id,
                user_id,
                MemoryObjectType.BELIEF.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_tension(belief_id, user_id=user_id)

    async def get_beliefs_above_tension_threshold(
        self,
        user_id: str,
        threshold: float,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND object_type = ?
              AND status = ?
              AND tension_score >= ?
            ORDER BY tension_score DESC, updated_at DESC, id ASC
            LIMIT ?
            """,
            (
                user_id,
                MemoryObjectType.BELIEF.value,
                MemoryStatus.ACTIVE.value,
                float(threshold),
                limit,
            ),
        )

    async def add_tension_evidence_ids(
        self,
        belief_id: str,
        evidence_memory_ids: list[str],
        *,
        user_id: str,
        commit: bool = True,
    ) -> list[str]:
        payload = await self._belief_payload(belief_id, user_id=user_id)
        existing_ids = self._payload_tension_evidence_ids(payload)
        merged_ids = list(existing_ids)
        for memory_id in evidence_memory_ids:
            normalized = str(memory_id)
            if normalized not in merged_ids:
                merged_ids.append(normalized)
        payload["tension_evidence_memory_ids"] = merged_ids
        await self._write_belief_payload(
            belief_id,
            user_id=user_id,
            payload=payload,
            commit=commit,
        )
        return merged_ids

    async def get_tension_evidence_ids(self, belief_id: str, *, user_id: str) -> list[str]:
        """Return the currently buffered tension evidence ids without mutating the belief."""
        payload = await self._belief_payload(belief_id, user_id=user_id)
        return self._payload_tension_evidence_ids(payload)

    async def pop_tension_evidence_ids(
        self,
        belief_id: str,
        *,
        user_id: str,
        commit: bool = True,
    ) -> list[str]:
        payload = await self._belief_payload(belief_id, user_id=user_id)
        evidence_ids = self._payload_tension_evidence_ids(payload)
        payload.pop("tension_evidence_memory_ids", None)
        await self._write_belief_payload(
            belief_id,
            user_id=user_id,
            payload=payload,
            commit=commit,
        )
        return evidence_ids

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
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        conversation_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        rows = await self._active_belief_rows(user_id)
        normalized = claim_key.strip().lower()
        namespace_requested = self._namespace_filter_requested(
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        return [
            row
            for row in rows
            if str(row["claim_key"]).strip().lower() == normalized
            and (
                not namespace_requested
                or self._row_matches_namespace(
                    row,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    conversation_id=conversation_id,
                    incognito=incognito,
                    remember_across_chats=remember_across_chats,
                    remember_across_devices=remember_across_devices,
                )
            )
        ]

    async def find_active_belief_candidates_by_claim_key(
        self,
        user_id: str,
        claim_key: str,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        conversation_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = await self._active_belief_rows(user_id)
        normalized = claim_key.strip().lower()
        target_tokens = _claim_key_tokens(normalized)
        namespace_requested = self._namespace_filter_requested(
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        ranked: list[tuple[tuple[int, int, str, str], dict[str, Any]]] = []
        for row in rows:
            if namespace_requested and not self._row_matches_namespace(
                row,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
            ):
                continue
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
                mo.scope_canonical,
                mo.user_persona_id,
                mo.platform_id,
                mo.character_id,
                mo.sensitivity,
                mo.platform_locked,
                mo.platform_id_lock,
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
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        conversation_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> dict[str, Any]:
        cursor = await self._connection.execute(
            """
            WITH support_candidates AS (
                SELECT
                    mo.id AS memory_id,
                    mo.conversation_id,
                    mo.assistant_mode_id,
                    mo.created_at,
                    mo.scope,
                    mo.scope_canonical,
                    mo.user_persona_id,
                    mo.platform_id,
                    mo.character_id,
                    mo.sensitivity,
                    mo.platform_locked,
                    mo.platform_id_lock
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
                    mo.created_at,
                    mo.scope,
                    mo.scope_canonical,
                    mo.user_persona_id,
                    mo.platform_id,
                    mo.character_id,
                    mo.sensitivity,
                    mo.platform_locked,
                    mo.platform_id_lock
                FROM memory_objects AS mo
                WHERE mo.user_id = ?
                  AND mo.status = ?
                  AND mo.object_type = ?
                  AND json_extract(mo.payload_json, '$.claim_key') = ?
            )
            SELECT *
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
        namespace_requested = self._namespace_filter_requested(
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        rows = []
        for row in await cursor.fetchall():
            payload = dict(row)
            if namespace_requested and not self._row_matches_namespace(
                payload,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
            ):
                continue
            rows.append(payload)
        stats = {
            "total_evidence": len(rows),
            "distinct_conversations": len({row.get("conversation_id") for row in rows if row.get("conversation_id")}),
            "distinct_sessions": len({str(row.get("created_at") or "")[:10] for row in rows if row.get("created_at")}),
            "oldest_at": min((row.get("created_at") for row in rows if row.get("created_at")), default=None),
            "newest_at": max((row.get("created_at") for row in rows if row.get("created_at")), default=None),
        }
        if stats["distinct_conversations"] < min_conversations:
            return stats
        return stats

    @staticmethod
    def _namespace_filter_requested(
        *,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        conversation_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
    ) -> bool:
        return any(
            (
                user_persona_id is not None,
                platform_id is not None,
                character_id is not None,
                conversation_id is not None,
                incognito,
                not remember_across_chats,
                not remember_across_devices,
            )
        )

    @staticmethod
    def _row_matches_namespace(
        row: dict[str, Any],
        *,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        conversation_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
    ) -> bool:
        if row.get("user_persona_id") != user_persona_id:
            return False
        if str(row.get("sensitivity") or "unknown") != "public":
            return False
        active_platform_id = str(platform_id or "default")
        if bool(row.get("platform_locked")):
            if row.get("platform_id_lock") != active_platform_id:
                return False
        elif not remember_across_devices and row.get("platform_id") != active_platform_id:
            return False
        scope = str(row.get("scope_canonical") or row.get("scope") or "")
        if scope in {MemoryScope.CONVERSATION.value, MemoryScope.EPHEMERAL_SESSION.value}:
            scope = MemoryScope.CHAT.value
        elif scope in {MemoryScope.WORKSPACE.value, "legacy_workspace"}:
            scope = MemoryScope.CHARACTER.value
        elif scope in {MemoryScope.GLOBAL_USER.value, MemoryScope.ASSISTANT_MODE.value, "legacy_assistant_mode"}:
            scope = MemoryScope.USER.value
        if scope == MemoryScope.CHAT.value:
            if conversation_id is not None:
                return row.get("conversation_id") == conversation_id
            return character_id is not None and row.get("character_id") == character_id
        if incognito or not remember_across_chats:
            return False
        if scope == MemoryScope.CHARACTER.value:
            return row.get("character_id") == character_id
        if scope == MemoryScope.USER.value:
            return True
        return False

    async def _update_tension(
        self,
        belief_id: str,
        *,
        user_id: str,
        delta: float,
        commit: bool,
    ) -> float:
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET tension_score = MAX(0.0, COALESCE(tension_score, 0.0) + ?),
                tension_updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND object_type = ?
            """,
            (
                float(delta),
                self._timestamp(),
                belief_id,
                user_id,
                MemoryObjectType.BELIEF.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_tension(belief_id, user_id=user_id)

    async def _belief_payload(
        self,
        belief_id: str,
        *,
        user_id: str,
    ) -> dict[str, Any]:
        row = await self._fetch_one(
            """
            SELECT payload_json
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
              AND object_type = ?
            LIMIT 1
            """,
            (
                belief_id,
                user_id,
                MemoryObjectType.BELIEF.value,
            ),
        )
        payload = {} if row is None else row.get("payload_json")
        return dict(payload) if isinstance(payload, dict) else {}

    async def _write_belief_payload(
        self,
        belief_id: str,
        *,
        user_id: str,
        payload: dict[str, Any],
        commit: bool,
    ) -> None:
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET payload_json = ?,
                tension_updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND object_type = ?
            """,
            (
                _encode_json(payload),
                self._timestamp(),
                belief_id,
                user_id,
                MemoryObjectType.BELIEF.value,
            ),
        )
        if commit:
            await self._connection.commit()

    @staticmethod
    def _payload_tension_evidence_ids(payload: dict[str, Any]) -> list[str]:
        raw_ids = payload.get("tension_evidence_memory_ids", [])
        if not isinstance(raw_ids, list):
            return []
        return [str(memory_id) for memory_id in raw_ids]

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
            """
            SELECT
                id,
                user_id,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                sensitivity,
                platform_locked,
                platform_id_lock,
                scope_canonical
            FROM memory_objects
            WHERE id = ?
            """,
            (source_id,),
        )
        target_row = await self._fetch_one(
            """
            SELECT
                id,
                user_id,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                sensitivity,
                platform_locked,
                platform_id_lock,
                scope_canonical
            FROM memory_objects
            WHERE id = ?
            """,
            (target_id,),
        )
        if source_row is None or target_row is None:
            raise ValueError("Memory links require existing source and target memory objects")
        if source_row["user_id"] != target_row["user_id"]:
            raise ValueError("Memory links cannot cross user boundaries")
        self._validate_link_namespace(source_row, target_row)
        source_lock = source_row.get("platform_id_lock")
        target_lock = target_row.get("platform_id_lock")
        if source_lock is not None and target_lock is not None and source_lock != target_lock:
            raise ValueError("Memory links cannot cross platform-lock boundaries")

        link_id = generate_prefixed_id("lnk")
        created_at = self._timestamp()
        sensitivity = self._combined_link_sensitivity(
            source_row.get("sensitivity"),
            target_row.get("sensitivity"),
        )
        platform_locked = bool(source_row.get("platform_locked")) or bool(target_row.get("platform_locked"))
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
                created_at,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                sensitivity,
                platform_locked,
                platform_id_lock,
                policy_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                source_row.get("user_persona_id") or target_row.get("user_persona_id"),
                source_row.get("platform_id") or target_row.get("platform_id"),
                source_row.get("character_id") or target_row.get("character_id"),
                self._shared_link_conversation_id(source_row, target_row),
                sensitivity,
                1 if platform_locked else 0,
                source_lock or target_lock,
                _encode_json(
                    {
                        "src_scope": source_row.get("scope_canonical"),
                        "dst_scope": target_row.get("scope_canonical"),
                    }
                ),
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

    @staticmethod
    def _validate_link_namespace(source_row: dict[str, Any], target_row: dict[str, Any]) -> None:
        for field_name in ("user_persona_id", "platform_id", "character_id"):
            source_value = source_row.get(field_name)
            target_value = target_row.get(field_name)
            if source_value is not None and target_value is not None and source_value != target_value:
                raise ValueError("Memory links cannot cross namespace boundaries")
        if (
            str(source_row.get("scope_canonical") or "") == MemoryScope.CHAT.value
            and str(target_row.get("scope_canonical") or "") == MemoryScope.CHAT.value
            and source_row.get("conversation_id") is not None
            and target_row.get("conversation_id") is not None
            and source_row.get("conversation_id") != target_row.get("conversation_id")
        ):
            raise ValueError("Memory links cannot cross chat conversation boundaries")

    @staticmethod
    def _shared_link_conversation_id(source_row: dict[str, Any], target_row: dict[str, Any]) -> object | None:
        source_conversation_id = source_row.get("conversation_id")
        target_conversation_id = target_row.get("conversation_id")
        if source_conversation_id is not None and source_conversation_id == target_conversation_id:
            return source_conversation_id
        return None

    @staticmethod
    def _combined_link_sensitivity(source: object, target: object) -> str:
        values = {str(value or MemorySensitivity.UNKNOWN.value) for value in (source, target)}
        if MemorySensitivity.UNKNOWN.value in values:
            return MemorySensitivity.UNKNOWN.value
        if MemorySensitivity.SECRET.value in values:
            return MemorySensitivity.SECRET.value
        if MemorySensitivity.PRIVATE.value in values:
            return MemorySensitivity.PRIVATE.value
        if values == {MemorySensitivity.PUBLIC.value}:
            return MemorySensitivity.PUBLIC.value
        return MemorySensitivity.UNKNOWN.value
