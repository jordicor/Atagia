"""SQLite repository helpers for consequence chains."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.core.repositories import BaseRepository, MemoryObjectRepository, _encode_json
from atagia.models.schemas_memory import MemoryScope, MemoryStatus


class _ConsequenceChainCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    action_memory_id: str
    outcome_memory_id: str
    tendency_belief_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: str
    created_at: str
    updated_at: str


class ConsequenceRepository(BaseRepository):
    """Persistence helpers for action -> outcome -> tendency chains."""

    async def create_chain(
        self,
        chain_data: dict[str, Any],
        *,
        commit: bool = True,
    ) -> str:
        payload = _ConsequenceChainCreate.model_validate(chain_data)
        source_memories = [
            await self._load_memory_for_chain(payload.action_memory_id, payload.user_id),
            await self._load_memory_for_chain(payload.outcome_memory_id, payload.user_id),
        ]
        if payload.tendency_belief_id is not None:
            source_memories.append(
                await self._load_memory_for_chain(payload.tendency_belief_id, payload.user_id)
            )
        chain_policy = self._derive_chain_policy(source_memories)
        status = payload.status
        if chain_policy["sensitivity"] != "public":
            status = MemoryStatus.REVIEW_REQUIRED.value

        await self._connection.execute(
            """
            INSERT INTO consequence_chains(
                id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                action_memory_id,
                outcome_memory_id,
                tendency_belief_id,
                confidence,
                status,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                scope_canonical,
                incognito_snapshot,
                policy_snapshot_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.id,
                payload.user_id,
                payload.workspace_id,
                payload.conversation_id,
                payload.assistant_mode_id,
                payload.action_memory_id,
                payload.outcome_memory_id,
                payload.tendency_belief_id,
                payload.confidence,
                status,
                chain_policy["user_persona_id"],
                chain_policy["platform_id"],
                chain_policy["character_id"],
                chain_policy["sensitivity"],
                _encode_json(chain_policy["themes"]),
                int(chain_policy["platform_locked"]),
                chain_policy["platform_id_lock"],
                chain_policy["scope_canonical"],
                int(chain_policy["incognito"]),
                _encode_json(chain_policy["policy_snapshot"]),
                payload.created_at,
                payload.updated_at,
            ),
        )
        if commit:
            await self._connection.commit()
        return payload.id

    async def find_chains_for_action(
        self,
        user_id: str,
        action_memory_id: str,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        visibility_clause, visibility_parameters = self._chain_visibility_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        return await self._fetch_all(
            f"""
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND action_memory_id = ?
              AND status = 'active'
              {visibility_clause}
            ORDER BY confidence DESC, updated_at DESC, id ASC
            """,
            (user_id, action_memory_id, *visibility_parameters),
        )

    async def find_chains_for_workspace(
        self,
        user_id: str,
        workspace_id: str,
        *,
        limit: int = 20,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        visibility_clause, visibility_parameters = self._chain_visibility_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        return await self._fetch_all(
            f"""
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND workspace_id = ?
              AND status = 'active'
              {visibility_clause}
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT ?
            """,
            (user_id, workspace_id, *visibility_parameters, limit),
        )

    async def find_chains_by_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        visibility_clause, visibility_parameters = self._chain_visibility_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        return await self._fetch_all(
            f"""
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND status = 'active'
              {visibility_clause}
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT ?
            """,
            (user_id, *visibility_parameters, limit),
        )

    async def update_chain_confidence(
        self,
        chain_id: str,
        user_id: str,
        confidence: float,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        commit: bool = True,
    ) -> None:
        visibility_clause, visibility_parameters = self._chain_visibility_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        await self._connection.execute(
            f"""
            UPDATE consequence_chains
            SET confidence = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              {visibility_clause}
            """,
            (confidence, self._timestamp(), chain_id, user_id, *visibility_parameters),
        )
        if commit:
            await self._connection.commit()

    async def archive_chain(
        self,
        chain_id: str,
        user_id: str,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        commit: bool = True,
    ) -> None:
        visibility_clause, visibility_parameters = self._chain_visibility_sql(
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
        )
        await self._connection.execute(
            f"""
            UPDATE consequence_chains
            SET status = 'archived',
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              {visibility_clause}
            """,
            (self._timestamp(), chain_id, user_id, *visibility_parameters),
        )
        if commit:
            await self._connection.commit()

    async def _ensure_memory_belongs_to_user(self, memory_id: str, user_id: str) -> None:
        await self._load_memory_for_chain(memory_id, user_id)

    async def _load_memory_for_chain(self, memory_id: str, user_id: str) -> dict[str, Any]:
        row = await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )
        if row is None:
            raise ValueError(f"Memory object {memory_id} does not exist for user {user_id}")
        return row

    @classmethod
    def _derive_chain_policy(cls, source_memories: list[dict[str, Any]]) -> dict[str, Any]:
        personas = {memory.get("user_persona_id") for memory in source_memories}
        if len(personas) > 1:
            raise ValueError("Consequence source memories disagree on user persona")

        policy_snapshots = [cls._source_policy(memory) for memory in source_memories]
        policy_keys = (
            "incognito",
            "remember_across_chats",
            "remember_across_devices",
            "temporary",
            "purge_on_close",
        )
        for key in policy_keys:
            if len({snapshot.get(key) for snapshot in policy_snapshots}) > 1:
                raise ValueError(f"Consequence source memories disagree on {key} policy")

        platform_locks = {
            snapshot.get("platform_id_lock")
            for snapshot in policy_snapshots
            if bool(snapshot.get("platform_locked"))
        }
        if len(platform_locks) > 1:
            raise ValueError("Consequence source memories disagree on platform lock")
        source_platforms = {memory.get("platform_id") or "default" for memory in source_memories}
        if len(source_platforms) > 1 and not platform_locks:
            raise ValueError("Consequence source memories disagree on platform")

        tendency = source_memories[2] if len(source_memories) > 2 else None
        scope_canonical = cls._canonical_scope_for_memory(tendency or source_memories[1])
        character_id = (tendency or source_memories[1]).get("character_id")
        if scope_canonical == MemoryScope.CHARACTER.value:
            expected_character = character_id
            if expected_character is None:
                raise ValueError("Character-scoped consequence tendency requires character_id")
            for memory in source_memories:
                memory_character = memory.get("character_id")
                if memory_character not in {None, expected_character}:
                    raise ValueError("Consequence source memories disagree on character")
        elif scope_canonical == MemoryScope.CHAT.value:
            conversations = {memory.get("conversation_id") for memory in source_memories[:2]}
            if len(conversations) > 1:
                raise ValueError("Chat-scoped consequence chain cannot bridge conversations")

        sensitivity = cls._strictest_sensitivity(source_memories)
        themes: list[str] = []
        for memory in source_memories:
            raw_themes = memory.get("themes_json") or []
            if isinstance(raw_themes, list):
                for theme in raw_themes:
                    normalized = str(theme).strip()
                    if normalized and normalized not in themes:
                        themes.append(normalized)
        policy_snapshot = {
            **policy_snapshots[0],
            "source_memory_ids": [str(memory["id"]) for memory in source_memories],
            "source_scope_canonical": [
                cls._canonical_scope_for_memory(memory) for memory in source_memories
            ],
        }
        platform_locked = any(bool(snapshot.get("platform_locked")) for snapshot in policy_snapshots)
        platform_id_lock = next(iter(platform_locks), None)
        return {
            "user_persona_id": next(iter(personas)),
            "platform_id": next(iter(source_platforms)),
            "character_id": character_id,
            "sensitivity": sensitivity,
            "themes": themes,
            "platform_locked": platform_locked,
            "platform_id_lock": platform_id_lock,
            "scope_canonical": scope_canonical,
            "incognito": bool(policy_snapshots[0].get("incognito")),
            "policy_snapshot": policy_snapshot,
        }

    @staticmethod
    def _source_policy(memory: dict[str, Any]) -> dict[str, Any]:
        payload = memory.get("payload_json") or {}
        policy = payload.get("source_turn_policy") if isinstance(payload, dict) else None
        if isinstance(policy, dict):
            return dict(policy)
        return {
            "user_persona_id": memory.get("user_persona_id"),
            "platform_id": memory.get("platform_id") or "default",
            "character_id": memory.get("character_id"),
            "conversation_id": memory.get("conversation_id"),
            "incognito": False,
            "remember_across_chats": True,
            "remember_across_devices": True,
            "temporary": False,
            "purge_on_close": False,
            "intended_scope": memory.get("scope_canonical") or memory.get("scope"),
            "platform_locked": bool(memory.get("platform_locked")),
            "platform_id_lock": memory.get("platform_id_lock"),
        }

    @staticmethod
    def _canonical_scope_for_memory(memory: dict[str, Any]) -> str:
        scope_value = str(memory.get("scope_canonical") or memory.get("scope") or "")
        try:
            scope = MemoryScope(scope_value)
        except ValueError:
            return scope_value
        if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
            return MemoryScope.CHAT.value
        if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
            return MemoryScope.CHARACTER.value
        return MemoryScope.USER.value

    @staticmethod
    def _strictest_sensitivity(source_memories: list[dict[str, Any]]) -> str:
        rank = {"public": 1, "private": 2, "secret": 3, "unknown": 4}
        strongest = "public"
        for memory in source_memories:
            sensitivity = str(memory.get("sensitivity") or "unknown")
            if rank.get(sensitivity, 0) > rank[strongest]:
                strongest = sensitivity
        return strongest

    @staticmethod
    def _chain_visibility_sql(
        *,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
    ) -> tuple[str, list[Any]]:
        if platform_id is None or conversation_id is None:
            return "", []
        clauses, parameters = MemoryObjectRepository.namespace_visibility_clauses(
            [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            incognito=incognito,
            table_alias="consequence_chains",
        )
        if not clauses:
            return " AND 0", []
        return " AND " + " AND ".join(clauses), parameters
