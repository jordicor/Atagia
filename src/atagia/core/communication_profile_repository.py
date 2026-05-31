"""SQLite repository helpers for user communication profiles."""

from __future__ import annotations

from typing import Any

from atagia.core import json_utils
from atagia.core.ids import generate_prefixed_id
from atagia.core.language_codes import normalize_optional_iso_639_1_code
from atagia.core.repositories import BaseRepository, _decode_json_columns
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryScope,
    UserCommunicationProfile,
)

USER_LANGUAGE_PROFILE_KIND = "user_language_profile"


class CommunicationProfileRepository(BaseRepository):
    """Persistence operations for non-FTS user communication profiles."""

    async def get_user_language_profile_for_context(
        self,
        context: ExtractionConversationContext,
    ) -> UserCommunicationProfile | None:
        """Return the most specific active, non-stale profile visible in context."""
        if self.target_scope_for_context(context) is None:
            return None
        clauses, parameters = self._visible_context_clauses(context)
        if not clauses:
            return None
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM user_communication_profiles AS ucp
            WHERE ucp.user_id = ?
              AND ucp.profile_kind = ?
              AND ucp.status = 'active'
              AND ucp.stale = 0
              AND ({visibility_clause})
            ORDER BY
              CASE ucp.scope_canonical
                WHEN 'chat' THEN 0
                WHEN 'character' THEN 1
                ELSE 2
              END ASC,
              ucp.updated_at DESC
            LIMIT 1
            """.format(visibility_clause=" OR ".join(clauses)),
            (
                context.user_id,
                USER_LANGUAGE_PROFILE_KIND,
                *parameters,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._profile_from_row(row)

    async def get_exact_user_language_profile(
        self,
        context: ExtractionConversationContext,
        *,
        scope: MemoryScope,
    ) -> UserCommunicationProfile | None:
        """Return the exact profile row for the context target coordinates."""
        target = self._target_coordinates(context, scope=scope)
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM user_communication_profiles
            WHERE user_id = ?
              AND profile_kind = ?
              AND scope_canonical = ?
              AND {user_persona_clause}
              AND {platform_clause}
              AND {character_clause}
              AND {conversation_clause}
              AND {space_clause}
              AND {memory_owner_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND status = 'active'
              AND stale = 0
            LIMIT 1
            """.format(
                user_persona_clause=self._nullable_match_clause("user_persona_id", target["user_persona_id"]),
                platform_clause=self._nullable_match_clause("platform_id", target["platform_id"]),
                character_clause=self._nullable_match_clause("character_id", target["character_id"]),
                conversation_clause=self._nullable_match_clause("conversation_id", target["conversation_id"]),
                space_clause=self._nullable_match_clause("space_id", target["space_id"]),
                memory_owner_clause=self._nullable_match_clause("memory_owner_id", target["memory_owner_id"]),
                embodiment_clause=self._nullable_match_clause("embodiment_id", target["embodiment_id"]),
                realm_clause=self._nullable_match_clause("realm_id", target["realm_id"]),
            ),
            (
                target["user_id"],
                USER_LANGUAGE_PROFILE_KIND,
                scope.value,
                *self._nullable_match_parameters(target["user_persona_id"]),
                *self._nullable_match_parameters(target["platform_id"]),
                *self._nullable_match_parameters(target["character_id"]),
                *self._nullable_match_parameters(target["conversation_id"]),
                *self._nullable_match_parameters(target["space_id"]),
                *self._nullable_match_parameters(target["memory_owner_id"]),
                *self._nullable_match_parameters(target["embodiment_id"]),
                *self._nullable_match_parameters(target["realm_id"]),
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._profile_from_row(row)

    async def upsert_user_language_profile(
        self,
        context: ExtractionConversationContext,
        profile: UserCommunicationProfile,
        *,
        scope: MemoryScope,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Upsert a non-FTS user language profile for one target context."""
        target = self._target_coordinates(context, scope=scope)
        timestamp = self._timestamp()
        source_refs = self._source_refs(profile)
        payload = profile.model_dump(mode="json")
        row_id = generate_prefixed_id("ucp")
        await self._connection.execute(
            """
            INSERT INTO user_communication_profiles(
                id,
                user_id,
                profile_kind,
                scope_canonical,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                user_persona_id,
                platform_id,
                character_id,
                subject_presence_id,
                space_id,
                space_boundary_mode,
                memory_owner_id,
                source_mind_id,
                embodiment_id,
                realm_id,
                profile_json,
                source_refs_json,
                status,
                stale,
                stale_reason,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            ON CONFLICT(
                user_id,
                profile_kind,
                scope_canonical,
                user_persona_key,
                platform_key,
                character_key,
                conversation_key,
                space_key,
                memory_owner_key,
                embodiment_key,
                realm_key
            )
            DO UPDATE SET
                workspace_id = excluded.workspace_id,
                assistant_mode_id = excluded.assistant_mode_id,
                subject_presence_id = excluded.subject_presence_id,
                space_boundary_mode = excluded.space_boundary_mode,
                source_mind_id = excluded.source_mind_id,
                profile_json = excluded.profile_json,
                source_refs_json = excluded.source_refs_json,
                status = 'active',
                stale = excluded.stale,
                stale_reason = excluded.stale_reason,
                updated_at = excluded.updated_at
            """,
            (
                row_id,
                target["user_id"],
                USER_LANGUAGE_PROFILE_KIND,
                scope.value,
                target["workspace_id"],
                target["conversation_id"],
                target["assistant_mode_id"],
                target["user_persona_id"],
                target["platform_id"],
                target["character_id"],
                profile.subject_presence_id,
                target["space_id"],
                target["space_boundary_mode"],
                target["memory_owner_id"],
                target["source_mind_id"],
                target["embodiment_id"],
                target["realm_id"],
                json_utils.dumps(payload, sort_keys=True),
                json_utils.dumps(source_refs, sort_keys=True),
                1 if profile.stale else 0,
                profile.stale_reason,
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_profile_row_by_target(context, scope=scope)

    async def get_profile_row_by_target(
        self,
        context: ExtractionConversationContext,
        *,
        scope: MemoryScope,
    ) -> dict[str, Any]:
        target = self._target_coordinates(context, scope=scope)
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM user_communication_profiles
            WHERE user_id = ?
              AND profile_kind = ?
              AND scope_canonical = ?
              AND {user_persona_clause}
              AND {platform_clause}
              AND {character_clause}
              AND {conversation_clause}
              AND {space_clause}
              AND {memory_owner_clause}
              AND {embodiment_clause}
              AND {realm_clause}
            LIMIT 1
            """.format(
                user_persona_clause=self._nullable_match_clause("user_persona_id", target["user_persona_id"]),
                platform_clause=self._nullable_match_clause("platform_id", target["platform_id"]),
                character_clause=self._nullable_match_clause("character_id", target["character_id"]),
                conversation_clause=self._nullable_match_clause("conversation_id", target["conversation_id"]),
                space_clause=self._nullable_match_clause("space_id", target["space_id"]),
                memory_owner_clause=self._nullable_match_clause("memory_owner_id", target["memory_owner_id"]),
                embodiment_clause=self._nullable_match_clause("embodiment_id", target["embodiment_id"]),
                realm_clause=self._nullable_match_clause("realm_id", target["realm_id"]),
            ),
            (
                target["user_id"],
                USER_LANGUAGE_PROFILE_KIND,
                scope.value,
                *self._nullable_match_parameters(target["user_persona_id"]),
                *self._nullable_match_parameters(target["platform_id"]),
                *self._nullable_match_parameters(target["character_id"]),
                *self._nullable_match_parameters(target["conversation_id"]),
                *self._nullable_match_parameters(target["space_id"]),
                *self._nullable_match_parameters(target["memory_owner_id"]),
                *self._nullable_match_parameters(target["embodiment_id"]),
                *self._nullable_match_parameters(target["realm_id"]),
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ValueError("User communication profile upsert did not produce a row")
        return _decode_json_columns(row)

    async def mark_stale_for_source_message(
        self,
        *,
        user_id: str,
        source_message_id: str,
        reason: str,
        commit: bool = True,
    ) -> int:
        """Mark profiles stale when their source-message provenance is invalidated."""
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE user_communication_profiles
            SET stale = 1,
                stale_reason = ?,
                updated_at = ?
            WHERE user_id = ?
              AND status = 'active'
              AND stale = 0
              AND EXISTS (
                    SELECT 1
                    FROM json_each(user_communication_profiles.source_refs_json) AS source_ref
                    WHERE json_extract(source_ref.value, '$.source_message_id') = ?
                  )
            """,
            (reason, timestamp, user_id, source_message_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        reason: str,
        commit: bool = True,
    ) -> int:
        """Mark profiles stale when their conversation target or sources are invalidated."""
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE user_communication_profiles
            SET stale = 1,
                stale_reason = ?,
                updated_at = ?
            WHERE user_id = ?
              AND status = 'active'
              AND stale = 0
              AND (
                    conversation_id = ?
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(user_communication_profiles.source_refs_json) AS source_ref
                        WHERE json_extract(source_ref.value, '$.conversation_id') = ?
                    )
                  )
            """,
            (reason, timestamp, user_id, conversation_id, conversation_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        reason: str,
        commit: bool = True,
    ) -> int:
        """Mark profiles stale when a source memory object is changed or removed."""
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE user_communication_profiles
            SET stale = 1,
                stale_reason = ?,
                updated_at = ?
            WHERE user_id = ?
              AND status = 'active'
              AND stale = 0
              AND EXISTS (
                    SELECT 1
                    FROM json_each(user_communication_profiles.source_refs_json) AS source_ref
                    WHERE json_extract(source_ref.value, '$.memory_id') = ?
                  )
            """,
            (reason, timestamp, user_id, memory_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_memories(
        self,
        *,
        user_id: str,
        memory_ids: list[str],
        reason: str,
        commit: bool = True,
    ) -> int:
        """Mark profiles stale when any source memory object is changed or removed."""
        normalized_ids = [str(memory_id).strip() for memory_id in memory_ids if str(memory_id).strip()]
        if not normalized_ids:
            return 0
        placeholders = ", ".join("?" for _memory_id in normalized_ids)
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            f"""
            UPDATE user_communication_profiles
            SET stale = 1,
                stale_reason = ?,
                updated_at = ?
            WHERE user_id = ?
              AND status = 'active'
              AND stale = 0
              AND EXISTS (
                    SELECT 1
                    FROM json_each(
                        CASE
                            WHEN json_valid(user_communication_profiles.source_refs_json) = 1
                            THEN user_communication_profiles.source_refs_json
                            ELSE '[]'
                        END
                    ) AS source_ref
                    WHERE json_extract(source_ref.value, '$.memory_id') IN ({placeholders})
                  )
            """,
            (reason, timestamp, user_id, *normalized_ids),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    def target_scope_for_context(context: ExtractionConversationContext) -> MemoryScope | None:
        """Return where the profile may be stored for this context."""
        if context.incognito or context.isolated_mode or context.temporary or context.purge_on_close:
            return None
        if not context.remember_across_chats:
            return MemoryScope.CHAT
        if CommunicationProfileRepository._effective_character_id(context) is None:
            return MemoryScope.USER
        return MemoryScope.CHARACTER

    @staticmethod
    def _profile_from_row(row: Any) -> UserCommunicationProfile:
        decoded = _decode_json_columns(row)
        profile = UserCommunicationProfile.model_validate(
            CommunicationProfileRepository._sanitize_profile_payload(
                decoded["profile_json"]
            )
        )
        return profile.model_copy(
            update={
                "stale": bool(decoded.get("stale")),
                "stale_reason": decoded.get("stale_reason"),
            }
        )

    @staticmethod
    def _sanitize_profile_payload(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        for field_name in (
            "observed_user_languages",
            "explicit_language_preferences",
            "explicit_language_abilities",
            "contextual_norms",
        ):
            items = normalized.get(field_name)
            if not isinstance(items, list):
                if field_name in normalized:
                    normalized[field_name] = []
                continue
            filtered: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                language_code = normalize_optional_iso_639_1_code(
                    item.get("language_code")
                )
                if language_code is None:
                    continue
                filtered.append({**item, "language_code": language_code})
            normalized[field_name] = filtered
        return normalized

    @staticmethod
    def _source_refs(profile: UserCommunicationProfile) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [
            *profile.observed_user_languages,
            *profile.explicit_language_preferences,
            *profile.explicit_language_abilities,
            *profile.contextual_norms,
        ]:
            for source_ref in item.source_refs:
                payload = source_ref.model_dump(mode="json")
                key = json_utils.dumps(payload, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(payload)
        return refs

    @staticmethod
    def _effective_character_id(context: ExtractionConversationContext) -> str | None:
        return context.character_id or context.workspace_id

    @classmethod
    def _target_coordinates(
        cls,
        context: ExtractionConversationContext,
        *,
        scope: MemoryScope,
    ) -> dict[str, Any]:
        conversation_id = context.conversation_id if scope is MemoryScope.CHAT else None
        character_id = cls._effective_character_id(context) if scope is MemoryScope.CHARACTER else None
        return {
            "user_id": context.user_id,
            "workspace_id": context.workspace_id,
            "conversation_id": conversation_id,
            "assistant_mode_id": context.assistant_mode_id,
            "user_persona_id": context.user_persona_id,
            "platform_id": context.platform_id,
            "character_id": character_id,
            "space_id": context.active_space_id,
            "space_boundary_mode": (
                context.active_space_boundary_mode.value
                if context.active_space_id is not None
                else None
            ),
            "memory_owner_id": context.active_mind_id,
            "source_mind_id": context.source_mind_id or context.active_mind_id,
            "embodiment_id": context.active_embodiment_id,
            "realm_id": context.active_realm_id,
        }

    @classmethod
    def _visible_context_clauses(
        cls,
        context: ExtractionConversationContext,
    ) -> tuple[list[str], list[Any]]:
        base_clauses, base_parameters = cls._visibility_base_context_clauses(context)
        clauses: list[str] = []
        parameters: list[Any] = []
        if context.conversation_id:
            clauses.append(
                "(ucp.scope_canonical = 'chat' AND ucp.conversation_id = ? AND {base})".format(
                    base=" AND ".join(base_clauses)
                )
            )
            parameters.extend([context.conversation_id, *base_parameters])
        if not context.incognito and context.remember_across_chats:
            character_id = cls._effective_character_id(context)
            if character_id is not None:
                clauses.append(
                    "(ucp.scope_canonical = 'character' AND ucp.character_id = ? AND {base})".format(
                        base=" AND ".join(base_clauses)
                    )
                )
                parameters.extend([character_id, *base_parameters])
            clauses.append(
                "(ucp.scope_canonical = 'user' AND ucp.character_id IS NULL AND ucp.conversation_id IS NULL AND {base})".format(
                    base=" AND ".join(base_clauses)
                )
            )
            parameters.extend(base_parameters)
        return clauses, parameters

    @classmethod
    def _visibility_base_context_clauses(
        cls,
        context: ExtractionConversationContext,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        clauses.append(cls._nullable_visibility_clause("ucp.user_persona_id", context.user_persona_id))
        parameters.extend(cls._nullable_visibility_parameters(context.user_persona_id))
        if context.remember_across_devices:
            clauses.append("(ucp.platform_id IS NULL OR ucp.platform_id = ?)")
            parameters.append(context.platform_id)
        else:
            clauses.append("ucp.platform_id = ?")
            parameters.append(context.platform_id)
        for column, value in (
            ("ucp.space_id", context.active_space_id),
            ("ucp.memory_owner_id", context.active_mind_id),
            ("ucp.embodiment_id", context.active_embodiment_id),
            ("ucp.realm_id", context.active_realm_id),
        ):
            clauses.append(cls._nullable_visibility_clause(column, value))
            parameters.extend(cls._nullable_visibility_parameters(value))
        return clauses, parameters

    @staticmethod
    def _nullable_match_clause(column: str, value: str | None) -> str:
        if value is None:
            return f"{column} IS NULL"
        return f"{column} = ?"

    @staticmethod
    def _nullable_match_parameters(value: str | None) -> list[str]:
        return [] if value is None else [value]

    @staticmethod
    def _nullable_visibility_clause(column: str, value: str | None) -> str:
        if value is None:
            return f"{column} IS NULL"
        return f"({column} IS NULL OR {column} = ?)"

    @staticmethod
    def _nullable_visibility_parameters(value: str | None) -> list[str]:
        return [] if value is None else [value]
