"""SQLite repositories for the lightweight entity and relationship graph."""

from __future__ import annotations

import hashlib
from typing import Any

from atagia.core import json_utils
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import (
    BaseRepository,
    MemoryObjectRepository,
    _derive_sensitivity_from_privacy,
    _encode_json,
)
from atagia.memory.intimacy_boundary_policy import (
    memory_object_intimacy_sql_clause,
    strongest_intimacy_boundary,
)
from atagia.models.schemas_memory import IntimacyBoundary, MemoryCategory, MemoryScope, MemorySensitivity


def _compact_text(value: str, *, max_length: int | None = None) -> str:
    normalized = " ".join(str(value).split())
    if max_length is None:
        return normalized
    return normalized[:max_length].strip()


def graph_surface_key(surface_text: str) -> str:
    """Return a mechanical alias key without interpreting natural-language meaning."""
    return _compact_text(surface_text).casefold()


def graph_quote_hash(evidence_quote: str | None) -> str:
    normalized = _compact_text(evidence_quote or "")
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _stable_hash(parts: list[Any]) -> str:
    payload = json_utils.dumps(parts, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_mention_signature(
    *,
    source_kind: str,
    source_id: str,
    surface_text: str,
    evidence_quote: str | None,
    source_occurrence_key: str | None = None,
) -> str:
    return _stable_hash(
        [
            "mention",
            source_kind,
            source_id,
            source_occurrence_key or "",
            graph_surface_key(surface_text),
            graph_quote_hash(evidence_quote),
        ]
    )


def graph_relationship_dedupe_key(
    *,
    source_entity_id: str,
    predicate: str,
    target_entity_id: str | None,
    target_value: dict[str, Any] | list[Any] | str | int | float | bool | None,
    direction: str,
    scope: MemoryScope,
    workspace_id: str | None,
    conversation_id: str | None,
    assistant_mode_id: str | None,
    user_persona_id: str | None,
    platform_id: str | None,
    character_id: str | None,
    platform_locked: bool,
    platform_id_lock: str | None,
    sensitivity: str,
    scope_canonical: str,
    valid_from: str | None,
    valid_to: str | None,
) -> str:
    return _stable_hash(
        [
            "relationship",
            source_entity_id,
            predicate,
            target_entity_id,
            target_value,
            direction,
            scope.value,
            workspace_id,
            conversation_id,
            assistant_mode_id,
            user_persona_id,
            platform_id,
            character_id,
            bool(platform_locked),
            platform_id_lock,
            scope_canonical,
            valid_from,
            valid_to,
        ]
    )


def _canonical_graph_scope_value(scope: MemoryScope) -> str:
    if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
        return MemoryScope.CHAT.value
    if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
        return MemoryScope.CHARACTER.value
    return MemoryScope.USER.value


def _strongest_relationship_intimacy_boundary(
    current_value: Any,
    incoming_value: IntimacyBoundary,
) -> IntimacyBoundary:
    return strongest_intimacy_boundary(
        [
            {"intimacy_boundary": current_value},
            {"intimacy_boundary": incoming_value.value},
        ]
    )


class EntityGraphRepository(BaseRepository):
    """Persistence operations for graph entities, mentions, and relationships."""

    async def create_projection_run(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        source_message_id: str | None,
        source_memory_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_run_id = run_id or generate_prefixed_id("gpr")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO graph_projection_runs(
                id,
                user_id,
                conversation_id,
                source_message_id,
                source_memory_ids_json,
                status,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                resolved_run_id,
                user_id,
                conversation_id,
                source_message_id,
                _encode_json(source_memory_ids or []),
                _encode_json(metadata),
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self.get_projection_run(resolved_run_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create graph projection run {resolved_run_id}")
        return created

    async def finish_projection_run(
        self,
        *,
        run_id: str,
        user_id: str,
        status: str,
        entity_count: int = 0,
        mention_count: int = 0,
        relationship_count: int = 0,
        skipped_count: int = 0,
        error: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        await self._connection.execute(
            """
            UPDATE graph_projection_runs
            SET status = ?,
                entity_count = ?,
                mention_count = ?,
                relationship_count = ?,
                skipped_count = ?,
                error = ?,
                finished_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                status,
                entity_count,
                mention_count,
                relationship_count,
                skipped_count,
                error,
                self._timestamp(),
                run_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_projection_run(run_id, user_id)

    async def get_projection_run(self, run_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM graph_projection_runs
            WHERE id = ?
              AND user_id = ?
            """,
            (run_id, user_id),
        )

    async def create_entity(
        self,
        *,
        user_id: str,
        entity_type: str,
        display_name: str,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        description: str = "",
        confidence: float = 0.5,
        status: str = "active",
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
        entity_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_entity_id = entity_id or generate_prefixed_id("ent")
        timestamp = self._timestamp()
        resolved_sensitivity = sensitivity or _derive_sensitivity_from_privacy(
            privacy_level,
            intimacy_boundary,
            MemoryCategory.UNKNOWN,
        )
        await self._connection.execute(
            """
            INSERT INTO graph_entities(
                id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                entity_type,
                display_name,
                description,
                confidence,
                status,
                privacy_level,
                intimacy_boundary,
                intimacy_boundary_confidence,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_entity_id,
                user_id,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                entity_type,
                _compact_text(display_name, max_length=240),
                _compact_text(description, max_length=2000),
                confidence,
                status,
                privacy_level,
                intimacy_boundary.value,
                intimacy_boundary_confidence,
                _encode_json(metadata),
                user_persona_id,
                platform_id or "default",
                character_id if character_id is not None else workspace_id,
                resolved_sensitivity.value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self.get_entity(resolved_entity_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create graph entity {resolved_entity_id}")
        return created

    async def get_entity(self, entity_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM graph_entities
            WHERE id = ?
              AND user_id = ?
            """,
            (entity_id, user_id),
        )

    async def list_entities(
        self,
        *,
        user_id: str,
        entity_type: str | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if entity_type is not None:
            clauses.append("entity_type = ?")
            parameters.append(entity_type)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(statuses)
        query = f"""
            SELECT *
            FROM graph_entities
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, id ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return await self._fetch_all(query, tuple(parameters))

    async def list_entity_cards(
        self,
        *,
        user_id: str,
        allowed_scopes: list[MemoryScope] | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        cross_chat_allowed: bool = True,
        privacy_ceiling: int = 3,
        allow_intimacy_context: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        entities = await self._list_entities_for_projection_context(
            user_id=user_id,
            allowed_scopes=allowed_scopes,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            cross_chat_allowed=cross_chat_allowed,
            privacy_ceiling=privacy_ceiling,
            allow_intimacy_context=allow_intimacy_context,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            limit=limit,
        )
        cards: list[dict[str, Any]] = []
        for entity in entities:
            aliases = await self._list_aliases_for_context(
                user_id=user_id,
                entity=entity,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
            )
            cards.append(
                {
                    "id": entity["id"],
                    "entity_type": entity["entity_type"],
                    "display_name": entity["display_name"],
                    "aliases": [
                        alias["surface_text"]
                        for alias in aliases
                        if alias["status"] == "active"
                    ][:8],
                    "confidence": entity["confidence"],
                    "status": entity["status"],
                }
            )
        return cards

    async def _list_entities_for_projection_context(
        self,
        *,
        user_id: str,
        allowed_scopes: list[MemoryScope] | None,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        cross_chat_allowed: bool,
        privacy_ceiling: int,
        allow_intimacy_context: bool,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        if allowed_scopes is None:
            return await self.list_entities(user_id=user_id, statuses=["active"], limit=limit)

        scope_clauses: list[str] = []
        scope_parameters: list[Any] = []
        allowed = set(MemoryObjectRepository.canonical_retrieval_scopes(allowed_scopes))
        cross_chat = cross_chat_allowed and remember_across_chats and not incognito
        if MemoryScope.USER in allowed and cross_chat:
            scope_clauses.append(
                "(ge.conversation_id IS NULL AND ge.workspace_id IS NULL AND ge.assistant_mode_id IS NULL)"
            )
        if MemoryScope.CHARACTER in allowed and cross_chat and workspace_id is not None:
            scope_clauses.append(
                "(ge.conversation_id IS NULL AND (ge.character_id = ? OR (ge.character_id IS NULL AND ge.workspace_id = ?)))"
            )
            scope_parameters.extend([character_id if character_id is not None else workspace_id, workspace_id])
        if MemoryScope.CHAT in allowed and conversation_id is not None:
            scope_clauses.append("(ge.conversation_id = ?)")
            scope_parameters.append(conversation_id)
        if not scope_clauses:
            return []
        phase7_clauses: list[str] = ["ge.sensitivity = 'public'"]
        phase7_parameters: list[Any] = []
        if platform_id is not None:
            phase7_clauses.append("ge.user_persona_id IS ?")
            phase7_parameters.append(user_persona_id)
            if incognito or not remember_across_chats:
                phase7_clauses.append("ge.conversation_id = ?")
                phase7_parameters.append(conversation_id)
            elif character_id is not None:
                phase7_clauses.append(
                    "("
                    "(ge.conversation_id = ?) "
                    "OR (ge.conversation_id IS NULL AND ge.character_id = ?) "
                    "OR (ge.conversation_id IS NULL AND ge.character_id IS NULL)"
                    ")"
                )
                phase7_parameters.extend([conversation_id, character_id])
            else:
                phase7_clauses.append(
                    "(ge.conversation_id = ? OR (ge.conversation_id IS NULL AND ge.character_id IS NULL))"
                )
                phase7_parameters.append(conversation_id)
            if remember_across_devices:
                phase7_clauses.append("(ge.platform_locked = 0 OR ge.platform_id_lock = ?)")
                phase7_parameters.append(platform_id)
            else:
                phase7_clauses.append(
                    "(ge.platform_id_lock = ? OR (ge.platform_locked = 0 AND ge.platform_id = ?))"
                )
                phase7_parameters.extend([platform_id, platform_id])

        return await self._fetch_all(
            """
            SELECT ge.*
            FROM graph_entities AS ge
            LEFT JOIN conversations AS conv
              ON conv.id = ge.conversation_id
             AND conv.user_id = ge.user_id
            WHERE ge.user_id = ?
              AND ge.status = 'active'
              AND ge.privacy_level <= ?
              AND {phase7_clauses}
              AND {intimacy_filter}
              AND ({scope_clauses})
              AND (
                  ge.conversation_id IS NULL
                  OR ge.conversation_id = ?
                  OR (
                      COALESCE(conv.temporary, 0) = 0
                      AND COALESCE(conv.isolated_mode, 0) = 0
                      AND COALESCE(conv.status, 'active') NOT IN ('archived', 'pending_deletion')
                  )
              )
            ORDER BY ge.updated_at DESC, ge.id ASC
            LIMIT ?
            """.format(
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "ge",
                    allow_intimacy_context=allow_intimacy_context,
                ),
                scope_clauses=" OR ".join(scope_clauses),
                phase7_clauses=" AND ".join(phase7_clauses),
            ),
            (
                user_id,
                privacy_ceiling,
                *phase7_parameters,
                *scope_parameters,
                conversation_id,
                limit,
            ),
        )

    async def upsert_alias(
        self,
        *,
        user_id: str,
        entity_id: str,
        surface_text: str,
        confidence: float = 0.5,
        status: str = "active",
        source_mention_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if await self.get_entity(entity_id, user_id) is None:
            raise ValueError(f"Entity {entity_id} does not belong to user {user_id}")
        surface_key = graph_surface_key(surface_text)
        alias_id = generate_prefixed_id("als")
        timestamp = self._timestamp()
        resolved_sensitivity = sensitivity or MemorySensitivity.UNKNOWN
        cursor = await self._connection.execute(
            """
            INSERT OR IGNORE INTO graph_entity_aliases(
                id,
                user_id,
                entity_id,
                source_mention_id,
                surface_text,
                surface_key,
                confidence,
                status,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alias_id,
                user_id,
                entity_id,
                source_mention_id,
                _compact_text(surface_text, max_length=240),
                surface_key,
                confidence,
                status,
                _encode_json(metadata),
                user_persona_id,
                platform_id,
                character_id,
                resolved_sensitivity.value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                timestamp,
                timestamp,
            ),
        )
        if cursor.rowcount == 0:
            await self._connection.execute(
                """
                UPDATE graph_entity_aliases
                SET confidence = MAX(confidence, ?),
                    status = CASE
                        WHEN status = 'review_required' AND ? = 'active' THEN 'active'
                        ELSE status
                    END,
                    source_mention_id = COALESCE(source_mention_id, ?),
                    updated_at = ?
                WHERE user_id = ?
                  AND entity_id = ?
                  AND surface_key = ?
                """,
                (
                    confidence,
                    status,
                    source_mention_id,
                    timestamp,
                    user_id,
                    entity_id,
                    surface_key,
                ),
            )
        if commit:
            await self._connection.commit()
        alias = await self._fetch_one(
            """
            SELECT *
            FROM graph_entity_aliases
            WHERE user_id = ?
              AND entity_id = ?
              AND surface_key = ?
            """,
            (user_id, entity_id, surface_key),
        )
        if alias is None:
            raise RuntimeError(f"Failed to upsert graph alias for entity {entity_id}")
        return alias

    async def _list_aliases_for_context(
        self,
        *,
        user_id: str,
        entity: dict[str, Any],
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        conversation_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
    ) -> list[dict[str, Any]]:
        if platform_id is None:
            return await self.list_aliases(user_id=user_id, entity_id=str(entity["id"]))
        clauses = [
            "gea.user_id = ?",
            "gea.entity_id = ?",
            "gea.status = 'active'",
            "gea.sensitivity = 'public'",
            "gea.user_persona_id IS ?",
        ]
        parameters: list[Any] = [user_id, str(entity["id"]), user_persona_id]
        if incognito or not remember_across_chats:
            clauses.append("gem.conversation_id = ?")
            parameters.append(conversation_id)
        elif character_id is None:
            clauses.append("gea.character_id IS NULL")
        else:
            clauses.append("(gea.character_id IS NULL OR gea.character_id = ?)")
            parameters.append(character_id)
        if remember_across_devices:
            clauses.append("(gea.platform_locked = 0 OR gea.platform_id_lock = ?)")
            parameters.append(platform_id)
        else:
            clauses.append("(gea.platform_id_lock = ? OR (gea.platform_locked = 0 AND gea.platform_id = ?))")
            parameters.extend([platform_id, platform_id])
        mention_clauses = [
            "gea.source_mention_id IS NULL",
            "("
            "gem.id IS NOT NULL "
            "AND gem.status = 'active' "
            "AND gem.sensitivity = 'public' "
            "AND gem.user_persona_id IS ?"
            ")",
        ]
        parameters.append(user_persona_id)
        clauses.append("(" + " OR ".join(mention_clauses) + ")")
        return await self._fetch_all(
            f"""
            SELECT gea.*
            FROM graph_entity_aliases AS gea
            LEFT JOIN graph_entity_mentions AS gem
              ON gem.id = gea.source_mention_id
             AND gem.user_id = gea.user_id
            WHERE {" AND ".join(clauses)}
            ORDER BY gea.created_at ASC, gea.id ASC
            """,
            tuple(parameters),
        )

    async def list_aliases(self, *, user_id: str, entity_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM graph_entity_aliases
            WHERE user_id = ?
              AND entity_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, entity_id),
        )

    async def find_aliases(self, *, user_id: str, surface_text: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM graph_entity_aliases
            WHERE user_id = ?
              AND surface_key = ?
            ORDER BY confidence DESC, created_at ASC, id ASC
            """,
            (user_id, graph_surface_key(surface_text)),
        )

    async def entity_id_for_existing_mention(
        self,
        *,
        user_id: str,
        source_kind: str,
        source_id: str,
        surface_text: str,
        evidence_quote: str | None,
        source_occurrence_key: str | None = None,
    ) -> str | None:
        signature = graph_mention_signature(
            source_kind=source_kind,
            source_id=source_id,
            surface_text=surface_text,
            evidence_quote=evidence_quote,
            source_occurrence_key=source_occurrence_key,
        )
        cursor = await self._connection.execute(
            """
            SELECT entity_id
            FROM graph_entity_mentions
            WHERE user_id = ?
              AND source_signature = ?
              AND entity_id IS NOT NULL
            """,
            (user_id, signature),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return str(row["entity_id"])

    async def upsert_mention(
        self,
        *,
        user_id: str,
        entity_id: str | None,
        source_kind: str,
        source_id: str,
        surface_text: str,
        evidence_quote: str | None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        memory_id: str | None = None,
        projection_run_id: str | None = None,
        source_occurrence_key: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        confidence: float = 0.5,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if entity_id is not None and await self.get_entity(entity_id, user_id) is None:
            raise ValueError(f"Entity {entity_id} does not belong to user {user_id}")
        source_signature = graph_mention_signature(
            source_kind=source_kind,
            source_id=source_id,
            surface_text=surface_text,
            evidence_quote=evidence_quote,
            source_occurrence_key=source_occurrence_key,
        )
        mention_id = generate_prefixed_id("mnt")
        surface_key = graph_surface_key(surface_text)
        quote_hash = graph_quote_hash(evidence_quote)
        resolved_sensitivity = sensitivity or MemorySensitivity.UNKNOWN
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            INSERT OR IGNORE INTO graph_entity_mentions(
                id,
                user_id,
                entity_id,
                conversation_id,
                message_id,
                memory_id,
                projection_run_id,
                source_kind,
                source_id,
                source_signature,
                source_occurrence_key,
                surface_text,
                surface_key,
                span_start,
                span_end,
                quote_hash,
                confidence,
                status,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mention_id,
                user_id,
                entity_id,
                conversation_id,
                message_id,
                memory_id,
                projection_run_id,
                source_kind,
                source_id,
                source_signature,
                source_occurrence_key or "",
                _compact_text(surface_text, max_length=240),
                surface_key,
                span_start,
                span_end,
                quote_hash,
                confidence,
                status,
                _encode_json(metadata),
                user_persona_id,
                platform_id,
                character_id,
                resolved_sensitivity.value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                timestamp,
                timestamp,
            ),
        )
        if cursor.rowcount == 0:
            await self._connection.execute(
                """
                UPDATE graph_entity_mentions
                SET entity_id = COALESCE(entity_id, ?),
                    confidence = MAX(confidence, ?),
                    status = CASE
                        WHEN status = 'review_required' AND ? = 'active' THEN 'active'
                        ELSE status
                    END,
                    projection_run_id = COALESCE(projection_run_id, ?),
                    metadata_json = ?,
                    sensitivity = CASE
                        WHEN sensitivity = 'secret' OR ? = 'secret' THEN 'secret'
                        WHEN sensitivity = 'private' OR ? = 'private' THEN 'private'
                        WHEN sensitivity = 'public' OR ? = 'public' THEN 'public'
                        ELSE sensitivity
                    END,
                    platform_locked = MAX(platform_locked, ?),
                    platform_id_lock = COALESCE(platform_id_lock, ?),
                    updated_at = ?
                WHERE user_id = ?
                  AND source_signature = ?
                """,
                (
                    entity_id,
                    confidence,
                    status,
                    projection_run_id,
                    _encode_json(metadata),
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    int(platform_locked),
                    platform_id_lock,
                    timestamp,
                    user_id,
                    source_signature,
                ),
            )
        if commit:
            await self._connection.commit()
        mention = await self._fetch_one(
            """
            SELECT *
            FROM graph_entity_mentions
            WHERE user_id = ?
              AND source_signature = ?
            """,
            (user_id, source_signature),
        )
        if mention is None:
            raise RuntimeError("Failed to upsert graph entity mention")
        return mention

    async def list_mentions_for_source(
        self,
        *,
        user_id: str,
        source_kind: str,
        source_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM graph_entity_mentions
            WHERE user_id = ?
              AND source_kind = ?
              AND source_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, source_kind, source_id),
        )

    async def upsert_relationship(
        self,
        *,
        user_id: str,
        source_entity_id: str,
        predicate: str,
        target_entity_id: str | None = None,
        target_value: dict[str, Any] | list[Any] | str | int | float | bool | None = None,
        direction: str = "directed",
        scope: MemoryScope = MemoryScope.CONVERSATION,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        confidence: float = 0.5,
        status: str = "active",
        valid_from: str | None = None,
        valid_to: str | None = None,
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        supersedes_relationship_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if await self.get_entity(source_entity_id, user_id) is None:
            raise ValueError(f"Source entity {source_entity_id} does not belong to user {user_id}")
        if target_entity_id is not None and await self.get_entity(target_entity_id, user_id) is None:
            raise ValueError(f"Target entity {target_entity_id} does not belong to user {user_id}")
        if target_entity_id is None and target_value is None:
            raise ValueError("Graph relationships require a target entity or target value")
        target_value_json = None if target_value is None else json_utils.dumps(target_value, sort_keys=True)
        resolved_sensitivity = sensitivity or _derive_sensitivity_from_privacy(
            privacy_level,
            intimacy_boundary,
            MemoryCategory.UNKNOWN,
        )
        resolved_scope_canonical = scope_canonical or _canonical_graph_scope_value(scope)
        if resolved_scope_canonical not in {
            MemoryScope.CHAT.value,
            MemoryScope.CHARACTER.value,
            MemoryScope.USER.value,
        }:
            resolved_scope_canonical = _canonical_graph_scope_value(scope)
        dedupe_key = graph_relationship_dedupe_key(
            source_entity_id=source_entity_id,
            predicate=predicate,
            target_entity_id=target_entity_id,
            target_value=target_value,
            direction=direction,
            scope=scope,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            platform_locked=platform_locked,
            platform_id_lock=platform_id_lock,
            sensitivity=resolved_sensitivity.value,
            scope_canonical=resolved_scope_canonical,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        relationship_id = generate_prefixed_id("rel")
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            INSERT OR IGNORE INTO graph_relationships(
                id,
                user_id,
                source_entity_id,
                target_entity_id,
                target_value_json,
                predicate,
                direction,
                scope,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                confidence,
                status,
                valid_from,
                valid_to,
                privacy_level,
                intimacy_boundary,
                intimacy_boundary_confidence,
                supersedes_relationship_id,
                dedupe_key,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                scope_canonical,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relationship_id,
                user_id,
                source_entity_id,
                target_entity_id,
                target_value_json,
                predicate,
                direction,
                resolved_scope_canonical,
                workspace_id,
                conversation_id,
                assistant_mode_id,
                confidence,
                status,
                valid_from,
                valid_to,
                privacy_level,
                intimacy_boundary.value,
                intimacy_boundary_confidence,
                supersedes_relationship_id,
                dedupe_key,
                _encode_json(metadata),
                user_persona_id,
                platform_id,
                character_id,
                resolved_sensitivity.value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                resolved_scope_canonical,
                timestamp,
                timestamp,
            ),
        )
        if cursor.rowcount == 0:
            existing_relationship = await self._fetch_one(
                """
                SELECT intimacy_boundary
                FROM graph_relationships
                WHERE user_id = ?
                  AND dedupe_key = ?
                """,
                (user_id, dedupe_key),
            )
            merged_intimacy_boundary = _strongest_relationship_intimacy_boundary(
                existing_relationship["intimacy_boundary"] if existing_relationship else None,
                intimacy_boundary,
            )
            await self._connection.execute(
                """
                UPDATE graph_relationships
                SET confidence = MAX(confidence, ?),
                    status = CASE
                        WHEN status = 'review_required' AND ? = 'active' THEN 'active'
                        WHEN ? IN ('superseded', 'conflicted', 'archived', 'deleted') THEN ?
                        ELSE status
                    END,
                    privacy_level = MAX(privacy_level, ?),
                    intimacy_boundary = ?,
                    intimacy_boundary_confidence = MAX(intimacy_boundary_confidence, ?),
                    sensitivity = CASE
                        WHEN sensitivity = 'secret' OR ? = 'secret' THEN 'secret'
                        WHEN sensitivity = 'private' OR ? = 'private' THEN 'private'
                        WHEN sensitivity = 'public' OR ? = 'public' THEN 'public'
                        ELSE sensitivity
                    END,
                    platform_locked = MAX(platform_locked, ?),
                    platform_id_lock = COALESCE(platform_id_lock, ?),
                    metadata_json = ?,
                    updated_at = ?
                WHERE user_id = ?
                  AND dedupe_key = ?
                """,
                (
                    confidence,
                    status,
                    status,
                    status,
                    privacy_level,
                    merged_intimacy_boundary.value,
                    intimacy_boundary_confidence,
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    int(platform_locked),
                    platform_id_lock,
                    _encode_json(metadata),
                    timestamp,
                    user_id,
                    dedupe_key,
                ),
            )
        if commit:
            await self._connection.commit()
        relationship = await self._fetch_one(
            """
            SELECT *
            FROM graph_relationships
            WHERE user_id = ?
              AND dedupe_key = ?
            """,
            (user_id, dedupe_key),
        )
        if relationship is None:
            raise RuntimeError("Failed to upsert graph relationship")
        return relationship

    async def link_relationship_source(
        self,
        *,
        user_id: str,
        relationship_id: str,
        source_kind: str,
        source_id: str,
        evidence_quote: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        memory_id: str | None = None,
        projection_run_id: str | None = None,
        source_occurrence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        relationship = await self._fetch_one(
            """
            SELECT id
            FROM graph_relationships
            WHERE id = ?
              AND user_id = ?
            """,
            (relationship_id, user_id),
        )
        if relationship is None:
            raise ValueError(f"Relationship {relationship_id} does not belong to user {user_id}")
        source_id_text = _compact_text(source_id, max_length=240)
        quote_hash = graph_quote_hash(evidence_quote)
        source_link_id = generate_prefixed_id("rls")
        resolved_sensitivity = sensitivity or MemorySensitivity.UNKNOWN
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            INSERT OR IGNORE INTO graph_relationship_sources(
                id,
                user_id,
                relationship_id,
                conversation_id,
                message_id,
                memory_id,
                projection_run_id,
                source_kind,
                source_id,
                source_occurrence_key,
                quote_hash,
                evidence_quote,
                metadata_json,
                user_persona_id,
                platform_id,
                character_id,
                sensitivity,
                themes_json,
                platform_locked,
                platform_id_lock,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_link_id,
                user_id,
                relationship_id,
                conversation_id,
                message_id,
                memory_id,
                projection_run_id,
                source_kind,
                source_id_text,
                source_occurrence_key or "",
                quote_hash,
                _compact_text(evidence_quote or "", max_length=1000) or None,
                _encode_json(metadata),
                user_persona_id,
                platform_id,
                character_id,
                resolved_sensitivity.value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                timestamp,
            ),
        )
        if cursor.rowcount == 0:
            await self._connection.execute(
                """
                UPDATE graph_relationship_sources
                SET projection_run_id = COALESCE(projection_run_id, ?),
                    metadata_json = ?,
                    sensitivity = CASE
                        WHEN sensitivity = 'secret' OR ? = 'secret' THEN 'secret'
                        WHEN sensitivity = 'private' OR ? = 'private' THEN 'private'
                        WHEN sensitivity = 'public' OR ? = 'public' THEN 'public'
                        ELSE sensitivity
                    END,
                    platform_locked = MAX(platform_locked, ?),
                    platform_id_lock = COALESCE(platform_id_lock, ?)
                WHERE user_id = ?
                  AND relationship_id = ?
                  AND source_kind = ?
                  AND source_id = ?
                  AND source_occurrence_key = ?
                  AND quote_hash = ?
                """,
                (
                    projection_run_id,
                    _encode_json(metadata),
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    resolved_sensitivity.value,
                    int(platform_locked),
                    platform_id_lock,
                    user_id,
                    relationship_id,
                    source_kind,
                    source_id_text,
                    source_occurrence_key or "",
                    quote_hash,
                ),
            )
        if commit:
            await self._connection.commit()
        source = await self._fetch_one(
            """
            SELECT *
            FROM graph_relationship_sources
            WHERE user_id = ?
              AND relationship_id = ?
              AND source_kind = ?
              AND source_id = ?
              AND source_occurrence_key = ?
              AND quote_hash = ?
            """,
            (
                user_id,
                relationship_id,
                source_kind,
                source_id_text,
                source_occurrence_key or "",
                quote_hash,
            ),
        )
        if source is None:
            raise RuntimeError(f"Failed to link graph relationship source {relationship_id}")
        return source

    async def list_relationships_for_entity(
        self,
        *,
        user_id: str,
        entity_id: str,
        statuses: list[str] | None = None,
        allowed_scopes: list[MemoryScope] | None = None,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["gr.user_id = ?", "(gr.source_entity_id = ? OR gr.target_entity_id = ?)"]
        parameters: list[Any] = [user_id, entity_id, entity_id]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"gr.status IN ({placeholders})")
            parameters.extend(statuses)
        if platform_id is not None and conversation_id is not None:
            visibility_clauses, visibility_parameters = MemoryObjectRepository.namespace_visibility_clauses(
                allowed_scopes or [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                table_alias="gr",
            )
            if not visibility_clauses:
                return []
            clauses.extend(visibility_clauses)
            parameters.extend(visibility_parameters)
        query = f"""
            SELECT gr.*
            FROM graph_relationships AS gr
            WHERE {" AND ".join(clauses)}
            ORDER BY gr.updated_at DESC, gr.id ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return await self._fetch_all(query, tuple(parameters))

    async def list_relationship_sources(
        self,
        *,
        user_id: str,
        relationship_id: str,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        remember_across_devices: bool = True,
    ) -> list[dict[str, Any]]:
        clauses = ["grs.user_id = ?", "grs.relationship_id = ?"]
        parameters: list[Any] = [user_id, relationship_id]
        if platform_id is not None and conversation_id is not None:
            clauses.extend(
                [
                    "grs.conversation_id = ?",
                    "grs.user_persona_id IS ?",
                    "grs.sensitivity = 'public'",
                ]
            )
            parameters.extend([conversation_id, user_persona_id])
            if character_id is None:
                clauses.append("grs.character_id IS NULL")
            else:
                clauses.append("(grs.character_id IS NULL OR grs.character_id = ?)")
                parameters.append(character_id)
            if remember_across_devices:
                clauses.append("(grs.platform_locked = 0 OR grs.platform_id_lock = ?)")
                parameters.append(platform_id)
            else:
                clauses.append("(grs.platform_id_lock = ? OR (grs.platform_locked = 0 AND grs.platform_id = ?))")
                parameters.extend([platform_id, platform_id])
        return await self._fetch_all(
            f"""
            SELECT grs.*
            FROM graph_relationship_sources AS grs
            WHERE {" AND ".join(clauses)}
            ORDER BY grs.created_at ASC, grs.id ASC
            """,
            tuple(parameters),
        )
