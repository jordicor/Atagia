"""Admin inspection helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository
from atagia.memory.mind_policy import OVERSEER_CONTEXT_GRANT_KINDS
from atagia.memory.retrieval_custody import build_coordinate_trace
from atagia.models.schemas_memory import RetrievalPlan, SpaceBoundaryMode

_COORDINATE_UPDATE_FIELDS: tuple[str, ...] = (
    "active_presence_id",
    "source_presence_id",
    "presence_cluster_id",
    "space_id",
    "space_boundary_mode",
    "memory_owner_id",
    "source_mind_id",
    "embodiment_id",
    "realm_id",
)

_MEMORY_INSPECTION_FIELDS: tuple[str, ...] = (
    "id",
    "user_id",
    "object_type",
    "scope",
    "scope_canonical",
    "status",
    "conversation_id",
    "assistant_mode_id",
    "workspace_id",
    "user_persona_id",
    "platform_id",
    "character_id",
    "privacy_level",
    "sensitivity",
    "source_kind",
    "confidence",
    "created_at",
    "updated_at",
    "canonical_text",
)

_SPACE_BOUNDARY_VALUES: frozenset[str] = frozenset(mode.value for mode in SpaceBoundaryMode)


def _canonical_scope_filter(scope: str) -> str:
    if scope in {"conversation", "ephemeral_session"}:
        return "chat"
    if scope == "workspace":
        return "character"
    if scope in {"global_user", "assistant_mode"}:
        return "user"
    return scope


class _InspectionRepository(BaseRepository):
    """Read-oriented queries used by the admin inspector."""

    async def get_retrieval_event_by_id(self, event_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM retrieval_events
            WHERE id = ?
            """,
            (event_id,),
        )

    async def get_memory(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )

    async def get_memory_coordinate_view(
        self,
        memory_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT
                mo.*,
                active_presence.id AS active_presence_row_id,
                active_presence.kind AS active_presence_kind,
                active_presence.display_name AS active_presence_display_name,
                active_presence.source_kind AS active_presence_source_kind,
                active_presence.source_id AS active_presence_source_id,
                source_presence.id AS source_presence_row_id,
                source_presence.kind AS source_presence_kind,
                source_presence.display_name AS source_presence_display_name,
                source_presence.source_kind AS source_presence_source_kind,
                source_presence.source_id AS source_presence_source_id,
                space.id AS space_row_id,
                space.boundary_mode AS space_record_boundary_mode,
                space.display_name AS space_display_name,
                space.source_kind AS space_source_kind,
                space.source_id AS space_source_id,
                owner_mind.id AS memory_owner_row_id,
                owner_mind.kind AS memory_owner_kind,
                owner_mind.display_name AS memory_owner_display_name,
                owner_mind.source_kind AS memory_owner_source_kind,
                owner_mind.source_id AS memory_owner_source_id,
                source_mind.id AS source_mind_row_id,
                source_mind.kind AS source_mind_kind,
                source_mind.display_name AS source_mind_display_name,
                source_mind.source_kind AS source_mind_source_kind,
                source_mind.source_id AS source_mind_source_id,
                embodiment.id AS embodiment_row_id,
                embodiment.display_name AS embodiment_display_name,
                embodiment.cross_embodiment_mode AS embodiment_cross_embodiment_mode,
                embodiment.source_kind AS embodiment_source_kind,
                embodiment.source_id AS embodiment_source_id,
                realm.id AS realm_row_id,
                realm.display_name AS realm_display_name,
                realm.cross_realm_mode AS realm_cross_realm_mode,
                realm.source_kind AS realm_source_kind,
                realm.source_id AS realm_source_id
            FROM memory_objects AS mo
            LEFT JOIN presences AS active_presence
              ON active_presence.owner_user_id = mo.user_id
             AND active_presence.id = mo.active_presence_id
            LEFT JOIN presences AS source_presence
              ON source_presence.owner_user_id = mo.user_id
             AND source_presence.id = mo.source_presence_id
            LEFT JOIN spaces AS space
              ON space.owner_user_id = mo.user_id
             AND space.id = mo.space_id
            LEFT JOIN minds AS owner_mind
              ON owner_mind.owner_user_id = mo.user_id
             AND owner_mind.id = mo.memory_owner_id
            LEFT JOIN minds AS source_mind
              ON source_mind.owner_user_id = mo.user_id
             AND source_mind.id = mo.source_mind_id
            LEFT JOIN embodiments AS embodiment
              ON embodiment.owner_user_id = mo.user_id
             AND embodiment.id = mo.embodiment_id
            LEFT JOIN realms AS realm
              ON realm.owner_user_id = mo.user_id
             AND realm.id = mo.realm_id
            WHERE mo.id = ?
              AND mo.user_id = ?
            """,
            (memory_id, user_id),
        )

    async def list_applicable_overseer_grants(
        self,
        *,
        user_id: str,
        memory_owner_id: str | None,
        space_id: str | None,
        realm_id: str | None,
        as_of: str,
    ) -> list[dict[str, Any]]:
        targets: list[tuple[str, str]] = []
        if memory_owner_id is not None:
            targets.append(("mind", memory_owner_id))
        if space_id is not None:
            targets.append(("space", space_id))
        if realm_id is not None:
            targets.append(("realm", realm_id))
        if not targets:
            return []
        clauses = " OR ".join("(og.target_kind = ? AND og.target_id = ?)" for _ in targets)
        parameters: list[Any] = [user_id, as_of]
        for target_kind, target_id in targets:
            parameters.extend((target_kind, target_id))
        return await self._fetch_all(
            f"""
            SELECT
                og.*,
                overseer.display_name AS overseer_display_name,
                overseer.kind AS overseer_kind
            FROM overseer_grants AS og
            JOIN minds AS overseer
              ON overseer.owner_user_id = og.owner_user_id
             AND overseer.id = og.overseer_mind_id
             AND overseer.kind = 'overseer'
            WHERE og.owner_user_id = ?
              AND og.revoked_at IS NULL
              AND (og.expires_at IS NULL OR og.expires_at > ?)
              AND ({clauses})
              AND (
                  (
                      og.target_kind = 'mind'
                      AND EXISTS (
                          SELECT 1 FROM minds AS target_mind
                          WHERE target_mind.owner_user_id = og.owner_user_id
                            AND target_mind.id = og.target_id
                      )
                  )
                  OR (
                      og.target_kind = 'space'
                      AND EXISTS (
                          SELECT 1 FROM spaces AS target_space
                          WHERE target_space.owner_user_id = og.owner_user_id
                            AND target_space.id = og.target_id
                      )
                  )
                  OR (
                      og.target_kind = 'realm'
                      AND EXISTS (
                          SELECT 1 FROM realms AS target_realm
                          WHERE target_realm.owner_user_id = og.owner_user_id
                            AND target_realm.id = og.target_id
                      )
                  )
              )
            ORDER BY og.target_kind ASC, og.target_id ASC, og.grant_kind ASC
            """,
            tuple(parameters),
        )

    async def get_space(self, user_id: str, space_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM spaces
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (user_id, space_id),
        )

    async def get_realm_bridge(
        self,
        *,
        owner_user_id: str,
        source_realm_id: str,
        target_realm_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM realm_bridges
            WHERE owner_user_id = ?
              AND source_realm_id = ?
              AND target_realm_id = ?
              AND cross_realm_mode IN ('attributed', 'applicable')
            """,
            (owner_user_id, source_realm_id, target_realm_id),
        )

    async def coordinate_exists(
        self,
        *,
        user_id: str,
        coordinate_kind: str,
        coordinate_id: str,
    ) -> bool:
        if coordinate_kind == "presence":
            table = "presences"
        elif coordinate_kind == "mind":
            table = "minds"
        elif coordinate_kind == "embodiment":
            table = "embodiments"
        elif coordinate_kind == "realm":
            table = "realms"
        else:
            raise ValueError(f"Unsupported coordinate kind: {coordinate_kind}")
        row = await self._fetch_one(
            f"""
            SELECT id
            FROM {table}
            WHERE owner_user_id = ?
              AND id = ?
            """,
            (user_id, coordinate_id),
        )
        return row is not None

    async def update_memory_coordinates(
        self,
        *,
        memory_id: str,
        user_id: str,
        updates: dict[str, str | None],
        timestamp: str,
    ) -> dict[str, Any] | None:
        unknown = sorted(set(updates) - set(_COORDINATE_UPDATE_FIELDS))
        if unknown:
            raise ValueError(f"Unsupported coordinate fields: {', '.join(unknown)}")
        assignments = [f"{field} = ?" for field in updates]
        parameters = [updates[field] for field in updates]
        parameters.extend((timestamp, memory_id, user_id))
        await self._connection.execute(
            f"""
            UPDATE memory_objects
            SET {", ".join(assignments)},
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            tuple(parameters),
        )
        return await self.get_memory_coordinate_view(memory_id, user_id)

    async def insert_coordinate_correction_audit(
        self,
        *,
        admin_user_id: str,
        memory_id: str,
        metadata: dict[str, Any],
        timestamp: str,
    ) -> dict[str, Any] | None:
        audit_id = generate_prefixed_id("aud")
        await self._connection.execute(
            """
            INSERT INTO admin_audit_log(
                id,
                admin_user_id,
                action,
                target_type,
                target_id,
                metadata_json,
                created_at
            )
            VALUES (?, ?, 'correct_memory_coordinates', 'memory_object', ?, ?, ?)
            """,
            (
                audit_id,
                admin_user_id,
                memory_id,
                json_utils.dumps(metadata, sort_keys=True),
                timestamp,
            ),
        )
        return await self._fetch_one(
            """
            SELECT *
            FROM admin_audit_log
            WHERE id = ?
            """,
            (audit_id,),
        )

    async def list_coordinate_correction_history(
        self,
        memory_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM admin_audit_log
            WHERE action = 'correct_memory_coordinates'
              AND target_type = 'memory_object'
              AND target_id = ?
              AND json_extract(metadata_json, '$.user_id') = ?
            ORDER BY created_at ASC, _rowid ASC
            """,
            (memory_id, user_id),
        )

    async def list_memories(
        self,
        user_id: str,
        object_type: str | None,
        scope: str | None,
        status: str | None,
        intimacy_boundary: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if object_type is not None:
            clauses.append("object_type = ?")
            parameters.append(object_type)
        if scope is not None:
            clauses.append("scope = ?")
            parameters.append(_canonical_scope_filter(scope))
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if intimacy_boundary is not None:
            clauses.append("intimacy_boundary = ?")
            parameters.append(intimacy_boundary)
        parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE {clauses}
            ORDER BY created_at DESC, id ASC
            LIMIT ?
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def list_belief_history(self, belief_id: str, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT
                bv.*,
                mo.intimacy_boundary AS parent_intimacy_boundary,
                mo.intimacy_boundary_confidence AS parent_intimacy_boundary_confidence
            FROM belief_versions AS bv
            JOIN memory_objects AS mo ON mo.id = bv.belief_id
            WHERE bv.belief_id = ?
              AND mo.user_id = ?
            ORDER BY bv.version ASC
            """,
            (belief_id, user_id),
        )

    async def list_consequence_chains(
        self,
        user_id: str,
        workspace_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["cc.user_id = ?", "cc.status = 'active'"]
        parameters: list[Any] = [user_id]
        if workspace_id is not None:
            clauses.append("cc.workspace_id = ?")
            parameters.append(workspace_id)
        parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT
                cc.*,
                action.canonical_text AS action_canonical_text,
                action.intimacy_boundary AS action_intimacy_boundary,
                action.intimacy_boundary_confidence AS action_intimacy_boundary_confidence,
                outcome.canonical_text AS outcome_canonical_text,
                outcome.intimacy_boundary AS outcome_intimacy_boundary,
                outcome.intimacy_boundary_confidence AS outcome_intimacy_boundary_confidence,
                tendency.canonical_text AS tendency_canonical_text,
                tendency.intimacy_boundary AS tendency_intimacy_boundary,
                tendency.intimacy_boundary_confidence AS tendency_intimacy_boundary_confidence
            FROM consequence_chains AS cc
            JOIN memory_objects AS action ON action.id = cc.action_memory_id
            JOIN memory_objects AS outcome ON outcome.id = cc.outcome_memory_id
            LEFT JOIN memory_objects AS tendency ON tendency.id = cc.tendency_belief_id
            WHERE {clauses}
            ORDER BY cc.confidence DESC, cc.updated_at DESC, cc.id ASC
            LIMIT ?
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )


class MemoryInspector:
    """Admin inspection utilities with lightweight audit logging."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock
        self._inspection_repository = _InspectionRepository(connection, clock)
        self._event_repository = RetrievalEventRepository(connection, clock)
        self._audit_repository = AdminAuditRepository(connection, clock)

    async def inspect_memory(
        self,
        memory_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> dict[str, Any] | None:
        memory = await self._inspection_repository.get_memory(memory_id, user_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_memory",
            target_type="memory_object",
            target_id=memory_id,
            metadata={"user_id": user_id, "found": memory is not None},
        )
        return memory

    async def inspect_memory_coordinates(
        self,
        memory_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> dict[str, Any] | None:
        row = await self._inspection_repository.get_memory_coordinate_view(memory_id, user_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_memory_coordinates",
            target_type="memory_object",
            target_id=memory_id,
            metadata={"user_id": user_id, "found": row is not None},
        )
        if row is None:
            return None
        grants = await self._inspection_repository.list_applicable_overseer_grants(
            user_id=user_id,
            memory_owner_id=_normalize_optional_text(row.get("memory_owner_id")),
            space_id=_normalize_optional_text(row.get("space_id")),
            realm_id=_normalize_optional_text(row.get("realm_id")),
            as_of=self._clock.now().isoformat(),
        )
        return _coordinate_inspection(row, overseer_grants=grants)

    async def inspect_retrieval_memory_decision(
        self,
        event_id: str,
        memory_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> dict[str, Any] | None:
        event = await self._event_repository.get_event(event_id, user_id)
        memory = await self._inspection_repository.get_memory_coordinate_view(memory_id, user_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_retrieval_memory_decision",
            target_type="retrieval_event",
            target_id=event_id,
            metadata={
                "user_id": user_id,
                "memory_id": memory_id,
                "event_found": event is not None,
                "memory_found": memory is not None,
            },
        )
        if event is None:
            return None
        custody_record = _find_custody_record(event, memory_id)
        grants: list[dict[str, Any]] = []
        if memory is not None:
            grants = await self._inspection_repository.list_applicable_overseer_grants(
                user_id=user_id,
                memory_owner_id=_normalize_optional_text(memory.get("memory_owner_id")),
                space_id=_normalize_optional_text(memory.get("space_id")),
                realm_id=_normalize_optional_text(memory.get("realm_id")),
                as_of=self._clock.now().isoformat(),
            )
        coordinate_trace, trace_source, unavailable_reason = await self._decision_coordinate_trace(
            event=event,
            memory=memory,
            custody_record=custody_record,
            grants=grants,
        )
        memory_coordinates = (
            None
            if memory is None
            else _coordinate_inspection(memory, overseer_grants=grants)
        )
        return {
            "event_id": event_id,
            "user_id": user_id,
            "memory_id": memory_id,
            "event_found": True,
            "memory_found": memory is not None,
            "decision": _retrieval_decision_label(custody_record, memory),
            "custody_record": custody_record,
            "coordinate_trace_v1": coordinate_trace,
            "coordinate_trace_source": trace_source,
            "coordinate_trace_unavailable_reason": unavailable_reason,
            "boundary_explanation": _boundary_explanation(coordinate_trace),
            "memory_coordinates": memory_coordinates,
        }

    async def _decision_coordinate_trace(
        self,
        *,
        event: dict[str, Any],
        memory: dict[str, Any] | None,
        custody_record: dict[str, Any] | None,
        grants: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        if custody_record is not None:
            trace = custody_record.get("coordinate_trace_v1")
            if isinstance(trace, dict):
                return trace, "retrieval_custody_v2", None
            return None, "unavailable", "custody_record_missing_coordinate_trace"
        if memory is None:
            return None, "unavailable", "memory_not_found_for_user"
        plan, error = _retrieval_plan_from_event(event)
        if plan is None:
            return None, "unavailable", error or "retrieval_plan_unavailable"
        candidate = _candidate_from_memory(memory)
        _annotate_reconstructed_mind_grant(candidate, plan, grants)
        await self._annotate_reconstructed_realm_bridge(candidate, plan)
        return build_coordinate_trace(candidate, plan), "reconstructed_from_event_plan", None

    async def _annotate_reconstructed_realm_bridge(
        self,
        candidate: dict[str, Any],
        plan: RetrievalPlan,
    ) -> None:
        active_realm_id = _normalize_optional_text(plan.active_realm_id)
        candidate_realm_id = _normalize_optional_text(
            candidate.get("realm_id") or candidate.get("active_realm_id")
        )
        owner_user_id = _normalize_optional_text(candidate.get("user_id"))
        if (
            active_realm_id is None
            or candidate_realm_id is None
            or owner_user_id is None
            or candidate_realm_id == active_realm_id
        ):
            return
        bridge = await self._inspection_repository.get_realm_bridge(
            owner_user_id=owner_user_id,
            source_realm_id=active_realm_id,
            target_realm_id=candidate_realm_id,
        )
        if bridge is None:
            return
        candidate["realm_relation"] = "cross"
        candidate["realm_bridge_mode"] = str(bridge["cross_realm_mode"])

    async def correct_memory_coordinates(
        self,
        memory_id: str,
        user_id: str,
        *,
        admin_user_id: str,
        updates: dict[str, Any],
        reason: str | None = None,
        invalidate_user_cache: Callable[[str], Awaitable[Any]] | None = None,
    ) -> dict[str, Any] | None:
        current = await self._inspection_repository.get_memory_coordinate_view(memory_id, user_id)
        if current is None:
            await self._audit_repository.create_audit_entry(
                admin_user_id=admin_user_id,
                action="correct_memory_coordinates",
                target_type="memory_object",
                target_id=memory_id,
                metadata={"user_id": user_id, "found": False},
            )
            return None

        normalized_updates = await self._normalize_coordinate_updates(
            current,
            user_id=user_id,
            updates=updates,
        )
        timestamp = self._clock.now().isoformat()
        before = _coordinate_values(current)
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            updated = await self._inspection_repository.update_memory_coordinates(
                memory_id=memory_id,
                user_id=user_id,
                updates=normalized_updates,
                timestamp=timestamp,
            )
            if updated is None:
                raise RuntimeError("Failed to update memory coordinates")
            await self._inspection_repository.insert_coordinate_correction_audit(
                admin_user_id=admin_user_id,
                memory_id=memory_id,
                metadata={
                    "user_id": user_id,
                    "found": True,
                    "reason": _normalize_optional_text(reason),
                    "before": before,
                    "after": _coordinate_values(updated),
                    "updated_fields": sorted(normalized_updates),
                },
                timestamp=timestamp,
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

        if invalidate_user_cache is not None:
            await invalidate_user_cache(user_id)

        grants = await self._inspection_repository.list_applicable_overseer_grants(
            user_id=user_id,
            memory_owner_id=_normalize_optional_text(updated.get("memory_owner_id")),
            space_id=_normalize_optional_text(updated.get("space_id")),
            realm_id=_normalize_optional_text(updated.get("realm_id")),
            as_of=self._clock.now().isoformat(),
        )
        return _coordinate_inspection(updated, overseer_grants=grants)

    async def inspect_coordinate_correction_history(
        self,
        memory_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> list[dict[str, Any]]:
        history = await self._inspection_repository.list_coordinate_correction_history(
            memory_id,
            user_id,
        )
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_coordinate_correction_history",
            target_type="memory_object",
            target_id=memory_id,
            metadata={"user_id": user_id, "result_count": len(history)},
        )
        return history

    async def _normalize_coordinate_updates(
        self,
        current: dict[str, Any],
        *,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, str | None]:
        if not updates:
            raise ValueError("At least one coordinate update is required")
        unknown = sorted(set(updates) - set(_COORDINATE_UPDATE_FIELDS))
        if unknown:
            raise ValueError(f"Unsupported coordinate fields: {', '.join(unknown)}")

        normalized: dict[str, str | None] = {
            field: _normalize_optional_text(updates[field])
            for field in updates
        }
        for field, coordinate_kind in (
            ("active_presence_id", "presence"),
            ("source_presence_id", "presence"),
            ("memory_owner_id", "mind"),
            ("source_mind_id", "mind"),
            ("embodiment_id", "embodiment"),
            ("realm_id", "realm"),
        ):
            value = normalized.get(field)
            if value is None:
                continue
            exists = await self._inspection_repository.coordinate_exists(
                user_id=user_id,
                coordinate_kind=coordinate_kind,
                coordinate_id=value,
            )
            if not exists:
                raise ValueError(f"Unknown {coordinate_kind} coordinate: {value}")

        final_space_id = normalized.get("space_id", _normalize_optional_text(current.get("space_id")))
        final_space_boundary = normalized.get(
            "space_boundary_mode",
            _normalize_optional_text(current.get("space_boundary_mode")),
        )
        if "space_id" in normalized and final_space_id is None:
            normalized.setdefault("space_boundary_mode", None)
            final_space_boundary = None
        if final_space_id is None:
            if final_space_boundary is not None:
                raise ValueError("space_boundary_mode requires space_id")
            return normalized
        space = await self._inspection_repository.get_space(user_id, final_space_id)
        if space is None:
            raise ValueError(f"Unknown space coordinate: {final_space_id}")
        if "space_id" in normalized and "space_boundary_mode" not in normalized:
            normalized["space_boundary_mode"] = str(space["boundary_mode"])
            final_space_boundary = str(space["boundary_mode"])
        if final_space_boundary is None:
            normalized["space_boundary_mode"] = str(space["boundary_mode"])
            final_space_boundary = str(space["boundary_mode"])
        if final_space_boundary not in _SPACE_BOUNDARY_VALUES:
            raise ValueError(f"Unsupported space_boundary_mode: {final_space_boundary}")
        return normalized

    async def inspect_retrieval_event(
        self,
        event_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> dict[str, Any] | None:
        event = await self._event_repository.get_event(event_id, user_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_retrieval_event",
            target_type="retrieval_event",
            target_id=event_id,
            metadata={"user_id": user_id, "found": event is not None},
        )
        return event

    async def inspect_retrieval_event_by_id(
        self,
        event_id: str,
        *,
        admin_user_id: str,
    ) -> dict[str, Any] | None:
        event = await self._inspection_repository.get_retrieval_event_by_id(event_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_retrieval_event",
            target_type="retrieval_event",
            target_id=event_id,
            metadata={
                "user_id": None if event is None else event["user_id"],
                "found": event is not None,
            },
        )
        return event

    async def inspect_user_memories(
        self,
        user_id: str,
        *,
        admin_user_id: str,
        object_type: str | None = None,
        scope: str | None = None,
        status: str | None = None,
        intimacy_boundary: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        memories = await self._inspection_repository.list_memories(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            status=status,
            intimacy_boundary=intimacy_boundary,
            limit=limit,
        )
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_user_memories",
            target_type="user_memory_collection",
            target_id=user_id,
            metadata={
                "user_id": user_id,
                "object_type": object_type,
                "scope": scope,
                "status": status,
                "intimacy_boundary": intimacy_boundary,
                "limit": limit,
                "result_count": len(memories),
            },
        )
        return memories

    async def inspect_belief_history(
        self,
        belief_id: str,
        user_id: str,
        *,
        admin_user_id: str,
    ) -> list[dict[str, Any]]:
        history = await self._inspection_repository.list_belief_history(belief_id, user_id)
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_belief_history",
            target_type="belief_history",
            target_id=belief_id,
            metadata={"user_id": user_id, "result_count": len(history)},
        )
        return history

    async def list_consequence_chains(
        self,
        user_id: str,
        *,
        admin_user_id: str,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        chains = await self._inspection_repository.list_consequence_chains(
            user_id=user_id,
            workspace_id=workspace_id,
            limit=limit,
        )
        await self._audit_repository.create_audit_entry(
            admin_user_id=admin_user_id,
            action="inspect_consequence_chains",
            target_type="consequence_chain_collection",
            target_id=user_id,
            metadata={
                "user_id": user_id,
                "workspace_id": workspace_id,
                "limit": limit,
                "result_count": len(chains),
            },
        )
        return chains


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


def _coordinate_values(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        field: _normalize_optional_text(row.get(field))
        for field in _COORDINATE_UPDATE_FIELDS
    }


def _coordinate_inspection(
    row: dict[str, Any],
    *,
    overseer_grants: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "memory": {
            field: row.get(field)
            for field in _MEMORY_INSPECTION_FIELDS
            if field in row
        },
        "coordinates": {
            "namespace": {
                "user_id": row.get("user_id"),
                "conversation_id": row.get("conversation_id"),
                "assistant_mode_id": row.get("assistant_mode_id"),
                "workspace_id": row.get("workspace_id"),
                "user_persona_id": row.get("user_persona_id"),
                "platform_id": row.get("platform_id"),
                "character_id": row.get("character_id"),
                "scope": row.get("scope"),
                "scope_canonical": row.get("scope_canonical") or row.get("scope"),
            },
            "presence": {
                "active_presence_id": row.get("active_presence_id"),
                "source_presence_id": row.get("source_presence_id"),
                "presence_cluster_id": row.get("presence_cluster_id"),
                "active_presence": _presence_ref(row, "active_presence"),
                "source_presence": _presence_ref(row, "source_presence"),
            },
            "space": {
                "space_id": row.get("space_id"),
                "space_boundary_mode": row.get("space_boundary_mode"),
                "space": _space_ref(row),
            },
            "mind": {
                "memory_owner_id": row.get("memory_owner_id"),
                "source_mind_id": row.get("source_mind_id"),
                "memory_owner": _mind_ref(row, "memory_owner"),
                "source_mind": _mind_ref(row, "source_mind"),
            },
            "embodiment": {
                "embodiment_id": row.get("embodiment_id"),
                "embodiment": _embodiment_ref(row),
            },
            "realm": {
                "realm_id": row.get("realm_id"),
                "realm": _realm_ref(row),
            },
            "overseer_grants": [_grant_ref(grant) for grant in overseer_grants],
        },
        "provenance": {
            "source_kind": row.get("source_kind"),
            "payload_coordinates": _payload_coordinate_provenance(row.get("payload_json")),
        },
    }


def _presence_ref(row: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    presence_id = _normalize_optional_text(row.get(f"{prefix}_id"))
    if presence_id is None:
        presence_id = _normalize_optional_text(
            row.get("active_presence_id" if prefix == "active_presence" else "source_presence_id")
        )
    if presence_id is None:
        return None
    return {
        "id": presence_id,
        "found": row.get(f"{prefix}_row_id") is not None,
        "kind": row.get(f"{prefix}_kind"),
        "display_name": row.get(f"{prefix}_display_name"),
        "source_kind": row.get(f"{prefix}_source_kind"),
        "source_id": row.get(f"{prefix}_source_id"),
    }


def _space_ref(row: dict[str, Any]) -> dict[str, Any] | None:
    space_id = _normalize_optional_text(row.get("space_id"))
    if space_id is None:
        return None
    return {
        "id": space_id,
        "found": row.get("space_row_id") is not None,
        "boundary_mode": row.get("space_record_boundary_mode"),
        "display_name": row.get("space_display_name"),
        "source_kind": row.get("space_source_kind"),
        "source_id": row.get("space_source_id"),
    }


def _mind_ref(row: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    mind_id = _normalize_optional_text(row.get(f"{prefix}_id"))
    if mind_id is None:
        mind_id = _normalize_optional_text(
            row.get("memory_owner_id" if prefix == "memory_owner" else "source_mind_id")
        )
    if mind_id is None:
        return None
    return {
        "id": mind_id,
        "found": row.get(f"{prefix}_row_id") is not None,
        "kind": row.get(f"{prefix}_kind"),
        "display_name": row.get(f"{prefix}_display_name"),
        "source_kind": row.get(f"{prefix}_source_kind"),
        "source_id": row.get(f"{prefix}_source_id"),
    }


def _embodiment_ref(row: dict[str, Any]) -> dict[str, Any] | None:
    embodiment_id = _normalize_optional_text(row.get("embodiment_id"))
    if embodiment_id is None:
        return None
    return {
        "id": embodiment_id,
        "found": row.get("embodiment_row_id") is not None,
        "display_name": row.get("embodiment_display_name"),
        "cross_embodiment_mode": row.get("embodiment_cross_embodiment_mode"),
        "source_kind": row.get("embodiment_source_kind"),
        "source_id": row.get("embodiment_source_id"),
    }


def _realm_ref(row: dict[str, Any]) -> dict[str, Any] | None:
    realm_id = _normalize_optional_text(row.get("realm_id"))
    if realm_id is None:
        return None
    return {
        "id": realm_id,
        "found": row.get("realm_row_id") is not None,
        "display_name": row.get("realm_display_name"),
        "cross_realm_mode": row.get("realm_cross_realm_mode"),
        "source_kind": row.get("realm_source_kind"),
        "source_id": row.get("realm_source_id"),
    }


def _grant_ref(grant: dict[str, Any]) -> dict[str, Any]:
    return {
        "overseer_mind_id": grant.get("overseer_mind_id"),
        "overseer_display_name": grant.get("overseer_display_name"),
        "target_kind": grant.get("target_kind"),
        "target_id": grant.get("target_id"),
        "grant_kind": grant.get("grant_kind"),
        "visibility": grant.get("visibility"),
        "expires_at": grant.get("expires_at"),
    }


def _payload_coordinate_provenance(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "presence_attribution",
        "space_boundary",
        "mind_perspective",
        "embodiment",
        "realm",
    )
    return {
        key: payload[key]
        for key in keys
        if isinstance(payload.get(key), dict)
    }


def _retrieval_plan_from_event(event: dict[str, Any]) -> tuple[RetrievalPlan | None, str | None]:
    payload = event.get("retrieval_plan_json")
    if not isinstance(payload, dict):
        return None, "retrieval_plan_json_missing"
    data = dict(payload)
    data.setdefault("assistant_mode_id", event.get("assistant_mode_id") or "default")
    data.setdefault("conversation_id", event.get("conversation_id") or "unknown")
    data.setdefault("platform_id", event.get("platform_id") or "default")
    data.setdefault("user_persona_id", event.get("user_persona_id"))
    data.setdefault("character_id", event.get("character_id"))
    data.setdefault("max_candidates", 0)
    data.setdefault("max_context_items", 1)
    data.setdefault("privacy_ceiling", 3)
    try:
        return RetrievalPlan.model_validate(data), None
    except ValueError as exc:
        return None, f"retrieval_plan_unparseable:{exc}"


def _candidate_from_memory(memory: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "user_id",
        "object_type",
        "scope",
        "scope_canonical",
        "conversation_id",
        "status",
        "sensitivity",
        "privacy_level",
        "active_presence_id",
        "source_presence_id",
        "presence_cluster_id",
        "space_id",
        "space_boundary_mode",
        "memory_owner_id",
        "source_mind_id",
        "embodiment_id",
        "realm_id",
        "active_realm_id",
    )
    return {
        field: memory.get(field)
        for field in fields
        if field in memory
    }


def _annotate_reconstructed_mind_grant(
    candidate: dict[str, Any],
    plan: RetrievalPlan,
    grants: list[dict[str, Any]],
) -> None:
    active_mind_id = _normalize_optional_text(plan.active_mind_id)
    topology = _normalize_optional_text(getattr(plan.mind_topology, "value", plan.mind_topology))
    if topology != "ojocentauri" or active_mind_id is None:
        return
    candidate_owner_id = _normalize_optional_text(candidate.get("memory_owner_id"))
    if candidate_owner_id is None or candidate_owner_id == active_mind_id:
        return
    targets = {
        ("mind", candidate_owner_id),
        ("space", _normalize_optional_text(candidate.get("space_id"))),
        ("realm", _normalize_optional_text(candidate.get("realm_id"))),
    }
    targets.discard(("space", None))
    targets.discard(("realm", None))
    for grant in grants:
        grant_kind = _normalize_optional_text(grant.get("grant_kind"))
        target_kind = _normalize_optional_text(grant.get("target_kind"))
        target_id = _normalize_optional_text(grant.get("target_id"))
        if (
            _normalize_optional_text(grant.get("overseer_mind_id")) == active_mind_id
            and grant_kind in OVERSEER_CONTEXT_GRANT_KINDS
            and (target_kind, target_id) in targets
        ):
            candidate["mind_relation"] = "granted"
            candidate["mind_grant_kind"] = grant_kind
            candidate["mind_grant_target_kind"] = target_kind
            candidate["mind_grant_target_id"] = target_id
            return


def _boundary_explanation(trace: dict[str, Any] | None) -> dict[str, Any]:
    if trace is None:
        return {
            "decision": "unknown",
            "reasons": [],
            "blocked_reasons": [],
            "attribution_reasons": [],
        }
    reasons: list[str] = []
    blocked: list[str] = []
    attributed: list[str] = []
    for gate_name in ("presence", "space", "mind", "embodiment", "realm"):
        gate = trace.get(gate_name)
        if not isinstance(gate, dict):
            continue
        reason = _normalize_optional_text(gate.get("reason"))
        if reason is None:
            continue
        reasons.append(reason)
        if gate.get("allowed") is False or gate.get("decision") == "blocked":
            blocked.append(reason)
        if _is_allowed_attribution_reason(gate, reason):
            attributed.append(reason)
    return {
        "decision": "blocked" if blocked else "allowed",
        "reasons": reasons,
        "blocked_reasons": blocked,
        "attribution_reasons": attributed,
    }


def _is_allowed_attribution_reason(gate: dict[str, Any], reason: str) -> bool:
    if gate.get("allowed") is not True and gate.get("decision") != "allowed":
        return False
    return (
        reason == "allowed_by_overseer_grant"
        or reason.startswith("allowed_by_realm_bridge_")
        or reason == "allowed_cross_presence_attributed"
    )


def _find_custody_record(event: dict[str, Any], memory_id: str) -> dict[str, Any] | None:
    outcome = event.get("outcome_json")
    if not isinstance(outcome, dict):
        return None
    custody = outcome.get("retrieval_custody_v2")
    if not isinstance(custody, list):
        return None
    for record in custody:
        if isinstance(record, dict) and str(record.get("candidate_id") or "") == memory_id:
            return dict(record)
    return None


def _retrieval_decision_label(
    custody_record: dict[str, Any] | None,
    memory: dict[str, Any] | None,
) -> str:
    if memory is None:
        return "memory_not_found_for_user"
    if custody_record is None:
        return "not_present_in_candidate_custody"
    if custody_record.get("selected") is True:
        return "selected"
    filter_reason = _normalize_optional_text(custody_record.get("filter_reason"))
    if filter_reason is not None:
        return f"filtered:{filter_reason}"
    return str(custody_record.get("composer_decision") or "candidate_not_selected")


__all__ = ["MemoryInspector"]
