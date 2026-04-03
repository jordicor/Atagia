"""Admin inspection helpers."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.repositories import BaseRepository
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository


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

    async def list_memories(
        self,
        user_id: str,
        object_type: str | None,
        scope: str | None,
        status: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if object_type is not None:
            clauses.append("object_type = ?")
            parameters.append(object_type)
        if scope is not None:
            clauses.append("scope = ?")
            parameters.append(scope)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
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
            SELECT bv.*
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
                outcome.canonical_text AS outcome_canonical_text,
                tendency.canonical_text AS tendency_canonical_text
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
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        memories = await self._inspection_repository.list_memories(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            status=status,
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
