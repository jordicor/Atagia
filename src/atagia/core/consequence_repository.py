"""SQLite repository helpers for consequence chains."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.core.repositories import BaseRepository


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
        await self._ensure_memory_belongs_to_user(payload.action_memory_id, payload.user_id)
        await self._ensure_memory_belongs_to_user(payload.outcome_memory_id, payload.user_id)
        if payload.tendency_belief_id is not None:
            await self._ensure_memory_belongs_to_user(payload.tendency_belief_id, payload.user_id)

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
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                payload.status,
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
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND action_memory_id = ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC, id ASC
            """,
            (user_id, action_memory_id),
        )

    async def find_chains_for_workspace(
        self,
        user_id: str,
        workspace_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND workspace_id = ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT ?
            """,
            (user_id, workspace_id, limit),
        )

    async def find_chains_by_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM consequence_chains
            WHERE user_id = ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC, id ASC
            LIMIT ?
            """,
            (user_id, limit),
        )

    async def update_chain_confidence(
        self,
        chain_id: str,
        user_id: str,
        confidence: float,
        *,
        commit: bool = True,
    ) -> None:
        await self._connection.execute(
            """
            UPDATE consequence_chains
            SET confidence = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (confidence, self._timestamp(), chain_id, user_id),
        )
        if commit:
            await self._connection.commit()

    async def archive_chain(
        self,
        chain_id: str,
        user_id: str,
        *,
        commit: bool = True,
    ) -> None:
        await self._connection.execute(
            """
            UPDATE consequence_chains
            SET status = 'archived',
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (self._timestamp(), chain_id, user_id),
        )
        if commit:
            await self._connection.commit()

    async def _ensure_memory_belongs_to_user(self, memory_id: str, user_id: str) -> None:
        row = await self._fetch_one(
            """
            SELECT id
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )
        if row is None:
            raise ValueError(f"Memory object {memory_id} does not exist for user {user_id}")
