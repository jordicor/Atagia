"""SQLite-owned lifecycle identity for one conversation's canonical sources."""

from __future__ import annotations

from dataclasses import dataclass

from atagia.core.repositories import BaseRepository


@dataclass(frozen=True, slots=True)
class ConversationLifecycleIdentity:
    """Immutable conversation epoch plus its monotonic source revision."""

    lifecycle_epoch: str
    source_revision: int


class ConversationLifecycleRepository(BaseRepository):
    """Capture and compare exact active-conversation source coordinates."""

    async def get_identity(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> ConversationLifecycleIdentity | None:
        """Return exact coordinates for an owned conversation in any status."""

        return await self._get_identity(
            user_id=user_id,
            conversation_id=conversation_id,
            active_only=False,
        )

    async def get_active_identity(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> ConversationLifecycleIdentity | None:
        return await self._get_identity(
            user_id=user_id,
            conversation_id=conversation_id,
            active_only=True,
        )

    async def _get_identity(
        self,
        *,
        user_id: str,
        conversation_id: str,
        active_only: bool,
    ) -> ConversationLifecycleIdentity | None:
        active_clause = "AND conversation.status = 'active'" if active_only else ""
        row = await self._fetch_one(
            f"""
            SELECT
                lifecycle.lifecycle_epoch,
                lifecycle.source_revision
            FROM conversation_lifecycles AS lifecycle
            JOIN conversations AS conversation
              ON conversation.id = lifecycle.conversation_id
             AND conversation.user_id = lifecycle.user_id
            WHERE lifecycle.user_id = ?
              AND lifecycle.conversation_id = ?
              {active_clause}
            LIMIT 1
            """,
            (user_id, conversation_id),
        )
        if row is None:
            return None
        return ConversationLifecycleIdentity(
            lifecycle_epoch=str(row["lifecycle_epoch"]),
            source_revision=int(row["source_revision"]),
        )

    async def matches_identity(
        self,
        *,
        user_id: str,
        conversation_id: str,
        identity: ConversationLifecycleIdentity,
    ) -> bool:
        current = await self.get_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return current == identity

    async def matches_active_identity(
        self,
        *,
        user_id: str,
        conversation_id: str,
        identity: ConversationLifecycleIdentity,
    ) -> bool:
        current = await self.get_active_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return current == identity
