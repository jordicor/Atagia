"""Async refresh orchestration for conversation Topic Working Sets."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import MessageRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.topic_working_set import TopicWorkingSetUpdater
from atagia.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TopicWorkingSetRefreshResult:
    """Result metadata for one non-blocking topic refresh attempt."""

    refreshed: bool
    reason: str
    changed_topic_count: int = 0
    processed_message_count: int = 0
    freshness: dict[str, Any] | None = None


@dataclass(slots=True)
class TopicWorkingSetRefreshService:
    """Refresh Topic Working Sets from async/post-response message processing."""

    connection: aiosqlite.Connection
    llm_client: LLMClient[Any]
    clock: Clock
    settings: Settings

    async def maybe_refresh_after_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
    ) -> TopicWorkingSetRefreshResult:
        if not self.settings.topic_working_set_enabled:
            return TopicWorkingSetRefreshResult(refreshed=False, reason="disabled")

        messages = MessageRepository(self.connection, self.clock)
        source_message = await messages.get_message(message_id, user_id)
        if source_message is None:
            return TopicWorkingSetRefreshResult(refreshed=False, reason="message_not_found")

        return await self.maybe_refresh(
            user_id=user_id,
            conversation_id=conversation_id,
            current_seq_upper_bound=int(source_message["seq"]),
        )

    async def maybe_refresh(
        self,
        *,
        user_id: str,
        conversation_id: str,
        current_seq_upper_bound: int | None = None,
    ) -> TopicWorkingSetRefreshResult:
        if not self.settings.topic_working_set_enabled:
            return TopicWorkingSetRefreshResult(refreshed=False, reason="disabled")

        topics = TopicRepository(self.connection, self.clock)
        messages = MessageRepository(self.connection, self.clock)
        snapshot = await self._snapshot(
            topics,
            user_id=user_id,
            conversation_id=conversation_id,
            current_seq_upper_bound=current_seq_upper_bound,
        )
        freshness = dict(snapshot.get("freshness") or {})
        if not self._should_refresh(freshness):
            return TopicWorkingSetRefreshResult(
                refreshed=False,
                reason=str(freshness.get("status") or "fresh"),
                freshness=freshness,
            )

        latest_seq = int(freshness.get("latest_message_seq") or 0)
        last_processed_seq = freshness.get("last_processed_seq")
        start_seq = int(last_processed_seq or 0) + 1
        end_seq = min(
            latest_seq,
            start_seq + self.settings.topic_working_set_refresh_batch_messages - 1,
        )
        if start_seq > end_seq:
            return TopicWorkingSetRefreshResult(
                refreshed=False,
                reason="no_unprocessed_messages",
                freshness=freshness,
            )

        message_batch = await messages.get_messages_in_seq_range(
            conversation_id,
            user_id,
            start_seq,
            end_seq,
        )
        if not message_batch:
            return TopicWorkingSetRefreshResult(
                refreshed=False,
                reason="empty_batch",
                freshness=freshness,
            )

        updater = TopicWorkingSetUpdater(
            llm_client=self.llm_client,
            clock=self.clock,
            topic_repository=topics,
            message_repository=messages,
            settings=self.settings,
        )
        changed_topics = await updater.update_from_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=message_batch,
        )
        logger.info(
            "Refreshed Topic Working Set for conversation_id=%s through seq=%s",
            conversation_id,
            end_seq,
        )
        return TopicWorkingSetRefreshResult(
            refreshed=True,
            reason=str(freshness.get("status") or "refresh_due"),
            changed_topic_count=len(changed_topics),
            processed_message_count=len(message_batch),
            freshness=freshness,
        )

    async def _snapshot(
        self,
        topics: TopicRepository,
        *,
        user_id: str,
        conversation_id: str,
        current_seq_upper_bound: int | None,
    ) -> dict[str, Any]:
        return await topics.get_topic_snapshot(
            user_id=user_id,
            conversation_id=conversation_id,
            current_seq_upper_bound=current_seq_upper_bound,
            refresh_message_threshold=self.settings.topic_working_set_refresh_message_lag,
            stale_message_threshold=self.settings.topic_working_set_stale_message_lag,
            refresh_token_threshold=self.settings.topic_working_set_refresh_token_lag,
            stale_token_threshold=self.settings.topic_working_set_stale_token_lag,
        )

    def _should_refresh(self, freshness: dict[str, Any]) -> bool:
        status = str(freshness.get("status") or "fresh")
        if status in {"slightly_stale", "stale"}:
            return True
        if status != "missing":
            return False
        return (
            int(freshness.get("lag_message_count") or 0)
            >= self.settings.topic_working_set_refresh_message_lag
            or int(freshness.get("lag_token_count") or 0)
            >= self.settings.topic_working_set_refresh_token_lag
        )
