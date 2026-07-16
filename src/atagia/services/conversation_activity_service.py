"""Conversation activity aggregation and warm-up orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import log1p
from statistics import mean, median
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from atagia.core.conversation_activity_repository import ConversationActivityRepository
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleIdentity,
    ConversationLifecycleRepository,
)
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    WorkspaceRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.storage_backend import build_recent_window_key
from atagia.core.transcript_rebuild_repository import (
    TranscriptRebuildRepository,
    UserAvailabilitySnapshot,
)
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.timestamps import (
    normalize_optional_timestamp,
    resolve_message_occurred_at,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.chat_support import default_operational_profile_snapshot
from atagia.services.errors import TranscriptRebuildInProgressError

if TYPE_CHECKING:
    from atagia.app import AppRuntime

_ACTIVITY_VERSION = 1
_RECENT_MESSAGE_BUDGET_DEFAULT = 12
_HOT_CONVERSATION_LIMIT_MAX = 100
_RECENT_MESSAGE_BUDGET_MAX = 100
_RECOMMENDED_TOTAL_MESSAGE_BUDGET_MAX = 200
_RETURN_INTERVAL_BUCKET_LIMITS_MINUTES = (
    60.0,
    6.0 * 60.0,
    12.0 * 60.0,
    24.0 * 60.0,
    2.0 * 24.0 * 60.0,
    4.0 * 24.0 * 60.0,
    7.0 * 24.0 * 60.0,
    14.0 * 24.0 * 60.0,
    30.0 * 24.0 * 60.0,
)


@dataclass(slots=True)
class _ConversationActivityPayload:
    user_id: str
    conversation_id: str
    as_of: str
    stats: dict[str, Any] | None
    recent_messages: list[dict[str, Any]]
    existing_cache_view: dict[str, Any] | None
    recent_window_key: str
    cache_key: str
    warmup_errors: list[str]
    lifecycle_epoch: str | None
    lifecycle_cleanup_key: str | None
    cache_revision: int | None
    derivation_revision: int | None
    conversation_lifecycle_epoch: str | None
    conversation_source_revision: int | None
    availability_snapshot: UserAvailabilitySnapshot
    user_persona_id: str | None
    platform_id: str | None
    character_id: str | None
    incognito: bool


@dataclass(frozen=True, slots=True)
class _WarmupEligibility:
    source: ConversationLifecycleIdentity | None
    eligible: bool


class ConversationActivityService:
    """Derive, persist, rank, and warm up conversation activity stats."""

    def __init__(self, runtime: AppRuntime) -> None:
        self.runtime = runtime

    async def refresh_user_activity_stats(
        self,
        connection: Any,
        user_id: str,
        *,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        stats_rows, _ = await self._refresh_user_activity_stats_with_sources(
            connection,
            user_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            as_of=as_of,
        )
        return stats_rows

    async def _refresh_user_activity_stats_with_sources(
        self,
        connection: Any,
        user_id: str,
        *,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, ConversationLifecycleIdentity],
    ]:
        rebuild_repository = TranscriptRebuildRepository(
            connection,
            self.runtime.clock,
        )
        activity_repository = ConversationActivityRepository(
            connection, self.runtime.clock
        )
        await connection.execute("BEGIN")
        try:
            availability_snapshot = (
                await rebuild_repository.capture_user_availability_snapshot(user_id)
            )
            rows, captured_sources = await self._capture_user_activity_sources(
                connection,
                user_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            stats_rows = [
                await self._compute_conversation_stats(
                    connection,
                    user_id=user_id,
                    conversation=row,
                    as_of=as_of,
                )
                for row in rows
            ]
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await rebuild_repository.require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            await self._require_user_activity_sources(
                connection,
                user_id,
                captured_sources,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            await activity_repository.upsert_activity_stats_bulk(
                stats_rows, commit=False
            )
            _, published_sources = await self._capture_user_activity_sources(
                connection,
                user_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            if published_sources.keys() != captured_sources.keys():
                raise TranscriptRebuildInProgressError(
                    "Conversation set changed while activity refresh was in progress; retry"
                )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._require_current_user_activity_sources(
            user_id,
            availability_snapshot,
            published_sources,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        return stats_rows, published_sources

    async def refresh_conversation_activity_stats(
        self,
        connection: Any,
        user_id: str,
        conversation_id: str,
        *,
        as_of: str | None = None,
    ) -> dict[str, Any] | None:
        stats, _ = await self._refresh_conversation_activity_stats_with_source(
            connection,
            user_id,
            conversation_id,
            as_of=as_of,
        )
        return stats

    async def _refresh_conversation_activity_stats_with_source(
        self,
        connection: Any,
        user_id: str,
        conversation_id: str,
        *,
        as_of: str | None = None,
        expected_source: ConversationLifecycleIdentity | None = None,
    ) -> tuple[
        dict[str, Any] | None,
        ConversationLifecycleIdentity | None,
    ]:
        rebuild_repository = TranscriptRebuildRepository(
            connection,
            self.runtime.clock,
        )
        await connection.execute("BEGIN")
        try:
            availability_snapshot = (
                await rebuild_repository.capture_user_availability_snapshot(user_id)
            )
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            captured_source = await ConversationLifecycleRepository(
                connection,
                self.runtime.clock,
            ).get_identity(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if expected_source is not None and captured_source != expected_source:
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed before activity refresh; retry"
                )
            if (conversation is None) != (captured_source is None):
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed while activity refresh was in progress; retry"
                )
            stats = (
                None
                if conversation is None
                else await self._compute_conversation_stats(
                    connection,
                    user_id=user_id,
                    conversation=conversation,
                    as_of=as_of,
                )
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        if conversation is None:
            await self._require_current_conversation_activity_source(
                user_id,
                availability_snapshot,
                conversation_id,
                None,
            )
            return None, None
        assert captured_source is not None
        assert stats is not None
        activity_repository = ConversationActivityRepository(
            connection, self.runtime.clock
        )
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await rebuild_repository.require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            if not await ConversationLifecycleRepository(
                connection,
                self.runtime.clock,
            ).matches_identity(
                user_id=user_id,
                conversation_id=conversation_id,
                identity=captured_source,
            ):
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed while activity refresh was in progress; retry"
                )
            await activity_repository.upsert_activity_stats(stats, commit=False)
            published_source = await ConversationLifecycleRepository(
                connection,
                self.runtime.clock,
            ).get_identity(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if published_source is None:
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed while activity refresh was in progress; retry"
                )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._require_current_conversation_activity_source(
            user_id,
            availability_snapshot,
            conversation_id,
            published_source,
        )
        return stats, published_source

    async def recompute_after_authoritative_transcript_change(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Rebuild activity state inside an owning transcript transaction.

        The selected-transcript coordinator already owns the user's write fence,
        so this internal path intentionally does not apply the public availability
        guard or commit independently.
        """

        conversation = await ConversationRepository(
            connection,
            self.runtime.clock,
        ).get_conversation(conversation_id, user_id)
        if conversation is None:
            raise RuntimeError("Authoritative transcript conversation disappeared")
        stats = await self._compute_conversation_stats(
            connection,
            user_id=user_id,
            conversation=conversation,
            as_of=None,
        )
        await ConversationActivityRepository(
            connection,
            self.runtime.clock,
        ).upsert_activity_stats(stats, commit=False)
        return stats

    async def get_activity_snapshot(
        self,
        connection: Any,
        user_id: str,
        *,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
        refresh: bool = True,
    ) -> dict[str, Any]:
        rebuild_repository = TranscriptRebuildRepository(
            connection,
            self.runtime.clock,
        )
        availability_snapshot = (
            await rebuild_repository.capture_user_availability_snapshot(user_id)
        )
        refreshed_single_source: ConversationLifecycleIdentity | None = None
        refreshed_sources: dict[str, ConversationLifecycleIdentity] | None = None
        read_sources: dict[str, ConversationLifecycleIdentity] | None = None
        if conversation_id is not None:
            if refresh:
                (
                    row,
                    refreshed_single_source,
                ) = await self._refresh_conversation_activity_stats_with_source(
                    connection,
                    user_id,
                    conversation_id,
                    as_of=as_of,
                )
            else:
                started_transaction = not connection.in_transaction
                if started_transaction:
                    await connection.execute("BEGIN")
                try:
                    row = await ConversationActivityRepository(
                        connection, self.runtime.clock
                    ).get_activity_stats(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    read_sources = await self._capture_activity_membership_sources(
                        connection,
                        user_id,
                        conversation_id=conversation_id,
                    )
                    if started_transaction:
                        await connection.commit()
                except Exception:
                    if started_transaction:
                        await connection.rollback()
                    raise
            conversations = [row] if row is not None else []
        else:
            if refresh:
                (
                    _,
                    refreshed_sources,
                ) = await self._refresh_user_activity_stats_with_sources(
                    connection,
                    user_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=assistant_mode_id,
                    namespace_filter=namespace_filter,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    as_of=as_of,
                )
            started_transaction = not connection.in_transaction
            if started_transaction and not refresh:
                await connection.execute("BEGIN")
            try:
                conversations = await ConversationActivityRepository(
                    connection,
                    self.runtime.clock,
                ).list_activity_stats(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=assistant_mode_id,
                    namespace_filter=namespace_filter,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    as_of=as_of,
                    active_only=refresh,
                )
                if not refresh:
                    read_sources = await self._capture_activity_membership_sources(
                        connection,
                        user_id,
                        workspace_id=workspace_id,
                        assistant_mode_id=assistant_mode_id,
                        namespace_filter=namespace_filter,
                        user_persona_id=user_persona_id,
                        platform_id=platform_id,
                        character_id=character_id,
                        incognito=incognito,
                        active_only=False,
                    )
                if started_transaction and not refresh:
                    await connection.commit()
            except Exception:
                if started_transaction and not refresh:
                    await connection.rollback()
                raise
        result = {
            "user_id": user_id,
            "as_of": self._resolve_as_of(as_of).isoformat(),
            "filters": {
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "assistant_mode_id": assistant_mode_id,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "incognito": incognito,
            },
            "conversations": conversations,
            "conversation_count": len(conversations),
        }
        if refresh and conversation_id is not None:
            await self._require_current_conversation_activity_source(
                user_id,
                availability_snapshot,
                conversation_id,
                refreshed_single_source,
            )
        elif refreshed_sources is not None:
            await self._require_current_user_activity_sources(
                user_id,
                availability_snapshot,
                refreshed_sources,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
        elif read_sources is not None:
            await self._require_current_activity_membership(
                user_id,
                availability_snapshot,
                read_sources,
                conversation_id=conversation_id,
                workspace_id=None if conversation_id is not None else workspace_id,
                assistant_mode_id=(
                    None if conversation_id is not None else assistant_mode_id
                ),
                namespace_filter=(
                    False if conversation_id is not None else namespace_filter
                ),
                user_persona_id=(
                    None if conversation_id is not None else user_persona_id
                ),
                platform_id=None if conversation_id is not None else platform_id,
                character_id=(None if conversation_id is not None else character_id),
                incognito=False if conversation_id is not None else incognito,
                active_only=False,
            )
        else:
            await self._require_current_availability(
                user_id,
                availability_snapshot,
            )
        return result

    async def list_hot_conversations(
        self,
        connection: Any,
        user_id: str,
        *,
        limit: int = 5,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
        refresh: bool = True,
    ) -> list[dict[str, Any]]:
        rows, _ = await self._list_hot_conversations_with_sources(
            connection,
            user_id,
            limit=limit,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            as_of=as_of,
            refresh=refresh,
        )
        return rows

    async def _list_hot_conversations_with_sources(
        self,
        connection: Any,
        user_id: str,
        *,
        limit: int = 5,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
        refresh: bool = True,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, ConversationLifecycleIdentity] | None,
    ]:
        rebuild_repository = TranscriptRebuildRepository(
            connection,
            self.runtime.clock,
        )
        availability_snapshot = (
            await rebuild_repository.capture_user_availability_snapshot(user_id)
        )
        effective_limit = self._bounded_int(
            limit,
            default=5,
            minimum=1,
            maximum=_HOT_CONVERSATION_LIMIT_MAX,
        )
        refreshed_sources: dict[str, ConversationLifecycleIdentity] | None = None
        read_sources: dict[str, ConversationLifecycleIdentity] | None = None
        if refresh:
            _, refreshed_sources = await self._refresh_user_activity_stats_with_sources(
                connection,
                user_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=platform_id is not None,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                as_of=as_of,
            )
        started_transaction = not connection.in_transaction
        if started_transaction and not refresh:
            await connection.execute("BEGIN")
        try:
            rows = await ConversationActivityRepository(
                connection, self.runtime.clock
            ).list_activity_stats(
                user_id=user_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=platform_id is not None,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                limit=effective_limit,
                as_of=as_of,
                active_only=True,
            )
            if not refresh:
                read_sources = await self._capture_activity_membership_sources(
                    connection,
                    user_id,
                    workspace_id=workspace_id,
                    assistant_mode_id=assistant_mode_id,
                    namespace_filter=platform_id is not None,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    active_only=True,
                )
            if started_transaction and not refresh:
                await connection.commit()
        except Exception:
            if started_transaction and not refresh:
                await connection.rollback()
            raise
        if refreshed_sources is None:
            await self._require_current_activity_membership(
                user_id,
                availability_snapshot,
                read_sources or {},
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=platform_id is not None,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                active_only=True,
            )
        else:
            await self._require_current_user_activity_sources(
                user_id,
                availability_snapshot,
                refreshed_sources,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=platform_id is not None,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
        return rows, refreshed_sources

    async def warmup_conversation(
        self,
        connection: Any,
        user_id: str,
        conversation_id: str,
        *,
        max_messages: int = _RECENT_MESSAGE_BUDGET_DEFAULT,
        as_of: str | None = None,
        refresh_stats: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        _expected_stats_source: ConversationLifecycleIdentity | None = None,
        _published_payloads: list[_ConversationActivityPayload] | None = None,
    ) -> dict[str, Any]:
        effective_max_messages = self._bounded_int(
            max_messages,
            default=_RECENT_MESSAGE_BUDGET_DEFAULT,
            minimum=1,
            maximum=_RECENT_MESSAGE_BUDGET_MAX,
        )
        started_transaction = not connection.in_transaction
        if started_transaction:
            await connection.execute("BEGIN")
        try:
            availability_snapshot = await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).capture_user_availability_snapshot(user_id)
            conversation, eligibility = await self._capture_warmup_eligibility(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            if started_transaction:
                await connection.commit()
        except Exception:
            if started_transaction:
                await connection.rollback()
            raise
        if not eligibility.eligible:
            result = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "as_of": self._resolve_as_of(as_of).isoformat(),
                "recent_window_key": self._recent_window_key(user_id, conversation_id),
                "recent_message_count": 0,
                "recent_message_ids": [],
                "cached_context_key": None,
                "cached_context_available": False,
                "stats": None,
                "warmup_errors": ["conversation_not_found"],
            }
            await self._require_current_warmup_eligibility(
                user_id,
                availability_snapshot,
                conversation_id=conversation_id,
                expected=eligibility,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            return result

        assert conversation is not None
        assert eligibility.source is not None
        if (
            _expected_stats_source is not None
            and eligibility.source != _expected_stats_source
        ):
            raise TranscriptRebuildInProgressError(
                "Conversation source changed between hot ranking and warmup; retry"
            )

        resolved_as_of = self._resolve_as_of(as_of).isoformat()
        stats = None
        stats_source = _expected_stats_source or eligibility.source
        if refresh_stats:
            (
                stats,
                stats_source,
            ) = await self._refresh_conversation_activity_stats_with_source(
                connection,
                user_id,
                conversation_id,
                as_of=resolved_as_of,
                expected_source=eligibility.source,
            )
        else:
            stats = await ConversationActivityRepository(
                connection,
                self.runtime.clock,
            ).get_activity_stats(user_id=user_id, conversation_id=conversation_id)
        payload = await self._build_warmup_payload(
            connection,
            user_id=user_id,
            conversation=conversation,
            max_messages=effective_max_messages,
            as_of=resolved_as_of,
            availability_snapshot=availability_snapshot,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        if stats_source is not None:
            payload_source = (
                None
                if payload.conversation_lifecycle_epoch is None
                or payload.conversation_source_revision is None
                else ConversationLifecycleIdentity(
                    lifecycle_epoch=payload.conversation_lifecycle_epoch,
                    source_revision=payload.conversation_source_revision,
                )
            )
            if payload_source != stats_source:
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed between activity stats and warmup; retry"
                )
        payload.stats = stats or payload.stats
        result = await self._finalize_warmup_payload(payload)
        if (
            _published_payloads is not None
            and "recent_window_failed" not in payload.warmup_errors
        ):
            _published_payloads.append(payload)
        return result

    async def warmup_recommended_conversations(
        self,
        connection: Any,
        user_id: str,
        *,
        limit: int = 3,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        as_of: str | None = None,
        lead_time_minutes: int | None = None,
        total_message_budget: int = 24,
        per_conversation_message_budget: int = _RECENT_MESSAGE_BUDGET_DEFAULT,
    ) -> dict[str, Any]:
        rebuild_repository = TranscriptRebuildRepository(
            connection,
            self.runtime.clock,
        )
        availability_snapshot = (
            await rebuild_repository.capture_user_availability_snapshot(user_id)
        )
        effective_limit = self._bounded_int(
            limit,
            default=3,
            minimum=1,
            maximum=_HOT_CONVERSATION_LIMIT_MAX,
        )
        effective_total_budget = self._bounded_int(
            total_message_budget,
            default=24,
            minimum=0,
            maximum=_RECOMMENDED_TOTAL_MESSAGE_BUDGET_MAX,
        )
        effective_per_conversation_budget = self._bounded_int(
            per_conversation_message_budget,
            default=_RECENT_MESSAGE_BUDGET_DEFAULT,
            minimum=1,
            maximum=_RECENT_MESSAGE_BUDGET_MAX,
        )
        resolved_as_of = self._resolve_as_of(as_of, lead_time_minutes=lead_time_minutes)
        (
            hot_conversations,
            hot_sources,
        ) = await self._list_hot_conversations_with_sources(
            connection,
            user_id,
            limit=effective_limit,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            as_of=resolved_as_of.isoformat(),
            refresh=True,
        )
        assert hot_sources is not None

        warmed: list[dict[str, Any]] = []
        published_payloads: list[_ConversationActivityPayload] = []
        remaining_budget = effective_total_budget
        try:
            for stats in hot_conversations:
                if remaining_budget <= 0:
                    break
                conversation_id = str(stats["conversation_id"])
                stats_source = hot_sources.get(conversation_id)
                if stats_source is None:
                    raise TranscriptRebuildInProgressError(
                        "Conversation set changed between hot ranking and warmup; retry"
                    )
                warmup = await self.warmup_conversation(
                    connection,
                    user_id,
                    conversation_id,
                    max_messages=min(
                        effective_per_conversation_budget,
                        remaining_budget,
                    ),
                    as_of=resolved_as_of.isoformat(),
                    refresh_stats=False,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    _expected_stats_source=stats_source,
                    _published_payloads=published_payloads,
                )
                warmed.append(warmup)
                remaining_budget -= int(warmup["recent_message_count"])

            result = {
                "user_id": user_id,
                "as_of": resolved_as_of.isoformat(),
                "requested_limit": effective_limit,
                "total_message_budget": effective_total_budget,
                "per_conversation_message_budget": effective_per_conversation_budget,
                "workspace_id": workspace_id,
                "assistant_mode_id": assistant_mode_id,
                "user_persona_id": user_persona_id,
                "platform_id": platform_id,
                "character_id": character_id,
                "incognito": incognito,
                "hot_conversations": hot_conversations,
                "warmed_conversations": warmed,
                "warmed_conversation_count": len(warmed),
                "warmed_message_count": sum(
                    int(item["recent_message_count"]) for item in warmed
                ),
            }
            await self._require_current_user_activity_sources(
                user_id,
                availability_snapshot,
                hot_sources,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=platform_id is not None,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            return result
        except BaseException:
            cache_service = ContextCacheService(self.runtime)
            for payload in published_payloads:
                assert payload.lifecycle_cleanup_key is not None
                assert payload.lifecycle_epoch is not None
                assert payload.cache_revision is not None
                assert payload.derivation_revision is not None
                assert payload.conversation_lifecycle_epoch is not None
                assert payload.conversation_source_revision is not None
                await cache_service.discard_recent_window_publication(
                    user_id=payload.user_id,
                    conversation_id=payload.conversation_id,
                    lifecycle_cleanup_key=payload.lifecycle_cleanup_key,
                    lifecycle_epoch=payload.lifecycle_epoch,
                    cache_revision=payload.cache_revision,
                    derivation_revision=payload.derivation_revision,
                    conversation_lifecycle_epoch=(payload.conversation_lifecycle_epoch),
                    conversation_source_revision=(payload.conversation_source_revision),
                )
            raise

    async def _build_warmup_payload(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation: dict[str, Any],
        max_messages: int,
        as_of: str | None,
        availability_snapshot: UserAvailabilitySnapshot,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> _ConversationActivityPayload:
        resolved_as_of = self._resolve_as_of(as_of).isoformat()
        messages = MessageRepository(connection, self.runtime.clock)
        started_transaction = not connection.in_transaction
        if started_transaction:
            await connection.execute("BEGIN")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            active_conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_active_conversation(str(conversation["id"]), user_id)
            if (
                active_conversation is not None
                and not self._conversation_matches_namespace(
                    active_conversation,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                )
            ):
                active_conversation = None
            lifecycle_identity = await UserLifecycleRepository(
                connection,
                self.runtime.clock,
            ).get_active_identity(user_id)
            if (
                active_conversation is None
                or lifecycle_identity is None
                or lifecycle_identity.lifecycle_epoch
                != availability_snapshot.lifecycle_epoch
                or lifecycle_identity.derivation_revision
                != availability_snapshot.derivation_revision
            ):
                lifecycle_identity = None
            stored_messages = (
                []
                if active_conversation is None
                else await messages.get_recent_messages(
                    str(active_conversation["id"]),
                    user_id,
                    limit=max_messages,
                )
            )
            conversation_lifecycle_identity = (
                None
                if active_conversation is None
                else await ConversationLifecycleRepository(
                    connection,
                    self.runtime.clock,
                ).get_active_identity(
                    user_id=user_id,
                    conversation_id=str(active_conversation["id"]),
                )
            )
            if started_transaction:
                await connection.commit()
        except Exception:
            if started_transaction:
                await connection.rollback()
            raise
        if active_conversation is not None:
            conversation = active_conversation
        recent_messages = [
            {
                "id": str(message["id"]),
                "seq": int(message["seq"]),
                "role": str(message["role"]),
                "text": str(message["text"]),
                "occurred_at": message.get("occurred_at"),
            }
            for message in stored_messages
        ]
        recent_window_key = self._recent_window_key(user_id, str(conversation["id"]))
        cache_key = ContextCacheService.build_cache_key(
            user_id=user_id,
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=(
                str(conversation["workspace_id"])
                if conversation.get("workspace_id") is not None
                else None
            ),
            active_presence_id=(
                str(conversation["active_presence_id"])
                if conversation.get("active_presence_id") is not None
                else None
            ),
            active_space_id=(
                str(conversation["active_space_id"])
                if conversation.get("active_space_id") is not None
                else None
            ),
            active_mind_id=(
                str(conversation["active_mind_id"])
                if conversation.get("active_mind_id") is not None
                else None
            ),
            mind_topology=(
                str(conversation["mind_topology"])
                if conversation.get("mind_topology") is not None
                else None
            ),
            active_embodiment_id=(
                str(conversation["active_embodiment_id"])
                if conversation.get("active_embodiment_id") is not None
                else None
            ),
            active_realm_id=(
                str(conversation["active_realm_id"])
                if conversation.get("active_realm_id") is not None
                else None
            ),
            operational_profile_token=default_operational_profile_snapshot(
                loader=self.runtime.operational_profile_loader,
                settings=self.runtime.settings,
            ).token,
            lifecycle_epoch=(
                lifecycle_identity.lifecycle_epoch
                if lifecycle_identity is not None
                else None
            ),
            cache_revision=(
                lifecycle_identity.cache_revision
                if lifecycle_identity is not None
                else None
            ),
            derivation_revision=(
                lifecycle_identity.derivation_revision
                if lifecycle_identity is not None
                else None
            ),
        )
        warmup_errors = (
            [] if active_conversation is not None else ["conversation_not_active"]
        )
        existing_cache_view = None
        if lifecycle_identity is not None:
            try:
                existing_cache_view = (
                    await self.runtime.storage_backend.get_context_view(cache_key)
                )
            except Exception:
                warmup_errors.append("context_cache_read_failed")
        payload = _ConversationActivityPayload(
            user_id=user_id,
            conversation_id=str(conversation["id"]),
            as_of=resolved_as_of,
            stats=None,
            recent_messages=recent_messages,
            existing_cache_view=existing_cache_view,
            recent_window_key=recent_window_key,
            cache_key=cache_key,
            warmup_errors=warmup_errors,
            lifecycle_epoch=(
                lifecycle_identity.lifecycle_epoch
                if lifecycle_identity is not None
                else None
            ),
            lifecycle_cleanup_key=(
                lifecycle_identity.lifecycle_cleanup_key
                if lifecycle_identity is not None
                else None
            ),
            cache_revision=(
                lifecycle_identity.cache_revision
                if lifecycle_identity is not None
                else None
            ),
            derivation_revision=(
                lifecycle_identity.derivation_revision
                if lifecycle_identity is not None
                else None
            ),
            conversation_lifecycle_epoch=(
                conversation_lifecycle_identity.lifecycle_epoch
                if conversation_lifecycle_identity is not None
                else None
            ),
            conversation_source_revision=(
                conversation_lifecycle_identity.source_revision
                if conversation_lifecycle_identity is not None
                else None
            ),
            availability_snapshot=availability_snapshot,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        return payload

    @staticmethod
    def _conversation_matches_namespace(
        conversation: dict[str, Any],
        *,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> bool:
        if platform_id is None:
            return True
        return (
            conversation.get("platform_id") == platform_id
            and conversation.get("user_persona_id") == user_persona_id
            and conversation.get("character_id") == character_id
            and bool(conversation.get("incognito")) is bool(incognito)
        )

    async def _capture_warmup_eligibility(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> tuple[dict[str, Any] | None, _WarmupEligibility]:
        conversation = await ConversationRepository(
            connection,
            self.runtime.clock,
        ).get_conversation(conversation_id, user_id)
        source = await ConversationLifecycleRepository(
            connection,
            self.runtime.clock,
        ).get_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if (conversation is None) != (source is None):
            raise TranscriptRebuildInProgressError(
                "Conversation ownership changed while warmup eligibility was read; retry"
            )
        eligible = bool(
            conversation is not None
            and conversation.get("status") == "active"
            and self._conversation_matches_namespace(
                conversation,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
        )
        return conversation, _WarmupEligibility(source=source, eligible=eligible)

    async def _require_current_warmup_eligibility(
        self,
        user_id: str,
        availability_snapshot: UserAvailabilitySnapshot,
        *,
        conversation_id: str,
        expected: _WarmupEligibility,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> None:
        connection = await self.runtime.open_connection()
        try:
            await connection.execute("BEGIN")
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            _, current = await self._capture_warmup_eligibility(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
            if current != expected:
                raise TranscriptRebuildInProgressError(
                    "Conversation eligibility changed while warmup was in progress; retry"
                )
        finally:
            await connection.close()

    async def _finalize_warmup_payload(
        self,
        payload: _ConversationActivityPayload,
    ) -> dict[str, Any]:
        published_recent_window = False
        try:
            if (
                payload.lifecycle_epoch is None
                or payload.lifecycle_cleanup_key is None
                or payload.cache_revision is None
                or payload.derivation_revision is None
                or payload.conversation_lifecycle_epoch is None
                or payload.conversation_source_revision is None
            ):
                raise RuntimeError("user lifecycle is unavailable")
            published_recent_window = await ContextCacheService(
                self.runtime
            ).publish_recent_window(
                user_id=payload.user_id,
                conversation_id=payload.conversation_id,
                messages=payload.recent_messages,
                lifecycle_epoch=payload.lifecycle_epoch,
                lifecycle_cleanup_key=payload.lifecycle_cleanup_key,
                cache_revision=payload.cache_revision,
                derivation_revision=payload.derivation_revision,
                conversation_lifecycle_epoch=(payload.conversation_lifecycle_epoch),
                conversation_source_revision=payload.conversation_source_revision,
            )
            if not published_recent_window:
                raise RuntimeError("user lifecycle is not publishable")
        except Exception:
            payload.warmup_errors.append("recent_window_failed")

        try:
            await self._require_current_warmup_identity(
                payload,
            )
        except BaseException:
            if published_recent_window:
                assert payload.lifecycle_cleanup_key is not None
                assert payload.lifecycle_epoch is not None
                assert payload.cache_revision is not None
                assert payload.derivation_revision is not None
                assert payload.conversation_lifecycle_epoch is not None
                assert payload.conversation_source_revision is not None
                await ContextCacheService(
                    self.runtime
                ).discard_recent_window_publication(
                    user_id=payload.user_id,
                    conversation_id=payload.conversation_id,
                    lifecycle_cleanup_key=payload.lifecycle_cleanup_key,
                    lifecycle_epoch=payload.lifecycle_epoch,
                    cache_revision=payload.cache_revision,
                    derivation_revision=payload.derivation_revision,
                    conversation_lifecycle_epoch=(payload.conversation_lifecycle_epoch),
                    conversation_source_revision=(payload.conversation_source_revision),
                )
            raise

        return {
            "user_id": payload.user_id,
            "conversation_id": (
                str(payload.stats["conversation_id"])
                if payload.stats is not None and "conversation_id" in payload.stats
                else payload.conversation_id
            ),
            "as_of": payload.as_of,
            "recent_window_key": payload.recent_window_key,
            "recent_message_count": len(payload.recent_messages),
            "recent_message_ids": [
                str(message["id"]) for message in payload.recent_messages
            ],
            "recent_messages": payload.recent_messages,
            "cached_context_key": payload.cache_key,
            "cached_context_available": payload.existing_cache_view is not None,
            "stats": payload.stats,
            "warmup_errors": payload.warmup_errors,
        }

    async def _capture_user_activity_sources(
        self,
        connection: Any,
        user_id: str,
        *,
        workspace_id: str | None,
        assistant_mode_id: str | None,
        namespace_filter: bool,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, ConversationLifecycleIdentity],
    ]:
        rows = await ConversationRepository(
            connection,
            self.runtime.clock,
        ).list_conversations(
            user_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        lifecycle_repository = ConversationLifecycleRepository(
            connection,
            self.runtime.clock,
        )
        sources: dict[str, ConversationLifecycleIdentity] = {}
        for row in rows:
            conversation_id = str(row["id"])
            identity = await lifecycle_repository.get_active_identity(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if identity is None:
                raise TranscriptRebuildInProgressError(
                    "Conversation set changed while activity refresh was in progress; retry"
                )
            sources[conversation_id] = identity
        return rows, sources

    async def _capture_activity_membership_sources(
        self,
        connection: Any,
        user_id: str,
        *,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        active_only: bool = False,
    ) -> dict[str, ConversationLifecycleIdentity]:
        """Stamp the complete unpaged live membership of an activity read."""

        conversation_ids = await ConversationActivityRepository(
            connection,
            self.runtime.clock,
        ).list_activity_membership_ids(
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            active_only=active_only,
        )
        lifecycle_repository = ConversationLifecycleRepository(
            connection,
            self.runtime.clock,
        )
        sources: dict[str, ConversationLifecycleIdentity] = {}
        for member_conversation_id in conversation_ids:
            identity = await lifecycle_repository.get_identity(
                user_id=user_id,
                conversation_id=member_conversation_id,
            )
            if identity is None:
                raise TranscriptRebuildInProgressError(
                    "Conversation membership changed while activity rows were read; retry"
                )
            sources[member_conversation_id] = identity
        return sources

    async def _require_current_activity_membership(
        self,
        user_id: str,
        availability_snapshot: UserAvailabilitySnapshot,
        expected_sources: dict[str, ConversationLifecycleIdentity],
        *,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        active_only: bool = False,
    ) -> None:
        connection = await self.runtime.open_connection()
        try:
            await connection.execute("BEGIN")
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            current_sources = await self._capture_activity_membership_sources(
                connection,
                user_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                active_only=active_only,
            )
            if current_sources != expected_sources:
                raise TranscriptRebuildInProgressError(
                    "Conversation membership changed while activity rows were read; retry"
                )
        finally:
            await connection.close()

    async def _require_user_activity_sources(
        self,
        connection: Any,
        user_id: str,
        expected_sources: dict[str, ConversationLifecycleIdentity],
        *,
        workspace_id: str | None,
        assistant_mode_id: str | None,
        namespace_filter: bool,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> None:
        _, current_sources = await self._capture_user_activity_sources(
            connection,
            user_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            namespace_filter=namespace_filter,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        if current_sources != expected_sources:
            raise TranscriptRebuildInProgressError(
                "Conversation sources changed while activity refresh was in progress; retry"
            )

    async def _require_current_user_activity_sources(
        self,
        user_id: str,
        availability_snapshot: UserAvailabilitySnapshot,
        expected_sources: dict[str, ConversationLifecycleIdentity],
        *,
        workspace_id: str | None,
        assistant_mode_id: str | None,
        namespace_filter: bool,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
    ) -> None:
        connection = await self.runtime.open_connection()
        try:
            await connection.execute("BEGIN")
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            await self._require_user_activity_sources(
                connection,
                user_id,
                expected_sources,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                namespace_filter=namespace_filter,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
            )
        finally:
            await connection.close()

    async def _require_current_conversation_activity_source(
        self,
        user_id: str,
        availability_snapshot: UserAvailabilitySnapshot,
        conversation_id: str,
        expected_source: ConversationLifecycleIdentity | None,
    ) -> None:
        connection = await self.runtime.open_connection()
        try:
            await connection.execute("BEGIN")
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                availability_snapshot,
            )
            current_source = await ConversationLifecycleRepository(
                connection,
                self.runtime.clock,
            ).get_identity(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if current_source != expected_source:
                raise TranscriptRebuildInProgressError(
                    "Conversation source changed while activity refresh was in progress; retry"
                )
        finally:
            await connection.close()

    async def _require_current_availability(
        self,
        user_id: str,
        snapshot: UserAvailabilitySnapshot,
    ) -> None:
        """Validate a result against a fresh SQLite view immediately before return."""

        connection = await self.runtime.open_connection()
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                user_id,
                snapshot,
            )
        finally:
            await connection.close()

    async def _require_current_warmup_identity(
        self,
        payload: _ConversationActivityPayload,
    ) -> None:
        """Validate user and conversation coordinates in one fresh SQLite view."""

        if (
            payload.lifecycle_epoch is None
            or payload.cache_revision is None
            or payload.derivation_revision is None
            or payload.lifecycle_epoch != payload.availability_snapshot.lifecycle_epoch
            or payload.derivation_revision
            != payload.availability_snapshot.derivation_revision
            or payload.conversation_lifecycle_epoch is None
            or payload.conversation_source_revision is None
        ):
            raise TranscriptRebuildInProgressError(
                "Conversation sources changed while warmup was in progress; retry"
            )
        connection = await self.runtime.open_connection()
        try:
            await connection.execute("BEGIN")
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_availability_snapshot(
                payload.user_id,
                payload.availability_snapshot,
            )
            if not await UserLifecycleRepository(
                connection,
                self.runtime.clock,
            ).matches_active_context_identity(
                payload.user_id,
                lifecycle_epoch=payload.lifecycle_epoch,
                cache_revision=payload.cache_revision,
                derivation_revision=payload.derivation_revision,
            ):
                raise TranscriptRebuildInProgressError(
                    "User cache source changed while warmup was in progress; retry"
                )
            current_conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_active_conversation(
                payload.conversation_id,
                payload.user_id,
            )
            if current_conversation is None or not self._conversation_matches_namespace(
                current_conversation,
                user_persona_id=payload.user_persona_id,
                platform_id=payload.platform_id,
                character_id=payload.character_id,
                incognito=payload.incognito,
            ):
                raise TranscriptRebuildInProgressError(
                    "Conversation eligibility changed while warmup was in progress; retry"
                )
            if not await ConversationLifecycleRepository(
                connection,
                self.runtime.clock,
            ).matches_active_identity(
                user_id=payload.user_id,
                conversation_id=payload.conversation_id,
                identity=ConversationLifecycleIdentity(
                    lifecycle_epoch=payload.conversation_lifecycle_epoch,
                    source_revision=payload.conversation_source_revision,
                ),
            ):
                raise TranscriptRebuildInProgressError(
                    "Conversation sources changed while warmup was in progress; retry"
                )
        finally:
            await connection.close()

    async def _compute_conversation_stats(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation: dict[str, Any],
        as_of: str | None,
    ) -> dict[str, Any]:
        messages = MessageRepository(connection, self.runtime.clock)
        retrieval_events = RetrievalEventRepository(connection, self.runtime.clock)
        workspace = None
        workspace_id = conversation.get("workspace_id")
        if workspace_id is not None:
            workspace = await WorkspaceRepository(
                connection, self.runtime.clock
            ).get_workspace(
                str(workspace_id),
                user_id,
            )

        stored_messages = await messages.list_messages_for_conversation(
            str(conversation["id"]),
            user_id,
        )
        stored_retrieval_events = await retrieval_events.list_events_for_conversation(
            user_id,
            str(conversation["id"]),
        )

        timezone_name = self._resolve_timezone_name(conversation, workspace)
        tzinfo = self._safe_timezone(timezone_name)
        resolved_as_of = self._resolve_as_of(as_of)

        message_times: list[datetime] = []
        active_days: dict[datetime.date, datetime] = {}
        weekday_histogram = [0 for _ in range(7)]
        hour_histogram = [0 for _ in range(24)]
        hour_of_week_histogram = [0 for _ in range(24 * 7)]

        first_message_at: datetime | None = None
        last_message_at: datetime | None = None
        last_user_message_at: datetime | None = None
        user_message_count = 0
        assistant_message_count = 0

        for message in stored_messages:
            dt = self._parse_timestamp(
                resolve_message_occurred_at(message) or message["created_at"]
            )
            if dt is None:
                continue
            message_times.append(dt)
            local_dt = dt.astimezone(tzinfo)
            local_day = local_dt.date()
            active_days[local_day] = min(active_days.get(local_day, local_dt), local_dt)
            weekday_histogram[local_dt.weekday()] += 1
            hour_histogram[local_dt.hour] += 1
            hour_of_week_histogram[local_dt.weekday() * 24 + local_dt.hour] += 1
            if first_message_at is None or dt < first_message_at:
                first_message_at = dt
            if last_message_at is None or dt > last_message_at:
                last_message_at = dt
            if str(message["role"]) == "user":
                user_message_count += 1
                if last_user_message_at is None or dt > last_user_message_at:
                    last_user_message_at = dt
            elif str(message["role"]) == "assistant":
                assistant_message_count += 1

        retrieval_count = 0
        for event in stored_retrieval_events:
            dt = self._parse_timestamp(event["created_at"])
            if dt is None:
                continue
            retrieval_count += 1
            local_dt = dt.astimezone(tzinfo)
            local_day = local_dt.date()
            active_days[local_day] = min(active_days.get(local_day, local_dt), local_dt)
            weekday_histogram[local_dt.weekday()] += 1
            hour_histogram[local_dt.hour] += 1
            hour_of_week_histogram[local_dt.weekday() * 24 + local_dt.hour] += 1
            if first_message_at is None or dt < first_message_at:
                first_message_at = dt
            if last_message_at is None or dt > last_message_at:
                last_message_at = dt

        recent_1d_message_count = self._recent_message_count(
            message_times, resolved_as_of, 1
        )
        recent_7d_message_count = self._recent_message_count(
            message_times, resolved_as_of, 7
        )
        recent_30d_message_count = self._recent_message_count(
            message_times, resolved_as_of, 30
        )

        active_day_count = len(active_days)
        return_intervals = self._return_intervals_minutes(active_days)
        interval_histogram = self._return_interval_histogram(return_intervals)
        avg_return_interval_minutes = (
            mean(return_intervals) if return_intervals else None
        )
        median_return_interval_minutes = (
            median(return_intervals) if return_intervals else None
        )
        p90_return_interval_minutes = self._percentile(return_intervals, 0.9)
        return_habit_confidence = self._return_habit_confidence(
            active_day_count=active_day_count,
            return_intervals=return_intervals,
            median_return_interval_minutes=median_return_interval_minutes,
            p90_return_interval_minutes=p90_return_interval_minutes,
            recent_30d_message_count=recent_30d_message_count,
            resolved_as_of=resolved_as_of,
            last_activity_at=last_message_at,
        )
        main_thread_score = self._main_thread_score(
            message_count=len(stored_messages),
            retrieval_count=retrieval_count,
            active_day_count=active_day_count,
            recent_30d_message_count=recent_30d_message_count,
            return_habit_confidence=return_habit_confidence,
            resolved_as_of=resolved_as_of,
            last_activity_at=last_message_at,
        )
        likely_soon_score = self._likely_soon_score(
            resolved_as_of=resolved_as_of,
            timezone_info=tzinfo,
            last_activity_at=last_message_at,
            median_return_interval_minutes=median_return_interval_minutes,
            weekday_histogram=weekday_histogram,
            hour_histogram=hour_histogram,
            hour_of_week_histogram=hour_of_week_histogram,
            recent_7d_message_count=recent_7d_message_count,
            recent_30d_message_count=recent_30d_message_count,
        )
        schedule_pattern_kind = self._schedule_pattern_kind(
            active_day_count=active_day_count,
            return_habit_confidence=return_habit_confidence,
            median_return_interval_minutes=median_return_interval_minutes,
            p90_return_interval_minutes=p90_return_interval_minutes,
            weekday_histogram=weekday_histogram,
            hour_histogram=hour_histogram,
            recent_30d_message_count=recent_30d_message_count,
        )

        return {
            "user_id": user_id,
            "conversation_id": str(conversation["id"]),
            "workspace_id": conversation.get("workspace_id"),
            "assistant_mode_id": conversation.get("assistant_mode_id"),
            "user_persona_id": conversation.get("user_persona_id"),
            "platform_id": conversation.get("platform_id"),
            "character_id": conversation.get("character_id")
            or conversation.get("workspace_id"),
            "incognito": bool(
                conversation.get("incognito") or conversation.get("isolated_mode")
            ),
            "remember_across_chats": conversation.get("remember_across_chats", True),
            "remember_across_devices": conversation.get(
                "remember_across_devices", True
            ),
            "effective_policy_hash": conversation.get("effective_policy_hash"),
            "timezone": timezone_name,
            "first_message_at": first_message_at.isoformat()
            if first_message_at is not None
            else None,
            "last_message_at": last_message_at.isoformat()
            if last_message_at is not None
            else None,
            "last_user_message_at": (
                last_user_message_at.isoformat()
                if last_user_message_at is not None
                else None
            ),
            "message_count": len(stored_messages),
            "user_message_count": user_message_count,
            "assistant_message_count": assistant_message_count,
            "retrieval_count": retrieval_count,
            "active_day_count": active_day_count,
            "recent_1d_message_count": recent_1d_message_count,
            "recent_7d_message_count": recent_7d_message_count,
            "recent_30d_message_count": recent_30d_message_count,
            "weekday_histogram_json": weekday_histogram,
            "hour_histogram_json": hour_histogram,
            "hour_of_week_histogram_json": hour_of_week_histogram,
            "return_interval_histogram_json": interval_histogram,
            "avg_return_interval_minutes": avg_return_interval_minutes,
            "median_return_interval_minutes": median_return_interval_minutes,
            "p90_return_interval_minutes": p90_return_interval_minutes,
            "main_thread_score": main_thread_score,
            "likely_soon_score": likely_soon_score,
            "return_habit_confidence": return_habit_confidence,
            "schedule_pattern_kind": schedule_pattern_kind,
            "activity_version": _ACTIVITY_VERSION,
            "updated_at": self.runtime.clock.now().isoformat(),
        }

    def _resolve_as_of(
        self,
        as_of: str | None,
        *,
        lead_time_minutes: int | None = None,
    ) -> datetime:
        normalized = normalize_optional_timestamp(as_of)
        if normalized is None:
            resolved = self.runtime.clock.now().astimezone(timezone.utc)
        else:
            resolved = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            if resolved.tzinfo is None:
                resolved = resolved.replace(tzinfo=timezone.utc)
            else:
                resolved = resolved.astimezone(timezone.utc)
        if lead_time_minutes is not None and lead_time_minutes > 0:
            resolved = resolved + timedelta(minutes=lead_time_minutes)
        return resolved.astimezone(timezone.utc)

    @staticmethod
    def _safe_timezone(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            return ZoneInfo("UTC")

    @staticmethod
    def _resolve_timezone_name(
        conversation: dict[str, Any],
        workspace: dict[str, Any] | None,
    ) -> str:
        for candidate in (conversation, workspace):
            if candidate is None:
                continue
            metadata = candidate.get("metadata_json")
            if not isinstance(metadata, dict):
                continue
            for key in ("timezone", "time_zone"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return "UTC"

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _recent_message_count(
        message_times: list[datetime],
        as_of: datetime,
        window_days: int,
    ) -> int:
        window_start = as_of - timedelta(days=window_days)
        return sum(1 for dt in message_times if window_start <= dt <= as_of)

    def _return_intervals_minutes(
        self, active_days: dict[datetime.date, datetime]
    ) -> list[float]:
        if len(active_days) < 2:
            return []
        ordered_days = sorted(active_days.values())
        intervals: list[float] = []
        for previous, current in zip(ordered_days, ordered_days[1:], strict=False):
            intervals.append((current - previous).total_seconds() / 60.0)
        return intervals

    @staticmethod
    def _return_interval_histogram(intervals: list[float]) -> list[int]:
        histogram = [0 for _ in range(len(_RETURN_INTERVAL_BUCKET_LIMITS_MINUTES) + 1)]
        for interval in intervals:
            bucket_index = len(_RETURN_INTERVAL_BUCKET_LIMITS_MINUTES)
            for index, threshold in enumerate(_RETURN_INTERVAL_BUCKET_LIMITS_MINUTES):
                if interval <= threshold:
                    bucket_index = index
                    break
            histogram[bucket_index] += 1
        return histogram

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
        return ordered[rank]

    @staticmethod
    def _return_habit_confidence(
        *,
        active_day_count: int,
        return_intervals: list[float],
        median_return_interval_minutes: float | None,
        p90_return_interval_minutes: float | None,
        recent_30d_message_count: int,
        resolved_as_of: datetime,
        last_activity_at: datetime | None,
    ) -> float:
        if not return_intervals:
            return 0.0
        coverage = min(1.0, len(return_intervals) / 8.0)
        density = min(1.0, active_day_count / 12.0)
        if (
            median_return_interval_minutes is None
            or p90_return_interval_minutes is None
        ):
            stability = 0.0
        else:
            spread = max(
                0.0, p90_return_interval_minutes - median_return_interval_minutes
            )
            stability = max(
                0.0,
                1.0 - min(1.0, spread / max(median_return_interval_minutes, 1.0)),
            )
        recency = 0.0
        if last_activity_at is not None:
            age_minutes = max(
                0.0, (resolved_as_of - last_activity_at).total_seconds() / 60.0
            )
            reference = max(median_return_interval_minutes or 60.0, 60.0)
            recency = max(0.0, 1.0 - min(1.0, age_minutes / (reference * 2.0)))
        activity_bonus = min(1.0, recent_30d_message_count / 10.0)
        return max(
            0.0,
            min(1.0, (coverage + density + stability + recency + activity_bonus) / 5.0),
        )

    @staticmethod
    def _main_thread_score(
        *,
        message_count: int,
        retrieval_count: int,
        active_day_count: int,
        recent_30d_message_count: int,
        return_habit_confidence: float,
        resolved_as_of: datetime,
        last_activity_at: datetime | None,
    ) -> float:
        volume_score = min(1.0, log1p(message_count + retrieval_count) / log1p(50.0))
        day_score = min(1.0, active_day_count / 14.0)
        recent_score = min(1.0, recent_30d_message_count / 12.0)
        recency_score = 0.0
        if last_activity_at is not None:
            age_days = max(
                0.0, (resolved_as_of - last_activity_at).total_seconds() / 86400.0
            )
            recency_score = max(0.0, 1.0 - min(1.0, age_days / 30.0))
        score = (
            (0.3 * volume_score)
            + (0.2 * day_score)
            + (0.2 * recent_score)
            + (0.15 * return_habit_confidence)
            + (0.15 * recency_score)
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _likely_soon_score(
        *,
        resolved_as_of: datetime,
        timezone_info: ZoneInfo,
        last_activity_at: datetime | None,
        median_return_interval_minutes: float | None,
        weekday_histogram: list[int],
        hour_histogram: list[int],
        hour_of_week_histogram: list[int],
        recent_7d_message_count: int,
        recent_30d_message_count: int,
    ) -> float:
        if last_activity_at is None:
            return 0.0
        age_minutes = max(
            0.0, (resolved_as_of - last_activity_at).total_seconds() / 60.0
        )
        reference = max(median_return_interval_minutes or 24.0 * 60.0, 60.0)
        recency_component = max(0.0, 1.0 - min(1.0, age_minutes / (reference * 1.5)))

        local_now = resolved_as_of.astimezone(timezone_info)
        slot_index = local_now.weekday() * 24 + local_now.hour
        max_weekday = max(weekday_histogram) if weekday_histogram else 0
        max_hour = max(hour_histogram) if hour_histogram else 0
        max_slot = max(hour_of_week_histogram) if hour_of_week_histogram else 0
        slot_alignment = 0.0
        if max_slot > 0:
            slot_alignment = hour_of_week_histogram[slot_index] / max_slot
        weekday_alignment = 0.0
        if max_weekday > 0:
            weekday_alignment = weekday_histogram[local_now.weekday()] / max_weekday
        hour_alignment = 0.0
        if max_hour > 0:
            hour_alignment = hour_histogram[local_now.hour] / max_hour

        recent_bias = min(
            1.0, (recent_7d_message_count + recent_30d_message_count) / 20.0
        )
        score = (
            (0.5 * recency_component)
            + (0.25 * slot_alignment)
            + (0.15 * weekday_alignment)
            + (0.1 * hour_alignment)
        )
        score = score + (0.05 * recent_bias)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _schedule_pattern_kind(
        *,
        active_day_count: int,
        return_habit_confidence: float,
        median_return_interval_minutes: float | None,
        p90_return_interval_minutes: float | None,
        weekday_histogram: list[int],
        hour_histogram: list[int],
        recent_30d_message_count: int,
    ) -> str:
        total_events = sum(weekday_histogram)
        if total_events == 0:
            return "inactive"
        if active_day_count <= 1:
            return "single_day"
        if return_habit_confidence < 0.2:
            return "irregular"
        weekend_share = 0.0
        if total_events > 0:
            weekend_share = (weekday_histogram[5] + weekday_histogram[6]) / total_events
        if weekend_share <= 0.2 and active_day_count >= 4:
            return "workweek"
        if (
            median_return_interval_minutes is not None
            and median_return_interval_minutes <= 36.0 * 60.0
        ):
            return "daily"
        if (
            median_return_interval_minutes is not None
            and median_return_interval_minutes <= 9.0 * 24.0 * 60.0
        ):
            return "weekly"
        if (
            median_return_interval_minutes is not None
            and median_return_interval_minutes <= 18.0 * 24.0 * 60.0
        ):
            return "biweekly"
        if recent_30d_message_count >= 8 and max(hour_histogram or [0]) >= 3:
            return "bursty"
        if (
            p90_return_interval_minutes is not None
            and p90_return_interval_minutes > 30.0 * 24.0 * 60.0
        ):
            return "sparse"
        return "irregular"

    @staticmethod
    def _bounded_int(
        value: int,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            resolved = default
        return max(minimum, min(maximum, resolved))

    @staticmethod
    def _recent_window_key(user_id: str, conversation_id: str) -> str:
        return build_recent_window_key(user_id, conversation_id)
