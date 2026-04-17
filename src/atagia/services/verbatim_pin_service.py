"""Service layer for user-controlled verbatim pins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.models.schemas_memory import MemoryScope, VerbatimPinStatus, VerbatimPinTargetKind

if TYPE_CHECKING:
    from atagia.app import AppRuntime


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


class VerbatimPinService:
    """Resolve pin snapshots and persist verbatim pin lifecycle changes."""

    def __init__(self, runtime: AppRuntime) -> None:
        self.runtime = runtime

    async def create_verbatim_pin(
        self,
        connection: Any,
        *,
        user_id: str,
        scope: MemoryScope,
        target_kind: VerbatimPinTargetKind,
        target_id: str,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int = 0,
        reason: str | None = None,
        created_by: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_target_id = _normalize_text(target_id)
        if resolved_target_id is None:
            raise ValueError("target_id must be provided")

        source_row = await self._load_source_row(
            connection,
            user_id=user_id,
            target_kind=target_kind,
            target_id=resolved_target_id,
        )
        scope_anchors = await self._resolve_scope_anchors(
            connection,
            user_id=user_id,
            scope=scope,
            source_row=source_row,
            payload_json=payload_json,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
        )
        source_text = self._source_text_for_kind(source_row, target_kind)

        if canonical_text is not None:
            resolved_canonical_text = _normalize_text(canonical_text)
        elif source_text is not None:
            resolved_canonical_text = source_text
        else:
            raise ValueError("canonical_text is required when the pin target cannot be resolved")
        if resolved_canonical_text is None:
            raise ValueError("canonical_text must be non-empty")

        if target_span_start is not None or target_span_end is not None:
            if source_text is None:
                raise ValueError("target spans require a resolvable source text")
            span_start = target_span_start or 0
            span_end = target_span_end if target_span_end is not None else len(source_text)
            if span_end < span_start:
                raise ValueError("target_span_end must be greater than or equal to target_span_start")
            if span_start < 0 or span_end < 0:
                raise ValueError("target span offsets must be non-negative")
            if span_end > len(source_text):
                raise ValueError("target span exceeds the resolved source text length")
            resolved_canonical_text = source_text[span_start:span_end]
        else:
            span_start = target_span_start
            span_end = target_span_end

        resolved_created_by = _normalize_text(created_by) or user_id
        resolved_payload = dict(payload_json or {})
        resolved_payload.setdefault("source_target_kind", target_kind.value)
        resolved_payload.setdefault("source_target_id", resolved_target_id)
        if source_row is not None:
            resolved_payload.setdefault(
                "source_snapshot",
                self._source_snapshot_metadata(source_row, target_kind, resolved_target_id),
            )
        if span_start is not None:
            resolved_payload.setdefault("target_span_start", span_start)
        if span_end is not None:
            resolved_payload.setdefault("target_span_end", span_end)

        resolved_index_text = _normalize_text(index_text)
        if resolved_index_text is None:
            safe_label = _normalize_text(resolved_payload.get("safe_index_text"))
            if privacy_level >= 2:
                resolved_index_text = safe_label or f"{target_kind.value} pin"
            else:
                resolved_index_text = resolved_canonical_text

        repository = VerbatimPinRepository(connection, self.runtime.clock)
        await connection.execute("BEGIN")
        try:
            created = await repository.create_verbatim_pin(
                user_id=user_id,
                scope=scope,
                target_kind=target_kind,
                target_id=resolved_target_id,
                workspace_id=scope_anchors["workspace_id"],
                conversation_id=scope_anchors["conversation_id"],
                assistant_mode_id=scope_anchors["assistant_mode_id"],
                canonical_text=resolved_canonical_text,
                index_text=resolved_index_text,
                privacy_level=privacy_level,
                created_by=resolved_created_by,
                reason=reason,
                target_span_start=span_start,
                target_span_end=span_end,
                expires_at=expires_at,
                payload_json=resolved_payload,
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return created

    async def get_verbatim_pin(
        self,
        connection: Any,
        *,
        user_id: str,
        pin_id: str,
    ) -> dict[str, Any] | None:
        return await VerbatimPinRepository(connection, self.runtime.clock).get_verbatim_pin(pin_id, user_id)

    async def list_verbatim_pins(
        self,
        connection: Any,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        scope_filter: list[MemoryScope] | None = None,
        target_kind_filter: list[VerbatimPinTargetKind] | None = None,
        status_filter: list[VerbatimPinStatus] | None = None,
        target_id: str | None = None,
        include_deleted: bool = False,
        active_only: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        return await VerbatimPinRepository(connection, self.runtime.clock).list_verbatim_pins(
            user_id,
            limit=limit,
            offset=offset,
            scope_filter=scope_filter,
            target_kind_filter=target_kind_filter,
            status_filter=status_filter,
            target_id=target_id,
            include_deleted=include_deleted,
            active_only=active_only,
            as_of=as_of,
        )

    async def update_verbatim_pin(
        self,
        connection: Any,
        *,
        user_id: str,
        pin_id: str,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int | None = None,
        status: VerbatimPinStatus | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        repository = VerbatimPinRepository(connection, self.runtime.clock)
        await connection.execute("BEGIN")
        try:
            updated = await repository.update_verbatim_pin(
                pin_id,
                user_id,
                canonical_text=canonical_text,
                index_text=index_text,
                target_span_start=target_span_start,
                target_span_end=target_span_end,
                privacy_level=privacy_level,
                status=status,
                reason=reason,
                expires_at=expires_at,
                payload_json=payload_json,
                commit=False,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return updated

    async def delete_verbatim_pin(
        self,
        connection: Any,
        *,
        user_id: str,
        pin_id: str,
    ) -> dict[str, Any] | None:
        repository = VerbatimPinRepository(connection, self.runtime.clock)
        await connection.execute("BEGIN")
        try:
            deleted = await repository.delete_verbatim_pin(pin_id, user_id, commit=False)
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return deleted

    async def search_active_verbatim_pins(
        self,
        connection: Any,
        *,
        user_id: str,
        query: str,
        privacy_ceiling: int,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        limit: int,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        return await VerbatimPinRepository(connection, self.runtime.clock).search_active_verbatim_pins(
            user_id=user_id,
            query=query,
            privacy_ceiling=privacy_ceiling,
            scope_filter=scope_filter,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            limit=limit,
            as_of=as_of,
        )

    async def _load_source_row(
        self,
        connection: Any,
        *,
        user_id: str,
        target_kind: VerbatimPinTargetKind,
        target_id: str,
    ) -> dict[str, Any] | None:
        messages = MessageRepository(connection, self.runtime.clock)
        memories = MemoryObjectRepository(connection, self.runtime.clock)

        if target_kind is VerbatimPinTargetKind.MESSAGE:
            return await messages.get_message(target_id, user_id)
        if target_kind is VerbatimPinTargetKind.MEMORY_OBJECT:
            return await memories.get_memory_object(target_id, user_id)
        message = await messages.get_message(target_id, user_id)
        if message is not None:
            return message
        return await memories.get_memory_object(target_id, user_id)

    async def _resolve_scope_anchors(
        self,
        connection: Any,
        *,
        user_id: str,
        scope: MemoryScope,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
    ) -> dict[str, str | None]:
        payload = payload_json or {}
        resolved_workspace_id = _normalize_text(workspace_id or payload.get("workspace_id"))
        resolved_conversation_id = _normalize_text(conversation_id or payload.get("conversation_id"))
        resolved_assistant_mode_id = _normalize_text(
            assistant_mode_id or payload.get("assistant_mode_id")
        )

        if source_row is not None:
            resolved_workspace_id = _normalize_text(source_row.get("workspace_id")) or resolved_workspace_id
            resolved_conversation_id = (
                _normalize_text(source_row.get("conversation_id"))
                or resolved_conversation_id
            )
            resolved_assistant_mode_id = (
                _normalize_text(source_row.get("assistant_mode_id"))
                or resolved_assistant_mode_id
            )

        if resolved_conversation_id is not None and (
            resolved_workspace_id is None or resolved_assistant_mode_id is None
        ):
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(resolved_conversation_id, user_id)
            if conversation is not None:
                resolved_workspace_id = (
                    _normalize_text(conversation.get("workspace_id")) or resolved_workspace_id
                )
                resolved_assistant_mode_id = (
                    _normalize_text(conversation.get("assistant_mode_id"))
                    or resolved_assistant_mode_id
                )

        if scope is MemoryScope.ASSISTANT_MODE and resolved_assistant_mode_id is None:
            raise ValueError("assistant_mode_id is required for assistant_mode pins")
        if scope is MemoryScope.WORKSPACE and (
            resolved_workspace_id is None or resolved_assistant_mode_id is None
        ):
            raise ValueError("workspace_id and assistant_mode_id are required for workspace pins")
        if (
            scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION}
            and (resolved_conversation_id is None or resolved_assistant_mode_id is None)
        ):
            raise ValueError(
                "conversation_id and assistant_mode_id are required for conversation-scoped pins"
            )

        return {
            "workspace_id": resolved_workspace_id,
            "conversation_id": resolved_conversation_id,
            "assistant_mode_id": resolved_assistant_mode_id,
        }

    @staticmethod
    def _source_text_for_kind(
        source_row: dict[str, Any] | None,
        target_kind: VerbatimPinTargetKind,
    ) -> str | None:
        if source_row is None:
            return None
        if target_kind is VerbatimPinTargetKind.MESSAGE:
            return _normalize_text(source_row.get("text"))
        if target_kind is VerbatimPinTargetKind.MEMORY_OBJECT:
            return _normalize_text(source_row.get("canonical_text"))
        return _normalize_text(source_row.get("text") or source_row.get("canonical_text"))

    @staticmethod
    def _source_snapshot_metadata(
        source_row: dict[str, Any],
        target_kind: VerbatimPinTargetKind,
        target_id: str,
    ) -> dict[str, Any]:
        if target_kind is VerbatimPinTargetKind.MESSAGE:
            return {
                "message_id": str(source_row.get("id") or target_id),
                "conversation_id": source_row.get("conversation_id"),
                "role": source_row.get("role"),
                "seq": source_row.get("seq"),
                "occurred_at": source_row.get("occurred_at"),
            }
        if target_kind is VerbatimPinTargetKind.MEMORY_OBJECT:
            return {
                "memory_id": str(source_row.get("id") or target_id),
                "object_type": source_row.get("object_type"),
                "scope": source_row.get("scope"),
                "assistant_mode_id": source_row.get("assistant_mode_id"),
                "workspace_id": source_row.get("workspace_id"),
                "conversation_id": source_row.get("conversation_id"),
            }
        return {
            "source_id": str(source_row.get("id") or target_id),
            "source_kind": "text_span",
        }
