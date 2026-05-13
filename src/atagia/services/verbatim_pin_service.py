"""Service layer for user-controlled verbatim pins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atagia.core.repositories import ConversationRepository, MemoryObjectRepository
from atagia.core.space_repository import SpaceRepository, space_snapshot
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.intimacy_boundary_policy import (
    constrained_scope_for_intimacy_boundary,
    is_blocked_intimacy_boundary,
    is_restricted_intimacy_boundary,
    minimum_privacy_for_intimacy_boundary,
    normalize_intimacy_boundary,
)
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryScope,
    MemorySensitivity,
    MindTopology,
    SpaceBoundaryMode,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)

if TYPE_CHECKING:
    from atagia.app import AppRuntime


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _normalize_space_boundary_mode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SpaceBoundaryMode):
        return value.value
    try:
        return SpaceBoundaryMode(str(value)).value
    except ValueError:
        return SpaceBoundaryMode.FOCUS.value


def _normalize_mind_topology(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, MindTopology):
        return value.value
    try:
        return MindTopology(str(value)).value
    except ValueError:
        return MindTopology.UNIMIND.value


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
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        canonical_text: str | None = None,
        index_text: str | None = None,
        target_span_start: int | None = None,
        target_span_end: int | None = None,
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        reason: str | None = None,
        created_by: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_target_id = _normalize_text(target_id)
        if resolved_target_id is None:
            raise ValueError("target_id must be provided")
        active_space_context = await self._active_space_context(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
        )
        active_mind_context = await self._active_mind_context(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
        )
        active_embodiment_context = await self._active_embodiment_context(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            active_embodiment_id=active_embodiment_id,
        )
        active_realm_context = await self._active_realm_context(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            active_realm_id=active_realm_id,
        )

        namespace_payload = self._payload_with_namespace(
            payload_json,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_mind_id=active_mind_context["active_mind_id"],
            mind_topology=active_mind_context["mind_topology"],
            active_embodiment_id=active_embodiment_context["active_embodiment_id"],
            active_realm_id=active_realm_context["active_realm_id"],
        )
        source_row = await self._load_source_row(
            connection,
            user_id=user_id,
            target_kind=target_kind,
            target_id=resolved_target_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_context["space_id"],
            active_space_boundary_mode=active_space_context["space_boundary_mode"],
            active_mind_id=active_mind_context["active_mind_id"],
            mind_topology=active_mind_context["mind_topology"],
            active_embodiment_id=active_embodiment_context["active_embodiment_id"],
            active_realm_id=active_realm_context["active_realm_id"],
        )
        resolved_intimacy_boundary = self._resolve_intimacy_boundary(
            explicit_boundary=intimacy_boundary,
            payload_json=namespace_payload,
            source_row=source_row,
        )
        if is_blocked_intimacy_boundary(resolved_intimacy_boundary):
            raise ValueError("safety_blocked verbatim pins cannot be created")
        resolved_intimacy_boundary_confidence = self._resolve_intimacy_boundary_confidence(
            explicit_confidence=intimacy_boundary_confidence,
            resolved_boundary=resolved_intimacy_boundary,
            payload_json=namespace_payload,
            source_row=source_row,
        )
        scope = constrained_scope_for_intimacy_boundary(
            resolved_intimacy_boundary,
            scope=scope,
        )
        privacy_level = minimum_privacy_for_intimacy_boundary(
            resolved_intimacy_boundary,
            privacy_level=privacy_level,
        )
        scope_anchors = await self._resolve_scope_anchors(
            connection,
            user_id=user_id,
            scope=scope,
            source_row=source_row,
            payload_json=namespace_payload,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
        )
        source_namespace = self._source_namespace_snapshot(
            source_row=source_row,
            payload_json=namespace_payload,
            scope_anchors=scope_anchors,
            scope=scope,
        )
        source_space = await self._source_space_snapshot(
            connection,
            user_id=user_id,
            source_row=source_row,
            payload_json=namespace_payload,
            active_space_id=active_space_context["space_id"],
            active_space_boundary_mode=active_space_context["space_boundary_mode"],
            conversation_id=scope_anchors["conversation_id"],
        )
        source_mind = self._source_mind_snapshot(
            source_row=source_row,
            payload_json=namespace_payload,
            active_mind_id=active_mind_context["active_mind_id"],
            mind_topology=active_mind_context["mind_topology"],
        )
        source_embodiment = self._source_embodiment_snapshot(
            source_row=source_row,
            payload_json=namespace_payload,
            active_embodiment_id=active_embodiment_context["active_embodiment_id"],
        )
        source_realm = self._source_realm_snapshot(
            source_row=source_row,
            payload_json=namespace_payload,
            active_realm_id=active_realm_context["active_realm_id"],
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
        resolved_payload = dict(namespace_payload)
        resolved_payload.setdefault("source_target_kind", target_kind.value)
        resolved_payload.setdefault("source_target_id", resolved_target_id)
        resolved_payload["intimacy_boundary"] = resolved_intimacy_boundary.value
        resolved_payload["intimacy_boundary_confidence"] = resolved_intimacy_boundary_confidence
        if is_restricted_intimacy_boundary(resolved_intimacy_boundary):
            resolved_payload.setdefault(
                "intimacy_boundary_policy",
                {"requires_explicit_intimacy_context": True},
            )
        if source_row is not None:
            resolved_payload.setdefault(
                "source_snapshot",
                self._source_snapshot_metadata(source_row, target_kind, resolved_target_id),
            )
        if source_space["space_id"] is not None:
            space_payload = {
                "active_space_id": source_space["space_id"],
                "boundary_mode": source_space["space_boundary_mode"],
            }
            if source_space.get("display_name") is not None:
                space_payload["display_name"] = source_space["display_name"]
            resolved_payload["space_boundary"] = space_payload
        if source_mind["memory_owner_id"] is not None:
            resolved_payload["mind_perspective"] = {
                "memory_owner_id": source_mind["memory_owner_id"],
                "source_mind_id": source_mind["source_mind_id"],
                "mind_topology": source_mind["mind_topology"],
            }
        if source_embodiment["embodiment_id"] is not None:
            resolved_payload["embodiment"] = {
                "active_embodiment_id": source_embodiment["embodiment_id"],
            }
        if source_realm["realm_id"] is not None:
            resolved_payload["realm"] = {
                "active_realm_id": source_realm["realm_id"],
            }
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
                intimacy_boundary=resolved_intimacy_boundary,
                intimacy_boundary_confidence=resolved_intimacy_boundary_confidence,
                created_by=resolved_created_by,
                reason=reason,
                target_span_start=span_start,
                target_span_end=span_end,
                expires_at=expires_at,
                payload_json=resolved_payload,
                user_persona_id=source_namespace["user_persona_id"],
                platform_id=source_namespace["platform_id"],
                character_id=source_namespace["character_id"],
                sensitivity=source_namespace["sensitivity"],
                themes=source_namespace["themes"],
                platform_locked=bool(source_namespace["platform_locked"]),
                platform_id_lock=source_namespace["platform_id_lock"],
                scope_canonical=source_namespace["scope_canonical"],
                incognito_snapshot=bool(source_namespace["incognito_snapshot"]),
                remember_across_chats_snapshot=bool(
                    source_namespace["remember_across_chats_snapshot"]
                ),
                remember_across_devices_snapshot=bool(
                    source_namespace["remember_across_devices_snapshot"]
                ),
                policy_snapshot=source_namespace["policy_snapshot"],
                space_id=source_space["space_id"],
                space_boundary_mode=source_space["space_boundary_mode"],
                memory_owner_id=source_mind["memory_owner_id"],
                source_mind_id=source_mind["source_mind_id"],
                embodiment_id=source_embodiment["embodiment_id"],
                realm_id=source_realm["realm_id"],
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
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        return await VerbatimPinRepository(connection, self.runtime.clock).get_verbatim_pin(
            pin_id,
            user_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )

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
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
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
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
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
        intimacy_boundary: IntimacyBoundary | None = None,
        intimacy_boundary_confidence: float | None = None,
        status: VerbatimPinStatus | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
        payload_json: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
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
                intimacy_boundary=intimacy_boundary,
                intimacy_boundary_confidence=intimacy_boundary_confidence,
                status=status,
                reason=reason,
                expires_at=expires_at,
                payload_json=payload_json,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                active_space_id=active_space_id,
                active_space_boundary_mode=active_space_boundary_mode,
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
                active_embodiment_id=active_embodiment_id,
                active_realm_id=active_realm_id,
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
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        repository = VerbatimPinRepository(connection, self.runtime.clock)
        await connection.execute("BEGIN")
        try:
            deleted = await repository.delete_verbatim_pin(
                pin_id,
                user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                active_space_id=active_space_id,
                active_space_boundary_mode=active_space_boundary_mode,
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
                active_embodiment_id=active_embodiment_id,
                active_realm_id=active_realm_id,
                commit=False,
            )
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
        allow_intimacy_context: bool = False,
        as_of: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
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
            allow_intimacy_context=allow_intimacy_context,
            as_of=as_of,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )

    @staticmethod
    def _resolve_intimacy_boundary(
        *,
        explicit_boundary: IntimacyBoundary,
        payload_json: dict[str, Any] | None,
        source_row: dict[str, Any] | None,
    ) -> IntimacyBoundary:
        if explicit_boundary is not IntimacyBoundary.ORDINARY:
            return explicit_boundary
        if source_row is not None and source_row.get("intimacy_boundary") is not None:
            return normalize_intimacy_boundary(source_row.get("intimacy_boundary"))
        payload = payload_json or {}
        if payload.get("intimacy_boundary") is not None:
            return normalize_intimacy_boundary(payload.get("intimacy_boundary"))
        return IntimacyBoundary.ORDINARY

    @staticmethod
    def _resolve_intimacy_boundary_confidence(
        *,
        explicit_confidence: float,
        resolved_boundary: IntimacyBoundary,
        payload_json: dict[str, Any] | None,
        source_row: dict[str, Any] | None,
    ) -> float:
        if explicit_confidence:
            return float(explicit_confidence)
        if (
            source_row is not None
            and normalize_intimacy_boundary(source_row.get("intimacy_boundary")) is resolved_boundary
        ):
            return float(source_row.get("intimacy_boundary_confidence", 0.0) or 0.0)
        payload = payload_json or {}
        if normalize_intimacy_boundary(payload.get("intimacy_boundary")) is resolved_boundary:
            return float(payload.get("intimacy_boundary_confidence", 0.0) or 0.0)
        return 0.0

    async def _load_source_row(
        self,
        connection: Any,
        *,
        user_id: str,
        target_kind: VerbatimPinTargetKind,
        target_id: str,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        memories = MemoryObjectRepository(connection, self.runtime.clock)

        if target_kind is VerbatimPinTargetKind.MESSAGE:
            return await self._load_message_source_row(
                connection,
                user_id=user_id,
                message_id=target_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
                active_space_id=active_space_id,
                active_space_boundary_mode=active_space_boundary_mode,
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
                active_embodiment_id=active_embodiment_id,
                active_realm_id=active_realm_id,
            )
        if target_kind is VerbatimPinTargetKind.MEMORY_OBJECT:
            if platform_id is not None and conversation_id is not None:
                return await memories.get_visible_memory_object(
                    target_id,
                    user_id,
                    conversation_id=conversation_id,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=bool(incognito),
                    remember_across_chats=remember_across_chats,
                    remember_across_devices=remember_across_devices,
                    sensitivity_gates_enabled=True,
                    active_space_id=active_space_id,
                    active_space_boundary_mode=active_space_boundary_mode,
                    active_mind_id=active_mind_id,
                    mind_topology=mind_topology,
                    active_embodiment_id=active_embodiment_id,
                    active_realm_id=active_realm_id,
                )
            return await memories.get_memory_object(target_id, user_id)
        message = await self._load_message_source_row(
            connection,
            user_id=user_id,
            message_id=target_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
        if message is not None:
            return message
        if platform_id is not None and conversation_id is not None:
            return await memories.get_visible_memory_object(
                target_id,
                user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=bool(incognito),
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                sensitivity_gates_enabled=True,
                active_space_id=active_space_id,
                active_space_boundary_mode=active_space_boundary_mode,
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
                active_embodiment_id=active_embodiment_id,
                active_realm_id=active_realm_id,
            )
        return await memories.get_memory_object(target_id, user_id)

    async def _load_message_source_row(
        self,
        connection: Any,
        *,
        user_id: str,
        message_id: str,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool | None,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["m.id = ?", "c.user_id = ?"]
        parameters: list[Any] = [message_id, user_id]
        if platform_id is not None and conversation_id is not None:
            clauses.extend(
                [
                    "c.id = ?",
                    "c.platform_id = ?",
                    "c.user_persona_id IS ?",
                    "c.character_id IS ?",
                ]
            )
            parameters.extend([conversation_id, platform_id, user_persona_id, character_id])
            if incognito is not None:
                clauses.append("c.incognito = ?")
                parameters.append(1 if incognito else 0)
        space_clause, space_parameters = self._message_space_visibility_clause(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
        )
        clauses.append(space_clause)
        parameters.extend(space_parameters)
        mind_clause, mind_parameters = self._message_mind_visibility_clause(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
        )
        clauses.append(mind_clause)
        parameters.extend(mind_parameters)
        embodiment_clause, embodiment_parameters = self._message_embodiment_visibility_clause(
            active_embodiment_id=active_embodiment_id,
        )
        clauses.append(embodiment_clause)
        parameters.extend(embodiment_parameters)
        realm_clause, realm_parameters = self._message_realm_visibility_clause(
            active_realm_id=active_realm_id,
        )
        clauses.append(realm_clause)
        parameters.extend(realm_parameters)
        cursor = await connection.execute(
            """
            SELECT
                m.*,
                c.workspace_id AS workspace_id,
                c.assistant_mode_id AS assistant_mode_id,
                c.user_persona_id AS user_persona_id,
                c.platform_id AS platform_id,
                c.character_id AS character_id,
                c.mode AS mode,
                c.incognito AS incognito_snapshot,
                sp.boundary_mode AS space_boundary_mode,
                sp.display_name AS space_display_name,
                m.active_embodiment_id AS embodiment_id,
                m.active_realm_id AS realm_id
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            LEFT JOIN spaces AS sp
              ON sp.owner_user_id = c.user_id
             AND sp.id = m.space_id
            WHERE {where_clause}
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _payload_with_namespace(
        payload_json: dict[str, Any] | None,
        *,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool | None,
        remember_across_chats: bool,
        remember_across_devices: bool,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload_json or {})
        if user_persona_id is not None:
            payload["user_persona_id"] = user_persona_id
        if platform_id is not None:
            payload["platform_id"] = platform_id
        if character_id is not None:
            payload["character_id"] = character_id
        if (
            incognito is not None
            or user_persona_id is not None
            or platform_id is not None
            or character_id is not None
            or active_mind_id is not None
            or active_embodiment_id is not None
            or active_realm_id is not None
        ):
            raw_policy = payload.get("source_turn_policy") or payload.get("policy_snapshot_json")
            policy = dict(raw_policy) if isinstance(raw_policy, dict) else {}
            if incognito is not None:
                policy["incognito"] = bool(incognito)
            policy["remember_across_chats"] = bool(remember_across_chats)
            policy["remember_across_devices"] = bool(remember_across_devices)
            if active_mind_id is not None:
                policy["active_mind_id"] = active_mind_id
                policy["source_mind_id"] = active_mind_id
                policy["mind_topology"] = mind_topology or MindTopology.UNIMIND.value
            if active_embodiment_id is not None:
                policy["active_embodiment_id"] = active_embodiment_id
                policy["cross_embodiment_mode"] = "direct_if_same_body"
            if active_realm_id is not None:
                policy["active_realm_id"] = active_realm_id
                policy["cross_realm_mode"] = "none"
            payload["source_turn_policy"] = policy
        return payload

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
    def _source_namespace_snapshot(
        *,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        scope_anchors: dict[str, str | None],
        scope: MemoryScope,
    ) -> dict[str, Any]:
        payload = payload_json or {}
        source_policy = VerbatimPinService._source_policy_snapshot(source_row, payload)
        user_persona_id = VerbatimPinService._source_value(
            source_row,
            payload,
            "user_persona_id",
            "user_persona_id_snapshot",
        )
        platform_id = (
            VerbatimPinService._source_value(source_row, payload, "platform_id", "platform_id_snapshot")
            or "default"
        )
        character_id = (
            VerbatimPinService._source_value(source_row, payload, "character_id", "character_id_snapshot")
            or scope_anchors.get("workspace_id")
        )
        sensitivity = VerbatimPinService._source_sensitivity(source_row, payload)
        raw_themes = VerbatimPinService._source_value(source_row, payload, "themes_json", "themes_json")
        themes = raw_themes if isinstance(raw_themes, list) else []
        platform_locked = bool(
            VerbatimPinService._source_value(
                source_row,
                payload,
                "platform_locked",
                "platform_locked",
            )
            or source_policy.get("platform_locked")
        )
        platform_id_lock = (
            VerbatimPinService._source_value(
                source_row,
                payload,
                "platform_id_lock",
                "platform_id_lock",
            )
            or source_policy.get("platform_id_lock")
        )
        return {
            "user_persona_id": user_persona_id,
            "platform_id": platform_id,
            "character_id": character_id,
            "sensitivity": sensitivity,
            "themes": themes,
            "platform_locked": platform_locked,
            "platform_id_lock": platform_id_lock,
            "scope_canonical": VerbatimPinService._canonical_pin_scope(scope),
            "incognito_snapshot": bool(source_policy.get("incognito")),
            "remember_across_chats_snapshot": source_policy.get("remember_across_chats", True) is not False,
            "remember_across_devices_snapshot": source_policy.get("remember_across_devices", True) is not False,
            "policy_snapshot": source_policy,
        }

    async def _active_space_context(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str | None,
        active_space_id: str | None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None,
    ) -> dict[str, str | None]:
        resolved_space_id = _normalize_text(active_space_id)
        resolved_boundary_mode = _normalize_space_boundary_mode(active_space_boundary_mode)

        if resolved_space_id is None and conversation_id is not None:
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            if conversation is not None:
                resolved_space_id = _normalize_text(conversation.get("active_space_id"))

        if resolved_space_id is not None and resolved_boundary_mode is None:
            space_row = await SpaceRepository(connection, self.runtime.clock).get_space(
                owner_user_id=user_id,
                space_id=resolved_space_id,
            )
            if space_row is not None:
                resolved_boundary_mode = space_snapshot(space_row).boundary_mode.value

        return {
            "space_id": resolved_space_id,
            "space_boundary_mode": resolved_boundary_mode,
        }

    async def _source_space_snapshot(
        self,
        connection: Any,
        *,
        user_id: str,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        active_space_id: str | None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None,
        conversation_id: str | None,
    ) -> dict[str, str | None]:
        payload = payload_json or {}
        payload_boundary = payload.get("space_boundary")
        if not isinstance(payload_boundary, dict):
            payload_boundary = {}

        resolved_space_id = (
            _normalize_text(self._source_value(source_row, payload, "space_id", "space_id"))
            or _normalize_text(payload_boundary.get("active_space_id"))
            or _normalize_text(payload_boundary.get("space_id"))
            or _normalize_text(active_space_id)
        )
        resolved_boundary_mode = (
            _normalize_space_boundary_mode(
                self._source_value(
                    source_row,
                    payload,
                    "space_boundary_mode",
                    "space_boundary_mode",
                )
            )
            or _normalize_space_boundary_mode(payload_boundary.get("boundary_mode"))
            or _normalize_space_boundary_mode(active_space_boundary_mode)
        )
        resolved_display_name = (
            _normalize_text(
                self._source_value(source_row, payload, "space_display_name", "space_display_name")
            )
            or _normalize_text(payload_boundary.get("display_name"))
        )

        if resolved_space_id is None and conversation_id is not None:
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            if conversation is not None:
                resolved_space_id = _normalize_text(conversation.get("active_space_id"))

        if resolved_space_id is not None and (
            resolved_boundary_mode is None or resolved_display_name is None
        ):
            space_row = await SpaceRepository(connection, self.runtime.clock).get_space(
                owner_user_id=user_id,
                space_id=resolved_space_id,
            )
            if space_row is not None:
                snapshot = space_snapshot(space_row)
                resolved_boundary_mode = resolved_boundary_mode or snapshot.boundary_mode.value
                resolved_display_name = resolved_display_name or snapshot.display_name

        if resolved_space_id is None:
            return {"space_id": None, "space_boundary_mode": None, "display_name": None}
        return {
            "space_id": resolved_space_id,
            "space_boundary_mode": resolved_boundary_mode or SpaceBoundaryMode.FOCUS.value,
            "display_name": resolved_display_name,
        }

    async def _active_mind_context(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str | None,
        active_mind_id: str | None,
        mind_topology: MindTopology | str | None,
    ) -> dict[str, str | None]:
        resolved_mind_id = _normalize_text(active_mind_id)
        resolved_topology = _normalize_mind_topology(mind_topology)

        if conversation_id is not None and (resolved_mind_id is None or resolved_topology is None):
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            if conversation is not None:
                resolved_mind_id = resolved_mind_id or _normalize_text(
                    conversation.get("active_mind_id")
                )
                resolved_topology = resolved_topology or _normalize_mind_topology(
                    conversation.get("mind_topology")
                )

        return {
            "active_mind_id": resolved_mind_id,
            "mind_topology": resolved_topology or MindTopology.UNIMIND.value,
        }

    async def _active_embodiment_context(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str | None,
        active_embodiment_id: str | None,
    ) -> dict[str, str | None]:
        resolved_embodiment_id = _normalize_text(active_embodiment_id)
        if resolved_embodiment_id is None and conversation_id is not None:
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            if conversation is not None:
                resolved_embodiment_id = _normalize_text(
                    conversation.get("active_embodiment_id")
                )
        return {"active_embodiment_id": resolved_embodiment_id}

    async def _active_realm_context(
        self,
        connection: Any,
        *,
        user_id: str,
        conversation_id: str | None,
        active_realm_id: str | None,
    ) -> dict[str, str | None]:
        resolved_realm_id = _normalize_text(active_realm_id)
        if resolved_realm_id is None and conversation_id is not None:
            conversation = await ConversationRepository(
                connection,
                self.runtime.clock,
            ).get_conversation(conversation_id, user_id)
            if conversation is not None:
                resolved_realm_id = _normalize_text(
                    conversation.get("active_realm_id")
                )
        return {"active_realm_id": resolved_realm_id}

    @staticmethod
    def _source_mind_snapshot(
        *,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        active_mind_id: str | None,
        mind_topology: str | None,
    ) -> dict[str, str | None]:
        payload = payload_json or {}
        mind_payload = payload.get("mind_perspective")
        if not isinstance(mind_payload, dict):
            mind_payload = {}
        memory_owner_id = (
            _normalize_text(
                VerbatimPinService._source_value(
                    source_row,
                    payload,
                    "memory_owner_id",
                    "memory_owner_id",
                )
            )
            or _normalize_text(
                VerbatimPinService._source_value(
                    source_row,
                    payload,
                    "active_mind_id",
                    "active_mind_id",
                )
            )
            or _normalize_text(mind_payload.get("memory_owner_id"))
            or _normalize_text(active_mind_id)
        )
        source_mind_id = (
            _normalize_text(
                VerbatimPinService._source_value(
                    source_row,
                    payload,
                    "source_mind_id",
                    "source_mind_id",
                )
            )
            or _normalize_text(mind_payload.get("source_mind_id"))
            or memory_owner_id
        )
        return {
            "memory_owner_id": memory_owner_id,
            "source_mind_id": source_mind_id,
            "mind_topology": (
                _normalize_mind_topology(mind_payload.get("mind_topology"))
                or _normalize_mind_topology(mind_topology)
                or MindTopology.UNIMIND.value
            ),
        }

    @staticmethod
    def _source_embodiment_snapshot(
        *,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        active_embodiment_id: str | None,
    ) -> dict[str, str | None]:
        payload = payload_json or {}
        embodiment_payload = payload.get("embodiment")
        if not isinstance(embodiment_payload, dict):
            embodiment_payload = {}
        embodiment_id = (
            _normalize_text(
                VerbatimPinService._source_value(
                    source_row,
                    payload,
                    "embodiment_id",
                    "active_embodiment_id",
                )
            )
            or _normalize_text(embodiment_payload.get("active_embodiment_id"))
            or _normalize_text(active_embodiment_id)
        )
        return {"embodiment_id": embodiment_id}

    @staticmethod
    def _source_realm_snapshot(
        *,
        source_row: dict[str, Any] | None,
        payload_json: dict[str, Any] | None,
        active_realm_id: str | None,
    ) -> dict[str, str | None]:
        payload = payload_json or {}
        realm_payload = payload.get("realm")
        if not isinstance(realm_payload, dict):
            realm_payload = {}
        realm_id = (
            _normalize_text(
                VerbatimPinService._source_value(
                    source_row,
                    payload,
                    "realm_id",
                    "active_realm_id",
                )
            )
            or _normalize_text(realm_payload.get("active_realm_id"))
            or _normalize_text(active_realm_id)
        )
        return {"realm_id": realm_id}

    @staticmethod
    def _message_space_visibility_clause(
        *,
        active_space_id: str | None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None,
    ) -> tuple[str, list[Any]]:
        if active_space_id is None:
            return (
                "(m.space_id IS NULL OR COALESCE(sp.boundary_mode, 'focus') IN ('focus', 'tagged'))",
                [],
            )

        active_mode = _normalize_space_boundary_mode(active_space_boundary_mode)
        if active_mode == SpaceBoundaryMode.SEVERANCE.value:
            return "(m.space_id = ?)", [active_space_id]
        if active_mode == SpaceBoundaryMode.TAGGED.value:
            return (
                "(m.space_id IS NULL OR m.space_id = ? OR COALESCE(sp.boundary_mode, 'focus') IN ('focus', 'tagged'))",
                [active_space_id],
            )
        return (
            "(m.space_id IS NULL OR m.space_id = ? OR COALESCE(sp.boundary_mode, 'focus') = 'tagged')",
            [active_space_id],
        )

    @staticmethod
    def _message_mind_visibility_clause(
        *,
        active_mind_id: str | None,
        mind_topology: MindTopology | str | None,
    ) -> tuple[str, list[Any]]:
        normalized_active_mind = _normalize_text(active_mind_id)
        if normalized_active_mind is None:
            return "(m.active_mind_id IS NULL)", []
        topology = _normalize_mind_topology(mind_topology) or MindTopology.UNIMIND.value
        if topology == MindTopology.UNIMIND.value:
            return "(m.active_mind_id IS NULL OR m.active_mind_id = ?)", [normalized_active_mind]
        return "(m.active_mind_id = ?)", [normalized_active_mind]

    @staticmethod
    def _message_embodiment_visibility_clause(
        *,
        active_embodiment_id: str | None,
    ) -> tuple[str, list[Any]]:
        normalized_active_embodiment = _normalize_text(active_embodiment_id)
        if normalized_active_embodiment is None:
            return "(m.active_embodiment_id IS NULL)", []
        return (
            "(m.active_embodiment_id IS NULL OR m.active_embodiment_id = ?)",
            [normalized_active_embodiment],
        )

    @staticmethod
    def _message_realm_visibility_clause(
        *,
        active_realm_id: str | None,
    ) -> tuple[str, list[Any]]:
        normalized_active_realm = _normalize_text(active_realm_id)
        if normalized_active_realm is None:
            return "(m.active_realm_id IS NULL)", []
        return (
            "(m.active_realm_id IS NULL OR m.active_realm_id = ?)",
            [normalized_active_realm],
        )

    @staticmethod
    def _source_value(
        source_row: dict[str, Any] | None,
        payload: dict[str, Any],
        key: str,
        snapshot_key: str,
    ) -> Any:
        if source_row is not None:
            if source_row.get(key) is not None:
                return source_row.get(key)
            if source_row.get(snapshot_key) is not None:
                return source_row.get(snapshot_key)
        if payload.get(key) is not None:
            return payload.get(key)
        return payload.get(snapshot_key)

    @staticmethod
    def _source_sensitivity(
        source_row: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> MemorySensitivity | None:
        raw_value = VerbatimPinService._source_value(source_row, payload, "sensitivity", "sensitivity")
        if raw_value is None:
            return None
        try:
            return MemorySensitivity(str(raw_value))
        except ValueError:
            return None

    @staticmethod
    def _source_policy_snapshot(
        source_row: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_policy = None
        if source_row is not None:
            raw_policy = source_row.get("policy_snapshot_json")
            row_payload = source_row.get("payload_json")
            if raw_policy is None and isinstance(row_payload, dict):
                raw_policy = row_payload.get("source_turn_policy")
        if raw_policy is None:
            raw_policy = payload.get("policy_snapshot_json") or payload.get("source_turn_policy")
        return dict(raw_policy) if isinstance(raw_policy, dict) else {}

    @staticmethod
    def _canonical_pin_scope(scope: MemoryScope) -> str:
        if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
            return MemoryScope.CHAT.value
        if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
            return MemoryScope.CHARACTER.value
        return MemoryScope.USER.value

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
                "space_id": source_row.get("space_id"),
                "space_boundary_mode": source_row.get("space_boundary_mode"),
                "active_mind_id": source_row.get("active_mind_id"),
                "source_mind_id": source_row.get("source_mind_id"),
                "active_embodiment_id": source_row.get("embodiment_id")
                or source_row.get("active_embodiment_id"),
                "active_realm_id": source_row.get("realm_id")
                or source_row.get("active_realm_id"),
            }
        if target_kind is VerbatimPinTargetKind.MEMORY_OBJECT:
            return {
                "memory_id": str(source_row.get("id") or target_id),
                "object_type": source_row.get("object_type"),
                "scope": source_row.get("scope"),
                "assistant_mode_id": source_row.get("assistant_mode_id"),
                "workspace_id": source_row.get("workspace_id"),
                "conversation_id": source_row.get("conversation_id"),
                "space_id": source_row.get("space_id"),
                "space_boundary_mode": source_row.get("space_boundary_mode"),
                "memory_owner_id": source_row.get("memory_owner_id"),
                "source_mind_id": source_row.get("source_mind_id"),
                "embodiment_id": source_row.get("embodiment_id"),
                "realm_id": source_row.get("realm_id"),
            }
        return {
            "source_id": str(source_row.get("id") or target_id),
            "source_kind": "text_span",
        }
