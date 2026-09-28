"""Conversation, memory, and erasure lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import aiosqlite

from atagia.core import json_utils
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.conversation_namespace import (
    ConversationNamespaceSnapshot,
    capture_conversation_namespace_snapshot,
)
from atagia.core.ids import generate_prefixed_id
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_extraction_suppression_repository import (
    MemoryExtractionSuppressionRepository,
)
from atagia.core.repositories import (
    MemoryObjectRepository,
    _decode_json_columns,
    _encode_json,
    summary_mirror_id,
)
from atagia.core.user_erasure_repository import (
    ErasureCleanupTargetSpec,
    UserErasureRepository,
)
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.models.schemas_api import DeletionReport, ErasureReport
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_memory import (
    ConversationStatus,
    MemoryObjectType,
    MemoryStatus,
    SpaceBoundaryMode,
)
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import (
    ConversationAlreadyClosedError,
    ConversationNotFoundError,
    DeletionConfirmationError,
    InvalidConversationTransitionError,
    MemoryNotEditableError,
    MemoryNotFoundError,
    MemoryProvenanceRepairRequiredError,
    UserErasureCleanupPendingError,
    UserErasureReconciliationRequiredError,
)
from atagia.services.artifact_blob_migration import LegacyArtifactBlobStateError
from atagia.services.user_erasure_cleanup_service import UserErasureCleanupService

if TYPE_CHECKING:
    from atagia.app import AppRuntime


DELETE_CONVERSATION_CONFIRMATION = "DELETE_CONVERSATION"
PURGE_ON_CLOSE_CONFIRMATION = "PURGE_ON_CLOSE"
HARD_DELETE_MEMORY_CONFIRMATION = "HARD_DELETE_MEMORY"
ERASE_ALL_DATA_CONFIRMATION = "ERASE_ALL_DATA"


@dataclass(slots=True)
class ConversationLifecycleService:
    """High-level lifecycle operations with explicit destructive confirmations."""

    runtime: AppRuntime

    async def close_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        purge: bool | None = None,
        confirmation: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport | dict[str, Any]:
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            return await self._close_conversation_guarded(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                purge=purge,
                confirmation=confirmation,
                namespace_guard=namespace_guard,
            )

    async def _close_conversation_guarded(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        purge: bool | None = None,
        confirmation: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport | dict[str, Any]:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            conversation = await self._get_conversation(
                connection, user_id, conversation_id
            )
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found for user")
            if str(conversation["status"]) == ConversationStatus.CLOSED.value:
                raise ConversationAlreadyClosedError("Conversation is already closed")
            if str(conversation["status"]) in {
                ConversationStatus.ARCHIVED.value,
                ConversationStatus.PENDING_DELETION.value,
            }:
                raise InvalidConversationTransitionError(
                    "Conversation cannot be closed from its current state"
                )

            should_purge = (
                bool(conversation.get("purge_on_close"))
                if purge is None
                else bool(purge)
            )
            if should_purge:
                if confirmation != PURGE_ON_CLOSE_CONFIRMATION:
                    raise DeletionConfirmationError(
                        "Missing PURGE_ON_CLOSE confirmation"
                    )
                await connection.rollback()
                return await self._delete_conversation(
                    connection,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    deletion_reason="purge_on_close",
                    confirmation_override=True,
                    namespace_guard=namespace_guard,
                )

            timestamp = self.runtime.clock.now().isoformat()
            cursor = await connection.execute(
                """
                UPDATE conversations
                SET status = ?,
                    closed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND status = ?
                """,
                (
                    ConversationStatus.CLOSED.value,
                    timestamp,
                    timestamp,
                    conversation_id,
                    user_id,
                    ConversationStatus.ACTIVE.value,
                ),
            )
            if cursor.rowcount == 0:
                raise InvalidConversationTransitionError("Conversation is not active")
            await self._delete_initial_context_packages_for_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await ContextCacheService(self.runtime).invalidate_conversation_cache_by_id(
            user_id,
            conversation_id,
        )
        return await self._get_conversation(connection, user_id, conversation_id) or {}

    async def archive_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> dict[str, Any]:
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            return await self._archive_conversation_guarded(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )

    async def _archive_conversation_guarded(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> dict[str, Any]:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            conversation = await self._get_conversation(
                connection, user_id, conversation_id
            )
            if conversation is None:
                raise ConversationNotFoundError("Conversation not found for user")
            if bool(conversation.get("temporary")) and bool(
                conversation.get("purge_on_close")
            ):
                raise InvalidConversationTransitionError(
                    "Temporary purge-on-close conversations cannot be archived"
                )
            if str(conversation["status"]) == ConversationStatus.PENDING_DELETION.value:
                raise InvalidConversationTransitionError(
                    "Conversation is pending deletion"
                )

            timestamp = self.runtime.clock.now().isoformat()
            conversation_message_ids = await self._conversation_message_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            affected_memory_ids = await self._conversation_affected_memory_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await self._require_extracted_source_provenance(
                connection,
                user_id=user_id,
                memory_ids=affected_memory_ids,
                operation="archive conversation",
            )
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=affected_memory_ids,
            )
            affected_trace_ids = [
                *affected_memory_ids,
                *self._summary_mirror_ids(summary_ids),
            ]
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=affected_memory_ids,
                reason="conversation_archived",
            )
            await self._delete_embeddings_for_ids(connection, affected_trace_ids)
            await self._delete_summary_views(
                connection, user_id=user_id, summary_ids=summary_ids
            )
            await self._cleanup_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
                conversation_id=conversation_id,
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
            )
            if affected_memory_ids:
                placeholders = self._placeholders(affected_memory_ids)
                await connection.execute(
                    f"""
                    UPDATE memory_objects
                    SET status = ?,
                        archived_by_conversation_id = ?,
                        updated_at = ?
                    WHERE user_id = ?
                      AND id IN ({placeholders})
                    """,
                    (
                        MemoryStatus.ARCHIVED.value,
                        conversation_id,
                        timestamp,
                        user_id,
                        *affected_memory_ids,
                    ),
                )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                reason="source_conversation_archived",
                commit=False,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_memories(
                user_id=user_id,
                memory_ids=affected_trace_ids,
                reason="source_conversation_archived",
                commit=False,
            )
            await self._delete_initial_context_packages_for_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            cursor = await connection.execute(
                """
                UPDATE conversations
                SET status = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND status IN (?, ?)
                """,
                (
                    ConversationStatus.ARCHIVED.value,
                    timestamp,
                    conversation_id,
                    user_id,
                    ConversationStatus.ACTIVE.value,
                    ConversationStatus.CLOSED.value,
                ),
            )
            if cursor.rowcount == 0:
                raise InvalidConversationTransitionError(
                    "Conversation cannot be archived from its current state"
                )
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=conversation_message_ids,
                excluded_conversation_ids=[conversation_id],
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._invalidate_user_prompt_cache(user_id)
        await self._purge_conversation_jobs(user_id, conversation_id)
        await self._delete_embedding_index_entries(affected_trace_ids)
        return await self._get_conversation(connection, user_id, conversation_id) or {}

    async def expire_idle_temporary_conversations(
        self,
        connection: aiosqlite.Connection,
        *,
        dry_run: bool = False,
    ) -> int:
        """Close or purge temporary conversations that exceeded their idle TTL."""
        timestamp = self.runtime.clock.now().isoformat()
        cursor = await connection.execute(
            """
            SELECT id, user_id, purge_on_close
            FROM conversations
            WHERE temporary = 1
              AND status = ?
              AND temporary_ttl_seconds IS NOT NULL
              AND last_activity_at IS NOT NULL
              AND julianday(?) - julianday(last_activity_at) > temporary_ttl_seconds / 86400.0
            ORDER BY last_activity_at ASC, id ASC
            """,
            (ConversationStatus.ACTIVE.value, timestamp),
        )
        rows = await cursor.fetchall()
        if dry_run:
            return len(rows)
        for row in rows:
            if bool(row["purge_on_close"]):
                await self._delete_conversation(
                    connection,
                    user_id=str(row["user_id"]),
                    conversation_id=str(row["id"]),
                    deletion_reason="ttl_expiry",
                    confirmation_override=True,
                )
            else:
                await self.close_conversation(
                    connection,
                    user_id=str(row["user_id"]),
                    conversation_id=str(row["id"]),
                    purge=False,
                )
        return len(rows)

    async def process_pending_file_deletions(
        self,
        connection: aiosqlite.Connection,
        *,
        dry_run: bool = False,
        limit: int = 100,
    ) -> int:
        """Reject legacy local-file deletion work owned by the offline migrator."""
        cursor = await connection.execute(
            """
            SELECT *
            FROM pending_file_deletions
            WHERE deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        if dry_run:
            return len(rows)
        if rows:
            raise LegacyArtifactBlobStateError(
                "Legacy pending artifact-file deletions require "
                "`atagia-artifact-blob-migrate run` while Atagia is stopped"
            )
        return 0

    async def purge_pending_deleted_conversations(
        self,
        connection: aiosqlite.Connection,
        *,
        dry_run: bool = False,
        limit: int = 25,
    ) -> int:
        """Physically delete conversations hidden by the synchronous delete step."""
        cursor = await connection.execute(
            """
            SELECT id, user_id
            FROM conversations
            WHERE status = ?
            ORDER BY updated_at ASC, id ASC
            LIMIT ?
            """,
            (ConversationStatus.PENDING_DELETION.value, limit),
        )
        rows = await cursor.fetchall()
        if dry_run:
            return len(rows)
        purged = 0
        for row in rows:
            await self._purge_pending_conversation(
                connection,
                user_id=str(row["user_id"]),
                conversation_id=str(row["id"]),
            )
            purged += 1
        return purged

    async def delete_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        confirmation: str,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        if confirmation != DELETE_CONVERSATION_CONFIRMATION:
            raise DeletionConfirmationError("Missing DELETE_CONVERSATION confirmation")
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            return await self._delete_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                deletion_reason="user_request",
                confirmation_override=True,
                namespace_guard=namespace_guard,
            )

    async def edit_memory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
        new_text: str,
        edit_source: str = "api",
        edited_by: str = "system",
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> dict[str, Any]:
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            return await self._edit_memory_guarded(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                new_text=new_text,
                edit_source=edit_source,
                edited_by=edited_by,
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
                namespace_guard=namespace_guard,
            )

    async def _edit_memory_guarded(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
        new_text: str,
        edit_source: str = "api",
        edited_by: str = "system",
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> dict[str, Any]:
        normalized_text = " ".join(new_text.split()).strip()
        if not normalized_text:
            raise MemoryNotEditableError("Memory text cannot be empty")
        timestamp = self.runtime.clock.now().isoformat()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            memory = await self._get_memory(
                connection,
                user_id,
                memory_id,
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
            if memory is None:
                raise MemoryNotFoundError("Memory object not found for user")
            if (
                str(memory.get("status")) != MemoryStatus.ACTIVE.value
                or str(memory.get("object_type")) != MemoryObjectType.EVIDENCE.value
            ):
                raise MemoryNotEditableError(
                    "Only active evidence memories can be edited"
                )
            source_message_ids = await self._memory_source_message_ids(
                connection,
                user_id=user_id,
                memory_ids=[memory_id],
            )
            if (
                str(memory.get("source_kind") or "") == "extracted"
                and not source_message_ids
            ):
                raise MemoryProvenanceRepairRequiredError(
                    "Extracted memory is missing source provenance; repair is required before edit"
                )
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=[memory_id],
            )
            summary_mirror_ids = self._summary_mirror_ids(summary_ids)
            affected_trace_ids = [memory_id, *summary_mirror_ids]
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=[memory_id],
                reason="memory_edited",
                replacement_memory_id=memory_id,
            )
            await connection.execute(
                """
                INSERT INTO memory_edit_history(
                    memory_id,
                    previous_text,
                    new_text,
                    edited_by,
                    edit_source,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    str(memory["canonical_text"]),
                    normalized_text,
                    edited_by,
                    edit_source,
                    timestamp,
                ),
            )
            await connection.execute(
                """
                UPDATE memory_objects
                SET canonical_text = ?,
                    index_text = NULL,
                    extraction_hash = NULL,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                """,
                (normalized_text, timestamp, memory_id, user_id),
            )
            await self._delete_summary_views(
                connection,
                user_id=user_id,
                summary_ids=summary_ids,
            )
            await self._cleanup_memory_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
            )
            await MemoryEvidenceRepository(
                connection,
                self.runtime.clock,
            ).delete_for_memory_ids(
                user_id=user_id,
                memory_ids=[memory_id],
                commit=False,
            )
            await self._mark_retrieval_surfaces_stale_for_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                timestamp=timestamp,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_memories(
                user_id=user_id,
                memory_ids=affected_trace_ids,
                reason="source_memory_edited",
                commit=False,
            )
            await self._mark_initial_context_packages_stale_for_user(
                connection,
                user_id=user_id,
            )
            await self._delete_embeddings_for_ids(connection, affected_trace_ids)
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=source_message_ids,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._delete_embedding_index_entries(affected_trace_ids)
        await ContextCacheService(self.runtime).invalidate_user_cache(user_id)
        refreshed = await self._get_memory(
            connection,
            user_id,
            memory_id,
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
        if refreshed is None:
            raise MemoryNotFoundError("Memory object not found after edit")
        return refreshed

    async def delete_memory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
        hard: bool = False,
        confirmation: str | None = None,
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            return await self._delete_memory_guarded(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                hard=hard,
                confirmation=confirmation,
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
                namespace_guard=namespace_guard,
            )

    async def _delete_memory_guarded(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
        hard: bool = False,
        confirmation: str | None = None,
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        if not hard:
            return await self._archive_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
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
                namespace_guard=namespace_guard,
            )
        if confirmation != HARD_DELETE_MEMORY_CONFIRMATION:
            raise DeletionConfirmationError("Missing HARD_DELETE_MEMORY confirmation")
        return await self._hard_delete_memory(
            connection,
            user_id=user_id,
            memory_id=memory_id,
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
            namespace_guard=namespace_guard,
        )

    async def erase_user_data(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        confirmation: str,
    ) -> ErasureReport:
        if confirmation != ERASE_ALL_DATA_CONFIRMATION:
            raise DeletionConfirmationError("Missing ERASE_ALL_DATA confirmation")
        async with ContextCacheService(self.runtime).user_cache_guard(user_id):
            try:
                return await self._erase_user_data(connection, user_id=user_id)
            finally:
                # Canonical deletion commits before transient cleanup. If that
                # cleanup fails, _erase_user_data raises even though the source
                # rows are already gone; eviction must therefore live in the
                # finally path. The guard serializes the current runtime; the
                # DF cache's post-scan SQLite identity check rejects a
                # cross-runtime scan that overlaps this erasure.
                self.runtime.token_document_frequency_cache.forget(user_id)

    async def _delete_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
        deletion_reason: str,
        confirmation_override: bool,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        del confirmation_override
        timestamp = self.runtime.clock.now().isoformat()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            conversation_message_ids = await self._conversation_message_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            cursor = await connection.execute(
                """
                UPDATE conversations
                SET status = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND status IN (?, ?, ?)
                """,
                (
                    ConversationStatus.PENDING_DELETION.value,
                    timestamp,
                    conversation_id,
                    user_id,
                    ConversationStatus.ACTIVE.value,
                    ConversationStatus.CLOSED.value,
                    ConversationStatus.ARCHIVED.value,
                ),
            )
            if cursor.rowcount == 0:
                existing = await self._get_conversation(
                    connection, user_id, conversation_id
                )
                if existing is None:
                    await connection.rollback()
                    return DeletionReport(
                        conversation_id=conversation_id,
                        already_deleted=True,
                    )
                if str(existing["status"]) != ConversationStatus.PENDING_DELETION.value:
                    raise InvalidConversationTransitionError(
                        "Conversation cannot be deleted from its current state"
                    )
            memory_ids = await self._conversation_affected_memory_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=memory_ids,
            )
            summary_mirror_ids = self._summary_mirror_ids(summary_ids)
            all_memory_ids = [*memory_ids, *summary_mirror_ids]
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                reason="conversation_deleted",
            )
            artifact_ids = await self._conversation_artifact_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            message_count = await self._count(
                connection,
                "messages AS m JOIN conversations AS c ON c.id = m.conversation_id",
                "c.user_id = ? AND m.conversation_id = ?",
                (user_id, conversation_id),
            )
            tombstone_id = generate_prefixed_id("tmb")
            await self._delete_embeddings_for_ids(connection, all_memory_ids)
            await self._cleanup_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                conversation_id=conversation_id,
            )
            await self._delete_summary_views(
                connection, user_id=user_id, summary_ids=summary_ids
            )
            await self._tombstone_memory_rows(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                conversation_id=conversation_id,
                timestamp=timestamp,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                reason="source_conversation_deleted",
                commit=False,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_memories(
                user_id=user_id,
                memory_ids=all_memory_ids,
                reason="source_conversation_deleted",
                commit=False,
            )
            await self._delete_initial_context_packages_for_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await self._delete_artifacts(
                connection, user_id=user_id, artifact_ids=artifact_ids
            )
            await connection.execute(
                "DELETE FROM verbatim_pins WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await connection.execute(
                "DELETE FROM retrieval_events WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
            )
            await connection.execute(
                "DELETE FROM conversation_activity_stats WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await connection.execute(
                "DELETE FROM worker_job_runs WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await self._insert_tombstone(
                connection,
                tombstone_id=tombstone_id,
                entity_type="conversation",
                deletion_reason=deletion_reason,
                timestamp=timestamp,
                scope_summary={
                    "memory_count": len(all_memory_ids),
                    "message_count": message_count,
                    "summary_count": len(summary_ids),
                    "artifact_count": len(artifact_ids),
                },
            )
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=conversation_message_ids,
                excluded_conversation_ids=[conversation_id],
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        await self._invalidate_user_prompt_cache(user_id)
        await self._purge_conversation_jobs(user_id, conversation_id)
        await self._delete_embedding_index_entries(all_memory_ids)
        return DeletionReport(
            conversation_id=conversation_id,
            deleted_memories=len(all_memory_ids),
            deleted_messages=message_count,
            deleted_summaries=len(summary_ids),
            deleted_artifacts=len(artifact_ids),
            tombstone_id=tombstone_id,
        )

    async def _purge_pending_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> None:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            conversation = await self._get_conversation(
                connection, user_id, conversation_id
            )
            if conversation is None:
                await connection.rollback()
                return
            if str(conversation["status"]) != ConversationStatus.PENDING_DELETION.value:
                await connection.rollback()
                return

            conversation_message_ids = await self._conversation_message_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            memory_ids = await self._conversation_affected_memory_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=memory_ids,
            )
            summary_mirror_ids = self._summary_mirror_ids(summary_ids)
            all_memory_ids = [*memory_ids, *summary_mirror_ids]
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                reason="conversation_purged",
            )
            artifact_ids = await self._conversation_artifact_ids(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await self._delete_embeddings_for_ids(connection, all_memory_ids)
            await self._cleanup_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                conversation_id=conversation_id,
            )
            await self._delete_summary_views(
                connection, user_id=user_id, summary_ids=summary_ids
            )
            await self._delete_memory_rows(
                connection, user_id=user_id, memory_ids=all_memory_ids
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                reason="source_conversation_purged",
                commit=False,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_memories(
                user_id=user_id,
                memory_ids=all_memory_ids,
                reason="source_conversation_purged",
                commit=False,
            )
            await self._delete_initial_context_packages_for_conversation(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await self._delete_artifacts(
                connection, user_id=user_id, artifact_ids=artifact_ids
            )
            await connection.execute(
                "DELETE FROM verbatim_pins WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await connection.execute(
                "DELETE FROM retrieval_events WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
            )
            await connection.execute(
                "DELETE FROM conversation_activity_stats WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await connection.execute(
                "DELETE FROM worker_job_runs WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            )
            await connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            await connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=conversation_message_ids,
                excluded_conversation_ids=[conversation_id],
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        await self._delete_embedding_index_entries(all_memory_ids)
        await self._invalidate_user_prompt_cache(user_id)

    async def _archive_memory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        timestamp = self.runtime.clock.now().isoformat()
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            memory = await self._get_memory(
                connection,
                user_id,
                memory_id,
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
            if memory is None:
                raise MemoryNotFoundError("Memory object not found for user")
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=[memory_id],
            )
            summary_mirror_ids = self._summary_mirror_ids(summary_ids)
            affected_trace_ids = [memory_id, *summary_mirror_ids]
            source_message_ids = await self._memory_source_message_ids(
                connection,
                user_id=user_id,
                memory_ids=[memory_id],
            )
            if (
                str(memory.get("source_kind") or "") == "extracted"
                and not source_message_ids
            ):
                raise MemoryProvenanceRepairRequiredError(
                    "Extracted memory is missing source provenance; repair is required before archive"
                )
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=[memory_id],
                reason="memory_archived",
            )
            await self._delete_embeddings_for_ids(connection, affected_trace_ids)
            await self._delete_summary_views(
                connection,
                user_id=user_id,
                summary_ids=summary_ids,
            )
            await self._cleanup_memory_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=affected_trace_ids,
            )
            await connection.execute(
                """
                UPDATE memory_objects
                SET status = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                """,
                (MemoryStatus.ARCHIVED.value, timestamp, memory_id, user_id),
            )
            await self._mark_retrieval_surfaces_stale_for_memory(
                connection,
                user_id=user_id,
                memory_id=memory_id,
                timestamp=timestamp,
            )
            await CommunicationProfileRepository(
                connection,
                self.runtime.clock,
            ).mark_stale_for_memories(
                user_id=user_id,
                memory_ids=affected_trace_ids,
                reason="source_memory_archived",
                commit=False,
            )
            await self._mark_initial_context_packages_stale_for_user(
                connection,
                user_id=user_id,
            )
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=source_message_ids,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._delete_embedding_index_entries(affected_trace_ids)
        await ContextCacheService(self.runtime).invalidate_user_cache(user_id)
        return DeletionReport(
            memory_id=memory_id,
            deleted_memories=1,
            deleted_summaries=len(summary_ids),
        )

    async def _hard_delete_memory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        namespace_guard: ConversationNamespaceSnapshot | None = None,
    ) -> DeletionReport:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)
            await self._require_namespace_guard(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                namespace_guard=namespace_guard,
            )
            memory = await self._get_memory(
                connection,
                user_id,
                memory_id,
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
            if memory is None:
                raise MemoryNotFoundError("Memory object not found for user")
            summary_ids = await self._derived_summary_ids(
                connection,
                user_id=user_id,
                seed_object_ids=[memory_id],
            )
            summary_mirror_ids = self._summary_mirror_ids(summary_ids)
            all_memory_ids = [memory_id, *summary_mirror_ids]
            source_message_ids = await self._memory_source_message_ids(
                connection,
                user_id=user_id,
                memory_ids=[memory_id],
            )
            if (
                str(memory.get("source_kind") or "") == "extracted"
                and not source_message_ids
            ):
                raise MemoryProvenanceRepairRequiredError(
                    "Extracted memory is missing source provenance; repair is required before hard delete"
                )
            await self._suppress_memory_extractions(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
                reason="memory_hard_deleted",
            )
            await self._delete_embeddings_for_ids(connection, all_memory_ids)
            await self._cleanup_memory_projection_rows(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
            )
            await self._delete_summary_views(
                connection, user_id=user_id, summary_ids=summary_ids
            )
            await self._delete_retrieval_events_for_memory_ids(
                connection,
                user_id=user_id,
                memory_ids=all_memory_ids,
            )
            for source_memory_id in all_memory_ids:
                await CommunicationProfileRepository(
                    connection,
                    self.runtime.clock,
                ).mark_stale_for_memory(
                    user_id=user_id,
                    memory_id=source_memory_id,
                    reason="source_memory_deleted",
                    commit=False,
                )
            await self._mark_initial_context_packages_stale_for_user(
                connection,
                user_id=user_id,
            )
            await self._delete_memory_rows(
                connection, user_id=user_id, memory_ids=all_memory_ids
            )
            await self._bump_derivation_revision(
                connection,
                user_id=user_id,
                excluded_source_message_ids=source_message_ids,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        await self._delete_embedding_index_entries(all_memory_ids)
        await ContextCacheService(self.runtime).invalidate_user_cache(user_id)
        return DeletionReport(
            memory_id=memory_id,
            deleted_memories=len(all_memory_ids),
            deleted_summaries=len(summary_ids),
        )

    async def _erase_user_data(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
    ) -> ErasureReport:
        repository = UserErasureRepository(connection, self.runtime.clock)
        state = await repository.get_erasure_state_for_candidate(user_id)
        user = await self._fetch_one(
            connection,
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        )
        if user is None:
            if state is None:
                return ErasureReport(user_id=user_id, already_erased=True)
            cleanup_id = state.get("cleanup_id")
            if cleanup_id is not None:
                await self._complete_user_erasure_cleanup(
                    connection,
                    cleanup_id=str(cleanup_id),
                )
                counts = self._erasure_scope_counts(state.get("scope_summary"))
                return ErasureReport(
                    user_id=user_id,
                    deleted_memories=counts["memory_count"],
                    deleted_conversations=counts["conversation_count"],
                    deleted_artifacts=counts["artifact_count"],
                    tombstone_id=str(state["tombstone_id"]),
                )
            if state["erasure_cleanup_state"] == "verified":
                return ErasureReport(
                    user_id=user_id,
                    tombstone_id=str(state["tombstone_id"]),
                    already_erased=True,
                )
            raise UserErasureReconciliationRequiredError(
                "The retained legacy erasure marker requires explicit historical cleanup reconciliation"
            )

        identity = await UserLifecycleRepository(
            connection,
            self.runtime.clock,
        ).get_active_identity(user_id)
        if identity is None:
            raise UserErasureCleanupPendingError(
                "The user lifecycle is not active and cannot start a new erasure"
            )

        inventory: dict[str, list[str]] = {}

        async def inventory_scope() -> dict[str, int]:
            inventory["memory_ids"] = await self._ids(
                connection,
                "memory_objects",
                "id",
                "user_id = ?",
                (user_id,),
            )
            inventory["conversation_ids"] = await self._ids(
                connection,
                "conversations",
                "id",
                "user_id = ?",
                (user_id,),
            )
            inventory["workspace_ids"] = await self._ids(
                connection,
                "workspaces",
                "id",
                "user_id = ?",
                (user_id,),
            )
            inventory["retrieval_event_ids"] = await self._ids(
                connection,
                "retrieval_events",
                "id",
                "user_id = ?",
                (user_id,),
            )
            inventory["artifact_ids"] = await self._ids(
                connection,
                "artifacts",
                "id",
                "user_id = ?",
                (user_id,),
            )
            return {
                "memory_count": len(inventory["memory_ids"]),
                "conversation_count": len(inventory["conversation_ids"]),
                "artifact_count": len(inventory["artifact_ids"]),
            }

        async def canonical_delete() -> None:
            memory_ids = inventory["memory_ids"]
            await self._delete_embeddings_for_ids(connection, memory_ids)
            await self._delete_embeddings_for_user(connection, user_id)
            await self._delete_admin_audit_rows_for_erasure(
                connection,
                user_id=user_id,
                conversation_ids=inventory["conversation_ids"],
                workspace_ids=inventory["workspace_ids"],
                retrieval_event_ids=inventory["retrieval_event_ids"],
                memory_ids=memory_ids,
                artifact_ids=inventory["artifact_ids"],
            )
            await self._delete_user_child_tables(connection, user_id=user_id)
            await connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

        async def require_erasure_available() -> None:
            await TranscriptRebuildRepository(
                connection,
                self.runtime.clock,
            ).require_user_available(user_id)

        preparation = await repository.prepare_current_erasure(
            user_id=user_id,
            scope_counts=None,
            scope_counts_provider=inventory_scope,
            target_specs=(
                ErasureCleanupTargetSpec(
                    target_kind="transient_backend",
                    backend_name=self._storage_backend_name(),
                    target_key=identity.lifecycle_cleanup_key,
                    lifecycle_epoch=identity.lifecycle_epoch,
                ),
            ),
            canonical_delete=canonical_delete,
            transaction_precondition=require_erasure_available,
            expected_lifecycle_epoch=identity.lifecycle_epoch,
        )
        await self._complete_user_erasure_cleanup(
            connection,
            cleanup_id=preparation.cleanup_id,
        )
        return ErasureReport(
            user_id=user_id,
            deleted_memories=len(inventory["memory_ids"]),
            deleted_conversations=len(inventory["conversation_ids"]),
            deleted_artifacts=len(inventory["artifact_ids"]),
            tombstone_id=preparation.tombstone_id,
        )

    async def _complete_user_erasure_cleanup(
        self,
        connection: aiosqlite.Connection,
        *,
        cleanup_id: str,
    ) -> None:
        try:
            await UserErasureCleanupService(
                connection,
                self.runtime.clock,
                self.runtime.storage_backend,
                self._storage_backend_name(),
            ).complete_cleanup(cleanup_id)
        except Exception as exc:
            raise UserErasureCleanupPendingError(
                f"Canonical erasure completed; cleanup {cleanup_id} remains pending"
            ) from exc

    def _storage_backend_name(self) -> str:
        return str(getattr(self.runtime.settings, "storage_backend", "inprocess"))

    @staticmethod
    def _erasure_scope_counts(scope_summary: Any) -> dict[str, int]:
        if not isinstance(scope_summary, str):
            return {
                "memory_count": 0,
                "conversation_count": 0,
                "artifact_count": 0,
            }
        try:
            decoded = json_utils.loads(scope_summary)
        except json_utils.JSONDecodeError:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}
        counts: dict[str, int] = {}
        for key in ("memory_count", "conversation_count", "artifact_count"):
            value = decoded.get(key, 0)
            try:
                counts[key] = max(0, int(value))
            except (TypeError, ValueError):
                counts[key] = 0
        return counts

    async def _delete_admin_audit_rows_for_erasure(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_ids: list[str],
        workspace_ids: list[str],
        retrieval_event_ids: list[str],
        memory_ids: list[str],
        artifact_ids: list[str],
    ) -> None:
        await connection.execute(
            """
            DELETE FROM admin_audit_log
            WHERE admin_user_id = ?
               OR target_id = ?
               OR json_extract(metadata_json, '$.user_id') = ?
            """,
            (user_id, user_id, user_id),
        )
        for target_type, target_ids in (
            ("conversation", conversation_ids),
            ("workspace", workspace_ids),
            ("retrieval_event", retrieval_event_ids),
            ("memory_object", memory_ids),
            ("artifact", artifact_ids),
        ):
            if not target_ids:
                continue
            placeholders = self._placeholders(target_ids)
            await connection.execute(
                f"""
                DELETE FROM admin_audit_log
                WHERE target_type = ?
                  AND target_id IN ({placeholders})
                """,
                (target_type, *target_ids),
            )

    async def _delete_user_child_tables(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
    ) -> None:
        for statement in (
            "DELETE FROM memory_edit_history WHERE memory_id IN (SELECT id FROM memory_objects WHERE user_id = ?)",
            "DELETE FROM pending_memory_confirmations WHERE user_id = ?",
            "DELETE FROM memory_feedback_events WHERE user_id = ?",
            "DELETE FROM memory_links WHERE user_id = ?",
            "DELETE FROM contract_dimensions_current WHERE user_id = ?",
            "DELETE FROM user_communication_profiles WHERE user_id = ?",
            "DELETE FROM consequence_chains WHERE user_id = ?",
            "DELETE FROM graph_relationship_sources WHERE user_id = ?",
            "DELETE FROM graph_relationships WHERE user_id = ?",
            "DELETE FROM graph_entity_aliases WHERE user_id = ?",
            "DELETE FROM graph_entity_mentions WHERE user_id = ?",
            "DELETE FROM graph_projection_runs WHERE user_id = ?",
            "DELETE FROM graph_entities WHERE user_id = ?",
            "DELETE FROM initial_context_package_build_attempts WHERE user_id = ?",
            "DELETE FROM initial_context_packages WHERE user_id = ?",
            "DELETE FROM conversation_topic_sources WHERE user_id = ?",
            "DELETE FROM conversation_topic_events WHERE user_id = ?",
            "DELETE FROM conversation_topics WHERE user_id = ?",
            "DELETE FROM conversation_activity_stats WHERE user_id = ?",
            "DELETE FROM proxy_turn_runs WHERE user_id = ?",
            "DELETE FROM artifact_links WHERE user_id = ?",
            "DELETE FROM artifact_chunks WHERE user_id = ?",
            "DELETE FROM artifact_blobs WHERE artifact_id IN (SELECT id FROM artifacts WHERE user_id = ?)",
            "DELETE FROM artifacts WHERE user_id = ?",
            "DELETE FROM artifact_payload_blobs WHERE user_id = ?",
            "DELETE FROM verbatim_pins WHERE user_id = ?",
            "DELETE FROM retrieval_events WHERE user_id = ?",
            "DELETE FROM summary_views WHERE user_id = ?",
            "DELETE FROM memory_embedding_metadata WHERE user_id = ?",
            "DELETE FROM memory_objects WHERE user_id = ?",
            "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)",
            "DELETE FROM conversations WHERE user_id = ?",
            "DELETE FROM workspaces WHERE user_id = ?",
            "DELETE FROM memory_consent_profile WHERE user_id = ?",
            "DELETE FROM evaluation_metrics WHERE user_id = ?",
        ):
            await connection.execute(statement, (user_id,))
        await connection.execute(
            """
            DELETE FROM admin_audit_log
            WHERE admin_user_id = ?
               OR target_id = ?
            """,
            (user_id, user_id),
        )

    async def _get_conversation(
        self,
        connection: aiosqlite.Connection,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            connection,
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )

    async def _require_namespace_guard(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str | None,
        namespace_guard: ConversationNamespaceSnapshot | None,
    ) -> None:
        if namespace_guard is None:
            return
        if (
            namespace_guard.user_id != user_id
            or conversation_id != namespace_guard.conversation_id
        ):
            raise ConversationNotFoundError("Conversation namespace changed")
        current = await capture_conversation_namespace_snapshot(
            connection,
            self.runtime.clock,
            user_id=user_id,
            conversation_id=namespace_guard.conversation_id,
        )
        if current != namespace_guard:
            raise ConversationNotFoundError("Conversation namespace changed")

    async def _get_memory(
        self,
        connection: aiosqlite.Connection,
        user_id: str,
        memory_id: str,
        *,
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
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        if conversation_id is not None and platform_id is not None:
            return await MemoryObjectRepository(
                connection,
                self.runtime.clock,
            ).get_visible_memory_object(
                memory_id,
                user_id,
                conversation_id=conversation_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                incognito=incognito,
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
        return await self._fetch_one(
            connection,
            "SELECT * FROM memory_objects WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )

    async def _conversation_affected_memory_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[str]:
        cursor = await connection.execute(
            """
            SELECT DISTINCT mo.id
            FROM memory_objects AS mo
            LEFT JOIN json_each(
                json_extract(mo.payload_json, '$.source_message_ids')
            ) AS source_ids ON 1 = 1
            WHERE mo.user_id = ?
              AND (
                  mo.conversation_id = ?
                  OR CAST(source_ids.value AS TEXT) IN (
                      SELECT id
                      FROM messages
                      WHERE conversation_id = ?
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_evidence_spans AS evidence_span
                      LEFT JOIN messages AS evidence_message
                        ON evidence_message.id = evidence_span.message_id
                      WHERE evidence_span.user_id = mo.user_id
                        AND evidence_span.memory_id = mo.id
                        AND (
                            evidence_span.conversation_id = ?
                            OR evidence_message.conversation_id = ?
                        )
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM memory_fact_facets AS fact_facet
                      JOIN messages AS fact_message
                        ON fact_message.id = fact_facet.source_message_id
                      WHERE fact_facet.user_id = mo.user_id
                        AND fact_facet.memory_id = mo.id
                        AND (
                            fact_facet.conversation_id = ?
                            OR fact_message.conversation_id = ?
                        )
                  )
              )
            ORDER BY mo.id ASC
            """,
            (
                user_id,
                conversation_id,
                conversation_id,
                conversation_id,
                conversation_id,
                conversation_id,
                conversation_id,
            ),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]

    async def _bump_derivation_revision(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        excluded_source_message_ids: Iterable[str] = (),
        excluded_conversation_ids: Iterable[str] = (),
        allow_validated_requeue: bool = True,
    ) -> int:
        """Fence worker effects in the caller-owned source mutation transaction."""

        repository = UserLifecycleRepository(connection, self.runtime.clock)
        identity = await repository.get_active_identity(user_id)
        if identity is None:
            raise UserErasureCleanupPendingError(
                "User lifecycle is not active; source mutation cannot commit"
            )
        revision = await repository.bump_derivation_revision(
            user_id,
            expected_lifecycle_epoch=identity.lifecycle_epoch,
            commit=False,
        )
        if revision is None:
            raise UserErasureCleanupPendingError(
                "User lifecycle changed before source mutation could commit"
            )
        await JobRunRepository(
            connection,
            self.runtime.clock,
        ).reconcile_stale_root_jobs_after_revision_bump(
            user_id,
            identity.derivation_revision,
            revision,
            excluded_source_message_ids=excluded_source_message_ids,
            excluded_conversation_ids=excluded_conversation_ids,
            allow_validated_requeue=allow_validated_requeue,
        )
        return revision

    async def _memory_source_message_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> list[str]:
        stable_ids = list(dict.fromkeys(memory_ids))
        if not stable_ids:
            return []
        placeholders = self._placeholders(stable_ids)
        cursor = await connection.execute(
            f"""
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            (user_id, *stable_ids),
        )
        memories = [_decode_json_columns(row) or {} for row in await cursor.fetchall()]
        return await MemoryExtractionSuppressionRepository(
            connection,
            self.runtime.clock,
        ).source_message_ids_for_memories(memories)

    async def _require_extracted_source_provenance(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        operation: str,
    ) -> None:
        stable_ids = list(dict.fromkeys(memory_ids))
        if not stable_ids:
            return
        placeholders = self._placeholders(stable_ids)
        cursor = await connection.execute(
            f"""
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND source_kind = 'extracted'
              AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            (user_id, *stable_ids),
        )
        repository = MemoryExtractionSuppressionRepository(
            connection,
            self.runtime.clock,
        )
        for row in await cursor.fetchall():
            memory = _decode_json_columns(row) or {}
            if await repository.source_message_ids_for_memories([memory]):
                continue
            raise MemoryProvenanceRepairRequiredError(
                "Extracted memory "
                f"{memory.get('id')} is missing source provenance; repair is required "
                f"before {operation}"
            )

    @staticmethod
    async def _conversation_message_ids(
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[str]:
        cursor = await connection.execute(
            """
            SELECT message.id
            FROM messages AS message
            JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            WHERE conversation.user_id = ?
              AND message.conversation_id = ?
            ORDER BY message.seq ASC, message.id ASC
            """,
            (user_id, conversation_id),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]

    async def _suppress_memory_extractions(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        reason: str,
        replacement_memory_id: str | None = None,
    ) -> int:
        """Retire extraction identities before explicit source removal commits."""

        stable_ids = list(dict.fromkeys(memory_ids))
        if not stable_ids:
            return 0
        placeholders = self._placeholders(stable_ids)
        cursor = await connection.execute(
            f"""
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            (user_id, *stable_ids),
        )
        memories = [_decode_json_columns(row) or {} for row in await cursor.fetchall()]
        return await MemoryExtractionSuppressionRepository(
            connection,
            self.runtime.clock,
        ).suppress_memories(
            memories,
            reason=reason,
            replacement_memory_id=replacement_memory_id,
            commit=False,
        )

    async def _conversation_artifact_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[str]:
        cursor = await connection.execute(
            """
            SELECT DISTINCT a.id
            FROM artifacts AS a
            LEFT JOIN messages AS m ON m.id = a.message_id
            WHERE a.user_id = ?
              AND (
                  a.conversation_id = ?
                  OR m.conversation_id = ?
              )
            ORDER BY a.id ASC
            """,
            (user_id, conversation_id, conversation_id),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]

    async def _derived_summary_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        seed_object_ids: Iterable[str],
    ) -> list[str]:
        frontier = {str(item) for item in seed_object_ids if str(item).strip()}
        found: set[str] = set()
        while frontier:
            placeholders = self._placeholders(frontier)
            cursor = await connection.execute(
                f"""
                SELECT DISTINCT sv.id
                FROM summary_views AS sv
                JOIN json_each(sv.source_object_ids_json) AS source_ids ON 1 = 1
                WHERE sv.user_id = ?
                  AND CAST(source_ids.value AS TEXT) IN ({placeholders})
                ORDER BY sv.id ASC
                """,
                (user_id, *sorted(frontier)),
            )
            next_ids = {str(row["id"]) for row in await cursor.fetchall()}
            mirror_cursor = await connection.execute(
                f"""
                SELECT DISTINCT json_extract(mo.payload_json, '$.summary_view_id') AS summary_id
                FROM memory_objects AS mo
                JOIN json_each(
                    json_extract(mo.payload_json, '$.source_object_ids')
                ) AS source_ids ON 1 = 1
                WHERE mo.user_id = ?
                  AND mo.object_type = ?
                  AND CAST(source_ids.value AS TEXT) IN ({placeholders})
                  AND json_extract(mo.payload_json, '$.summary_view_id') IS NOT NULL
                ORDER BY summary_id ASC
                """,
                (user_id, MemoryObjectType.SUMMARY_VIEW.value, *sorted(frontier)),
            )
            next_ids.update(
                str(row["summary_id"]) for row in await mirror_cursor.fetchall()
            )
            next_ids = next_ids - found
            found.update(next_ids)
            frontier = {
                item
                for summary_id in next_ids
                for item in (summary_id, summary_mirror_id(summary_id))
            }
        return sorted(found)

    async def _cleanup_projection_rows(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        conversation_id: str,
    ) -> None:
        await connection.execute(
            "DELETE FROM graph_relationship_sources WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM graph_entity_mentions WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM graph_projection_runs WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM consequence_chains WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM conversation_topics WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await connection.execute(
            "DELETE FROM conversation_topic_events WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        await self._cleanup_memory_projection_rows(
            connection,
            user_id=user_id,
            memory_ids=memory_ids,
        )

    async def _cleanup_memory_projection_rows(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> None:
        """Remove only projections that directly depend on selected memories."""

        if not memory_ids:
            await self._delete_orphan_graph_rows(connection, user_id=user_id)
            return
        placeholders = self._placeholders(memory_ids)
        await connection.execute(
            f"DELETE FROM graph_relationship_sources WHERE user_id = ? AND memory_id IN ({placeholders})",
            (user_id, *memory_ids),
        )
        await connection.execute(
            f"DELETE FROM graph_entity_mentions WHERE user_id = ? AND memory_id IN ({placeholders})",
            (user_id, *memory_ids),
        )
        await connection.execute(
            f"DELETE FROM contract_dimensions_current WHERE user_id = ? AND source_memory_id IN ({placeholders})",
            (user_id, *memory_ids),
        )
        await connection.execute(
            f"""
            DELETE FROM consequence_chains
            WHERE user_id = ?
              AND (
                  action_memory_id IN ({placeholders})
                  OR outcome_memory_id IN ({placeholders})
                  OR tendency_belief_id IN ({placeholders})
              )
            """,
            (user_id, *memory_ids, *memory_ids, *memory_ids),
        )
        topic_cursor = await connection.execute(
            f"""
            SELECT DISTINCT topic_id
            FROM conversation_topic_sources
            WHERE user_id = ?
              AND source_kind = 'memory_object'
              AND source_id IN ({placeholders})
            """,
            (user_id, *memory_ids),
        )
        affected_topic_ids = [
            str(row["topic_id"]) for row in await topic_cursor.fetchall()
        ]
        await connection.execute(
            f"""
            DELETE FROM conversation_topic_sources
            WHERE user_id = ?
              AND source_kind = 'memory_object'
              AND source_id IN ({placeholders})
            """,
            (user_id, *memory_ids),
        )
        if affected_topic_ids:
            topic_placeholders = self._placeholders(affected_topic_ids)
            await connection.execute(
                f"""
                DELETE FROM conversation_topics
                WHERE user_id = ?
                  AND id IN ({topic_placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM conversation_topic_sources AS source
                      WHERE source.user_id = conversation_topics.user_id
                        AND source.topic_id = conversation_topics.id
                  )
                """,
                (user_id, *affected_topic_ids),
            )
        await self._delete_orphan_graph_rows(connection, user_id=user_id)

    async def _delete_summary_views(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        summary_ids: list[str],
    ) -> None:
        if not summary_ids:
            return
        placeholders = self._placeholders(summary_ids)
        mirror_ids = self._summary_mirror_ids(summary_ids)
        mirror_placeholders = self._placeholders(mirror_ids)
        await connection.execute(
            f"""
            DELETE FROM memory_objects
            WHERE user_id = ?
              AND id IN ({mirror_placeholders})
            """,
            (user_id, *mirror_ids),
        )
        await connection.execute(
            f"""
            DELETE FROM summary_views
            WHERE user_id = ?
              AND id IN ({placeholders})
            """,
            (user_id, *summary_ids),
        )

    async def _delete_memory_rows(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return
        placeholders = self._placeholders(memory_ids)
        await connection.execute(
            f"DELETE FROM memory_objects WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *memory_ids),
        )

    async def _mark_retrieval_surfaces_stale_for_memory(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_id: str,
        timestamp: str,
    ) -> None:
        await connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET status = 'stale',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id = ?
              AND status != 'deleted'
            """,
            (timestamp, user_id, memory_id),
        )

    async def _mark_retrieval_surfaces_deleted_for_memory_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        timestamp: str,
    ) -> None:
        if not memory_ids:
            return
        placeholders = self._placeholders(memory_ids)
        await connection.execute(
            f"""
            UPDATE memory_retrieval_surfaces
            SET status = 'deleted',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
            """,
            (timestamp, user_id, *memory_ids),
        )

    async def _tombstone_memory_rows(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
        conversation_id: str,
        timestamp: str,
    ) -> None:
        if not memory_ids:
            return
        placeholders = self._placeholders(memory_ids)
        await connection.execute(
            f"""
            UPDATE memory_objects
            SET status = ?,
                archived_by_conversation_id = ?,
                updated_at = ?
            WHERE user_id = ?
              AND id IN ({placeholders})
            """,
            (
                MemoryStatus.DELETED.value,
                conversation_id,
                timestamp,
                user_id,
                *memory_ids,
            ),
        )
        await self._mark_retrieval_surfaces_deleted_for_memory_ids(
            connection,
            user_id=user_id,
            memory_ids=memory_ids,
            timestamp=timestamp,
        )

    async def _delete_retrieval_events_for_memory_ids(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return
        placeholders = self._placeholders(memory_ids)
        await connection.execute(
            f"""
            DELETE FROM retrieval_events
            WHERE user_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM json_each(retrieval_events.selected_memory_ids_json) AS selected_ids
                  WHERE selected_ids.value IN ({placeholders})
              )
            """,
            (user_id, *memory_ids),
        )

    async def _invalidate_user_prompt_cache(self, user_id: str) -> None:
        await ContextCacheService(self.runtime).invalidate_user_cache(user_id)

    async def _delete_initial_context_packages_for_conversation(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        conversation_id: str,
    ) -> None:
        repository = InitialContextPackageRepository(connection, self.runtime.clock)
        await repository.delete_for_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            commit=False,
        )
        await repository.mark_stale_for_key_family(
            user_id=user_id,
            package_kind=InitialContextPackageKind.BASELINE,
            commit=False,
        )

    async def _mark_initial_context_packages_stale_for_user(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
    ) -> None:
        await InitialContextPackageRepository(
            connection,
            self.runtime.clock,
        ).mark_stale_for_user(user_id, commit=False)

    async def _delete_orphan_graph_rows(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
    ) -> None:
        await connection.execute(
            """
            DELETE FROM graph_relationships
            WHERE user_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM graph_relationship_sources AS grs
                  WHERE grs.user_id = graph_relationships.user_id
                    AND grs.relationship_id = graph_relationships.id
              )
            """,
            (user_id,),
        )
        await connection.execute(
            """
            DELETE FROM graph_entity_aliases
            WHERE user_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM graph_entities AS ge
                  WHERE ge.user_id = graph_entity_aliases.user_id
                    AND ge.id = graph_entity_aliases.entity_id
              )
            """,
            (user_id,),
        )
        await connection.execute(
            """
            DELETE FROM graph_entities
            WHERE user_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM graph_entity_mentions AS gem
                  WHERE gem.user_id = graph_entities.user_id
                    AND gem.entity_id = graph_entities.id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM graph_relationships AS gr
                  WHERE gr.user_id = graph_entities.user_id
                    AND (
                        gr.source_entity_id = graph_entities.id
                        OR gr.target_entity_id = graph_entities.id
                    )
              )
            """,
            (user_id,),
        )

    async def _delete_artifacts(
        self,
        connection: aiosqlite.Connection,
        *,
        user_id: str,
        artifact_ids: list[str],
    ) -> None:
        if not artifact_ids:
            return
        placeholders = self._placeholders(artifact_ids)
        cursor = await connection.execute(
            f"""
            SELECT DISTINCT payload_blob_id
            FROM artifacts
            WHERE user_id = ?
              AND id IN ({placeholders})
              AND payload_blob_id IS NOT NULL
            """,
            (user_id, *artifact_ids),
        )
        payload_blob_ids = [
            str(row["payload_blob_id"]) for row in await cursor.fetchall()
        ]
        await connection.execute(
            f"DELETE FROM artifact_links WHERE user_id = ? AND artifact_id IN ({placeholders})",
            (user_id, *artifact_ids),
        )
        await connection.execute(
            f"DELETE FROM artifact_chunks WHERE user_id = ? AND artifact_id IN ({placeholders})",
            (user_id, *artifact_ids),
        )
        await connection.execute(
            f"DELETE FROM artifact_blobs WHERE artifact_id IN ({placeholders})",
            tuple(artifact_ids),
        )
        await connection.execute(
            f"DELETE FROM artifacts WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *artifact_ids),
        )
        if payload_blob_ids:
            payload_placeholders = self._placeholders(payload_blob_ids)
            await connection.execute(
                f"""
                DELETE FROM artifact_payload_blobs
                WHERE user_id = ?
                  AND id IN ({payload_placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM artifacts
                      WHERE user_id = ?
                        AND payload_blob_id = artifact_payload_blobs.id
                        AND status NOT IN ('deleted', 'purged')
                  )
                """,
                (user_id, *payload_blob_ids, user_id),
            )

    async def _delete_embeddings_for_ids(
        self,
        connection: aiosqlite.Connection,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return
        placeholders = self._placeholders(memory_ids)
        for statement in (
            f"DELETE FROM vec_memory_embeddings WHERE memory_id IN ({placeholders})",
            f"DELETE FROM memory_embedding_metadata WHERE memory_id IN ({placeholders})",
        ):
            try:
                await connection.execute(statement, tuple(memory_ids))
            except aiosqlite.OperationalError as exc:
                if "no such table" not in str(exc).lower():
                    raise

    @staticmethod
    async def _delete_embeddings_for_user(
        connection: aiosqlite.Connection,
        user_id: str,
    ) -> None:
        try:
            await connection.execute(
                "DELETE FROM vec_memory_embeddings WHERE user_id = ?",
                (user_id,),
            )
        except aiosqlite.OperationalError as exc:
            normalized = str(exc).lower()
            if "no such table" not in normalized and "no such column" not in normalized:
                raise

    async def _delete_embedding_index_entries(self, memory_ids: list[str]) -> None:
        for memory_id in dict.fromkeys(memory_ids):
            try:
                await self.runtime.embedding_index.delete(memory_id)
            except Exception:
                continue

    async def _insert_tombstone(
        self,
        connection: aiosqlite.Connection,
        *,
        tombstone_id: str,
        entity_type: str,
        deletion_reason: str,
        timestamp: str,
        scope_summary: dict[str, Any],
    ) -> None:
        await connection.execute(
            """
            INSERT INTO deletion_tombstones(
                id,
                entity_type,
                deleted_at,
                deletion_reason,
                deleted_by,
                scope_summary
            )
            VALUES (?, ?, ?, ?, 'system', ?)
            """,
            (
                tombstone_id,
                entity_type,
                timestamp,
                deletion_reason,
                _encode_json(scope_summary),
            ),
        )

    async def _purge_conversation_jobs(
        self, user_id: str, conversation_id: str
    ) -> None:
        if self.runtime.settings.erasure_purge_streams:
            await self.runtime.storage_backend.purge_conversation_jobs(
                user_id, conversation_id
            )

    async def _fetch_one(
        self,
        connection: aiosqlite.Connection,
        query: str,
        parameters: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        cursor = await connection.execute(query, parameters)
        return _decode_json_columns(await cursor.fetchone())

    async def _ids(
        self,
        connection: aiosqlite.Connection,
        table: str,
        column: str,
        where_clause: str,
        parameters: tuple[Any, ...],
    ) -> list[str]:
        cursor = await connection.execute(
            f"SELECT {column} FROM {table} WHERE {where_clause} ORDER BY {column} ASC",
            parameters,
        )
        return [str(row[column]) for row in await cursor.fetchall()]

    async def _count(
        self,
        connection: aiosqlite.Connection,
        from_clause: str,
        where_clause: str,
        parameters: tuple[Any, ...],
    ) -> int:
        cursor = await connection.execute(
            f"SELECT COUNT(*) AS count FROM {from_clause} WHERE {where_clause}",
            parameters,
        )
        row = await cursor.fetchone()
        return int(row["count"])

    @staticmethod
    def _summary_mirror_ids(summary_ids: list[str]) -> list[str]:
        return [summary_mirror_id(summary_id) for summary_id in summary_ids]

    @staticmethod
    def _placeholders(values: Iterable[Any]) -> str:
        count = (
            len(list(values))
            if not isinstance(values, (list, tuple, set))
            else len(values)
        )
        if count <= 0:
            raise ValueError("Cannot build placeholders for an empty value set")
        return ", ".join("?" for _ in range(count))
