"""Confirmation flow orchestration for natural-memory consent UX."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.consent_repository import (
    MemoryConsentProfileRepository,
    PendingMemoryConfirmationRepository,
)
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.consent_confirmation import (
    ConsentResponseIntent,
    category_plural_label,
    classify_confirmation_response,
    safe_confirmation_label,
)
from atagia.models.schemas_memory import ConversationStatus, MemoryCategory, MemoryStatus
from atagia.services.embedding_payloads import build_embedding_upsert_payload
from atagia.services.embeddings import EmbeddingIndex, NoneBackend
from atagia.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PendingConfirmationTurnPlan:
    """Read-only decision plan for one chat turn."""

    prompt_text: str | None = None
    prompt_memory_ids: tuple[str, ...] = ()
    response_intent: ConsentResponseIntent | None = None
    response_memory_ids: tuple[str, ...] = ()
    response_category: MemoryCategory | None = None
    response_first_ask_memory_ids: tuple[str, ...] = ()
    response_reask_memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _EmbeddingUpsert:
    memory_id: str
    canonical_text: str
    index_text: str | None
    privacy_level: int
    intimacy_boundary: str
    preserve_verbatim: bool
    user_id: str
    object_type: str
    scope: str
    created_at: str


class PendingConfirmationService:
    """Plan and apply pending confirmation state transitions for one user turn."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        embedding_index: EmbeddingIndex | None = None,
        llm_client: LLMClient[Any] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._consent_repository = MemoryConsentProfileRepository(connection, clock)
        self._pending_repository = PendingMemoryConfirmationRepository(connection, clock)
        self._embedding_index = embedding_index or NoneBackend()
        self._clock = clock
        self._llm_client = llm_client
        self._settings = settings or Settings.from_env()

    async def plan_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
    ) -> PendingConfirmationTurnPlan:
        asked_marker = await self._pending_repository.get_oldest_asked_marker(user_id, conversation_id)
        if asked_marker is not None:
            return await self._plan_response_batch(
                user_id=user_id,
                conversation_id=conversation_id,
                message_text=message_text,
                category=MemoryCategory(str(asked_marker["memory_category"])),
            )

        unasked_marker = await self._pending_repository.get_oldest_unasked_marker(user_id, conversation_id)
        if unasked_marker is None:
            return PendingConfirmationTurnPlan()
        return await self._plan_prompt_batch(
            user_id=user_id,
            conversation_id=conversation_id,
            category=MemoryCategory(str(unasked_marker["memory_category"])),
        )

    async def apply_turn_plan(
        self,
        *,
        user_id: str,
        plan: PendingConfirmationTurnPlan,
        commit: bool = True,
    ) -> list[_EmbeddingUpsert]:
        embedding_upserts: list[_EmbeddingUpsert] = []
        if plan.response_category is not None and plan.response_memory_ids:
            if plan.response_intent is ConsentResponseIntent.CONFIRM:
                embedding_upserts = await self._confirm_memories(
                    user_id=user_id,
                    memory_ids=list(plan.response_memory_ids),
                    category=plan.response_category,
                )
            elif plan.response_intent is ConsentResponseIntent.DENY:
                await self._deny_memories(
                    user_id=user_id,
                    memory_ids=list(plan.response_memory_ids),
                    category=plan.response_category,
                )
            elif plan.response_intent is ConsentResponseIntent.AMBIGUOUS:
                if plan.response_first_ask_memory_ids:
                    await self._pending_repository.reset_after_ambiguous(
                        user_id,
                        list(plan.response_first_ask_memory_ids),
                        commit=False,
                    )
                if plan.response_reask_memory_ids:
                    await self._deny_memories(
                        user_id=user_id,
                        memory_ids=list(plan.response_reask_memory_ids),
                        category=plan.response_category,
                    )

        if plan.prompt_memory_ids:
            await self._pending_repository.mark_markers_asked(
                user_id,
                list(plan.prompt_memory_ids),
                asked_at=self._clock.now().isoformat(),
                commit=False,
            )

        if commit:
            await self._memory_repository.commit()
            await self._upsert_embeddings(embedding_upserts)
        return embedding_upserts

    async def apply_post_commit_embeddings(self, upserts: list[_EmbeddingUpsert]) -> None:
        """Run post-commit embedding side effects for confirmed memories."""
        await self._upsert_embeddings(upserts)

    async def list_pending_confirmations(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        platform_id: str | None = None,
        user_persona_id: str | None = None,
        character_id: str | None = None,
        category: MemoryCategory | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return safe pending-confirmation records for a host user UX."""

        markers = await self._pending_repository.list_pending_markers(
            user_id=user_id,
            conversation_id=conversation_id,
            platform_id=platform_id,
            user_persona_id=user_persona_id,
            character_id=character_id,
            category=category,
            limit=limit,
            offset=offset,
        )
        return [self._pending_marker_view(marker) for marker in markers]

    async def confirm_pending_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> dict[str, Any]:
        """Confirm one pending memory using the normal consent transition."""

        marker = await self._pending_repository.get_marker_for_memory(user_id, memory_id)
        if marker is None:
            raise ValueError("Pending confirmation not found")
        try:
            upserts = await self._confirm_memories(
                user_id=user_id,
                memory_ids=[memory_id],
                category=MemoryCategory(str(marker["memory_category"])),
            )
            memory = await self._memory_repository.get_memory_object(memory_id, user_id)
            await self._memory_repository.commit()
        except Exception:
            await self._memory_repository.rollback()
            raise
        await self._upsert_embeddings(upserts)
        if memory is None:
            raise ValueError("Pending confirmation not found")
        return memory

    async def decline_pending_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> dict[str, Any]:
        """Decline one pending memory using the normal consent transition."""

        marker = await self._pending_repository.get_marker_for_memory(user_id, memory_id)
        if marker is None:
            raise ValueError("Pending confirmation not found")
        try:
            await self._deny_memories(
                user_id=user_id,
                memory_ids=[memory_id],
                category=MemoryCategory(str(marker["memory_category"])),
            )
            memory = await self._memory_repository.get_memory_object(memory_id, user_id)
            await self._memory_repository.commit()
        except Exception:
            await self._memory_repository.rollback()
            raise
        if memory is None:
            raise ValueError("Pending confirmation not found")
        return memory

    @staticmethod
    def _pending_marker_view(marker: dict[str, Any]) -> dict[str, Any]:
        category = MemoryCategory(str(marker["memory_category"]))
        return {
            "memory_id": str(marker["memory_id"]),
            "user_id": str(marker["user_id"]),
            "conversation_id": str(marker["conversation_id"]),
            "category": category.value,
            "label": safe_confirmation_label(marker.get("index_text"), category),
            "created_at": str(marker["created_at"]),
            "asked_at": marker.get("asked_at"),
            "confirmation_asked_once": bool(int(marker.get("confirmation_asked_once") or 0)),
            "user_persona_id": marker.get("user_persona_id"),
            "platform_id": marker.get("platform_id"),
            "character_id": marker.get("character_id"),
            "mode": marker.get("mode"),
            "incognito_snapshot": bool(int(marker.get("incognito_snapshot") or 0)),
            "intended_scope": marker.get("intended_scope"),
            "intended_sensitivity": marker.get("intended_sensitivity"),
            "platform_locked": bool(int(marker.get("platform_locked") or 0)),
            "platform_id_lock": marker.get("platform_id_lock"),
            "policy_proven": bool(int(marker.get("policy_proven") or 0)),
            "memory_status": marker.get("memory_status"),
        }

    async def _plan_prompt_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        category: MemoryCategory,
    ) -> PendingConfirmationTurnPlan:
        markers = await self._pending_repository.list_markers_for_category(
            user_id,
            conversation_id,
            category,
            asked=False,
        )
        memory_ids = [str(marker["memory_id"]) for marker in markers]
        labels = await self._labels_for_memory_ids(user_id, memory_ids)
        if not labels:
            return PendingConfirmationTurnPlan()
        prompt_text = self._build_prompt_text(labels, category)
        return PendingConfirmationTurnPlan(
            prompt_text=prompt_text,
            prompt_memory_ids=tuple(memory_ids),
        )

    async def _plan_response_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_text: str,
        category: MemoryCategory,
    ) -> PendingConfirmationTurnPlan:
        markers = await self._pending_repository.list_markers_for_category(
            user_id,
            conversation_id,
            category,
            asked=True,
        )
        if not markers:
            return PendingConfirmationTurnPlan()
        first_ask_memory_ids = tuple(
            str(marker["memory_id"])
            for marker in markers
            if not bool(int(marker["confirmation_asked_once"]))
        )
        reask_memory_ids = tuple(
            str(marker["memory_id"])
            for marker in markers
            if bool(int(marker["confirmation_asked_once"]))
        )
        memory_ids = tuple(str(marker["memory_id"]) for marker in markers)
        labels = await self._labels_for_memory_ids(user_id, list(memory_ids))
        prompt_text = self._build_prompt_text(labels, category) if labels else None
        return PendingConfirmationTurnPlan(
            response_intent=await classify_confirmation_response(
                message_text,
                self._llm_client,
                prompt_text=prompt_text,
                settings=self._settings,
            ),
            response_memory_ids=memory_ids,
            response_category=category,
            response_first_ask_memory_ids=first_ask_memory_ids,
            response_reask_memory_ids=reask_memory_ids,
        )

    async def _labels_for_memory_ids(
        self,
        user_id: str,
        memory_ids: list[str],
    ) -> list[str]:
        memories = await self._memory_repository.list_memory_objects_by_ids(user_id, memory_ids)
        by_id = {str(memory["id"]): memory for memory in memories}
        labels: list[str] = []
        for memory_id in memory_ids:
            memory = by_id.get(memory_id)
            if memory is None:
                continue
            if memory["status"] != MemoryStatus.PENDING_USER_CONFIRMATION.value:
                continue
            labels.append(
                safe_confirmation_label(
                    memory.get("index_text"),
                    MemoryCategory(str(memory["memory_category"])),
                )
            )
        return labels

    @staticmethod
    def _build_prompt_text(labels: list[str], category: MemoryCategory) -> str:
        if len(labels) == 1:
            return f"Before I answer, I noted {labels[0]} earlier. Want me to keep it for next time?"
        rendered_labels = _human_join(labels)
        return (
            f"Before I answer, I noted {rendered_labels} earlier. "
            f"Want me to keep {category_plural_label(category)} for next time?"
        )

    async def _confirm_memories(
        self,
        *,
        user_id: str,
        memory_ids: list[str],
        category: MemoryCategory,
    ) -> list[_EmbeddingUpsert]:
        embedding_upserts: list[_EmbeddingUpsert] = []
        profile_user_persona_ids: set[str | None] = set()
        for memory_id in memory_ids:
            marker = await self._pending_repository.get_marker_for_memory(user_id, memory_id)
            if marker is None:
                continue
            profile_user_persona_ids.add(self._marker_persona(marker))
            if not self._marker_policy_is_proven(marker):
                await self._memory_repository.update_memory_object_status(
                    memory_id=memory_id,
                    user_id=user_id,
                    status=MemoryStatus.REVIEW_REQUIRED,
                    expected_current_status=MemoryStatus.PENDING_USER_CONFIRMATION,
                    commit=False,
                )
                continue
            memory = await self._memory_repository.get_memory_object(memory_id, user_id)
            if memory is None or not self._marker_matches_memory(marker, memory):
                await self._memory_repository.update_memory_object_status(
                    memory_id=memory_id,
                    user_id=user_id,
                    status=MemoryStatus.REVIEW_REQUIRED,
                    expected_current_status=MemoryStatus.PENDING_USER_CONFIRMATION,
                    commit=False,
                )
                continue
            if not await self._current_policy_allows_activation(marker, memory):
                await self._memory_repository.update_memory_object_status(
                    memory_id=memory_id,
                    user_id=user_id,
                    status=MemoryStatus.REVIEW_REQUIRED,
                    expected_current_status=MemoryStatus.PENDING_USER_CONFIRMATION,
                    commit=False,
                )
                continue
            updated = await self._memory_repository.update_memory_object_status(
                memory_id=memory_id,
                user_id=user_id,
                status=MemoryStatus.ACTIVE,
                expected_current_status=MemoryStatus.PENDING_USER_CONFIRMATION,
                commit=False,
            )
            if updated is None:
                continue
            if updated["status"] != MemoryStatus.ACTIVE.value:
                continue
            embedding_upserts.append(
                _EmbeddingUpsert(
                    memory_id=str(updated["id"]),
                    canonical_text=str(updated["canonical_text"]),
                    index_text=updated.get("index_text"),
                    privacy_level=int(updated["privacy_level"]),
                    intimacy_boundary=str(updated.get("intimacy_boundary") or "ordinary"),
                    preserve_verbatim=bool(int(updated["preserve_verbatim"])),
                    user_id=str(updated["user_id"]),
                    object_type=str(updated["object_type"]),
                    scope=str(updated["scope"]),
                    created_at=str(updated["created_at"]),
                )
            )
        await self._increment_profile_count(
            user_id=user_id,
            category=category,
            confirmed_delta=len(embedding_upserts),
            declined_delta=0,
            user_persona_id=self._single_profile_persona(profile_user_persona_ids),
        )
        await self._pending_repository.clear_markers(user_id, memory_ids, commit=False)
        return embedding_upserts

    @staticmethod
    def _marker_policy_is_proven(marker: dict[str, Any]) -> bool:
        return bool(int(marker.get("policy_proven") or 0))

    @staticmethod
    def _marker_matches_memory(marker: dict[str, Any], memory: dict[str, Any]) -> bool:
        comparisons = {
            "user_persona_id": memory.get("user_persona_id"),
            "platform_id": memory.get("platform_id"),
            "character_id": memory.get("character_id"),
            "intended_scope": memory.get("scope_canonical") or memory.get("scope"),
            "intended_sensitivity": memory.get("sensitivity"),
            "platform_id_lock": memory.get("platform_id_lock"),
        }
        for marker_key, memory_value in comparisons.items():
            marker_value = marker.get(marker_key)
            if marker_value is not None and marker_value != memory_value:
                return False
        if marker.get("intended_scope") == "chat" and marker.get("conversation_id") != memory.get(
            "conversation_id"
        ):
            return False
        if bool(int(marker.get("platform_locked") or 0)) and not bool(
            int(memory.get("platform_locked") or 0)
        ):
            return False
        return True

    async def _current_policy_allows_activation(
        self,
        marker: dict[str, Any],
        memory: dict[str, Any],
    ) -> bool:
        active_user = await self._user_repository.get_active_user(str(memory["user_id"]))
        if active_user is None:
            return False
        conversation_id = str(marker.get("conversation_id") or "")
        conversation = await self._conversation_repository.get_conversation(
            conversation_id,
            str(memory["user_id"]),
        )
        if conversation is None or str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
            return False
        source_chat_only = (
            bool(int(marker.get("incognito_snapshot") or 0))
            or not bool(int(marker.get("remember_across_chats_snapshot") or 1))
            or bool(int(marker.get("temporary_snapshot") or 0))
            or bool(int(marker.get("purge_on_close_snapshot") or 0))
        )
        current_chat_only = (
            bool(conversation.get("incognito"))
            or bool(conversation.get("isolated_mode"))
            or bool(conversation.get("temporary"))
            or bool(conversation.get("purge_on_close"))
            or not bool(active_user["remember_across_chats"])
        )
        scope = str(memory.get("scope_canonical") or memory.get("scope") or "")
        if scope in {"conversation", "ephemeral_session"}:
            scope = "chat"
        if source_chat_only or current_chat_only:
            return scope == "chat" and memory.get("conversation_id") == conversation_id
        source_platform_locked = (
            bool(int(marker.get("platform_locked") or 0))
            or not bool(int(marker.get("remember_across_devices_snapshot") or 1))
        )
        current_platform_locked = not bool(active_user["remember_across_devices"])
        if source_platform_locked or current_platform_locked:
            platform_id = str(conversation.get("platform_id") or marker.get("platform_id") or "default")
            if not bool(memory.get("platform_locked")):
                return False
            return memory.get("platform_id_lock") == platform_id
        return True

    async def _deny_memories(
        self,
        *,
        user_id: str,
        memory_ids: list[str],
        category: MemoryCategory,
    ) -> None:
        declined_count = 0
        profile_user_persona_ids: set[str | None] = set()
        for memory_id in memory_ids:
            marker = await self._pending_repository.get_marker_for_memory(user_id, memory_id)
            if marker is not None:
                profile_user_persona_ids.add(self._marker_persona(marker))
            updated = await self._memory_repository.update_memory_object_status(
                memory_id=memory_id,
                user_id=user_id,
                status=MemoryStatus.DECLINED,
                expected_current_status=MemoryStatus.PENDING_USER_CONFIRMATION,
                commit=False,
            )
            if updated is None:
                continue
            if updated["status"] == MemoryStatus.DECLINED.value:
                declined_count += 1
        await self._increment_profile_count(
            user_id=user_id,
            category=category,
            confirmed_delta=0,
            declined_delta=declined_count,
            user_persona_id=self._single_profile_persona(profile_user_persona_ids),
        )
        await self._pending_repository.clear_markers(user_id, memory_ids, commit=False)

    async def _increment_profile_count(
        self,
        *,
        user_id: str,
        category: MemoryCategory,
        confirmed_delta: int,
        declined_delta: int,
        user_persona_id: str | None,
    ) -> None:
        if confirmed_delta == 0 and declined_delta == 0:
            return
        profile = await self._consent_repository.get_profile(
            user_id,
            category,
            user_persona_id=user_persona_id,
        )
        confirmed_count = int(profile.get("confirmed_count", 0)) if profile is not None else 0
        declined_count = int(profile.get("declined_count", 0)) if profile is not None else 0
        timestamp = self._clock.now().isoformat()
        await self._consent_repository.upsert_profile(
            user_id=user_id,
            category=category,
            confirmed_count=confirmed_count + confirmed_delta,
            declined_count=declined_count + declined_delta,
            user_persona_id=user_persona_id,
            last_confirmed_at=timestamp if confirmed_delta > 0 else profile.get("last_confirmed_at") if profile else None,
            last_declined_at=timestamp if declined_delta > 0 else profile.get("last_declined_at") if profile else None,
            commit=False,
        )

    @staticmethod
    def _marker_persona(marker: dict[str, Any]) -> str | None:
        value = marker.get("user_persona_id")
        normalized = str(value).strip() if value is not None else ""
        return normalized or None

    @staticmethod
    def _single_profile_persona(values: set[str | None]) -> str | None:
        if len(values) == 1:
            return next(iter(values))
        return None

    async def _upsert_embeddings(self, upserts: list[_EmbeddingUpsert]) -> None:
        if self._embedding_index.vector_limit == 0:
            return
        for upsert in upserts:
            try:
                payload = build_embedding_upsert_payload(
                    canonical_text=upsert.canonical_text,
                    index_text=upsert.index_text,
                    privacy_level=upsert.privacy_level,
                    intimacy_boundary=upsert.intimacy_boundary,
                    preserve_verbatim=upsert.preserve_verbatim,
                )
                await self._embedding_index.upsert(
                    memory_id=upsert.memory_id,
                    text=payload.text,
                    metadata={
                        "user_id": upsert.user_id,
                        "object_type": upsert.object_type,
                        "scope": upsert.scope,
                        "created_at": upsert.created_at,
                        "index_text": payload.index_text,
                    },
                )
            except Exception:
                logger.warning(
                    "Embedding upsert failed after confirmation for memory_id=%s",
                    upsert.memory_id,
                    exc_info=True,
                )


def _human_join(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"
