"""Belief revision worker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.locking import acquire_belief_lock, belief_lock_key
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.storage_backend import StorageBackend
from atagia.memory.belief_reviser import BeliefReviser, RevisionContext
from atagia.memory.intent_classifier import are_claim_keys_equivalent, is_explicit_user_statement
from atagia.memory.scope_utils import resolve_scope_identifiers
from atagia.models.schemas_jobs import (
    JobEnvelope,
    JobType,
    REVISE_STREAM_NAME,
    RevisionJobPayload,
    StreamMessage,
    WORKER_GROUP_NAME,
    WorkerIterationResult,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    ConversationStatus,
)
from atagia.services.embeddings import EmbeddingIndex, NoneBackend
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import LLMClient, StructuredOutputError
from atagia.services.model_resolution import resolve_component_model
from atagia.services.worker_control_service import WorkerControlService, wait_if_worker_claims_paused

logger = logging.getLogger(__name__)
WORKER_ERROR_RETRY_SECONDS = 1.0
STREAM_RECLAIM_IDLE_MS = 1_000
MAX_STREAM_DELIVERIES = 3


class RevisionWorker:
    """Consumes belief revision jobs from the configured stream backend."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[object],
        clock: Clock,
        embedding_index: EmbeddingIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage_backend = storage_backend
        self._settings = settings or Settings.from_env()
        self._clock = clock
        self._worker_control = WorkerControlService(connection, clock)
        self._job_tracking = JobTrackingService(
            connection,
            clock,
            workers_enabled=self._settings.workers_enabled,
            settings=self._settings,
        )
        self._llm_client = llm_client
        self._embedding_index = embedding_index or NoneBackend()
        self._classifier_model = resolve_component_model(
            self._settings,
            "intent_classifier",
        )
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._belief_repository = BeliefRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)
        self._conversation_repository = ConversationRepository(connection, clock)
        self._user_repository = UserRepository(connection, clock)
        self._reviser = BeliefReviser(
            connection=connection,
            llm_client=llm_client,
            clock=clock,
            embedding_index=self._embedding_index,
            settings=self._settings,
        )

    async def run(self, consumer_name: str = "revise-1") -> None:
        await self._storage_backend.stream_ensure_group(REVISE_STREAM_NAME, WORKER_GROUP_NAME)
        while True:
            try:
                await self.run_once(consumer_name=consumer_name, block_ms=5000)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in revision worker loop")
                await asyncio.sleep(WORKER_ERROR_RETRY_SECONDS)

    async def run_once(
        self,
        *,
        consumer_name: str = "revise-1",
        block_ms: int | None = 0,
    ) -> WorkerIterationResult:
        if await wait_if_worker_claims_paused(self._worker_control, block_ms=block_ms):
            return WorkerIterationResult()
        messages = await self._next_messages(
            consumer_name=consumer_name,
            block_ms=block_ms,
        )
        if not messages:
            return WorkerIterationResult()

        acked = 0
        failed = 0
        dead_lettered = 0
        for message in messages:
            try:
                await self._job_tracking.mark_running(message)
                await self.process_job(message.payload)
                await self._job_tracking.mark_succeeded(message)
                await self._storage_backend.stream_ack(
                    REVISE_STREAM_NAME,
                    WORKER_GROUP_NAME,
                    message.message_id,
                )
                acked += 1
            except Exception as exc:
                failed += 1
                if isinstance(exc, StructuredOutputError):
                    details = "; ".join(exc.details) if exc.details else str(exc)
                    logger.warning(
                        "Failed to process revision job %s due to structured output: %s",
                        message.message_id,
                        details,
                    )
                else:
                    logger.exception("Failed to process revision job %s", message.message_id)
                if await self._dead_letter_if_exhausted(message, exc):
                    dead_lettered += 1
                else:
                    await self._job_tracking.mark_retrying(message, exc)
        return WorkerIterationResult(
            received=len(messages),
            acked=acked,
            failed=failed,
            dead_lettered=dead_lettered,
        )

    async def process_job(self, payload: dict[str, object]) -> dict[str, Any] | None:
        envelope = JobEnvelope.model_validate(payload)
        if envelope.job_type is not JobType.REVISE_BELIEFS:
            raise ValueError(f"Unsupported revision job type: {envelope.job_type}")
        raw_claim_key = str(envelope.payload.get("claim_key", "")).strip()
        if not raw_claim_key:
            logger.warning("Skipping revision job %s with empty claim_key", envelope.job_id)
            return {"status": "invalid_claim_key"}
        job_payload = RevisionJobPayload.model_validate(envelope.payload)
        job_payload = await self._strictest_payload(job_payload)
        if job_payload is None:
            return {"status": "skipped_inactive_source", "claim_key": raw_claim_key}

        lock_subject = job_payload.belief_id or job_payload.claim_key
        lock_token = await acquire_belief_lock(self._storage_backend, lock_subject)
        if lock_token is None:
            raise RuntimeError(f"Could not acquire belief lock for {lock_subject}")

        try:
            if job_payload.belief_id:
                belief = await self._memory_repository.get_memory_object(
                    job_payload.belief_id,
                    job_payload.user_id,
                )
                if belief is None or belief["status"] != MemoryStatus.ACTIVE.value:
                    return {"status": "skipped_inactive", "belief_id": job_payload.belief_id}
                if job_payload.isolated_mode and (
                    belief.get("conversation_id") != job_payload.conversation_id
                    or belief.get("scope") != MemoryScope.CONVERSATION.value
                ):
                    return {"status": "skipped_isolated_scope", "belief_id": job_payload.belief_id}
                if not self._belief_matches_payload_namespace(belief, job_payload):
                    return {"status": "skipped_namespace_mismatch", "belief_id": job_payload.belief_id}
                return await self._revise_existing_belief(job_payload)
            return await self._process_promotion(job_payload)
        finally:
            await self._storage_backend.release_lock(
                belief_lock_key(lock_subject),
                lock_token,
            )

    async def _strictest_payload(self, payload: RevisionJobPayload) -> RevisionJobPayload | None:
        active_user = await self._user_repository.get_active_user(payload.user_id)
        if active_user is None:
            return None
        updates: dict[str, Any] = {
            "remember_across_chats": (
                bool(payload.remember_across_chats)
                and bool(active_user["remember_across_chats"])
            ),
            "remember_across_devices": (
                bool(payload.remember_across_devices)
                and bool(active_user["remember_across_devices"])
            ),
        }
        if payload.conversation_id is not None:
            conversation = await self._conversation_repository.get_conversation(
                payload.conversation_id,
                payload.user_id,
            )
            if conversation is None or str(conversation.get("status")) != ConversationStatus.ACTIVE.value:
                return None
            updates.update(
                {
                    "temporary": bool(payload.temporary) or bool(conversation.get("temporary")),
                    "temporary_ttl_seconds": self._strictest_ttl(
                        payload.temporary_ttl_seconds,
                        conversation.get("temporary_ttl_seconds"),
                    ),
                    "purge_on_close": bool(payload.purge_on_close) or bool(conversation.get("purge_on_close")),
                    "isolated_mode": bool(payload.isolated_mode) or bool(conversation.get("isolated_mode")),
                    "incognito": bool(payload.incognito) or bool(conversation.get("incognito")),
                    "active_embodiment_id": (
                        payload.active_embodiment_id
                        or conversation.get("active_embodiment_id")
                    ),
                    "cross_embodiment_mode": (
                        payload.cross_embodiment_mode
                        or conversation.get("cross_embodiment_mode")
                        or "direct_if_same_body"
                    ),
                    "active_realm_id": (
                        payload.active_realm_id
                        or conversation.get("active_realm_id")
                    ),
                    "cross_realm_mode": (
                        payload.cross_realm_mode
                        or conversation.get("cross_realm_mode")
                        or "none"
                    ),
                }
            )
        return payload.model_copy(update=updates)

    @staticmethod
    def _strictest_ttl(source_ttl: int | None, current_ttl: object) -> int | None:
        values = [
            int(value)
            for value in (source_ttl, current_ttl)
            if value is not None and int(value) > 0
        ]
        if not values:
            return source_ttl
        return min(values)

    @staticmethod
    def _belief_matches_payload_namespace(
        belief: dict[str, Any],
        payload: RevisionJobPayload,
    ) -> bool:
        if not RevisionWorker._belief_matches_payload_mind(belief, payload):
            return False
        if not RevisionWorker._belief_matches_payload_embodiment(belief, payload):
            return False
        if not RevisionWorker._belief_matches_payload_realm(belief, payload):
            return False
        if belief.get("user_persona_id") != payload.user_persona_id:
            return False
        if str(belief.get("sensitivity") or "unknown") != str(payload.sensitivity or "unknown"):
            return False
        if bool(belief.get("platform_locked")):
            if belief.get("platform_id_lock") != payload.platform_id:
                return False
        elif not payload.remember_across_devices and belief.get("platform_id") != payload.platform_id:
            return False
        scope = str(belief.get("scope_canonical") or belief.get("scope") or "")
        if scope in {
            MemoryScope.CHAT.value,
            MemoryScope.CONVERSATION.value,
            MemoryScope.EPHEMERAL_SESSION.value,
        }:
            return belief.get("conversation_id") == payload.conversation_id
        if payload.incognito or payload.isolated_mode or not payload.remember_across_chats:
            return False
        if scope in {MemoryScope.WORKSPACE.value, MemoryScope.CHARACTER.value, "legacy_workspace"}:
            return belief.get("character_id") == payload.character_id
        if scope in {
            MemoryScope.GLOBAL_USER.value,
            MemoryScope.ASSISTANT_MODE.value,
            MemoryScope.USER.value,
            "legacy_assistant_mode",
        }:
            return True
        return False

    @staticmethod
    def _belief_matches_payload_mind(
        belief: dict[str, Any],
        payload: RevisionJobPayload,
    ) -> bool:
        belief_owner = belief.get("memory_owner_id")
        belief_owner_id = None if belief_owner is None else str(belief_owner)
        if payload.active_mind_id is None:
            return belief_owner_id is None
        if str(payload.mind_topology or "unimind") == "unimind":
            return belief_owner_id is None or belief_owner_id == payload.active_mind_id
        return belief_owner_id == payload.active_mind_id

    @staticmethod
    def _belief_matches_payload_embodiment(
        belief: dict[str, Any],
        payload: RevisionJobPayload,
    ) -> bool:
        belief_embodiment = belief.get("embodiment_id")
        belief_embodiment_id = None if belief_embodiment is None else str(belief_embodiment)
        if payload.active_embodiment_id is None:
            return belief_embodiment_id is None
        return (
            belief_embodiment_id is None
            or belief_embodiment_id == payload.active_embodiment_id
        )

    @staticmethod
    def _belief_matches_payload_realm(
        belief: dict[str, Any],
        payload: RevisionJobPayload,
    ) -> bool:
        belief_realm = belief.get("realm_id")
        belief_realm_id = None if belief_realm is None else str(belief_realm)
        if payload.active_realm_id is None:
            return belief_realm_id is None
        return belief_realm_id is None or belief_realm_id == payload.active_realm_id

    async def _revise_existing_belief(
        self,
        payload: RevisionJobPayload,
    ) -> dict[str, Any]:
        signal_type = await self._classify_revision_signal(payload)
        if signal_type == "contradictory":
            await self._belief_repository.add_tension_evidence_ids(
                payload.belief_id,
                payload.evidence_memory_ids,
                user_id=payload.user_id,
                commit=False,
            )
            tension_score = await self._belief_repository.increment_tension(
                payload.belief_id,
                self._settings.belief_tension_increment,
                user_id=payload.user_id,
                commit=False,
            )
            await self._belief_repository.commit()
            if tension_score < self._settings.belief_tension_threshold:
                logger.info(
                    "Deferring belief revision for belief_id=%s at tension=%s threshold=%s",
                    payload.belief_id,
                    tension_score,
                    self._settings.belief_tension_threshold,
                )
                return {
                    "status": "deferred_tension",
                    "belief_id": payload.belief_id,
                    "signal_type": signal_type,
                    "tension_score": tension_score,
                    "threshold": self._settings.belief_tension_threshold,
                }
            accumulated_evidence_ids = await self._belief_repository.get_tension_evidence_ids(
                payload.belief_id,
                user_id=payload.user_id,
            )
            threshold_payload = payload.model_copy(update={"evidence_memory_ids": accumulated_evidence_ids})
            threshold_preview = await self._preview_belief_revision(threshold_payload)
            try:
                popped_evidence_ids = await self._belief_repository.pop_tension_evidence_ids(
                    payload.belief_id,
                    user_id=payload.user_id,
                    commit=False,
                )
                if popped_evidence_ids != accumulated_evidence_ids:
                    raise RuntimeError("Tension evidence buffer changed during threshold revision")
                await self._belief_repository.reset_tension(
                    payload.belief_id,
                    user_id=payload.user_id,
                    commit=False,
                )
                result = await self._execute_belief_revision(
                    threshold_payload,
                    preview_action=threshold_preview,
                )
            except Exception:
                await self._belief_repository.rollback()
                raise
            result["signal_type"] = signal_type
            result["trigger_tension_score"] = tension_score
            result["tension_score"] = 0.0
            return result

        preview_decision = await self._preview_belief_revision(payload)
        if preview_decision.action.value != "REINFORCE":
            await self._belief_repository.add_tension_evidence_ids(
                payload.belief_id,
                payload.evidence_memory_ids,
                user_id=payload.user_id,
                commit=False,
            )
            tension_score = await self._belief_repository.increment_tension(
                payload.belief_id,
                self._settings.belief_tension_increment,
                user_id=payload.user_id,
                commit=False,
            )
            await self._belief_repository.commit()
            if tension_score < self._settings.belief_tension_threshold:
                return {
                    "status": "deferred_tension",
                    "belief_id": payload.belief_id,
                    "signal_type": "ambiguous",
                    "preview_action": preview_decision.action.value,
                    "tension_score": tension_score,
                    "threshold": self._settings.belief_tension_threshold,
                }
            accumulated_evidence_ids = await self._belief_repository.get_tension_evidence_ids(
                payload.belief_id,
                user_id=payload.user_id,
            )
            threshold_payload = payload.model_copy(update={"evidence_memory_ids": accumulated_evidence_ids})
            threshold_preview = await self._preview_belief_revision(threshold_payload)
            try:
                popped_evidence_ids = await self._belief_repository.pop_tension_evidence_ids(
                    payload.belief_id,
                    user_id=payload.user_id,
                    commit=False,
                )
                if popped_evidence_ids != accumulated_evidence_ids:
                    raise RuntimeError("Tension evidence buffer changed during threshold revision")
                await self._belief_repository.reset_tension(
                    payload.belief_id,
                    user_id=payload.user_id,
                    commit=False,
                )
                result = await self._execute_belief_revision(
                    threshold_payload,
                    preview_action=threshold_preview,
                )
            except Exception:
                await self._belief_repository.rollback()
                raise
            result["signal_type"] = "ambiguous"
            result["trigger_tension_score"] = tension_score
            result["tension_score"] = 0.0
            return result

        result = await self._execute_belief_revision(payload, preview_action=preview_decision)
        tension_score = await self._post_revision_tension_update(payload, result)
        result["signal_type"] = signal_type
        result["tension_score"] = tension_score
        return result

    async def _execute_belief_revision(
        self,
        payload: RevisionJobPayload,
        preview_action: Any | None = None,
    ) -> dict[str, Any]:
        evidence_rows = await self._load_evidence_rows(payload)
        revision_context = RevisionContext(
            user_id=payload.user_id,
            claim_key=payload.claim_key,
            claim_value=payload.claim_value,
            source_message_id=payload.source_message_id,
            assistant_mode_id=payload.assistant_mode_id,
            workspace_id=payload.workspace_id,
            conversation_id=payload.conversation_id,
            scope=MemoryScope(payload.scope),
            isolated_mode=payload.isolated_mode,
        )
        if preview_action is None:
            result = await self._reviser.revise(
                belief_id=payload.belief_id,
                new_evidence=evidence_rows,
                context=revision_context,
            )
        else:
            result = await self._reviser.apply_previewed_revision(
                belief_id=payload.belief_id,
                new_evidence=evidence_rows,
                context=revision_context,
                decision=preview_action,
                claim_key_already_validated=True,
            )
        return result.model_dump(mode="json")

    async def _classify_revision_signal(
        self,
        payload: RevisionJobPayload,
    ) -> str:
        current_version = await self._belief_repository.get_current_version(
            payload.belief_id,
            payload.user_id,
        )
        if current_version is None:
            return "unknown"
        current_value = self._normalize_claim_value(current_version.get("claim_value_json"))
        next_value = self._normalize_claim_value(self._parse_claim_value(payload.claim_value))
        if current_value != next_value:
            return "contradictory"
        return "ambiguous"

    async def _preview_belief_revision(self, payload: RevisionJobPayload) -> Any:
        evidence_rows = await self._load_evidence_rows(payload)
        return await self._reviser.preview_revision(
            belief_id=payload.belief_id,
            new_evidence=evidence_rows,
            context=RevisionContext(
                user_id=payload.user_id,
                claim_key=payload.claim_key,
                claim_value=payload.claim_value,
                source_message_id=payload.source_message_id,
                assistant_mode_id=payload.assistant_mode_id,
                workspace_id=payload.workspace_id,
                conversation_id=payload.conversation_id,
                scope=MemoryScope(payload.scope),
                isolated_mode=payload.isolated_mode,
            ),
        )

    async def _post_revision_tension_update(
        self,
        payload: RevisionJobPayload,
        result: dict[str, Any],
    ) -> float:
        if result.get("action") != "REINFORCE":
            return await self._belief_repository.get_tension(
                payload.belief_id,
                user_id=payload.user_id,
            )
        tension_score = await self._belief_repository.decrement_tension(
            payload.belief_id,
            self._settings.belief_tension_decrement,
            user_id=payload.user_id,
        )
        if tension_score <= 0.0:
            await self._belief_repository.pop_tension_evidence_ids(
                payload.belief_id,
                user_id=payload.user_id,
            )
        return tension_score

    async def _process_promotion(self, payload: RevisionJobPayload) -> dict[str, Any] | None:
        if self._promotion_blocked_by_policy(payload):
            return {"status": "blocked_by_source_policy", "claim_key": payload.claim_key}
        active_beliefs = await self._matching_beliefs_for_claim_key(payload)
        explicit_user_statement = await self._is_explicit_user_statement(payload)
        same_message_beliefs = [
            belief
            for belief in active_beliefs
            if payload.source_message_id in self._source_message_ids(belief)
        ]
        existing_belief_id = self._select_external_belief_id(active_beliefs, payload)
        if existing_belief_id is not None:
            return await self._revise_existing_belief(
                payload.model_copy(update={"belief_id": existing_belief_id}),
            )

        if same_message_beliefs and explicit_user_statement:
            return {"status": "already_promoted_fast", "claim_key": payload.claim_key}

        stats = await self._belief_repository.count_supporting_evidence(
            payload.user_id,
            payload.claim_key,
            min_conversations=self._settings.promotion_conv_to_ws_min_conversations,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            conversation_id=None,
            incognito=payload.incognito or payload.isolated_mode,
            remember_across_chats=payload.remember_across_chats,
            remember_across_devices=payload.remember_across_devices,
            active_embodiment_id=payload.active_embodiment_id,
            active_realm_id=payload.active_realm_id,
        )
        target_scope = self._promotion_target_scope(payload, active_beliefs, stats)
        if target_scope is None:
            if explicit_user_statement and payload.evidence_memory_ids and not same_message_beliefs:
                target_scope = self._explicit_statement_target_scope(payload)
                if target_scope is None:
                    return {"status": "below_threshold", "claim_key": payload.claim_key}
            else:
                return {"status": "below_threshold", "claim_key": payload.claim_key}

        if any(belief.get("scope") == target_scope.value for belief in same_message_beliefs):
            return {"status": "already_present", "claim_key": payload.claim_key, "scope": target_scope.value}

        promoted = await self._create_promoted_belief(
            payload=payload,
            target_scope=target_scope,
            stats=stats,
            seed_belief=same_message_beliefs[0] if same_message_beliefs else None,
        )
        return {
            "status": "promoted",
            "belief_id": promoted["id"],
            "scope": target_scope.value,
        }

    async def _matching_beliefs_for_claim_key(
        self,
        payload: RevisionJobPayload,
    ) -> list[dict[str, Any]]:
        candidates = await self._belief_repository.find_active_belief_candidates_by_claim_key(
            payload.user_id,
            payload.claim_key,
            user_persona_id=payload.user_persona_id,
            platform_id=payload.platform_id,
            character_id=payload.character_id,
            conversation_id=payload.conversation_id,
            incognito=payload.incognito or payload.isolated_mode,
            remember_across_chats=payload.remember_across_chats,
            remember_across_devices=payload.remember_across_devices,
        )
        matches: list[dict[str, Any]] = []
        for candidate in candidates:
            if await are_claim_keys_equivalent(
                self._llm_client,
                self._classifier_model,
                payload.claim_key,
                str(candidate["claim_key"]),
            ):
                if not self._belief_matches_payload_namespace(candidate, payload):
                    continue
                matches.append(candidate)
        return matches

    def _promotion_target_scope(
        self,
        payload: RevisionJobPayload,
        active_beliefs: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> MemoryScope | None:
        if payload.isolated_mode:
            return None
        current_scope = MemoryScope(payload.scope)
        if current_scope is MemoryScope.CONVERSATION:
            if stats["distinct_conversations"] >= self._settings.promotion_conv_to_ws_min_conversations:
                if payload.workspace_id is not None:
                    return MemoryScope.WORKSPACE
                return MemoryScope.ASSISTANT_MODE
            return None

        if current_scope in {MemoryScope.WORKSPACE, MemoryScope.ASSISTANT_MODE}:
            if stats["distinct_sessions"] < self._settings.promotion_ws_to_global_min_sessions:
                return None
            if self._settings.promotion_require_mode_consistency:
                distinct_modes = {
                    str(item["assistant_mode_id"])
                    for item in active_beliefs
                    if item.get("assistant_mode_id")
                }
                if len(distinct_modes) < 2:
                    return None
            return MemoryScope.GLOBAL_USER

        return None

    @staticmethod
    def _promotion_blocked_by_policy(payload: RevisionJobPayload) -> bool:
        return (
            payload.incognito
            or payload.isolated_mode
            or not payload.remember_across_chats
            or payload.temporary
            or payload.purge_on_close
            or str(payload.sensitivity or "unknown") != "public"
        )

    @staticmethod
    def _explicit_statement_target_scope(payload: RevisionJobPayload) -> MemoryScope | None:
        if payload.isolated_mode:
            return None
        current_scope = MemoryScope(payload.scope)
        if current_scope is MemoryScope.CONVERSATION:
            if payload.workspace_id is not None:
                return MemoryScope.WORKSPACE
            if payload.assistant_mode_id is not None:
                return MemoryScope.ASSISTANT_MODE
            return None
        if current_scope in {MemoryScope.WORKSPACE, MemoryScope.ASSISTANT_MODE}:
            return current_scope
        return None

    async def _create_promoted_belief(
        self,
        *,
        payload: RevisionJobPayload,
        target_scope: MemoryScope,
        stats: dict[str, Any],
        seed_belief: dict[str, Any] | None,
    ) -> dict[str, Any]:
        claim_value = self._parse_claim_value(payload.claim_value)
        canonical_text = self._canonical_text(payload, claim_value, seed_belief)
        now = self._timestamp()
        scope_identifiers = resolve_scope_identifiers(
            target_scope,
            assistant_mode_id=payload.assistant_mode_id,
            workspace_id=payload.workspace_id,
            conversation_id=payload.conversation_id,
        )
        if scope_identifiers is None:
            raise ValueError(f"Cannot resolve identifiers for promoted belief scope {target_scope.value}")
        try:
            created = await self._memory_repository.create_memory_object(
                user_id=payload.user_id,
                workspace_id=scope_identifiers["workspace_id"],
                conversation_id=scope_identifiers["conversation_id"],
                assistant_mode_id=scope_identifiers["assistant_mode_id"],
                object_type=MemoryObjectType.BELIEF,
                scope=target_scope,
                canonical_text=canonical_text,
                payload={
                    "claim_key": payload.claim_key,
                    "claim_value": claim_value,
                    "source_message_ids": [payload.source_message_id],
                    "promotion_stats": stats,
                    "mind_perspective": {
                        "memory_owner_id": payload.active_mind_id,
                        "source_mind_id": payload.source_mind_id or payload.active_mind_id,
                        "mind_topology": payload.mind_topology,
                    },
                    "embodiment": {
                        "active_embodiment_id": payload.active_embodiment_id,
                        "cross_embodiment_mode": payload.cross_embodiment_mode,
                    },
                    "realm": {
                        "active_realm_id": payload.active_realm_id,
                        "cross_realm_mode": payload.cross_realm_mode,
                    },
                },
                extraction_hash=None,
                source_kind=MemorySourceKind.INFERRED,
                confidence=float(seed_belief["confidence"]) if seed_belief is not None else 0.8,
                stability=float(seed_belief["stability"]) if seed_belief is not None else 0.65,
                vitality=float(seed_belief["vitality"]) if seed_belief is not None else 0.25,
                maya_score=float(seed_belief["maya_score"]) if seed_belief is not None else 1.0,
                privacy_level=int(seed_belief["privacy_level"]) if seed_belief is not None else 1,
                status=MemoryStatus.ACTIVE,
                user_persona_id=payload.user_persona_id,
                platform_id=payload.platform_id,
                character_id=payload.character_id if target_scope is MemoryScope.WORKSPACE else None,
                sensitivity=MemorySensitivity(str(payload.sensitivity or "unknown")),
                platform_locked=bool(payload.platform_locked) or not payload.remember_across_devices,
                platform_id_lock=payload.platform_id_lock or (
                    payload.platform_id if bool(payload.platform_locked) or not payload.remember_across_devices else None
                ),
                scope_canonical=self._canonical_scope_for_storage(target_scope).value,
                memory_owner_id=payload.active_mind_id,
                source_mind_id=payload.source_mind_id or payload.active_mind_id,
                embodiment_id=payload.active_embodiment_id,
                realm_id=payload.active_realm_id,
                commit=False,
            )
            await self._belief_repository.create_first_version(
                belief_id=str(created["id"]),
                claim_key=payload.claim_key,
                claim_value=claim_value,
                created_at=now,
                support_count=max(1, int(stats["total_evidence"] or 1)),
                commit=False,
            )
            for evidence_id in payload.evidence_memory_ids:
                evidence = await self._memory_repository.get_memory_object(evidence_id, payload.user_id)
                if evidence is None:
                    continue
                await self._belief_repository.create_memory_link(
                    source_id=str(evidence["id"]),
                    target_id=str(created["id"]),
                    relation_type="supports",
                    confidence=float(evidence.get("confidence", 1.0)),
                    commit=False,
                )
            await self._memory_repository.commit()
            return created
        except Exception:
            await self._memory_repository.rollback()
            raise

    @staticmethod
    def _canonical_scope_for_storage(scope: MemoryScope) -> MemoryScope:
        if scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
            return MemoryScope.CHAT
        if scope in {MemoryScope.WORKSPACE, MemoryScope.CHARACTER}:
            return MemoryScope.CHARACTER
        return MemoryScope.USER

    async def _load_evidence_rows(
        self,
        payload: RevisionJobPayload,
    ) -> list[dict[str, Any]]:
        evidence_rows: list[dict[str, Any]] = []
        for evidence_id in payload.evidence_memory_ids:
            row = await self._memory_repository.get_memory_object(evidence_id, payload.user_id)
            if row is None or row["object_type"] != MemoryObjectType.EVIDENCE.value:
                continue
            if not self._belief_matches_payload_namespace(row, payload):
                continue
            evidence_rows.append(row)
        return evidence_rows

    def _select_external_belief_id(
        self,
        beliefs: list[dict[str, Any]],
        payload: RevisionJobPayload,
    ) -> str | None:
        if payload.isolated_mode:
            return None
        ranked: list[tuple[int, str]] = []
        for belief in beliefs:
            if payload.source_message_id in self._source_message_ids(belief):
                continue
            belief_id = str(belief["belief_id"])
            score = 0
            if belief.get("scope") == payload.scope:
                score += 4
            if payload.conversation_id is not None and belief.get("conversation_id") == payload.conversation_id:
                score += 3
            if payload.workspace_id is not None and belief.get("workspace_id") == payload.workspace_id:
                score += 2
            if belief.get("assistant_mode_id") == payload.assistant_mode_id:
                score += 1
            ranked.append((score, belief_id))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1]

    async def _next_messages(
        self,
        *,
        consumer_name: str,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        reclaimed = await self._storage_backend.stream_claim_idle(
            REVISE_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            min_idle_ms=0 if block_ms == 0 else STREAM_RECLAIM_IDLE_MS,
            count=1,
        )
        if reclaimed:
            return reclaimed
        return await self._storage_backend.stream_read(
            REVISE_STREAM_NAME,
            WORKER_GROUP_NAME,
            consumer_name,
            count=1,
            block_ms=block_ms,
        )

    async def _dead_letter_if_exhausted(
        self,
        message: StreamMessage,
        exc: Exception,
    ) -> bool:
        if message.delivery_count < MAX_STREAM_DELIVERIES:
            return False
        await self._storage_backend.enqueue_job(
            f"dead_letter:{REVISE_STREAM_NAME}",
            {
                "message_id": message.message_id,
                "delivery_count": message.delivery_count,
                "payload": message.payload,
                "error": str(exc),
                "error_details": (
                    list(exc.details)
                    if isinstance(exc, StructuredOutputError)
                    else []
                ),
            },
        )
        await self._storage_backend.stream_ack(
            REVISE_STREAM_NAME,
            WORKER_GROUP_NAME,
            message.message_id,
        )
        await self._job_tracking.mark_dead_lettered(message, exc)
        return True

    async def _message_text(self, payload: RevisionJobPayload) -> str:
        message = await self._message_repository.get_message(payload.source_message_id, payload.user_id)
        if message is None:
            return ""
        return str(message["text"])

    async def _is_explicit_user_statement(self, payload: RevisionJobPayload) -> bool:
        message_text = await self._message_text(payload)
        if not message_text:
            return False
        return await is_explicit_user_statement(
            self._llm_client,
            self._classifier_model,
            message_text,
        )

    @staticmethod
    def _source_message_ids(belief: dict[str, Any]) -> list[str]:
        payload_json = belief.get("payload_json")
        if not isinstance(payload_json, dict):
            return []
        source_ids = payload_json.get("source_message_ids", [])
        if not isinstance(source_ids, list):
            return []
        return [str(item) for item in source_ids]

    @staticmethod
    def _parse_claim_value(claim_value: str) -> Any:
        try:
            return json_utils.loads(claim_value)
        except json_utils.JSONDecodeError:
            return claim_value

    @staticmethod
    def _normalize_claim_value(claim_value: Any) -> str:
        return json_utils.dumps(claim_value, sort_keys=True)

    def _canonical_text(
        self,
        payload: RevisionJobPayload,
        claim_value: Any,
        seed_belief: dict[str, Any] | None,
    ) -> str:
        if seed_belief is not None:
            return str(seed_belief["canonical_text"])
        rendered_value = claim_value if isinstance(claim_value, str) else json_utils.dumps(claim_value, sort_keys=True)
        return f"{payload.claim_key}: {rendered_value}"

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()
