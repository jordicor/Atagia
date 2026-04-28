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
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
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
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.embeddings import EmbeddingIndex, NoneBackend
from atagia.services.llm_client import LLMClient, StructuredOutputError
from atagia.services.model_resolution import resolve_component_model

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
        self._llm_client = llm_client
        self._embedding_index = embedding_index or NoneBackend()
        self._classifier_model = resolve_component_model(
            self._settings,
            "intent_classifier",
        )
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._belief_repository = BeliefRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)
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
                await self.process_job(message.payload)
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
                return await self._revise_existing_belief(job_payload)
            return await self._process_promotion(job_payload)
        finally:
            await self._storage_backend.release_lock(
                belief_lock_key(lock_subject),
                lock_token,
            )

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
            accumulated_evidence_ids = await self._belief_repository.pop_tension_evidence_ids(
                payload.belief_id,
                user_id=payload.user_id,
                commit=False,
            )
            try:
                await self._belief_repository.reset_tension(
                    payload.belief_id,
                    user_id=payload.user_id,
                    commit=False,
                )
                result = await self._execute_belief_revision(
                    payload.model_copy(update={"evidence_memory_ids": accumulated_evidence_ids}),
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
            accumulated_evidence_ids = await self._belief_repository.pop_tension_evidence_ids(
                payload.belief_id,
                user_id=payload.user_id,
                commit=False,
            )
            threshold_payload = payload.model_copy(update={"evidence_memory_ids": accumulated_evidence_ids})
            try:
                threshold_preview = await self._preview_belief_revision(threshold_payload)
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
        evidence_rows = await self._load_evidence_rows(payload.user_id, payload.evidence_memory_ids)
        revision_context = RevisionContext(
            user_id=payload.user_id,
            claim_key=payload.claim_key,
            claim_value=payload.claim_value,
            source_message_id=payload.source_message_id,
            assistant_mode_id=payload.assistant_mode_id,
            workspace_id=payload.workspace_id,
            conversation_id=payload.conversation_id,
            scope=MemoryScope(payload.scope),
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
        evidence_rows = await self._load_evidence_rows(payload.user_id, payload.evidence_memory_ids)
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
        active_beliefs = await self._matching_beliefs_for_claim_key(
            payload.user_id,
            payload.claim_key,
        )
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
        user_id: str,
        claim_key: str,
    ) -> list[dict[str, Any]]:
        candidates = await self._belief_repository.find_active_belief_candidates_by_claim_key(
            user_id,
            claim_key,
        )
        matches: list[dict[str, Any]] = []
        for candidate in candidates:
            if await are_claim_keys_equivalent(
                self._llm_client,
                self._classifier_model,
                claim_key,
                str(candidate["claim_key"]),
            ):
                matches.append(candidate)
        return matches

    def _promotion_target_scope(
        self,
        payload: RevisionJobPayload,
        active_beliefs: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> MemoryScope | None:
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
    def _explicit_statement_target_scope(payload: RevisionJobPayload) -> MemoryScope | None:
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
                },
                extraction_hash=None,
                source_kind=MemorySourceKind.INFERRED,
                confidence=float(seed_belief["confidence"]) if seed_belief is not None else 0.8,
                stability=float(seed_belief["stability"]) if seed_belief is not None else 0.65,
                vitality=float(seed_belief["vitality"]) if seed_belief is not None else 0.25,
                maya_score=float(seed_belief["maya_score"]) if seed_belief is not None else 1.0,
                privacy_level=int(seed_belief["privacy_level"]) if seed_belief is not None else 1,
                status=MemoryStatus.ACTIVE,
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

    async def _load_evidence_rows(
        self,
        user_id: str,
        evidence_ids: list[str],
    ) -> list[dict[str, Any]]:
        evidence_rows: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            row = await self._memory_repository.get_memory_object(evidence_id, user_id)
            if row is None or row["object_type"] != MemoryObjectType.EVIDENCE.value:
                continue
            evidence_rows.append(row)
        return evidence_rows

    def _select_external_belief_id(
        self,
        beliefs: list[dict[str, Any]],
        payload: RevisionJobPayload,
    ) -> str | None:
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
