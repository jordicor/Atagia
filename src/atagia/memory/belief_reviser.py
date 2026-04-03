"""Belief revision logic for Phase 2."""

from __future__ import annotations

import json
import html
from enum import Enum
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import MemoryObjectRepository
from atagia.memory.intent_classifier import are_claim_keys_equivalent
from atagia.memory.scope_utils import resolve_scope_identifiers
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage

DEFAULT_REVISION_MODEL = "claude-sonnet-4-6"
STABILITY_DELTA_PER_EVIDENCE = 0.03

REVISION_PROMPT_TEMPLATE = """You are deciding how an assistant memory belief should revise in light of new evidence.

Return JSON only, matching the schema exactly.
Choose exactly one action from the allowed list.

Current belief:
- belief_id: {belief_id}
- claim_key: {claim_key}
- claim_value_json: {claim_value_json}
- condition_json: {condition_json}
- support_count: {support_count}
- contradict_count: {contradict_count}
- scope: {scope}
- confidence: {confidence}
- stability: {stability}

New evidence:
- source_message_id: {source_message_id}
- assistant_mode_id: {assistant_mode_id}
- workspace_id: {workspace_id}
- conversation_id: {conversation_id}
- target_scope: {target_scope}

<new_evidence>
{evidence_block}
</new_evidence>

Allowed actions:
- REINFORCE: new evidence supports the belief.
- WEAKEN: new evidence partially contradicts the belief.
- SUPERSEDE: new evidence replaces the belief with a better successor.
- SPLIT_BY_MODE: the belief only holds for a specific assistant mode.
- SPLIT_BY_SCOPE: the belief only holds in a narrower workspace or conversation scope.
- SPLIT_BY_TIME: the belief was true before but has changed over time.
- MARK_EXCEPTION: the belief generally holds, but there is a specific exception.
- ARCHIVE: the belief is no longer useful or relevant.

Output schema:
- action: one of the allowed actions.
- explanation: a short explanation of why that action fits.
- successor_canonical_text: a concise canonical text for the successor belief when the chosen
  action creates a new belief. Use null for REINFORCE and ARCHIVE.
"""


class RevisionAction(str, Enum):
    REINFORCE = "REINFORCE"
    WEAKEN = "WEAKEN"
    SUPERSEDE = "SUPERSEDE"
    SPLIT_BY_MODE = "SPLIT_BY_MODE"
    SPLIT_BY_SCOPE = "SPLIT_BY_SCOPE"
    SPLIT_BY_TIME = "SPLIT_BY_TIME"
    MARK_EXCEPTION = "MARK_EXCEPTION"
    ARCHIVE = "ARCHIVE"


class RevisionDecision(BaseModel):
    """Structured LLM output for action selection."""

    model_config = ConfigDict(extra="forbid")

    action: RevisionAction
    explanation: str = Field(min_length=1)
    successor_canonical_text: str | None = None


class RevisionContext(BaseModel):
    """Context needed to apply a belief revision."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    claim_key: str
    claim_value: str
    source_message_id: str
    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    scope: MemoryScope


class RevisionResult(BaseModel):
    """Outcome of a belief revision action."""

    model_config = ConfigDict(extra="forbid")

    action: RevisionAction
    belief_id: str
    new_version: int | None = None
    new_belief_ids: list[str] = Field(default_factory=list)
    explanation: str


class BeliefReviser:
    """Applies LLM-selected belief revision actions."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._llm_client = llm_client
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._belief_repository = BeliefRepository(connection, clock)
        resolved_settings = settings or Settings.from_env()
        self._revision_model = (
            resolved_settings.llm_scoring_model
            or resolved_settings.llm_extraction_model
            or DEFAULT_REVISION_MODEL
        )
        self._classifier_model = (
            resolved_settings.llm_classifier_model
            or resolved_settings.llm_scoring_model
            or self._revision_model
        )

    async def revise(
        self,
        belief_id: str,
        new_evidence: list[dict[str, Any]],
        context: RevisionContext | dict[str, Any],
    ) -> RevisionResult:
        revision_context = RevisionContext.model_validate(context)
        belief_row = await self._memory_repository.get_memory_object(belief_id, revision_context.user_id)
        if belief_row is None:
            raise ValueError(f"Unknown belief_id: {belief_id}")
        current_version = await self._belief_repository.get_current_version(
            belief_id,
            revision_context.user_id,
        )
        if current_version is None:
            raise ValueError(f"Belief {belief_id} has no current version")
        if not await are_claim_keys_equivalent(
            self._llm_client,
            self._classifier_model,
            str(current_version["claim_key"]),
            revision_context.claim_key,
        ):
            raise ValueError(
                "Revision claim_key is not semantically equivalent to the current belief claim_key"
            )

        decision = await self._decide_action(
            belief_id=belief_id,
            belief_row=belief_row,
            current_version=current_version,
            new_evidence=new_evidence,
            context=revision_context,
        )
        try:
            if decision.action is RevisionAction.REINFORCE:
                result = await self._apply_reinforce(
                    belief_row=belief_row,
                    current_version=current_version,
                    new_evidence=new_evidence,
                    context=revision_context,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.WEAKEN:
                result = await self._apply_weaken(
                    belief_row=belief_row,
                    current_version=current_version,
                    new_evidence=new_evidence,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.SUPERSEDE:
                result = await self._apply_supersede(
                    belief_row=belief_row,
                    current_version=current_version,
                    new_evidence=new_evidence,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.SPLIT_BY_MODE:
                result = await self._apply_split_by_mode(
                    belief_row=belief_row,
                    current_version=current_version,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.SPLIT_BY_SCOPE:
                result = await self._apply_split_by_scope(
                    belief_row=belief_row,
                    current_version=current_version,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.SPLIT_BY_TIME:
                result = await self._apply_split_by_time(
                    belief_row=belief_row,
                    current_version=current_version,
                    new_evidence=new_evidence,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            elif decision.action is RevisionAction.MARK_EXCEPTION:
                result = await self._apply_mark_exception(
                    belief_row=belief_row,
                    current_version=current_version,
                    new_evidence=new_evidence,
                    context=revision_context,
                    decision=decision,
                    explanation=decision.explanation,
                )
            else:
                result = await self._apply_archive(
                    belief_row=belief_row,
                    user_id=revision_context.user_id,
                    explanation=decision.explanation,
                )
            await self._memory_repository.commit()
            return result
        except Exception:
            await self._memory_repository.rollback()
            raise

    async def _decide_action(
        self,
        *,
        belief_id: str,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
    ) -> RevisionDecision:
        prompt = self._build_prompt(
            belief_id=belief_id,
            belief_row=belief_row,
            current_version=current_version,
            new_evidence=new_evidence,
            context=context,
        )
        request = LLMCompletionRequest(
            model=self._revision_model,
            messages=[
                LLMMessage(role="system", content="Choose a belief revision action as grounded JSON only."),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            response_schema=RevisionDecision.model_json_schema(),
            metadata={
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "assistant_mode_id": context.assistant_mode_id,
                "purpose": "belief_revision",
            },
        )
        return await self._llm_client.complete_structured(request, RevisionDecision)

    def _build_prompt(
        self,
        *,
        belief_id: str,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
    ) -> str:
        evidence_lines = [
            f"- {html.escape(str(item.get('canonical_text', '')))}"
            for item in new_evidence
        ]
        evidence_block = "\n".join(evidence_lines) or "- (no extracted evidence rows)"
        return REVISION_PROMPT_TEMPLATE.format(
            belief_id=html.escape(belief_id),
            claim_key=html.escape(str(current_version["claim_key"])),
            claim_value_json=html.escape(self._json_text(current_version["claim_value_json"])),
            condition_json=html.escape(self._json_text(current_version["condition_json"])),
            support_count=html.escape(str(current_version["support_count"])),
            contradict_count=html.escape(str(current_version["contradict_count"])),
            scope=html.escape(str(belief_row["scope"])),
            confidence=html.escape(str(belief_row["confidence"])),
            stability=html.escape(str(belief_row["stability"])),
            source_message_id=html.escape(context.source_message_id),
            assistant_mode_id=html.escape(context.assistant_mode_id),
            workspace_id=html.escape(str(context.workspace_id)),
            conversation_id=html.escape(str(context.conversation_id)),
            target_scope=html.escape(context.scope.value),
            evidence_block=evidence_block,
        )

    async def _apply_reinforce(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
        explanation: str,
    ) -> RevisionResult:
        evidence_delta = max(1, len(new_evidence))
        new_support = int(current_version["support_count"]) + evidence_delta
        new_version_number: int | None = None
        current_claim_value = current_version["claim_value_json"]
        next_claim_value = self._parse_claim_value(context.claim_value)
        if self._json_text(current_claim_value) != self._json_text(next_claim_value):
            new_version_number = int(current_version["version"]) + 1
            await self._belief_repository.create_new_version(
                belief_id=str(belief_row["id"]),
                user_id=context.user_id,
                version=new_version_number,
                claim_key=str(current_version["claim_key"]),
                claim_value=next_claim_value,
                condition=current_version["condition_json"],
                support_count=new_support,
                contradict_count=int(current_version["contradict_count"]),
                supersedes_version=int(current_version["version"]),
                created_at=self._timestamp(),
                commit=False,
            )
        else:
            await self._connection.execute(
                """
                UPDATE belief_versions
                SET support_count = ?
                WHERE belief_id IN (
                    SELECT id
                    FROM memory_objects
                    WHERE id = ?
                      AND user_id = ?
                )
                  AND is_current = 1
                """,
                (new_support, belief_row["id"], context.user_id),
            )
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET confidence = ?, stability = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                min(1.0, float(belief_row["confidence"]) + (0.05 * evidence_delta)),
                min(1.0, float(belief_row["stability"]) + (STABILITY_DELTA_PER_EVIDENCE * evidence_delta)),
                self._timestamp(),
                belief_row["id"],
                context.user_id,
            ),
        )
        await self._link_evidence(new_evidence, str(belief_row["id"]), "reinforces")
        return RevisionResult(
            action=RevisionAction.REINFORCE,
            belief_id=str(belief_row["id"]),
            new_version=new_version_number,
            explanation=explanation,
        )

    async def _apply_weaken(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        del decision
        del context
        evidence_delta = max(1, len(new_evidence))
        support = int(current_version["support_count"])
        new_contradict = int(current_version["contradict_count"]) + evidence_delta
        new_confidence = float(belief_row["confidence"]) * (
            support / max(1, support + new_contradict)
        )
        await self._connection.execute(
            """
            UPDATE belief_versions
            SET contradict_count = ?
            WHERE belief_id IN (
                SELECT id
                FROM memory_objects
                WHERE id = ?
                  AND user_id = ?
            )
              AND is_current = 1
            """,
            (new_contradict, belief_row["id"], str(belief_row["user_id"])),
        )
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET confidence = ?, stability = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                max(0.0, min(1.0, new_confidence)),
                max(0.0, float(belief_row["stability"]) - (STABILITY_DELTA_PER_EVIDENCE * evidence_delta)),
                self._timestamp(),
                belief_row["id"],
                str(belief_row["user_id"]),
            ),
        )
        await self._link_evidence(new_evidence, str(belief_row["id"]), "weakens")
        return RevisionResult(
            action=RevisionAction.WEAKEN,
            belief_id=str(belief_row["id"]),
            explanation=explanation,
        )

    async def _apply_supersede(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        successor = await self._create_belief_memory(
            base_belief=belief_row,
            claim_key=str(current_version["claim_key"]),
            claim_value=self._parse_claim_value(context.claim_value),
            scope=context.scope,
            canonical_text=self._successor_text(decision, belief_row),
            source_message_id=context.source_message_id,
            assistant_mode_id=context.assistant_mode_id,
            workspace_id=context.workspace_id,
            conversation_id=context.conversation_id,
            support_count=int(current_version["support_count"]) + max(1, len(new_evidence)),
            contradict_count=int(current_version["contradict_count"]),
            condition={},
        )
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                MemoryStatus.SUPERSEDED.value,
                self._timestamp(),
                belief_row["id"],
                context.user_id,
            ),
        )
        await self._belief_repository.create_memory_link(
            source_id=str(successor["id"]),
            target_id=str(belief_row["id"]),
            relation_type="supersedes",
            confidence=float(successor["confidence"]),
            commit=False,
        )
        await self._link_evidence(new_evidence, str(successor["id"]), "reinforces")
        return RevisionResult(
            action=RevisionAction.SUPERSEDE,
            belief_id=str(belief_row["id"]),
            new_version=1,
            new_belief_ids=[str(successor["id"])],
            explanation=explanation,
        )

    async def _apply_split_by_mode(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        successor = await self._create_belief_memory(
            base_belief=belief_row,
            claim_key=str(current_version["claim_key"]),
            claim_value=current_version["claim_value_json"],
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text=self._successor_text(decision, belief_row),
            source_message_id=context.source_message_id,
            assistant_mode_id=context.assistant_mode_id,
            workspace_id=context.workspace_id,
            conversation_id=context.conversation_id,
            support_count=max(1, int(current_version["support_count"])),
            contradict_count=int(current_version["contradict_count"]),
            condition={"mode": context.assistant_mode_id},
        )
        await self._archive_belief(str(belief_row["id"]), context.user_id)
        await self._belief_repository.create_memory_link(
            source_id=str(successor["id"]),
            target_id=str(belief_row["id"]),
            relation_type="derived_from",
            confidence=float(successor["confidence"]),
            commit=False,
        )
        return RevisionResult(
            action=RevisionAction.SPLIT_BY_MODE,
            belief_id=str(belief_row["id"]),
            new_version=1,
            new_belief_ids=[str(successor["id"])],
            explanation=explanation,
        )

    async def _apply_split_by_scope(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        successor = await self._create_belief_memory(
            base_belief=belief_row,
            claim_key=str(current_version["claim_key"]),
            claim_value=current_version["claim_value_json"],
            scope=context.scope,
            canonical_text=self._successor_text(decision, belief_row),
            source_message_id=context.source_message_id,
            assistant_mode_id=context.assistant_mode_id,
            workspace_id=context.workspace_id,
            conversation_id=context.conversation_id,
            support_count=max(1, int(current_version["support_count"])),
            contradict_count=int(current_version["contradict_count"]),
            condition={"scope": context.scope.value},
        )
        await self._archive_belief(str(belief_row["id"]), context.user_id)
        await self._belief_repository.create_memory_link(
            source_id=str(successor["id"]),
            target_id=str(belief_row["id"]),
            relation_type="derived_from",
            confidence=float(successor["confidence"]),
            commit=False,
        )
        return RevisionResult(
            action=RevisionAction.SPLIT_BY_SCOPE,
            belief_id=str(belief_row["id"]),
            new_version=1,
            new_belief_ids=[str(successor["id"])],
            explanation=explanation,
        )

    async def _apply_split_by_time(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        now = self._timestamp()
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET valid_to = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (now, now, belief_row["id"], context.user_id),
        )
        successor = await self._create_belief_memory(
            base_belief=belief_row,
            claim_key=str(current_version["claim_key"]),
            claim_value=self._parse_claim_value(context.claim_value),
            scope=context.scope,
            canonical_text=self._successor_text(decision, belief_row),
            source_message_id=context.source_message_id,
            assistant_mode_id=context.assistant_mode_id,
            workspace_id=context.workspace_id,
            conversation_id=context.conversation_id,
            support_count=max(1, int(current_version["support_count"])),
            contradict_count=0,
            condition={},
            valid_from=now,
        )
        await self._belief_repository.create_memory_link(
            source_id=str(successor["id"]),
            target_id=str(belief_row["id"]),
            relation_type="supersedes",
            confidence=float(successor["confidence"]),
            commit=False,
        )
        return RevisionResult(
            action=RevisionAction.SPLIT_BY_TIME,
            belief_id=str(belief_row["id"]),
            new_version=1,
            new_belief_ids=[str(successor["id"])],
            explanation=explanation,
        )

    async def _apply_mark_exception(
        self,
        *,
        belief_row: dict[str, Any],
        current_version: dict[str, Any],
        new_evidence: list[dict[str, Any]],
        context: RevisionContext,
        decision: RevisionDecision,
        explanation: str,
    ) -> RevisionResult:
        exception = await self._create_belief_memory(
            base_belief=belief_row,
            claim_key=str(current_version["claim_key"]),
            claim_value=self._parse_claim_value(context.claim_value),
            scope=context.scope,
            canonical_text=self._successor_text(decision, belief_row, prefix="Exception: "),
            source_message_id=context.source_message_id,
            assistant_mode_id=context.assistant_mode_id,
            workspace_id=context.workspace_id,
            conversation_id=context.conversation_id,
            support_count=1,
            contradict_count=0,
            condition=self._exception_condition(context),
        )
        await self._belief_repository.create_memory_link(
            source_id=str(exception["id"]),
            target_id=str(belief_row["id"]),
            relation_type="exception_to",
            confidence=float(exception["confidence"]),
            commit=False,
        )
        return RevisionResult(
            action=RevisionAction.MARK_EXCEPTION,
            belief_id=str(belief_row["id"]),
            new_version=1,
            new_belief_ids=[str(exception["id"])],
            explanation=explanation,
        )

    async def _apply_archive(
        self,
        *,
        belief_row: dict[str, Any],
        user_id: str,
        explanation: str,
    ) -> RevisionResult:
        await self._archive_belief(str(belief_row["id"]), user_id)
        return RevisionResult(
            action=RevisionAction.ARCHIVE,
            belief_id=str(belief_row["id"]),
            explanation=explanation,
        )

    async def _create_belief_memory(
        self,
        *,
        base_belief: dict[str, Any],
        claim_key: str,
        claim_value: Any,
        scope: MemoryScope,
        canonical_text: str,
        source_message_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        support_count: int,
        contradict_count: int,
        condition: dict[str, Any],
        valid_from: str | None = None,
    ) -> dict[str, Any]:
        payload = dict(base_belief.get("payload_json") or {})
        payload["claim_key"] = claim_key
        payload["claim_value"] = claim_value
        payload["source_message_ids"] = [source_message_id]
        scope_identifiers = resolve_scope_identifiers(
            scope,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if scope_identifiers is None:
            raise ValueError(f"Cannot resolve identifiers for belief scope {scope.value}")
        created = await self._memory_repository.create_memory_object(
            user_id=str(base_belief["user_id"]),
            workspace_id=scope_identifiers["workspace_id"],
            conversation_id=scope_identifiers["conversation_id"],
            assistant_mode_id=scope_identifiers["assistant_mode_id"],
            object_type=MemoryObjectType.BELIEF,
            scope=scope,
            canonical_text=canonical_text,
            payload=payload,
            extraction_hash=None,
            source_kind=MemorySourceKind.INFERRED,
            confidence=float(base_belief["confidence"]),
            stability=float(base_belief["stability"]),
            vitality=float(base_belief["vitality"]),
            maya_score=float(base_belief["maya_score"]),
            privacy_level=int(base_belief["privacy_level"]),
            status=MemoryStatus.ACTIVE,
            valid_from=valid_from,
            valid_to=None,
            commit=False,
        )
        await self._belief_repository.create_first_version(
            belief_id=str(created["id"]),
            claim_key=claim_key,
            claim_value=claim_value,
            created_at=str(created["created_at"]),
            condition=condition,
            support_count=support_count,
            contradict_count=contradict_count,
            commit=False,
        )
        return created

    async def _link_evidence(
        self,
        evidence_rows: list[dict[str, Any]],
        target_belief_id: str,
        relation_type: str,
    ) -> None:
        for evidence in evidence_rows:
            await self._belief_repository.create_memory_link(
                source_id=str(evidence["id"]),
                target_id=target_belief_id,
                relation_type=relation_type,
                confidence=float(evidence.get("confidence", 1.0)),
                commit=False,
            )

    async def _archive_belief(self, belief_id: str, user_id: str) -> None:
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (MemoryStatus.ARCHIVED.value, self._timestamp(), belief_id, user_id),
        )

    def _exception_condition(self, context: RevisionContext) -> dict[str, Any]:
        condition: dict[str, Any] = {"scope": context.scope.value}
        if context.assistant_mode_id:
            condition["mode"] = context.assistant_mode_id
        if context.workspace_id is not None:
            condition["workspace_id"] = context.workspace_id
        if context.conversation_id is not None:
            condition["conversation_id"] = context.conversation_id
        return condition

    def _successor_text(
        self,
        decision: RevisionDecision,
        belief_row: dict[str, Any],
        *,
        prefix: str = "",
    ) -> str:
        if decision.successor_canonical_text and decision.successor_canonical_text.strip():
            return f"{prefix}{decision.successor_canonical_text.strip()}"
        return f"{prefix}{belief_row['canonical_text']}"

    @staticmethod
    def _parse_claim_value(claim_value: str) -> Any:
        try:
            return json.loads(claim_value)
        except json.JSONDecodeError:
            return claim_value

    @staticmethod
    def _json_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()
