"""Tests for evaluation metric computation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.core.retrieval_event_repository import MemoryFeedbackRepository, RetrievalEventRepository
from atagia.memory.metrics_computer import MetricsComputer
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from tests.turn_telemetry_support import sample_turn_telemetry
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus, TurnSurface
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"


class CCRProvider(LLMProvider):
    name = "metrics-computer-tests"

    def __init__(self, outputs: list[dict[str, object]] | None = None, *, fail: bool = False) -> None:
        self.outputs = list(outputs or [])
        self.fail = fail
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("synthetic ccr failure")
        if not self.outputs:
            raise AssertionError("No queued CCR output left")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.outputs.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in metrics computer tests")


async def _build_runtime() -> tuple[
    aiosqlite.Connection,
    FrozenClock,
    MessageRepository,
    MemoryObjectRepository,
    RetrievalEventRepository,
    MemoryFeedbackRepository,
    BeliefRepository,
    MetricsComputer,
]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 31, 8, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    feedback = MemoryFeedbackRepository(connection, clock)
    beliefs = BeliefRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_dbg_1", "usr_1", "wrk_1", "coding_debug", "Debug Chat")
    await conversations.create_conversation("cnv_qa_1", "usr_1", None, "general_qa", "QA Chat")
    await conversations.create_conversation("cnv_dbg_2", "usr_2", None, "coding_debug", "Other Chat")
    return connection, clock, messages, memories, events, feedback, beliefs, MetricsComputer(connection, clock)


def _dt(hour: int, minute: int, second: int = 0, *, day: int = 31) -> datetime:
    return datetime(2026, 3, day, hour, minute, second, tzinfo=timezone.utc)


async def _create_message_at(
    messages: MessageRepository,
    clock: FrozenClock,
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    seq: int,
    text: str,
    at: datetime,
) -> dict[str, object]:
    clock.current = at
    return await messages.create_message(message_id, conversation_id, role, seq, text, len(text.split()), {})


async def _create_memory_at(
    memories: MemoryObjectRepository,
    clock: FrozenClock,
    *,
    memory_id: str,
    user_id: str,
    object_type: MemoryObjectType,
    scope: MemoryScope,
    canonical_text: str,
    assistant_mode_id: str,
    at: datetime,
    conversation_id: str | None = None,
    workspace_id: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> dict[str, object]:
    clock.current = at
    return await memories.create_memory_object(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED if object_type is not MemoryObjectType.BELIEF else MemorySourceKind.INFERRED,
        confidence=0.8,
        privacy_level=0,
        status=status,
        memory_id=memory_id,
    )


async def _create_event_at(
    events: RetrievalEventRepository,
    *,
    event_id: str,
    user_id: str,
    conversation_id: str,
    request_message_id: str,
    response_message_id: str | None,
    assistant_mode_id: str,
    selected_memory_ids: list[str],
    at: datetime,
    contract_block: str = "",
    items_included: int | None = None,
    items_dropped: int = 0,
    total_tokens_estimate: int = 0,
    outcome: dict[str, object] | None = None,
    surface: TurnSurface = TurnSurface.CHAT,
    retrieval_duration_ms: float | None = None,
) -> dict[str, object]:
    telemetry = sample_turn_telemetry(surface)
    if retrieval_duration_ms is not None:
        # A turn contains its retrieval, and TurnTelemetry enforces that, so the
        # turn duration has to move with the retrieval duration it wraps.
        telemetry = replace(
            telemetry,
            retrieval_duration_ms=retrieval_duration_ms,
            turn_to_event_write_wall_ms=retrieval_duration_ms + telemetry.turn_to_event_write_wall_ms,
        )
    return await events.create_event(
        {
            "id": event_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "request_message_id": request_message_id,
            "response_message_id": response_message_id,
            "assistant_mode_id": assistant_mode_id,
            "retrieval_plan_json": {"fts_queries": ["retry"]},
            "selected_memory_ids_json": selected_memory_ids,
            "context_view_json": {
                "contract_block": contract_block,
                "selected_memory_ids": selected_memory_ids,
                "items_included": len(selected_memory_ids) if items_included is None else items_included,
                "items_dropped": items_dropped,
                "total_tokens_estimate": total_tokens_estimate,
            },
            "outcome_json": outcome or {},
            "created_at": at.isoformat(),
        },
        telemetry=telemetry,
    )


async def _strip_persisted_telemetry(
    connection: aiosqlite.Connection,
    event_id: str,
    user_id: str,
) -> None:
    """Turn a seeded event into a pre-migration row.

    Rows written before the turn-telemetry migration have NULL measurement
    columns, and the repository refuses to create one that way on purpose, so
    the only honest way to exercise that path is to clear the columns after the
    fact. ``turn_surface`` keeps its backfilled 'chat' value, exactly like a
    real pre-migration row.
    """
    await connection.execute(
        """
        UPDATE retrieval_events
        SET turn_to_event_write_wall_ms = NULL,
            retrieval_duration_ms = NULL,
            llm_total_calls = NULL,
            llm_failed_calls = NULL,
            llm_total_latency_ms = NULL,
            llm_by_purpose_json = NULL,
            stage_timings_ms_json = NULL
        WHERE id = ?
          AND user_id = ?
        """,
        (event_id, user_id),
    )
    await connection.commit()


async def _create_feedback_at(
    feedback: MemoryFeedbackRepository,
    clock: FrozenClock,
    *,
    retrieval_event_id: str,
    memory_id: str,
    user_id: str,
    feedback_type: str,
    at: datetime,
) -> None:
    clock.current = at
    await feedback.create_feedback(
        retrieval_event_id=retrieval_event_id,
        memory_id=memory_id,
        user_id=user_id,
        feedback_type=feedback_type,
        score=None,
        metadata={},
    )


@pytest.mark.asyncio
async def test_compute_mur_correct_ratio_with_mixed_feedback_and_user_filter() -> None:
    connection, clock, messages, memories, events, feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_1",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Retry advice",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_dbg_1",
            workspace_id="wrk_1",
            at=_dt(9, 0),
        )
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_2",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Queue advice",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_dbg_1",
            workspace_id="wrk_1",
            at=_dt(9, 1),
        )
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_3",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Backoff advice",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_dbg_1",
            workspace_id="wrk_1",
            at=_dt(9, 2),
        )
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_other",
            user_id="usr_2",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Other user memory",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_dbg_2",
            at=_dt(9, 3),
        )

        for seq in range(1, 9):
            conversation_id = "cnv_dbg_1" if seq <= 6 else "cnv_dbg_2"
            role = "user" if seq % 2 == 1 else "assistant"
            await _create_message_at(
                messages,
                clock,
                message_id=f"msg_{seq}",
                conversation_id=conversation_id,
                role=role,
                seq=(seq if conversation_id == "cnv_dbg_1" else seq - 6),
                text=f"message {seq}",
                at=_dt(10, seq),
            )

        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_1"],
            at=_dt(10, 10),
        )
        await _create_event_at(
            events,
            event_id="ret_2",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_3",
            response_message_id="msg_4",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_2"],
            at=_dt(10, 20),
        )
        await _create_event_at(
            events,
            event_id="ret_3",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_5",
            response_message_id="msg_6",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_3"],
            at=_dt(10, 30),
        )
        await _create_event_at(
            events,
            event_id="ret_4",
            user_id="usr_2",
            conversation_id="cnv_dbg_2",
            request_message_id="msg_7",
            response_message_id="msg_8",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_other"],
            at=_dt(10, 40),
        )

        await _create_feedback_at(
            feedback,
            clock,
            retrieval_event_id="ret_1",
            memory_id="mem_1",
            user_id="usr_1",
            feedback_type="useful",
            at=_dt(10, 11),
        )
        await _create_feedback_at(
            feedback,
            clock,
            retrieval_event_id="ret_2",
            memory_id="mem_2",
            user_id="usr_1",
            feedback_type="irrelevant",
            at=_dt(10, 21),
        )
        await _create_feedback_at(
            feedback,
            clock,
            retrieval_event_id="ret_4",
            memory_id="mem_other",
            user_id="usr_2",
            feedback_type="used",
            at=_dt(10, 41),
        )

        result = await computer.compute_mur("usr_1", "coding_debug", "2026-03-31")
        other_user_result = await computer.compute_mur("usr_2", "coding_debug", "2026-03-31")

        assert result.value == pytest.approx(1 / 3)
        assert result.sample_count == 3
        assert other_user_result.value == pytest.approx(1.0)
        assert other_user_result.sample_count == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_mur_returns_zero_when_no_feedback_exists() -> None:
    connection, clock, messages, memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_1",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Transient memory",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_dbg_1",
            workspace_id="wrk_1",
            at=_dt(9, 0),
        )
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Try this", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_1"],
            at=_dt(10, 2),
        )

        result = await computer.compute_mur("usr_1", "coding_debug", "2026-03-31")

        assert result.value == 0.0
        assert result.sample_count == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_ipr_counts_irrelevant_and_intrusive_feedback() -> None:
    connection, clock, messages, memories, events, feedback, _beliefs, computer = await _build_runtime()
    try:
        for index in range(1, 4):
            await _create_memory_at(
                memories,
                clock,
                memory_id=f"mem_{index}",
                user_id="usr_1",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text=f"Memory {index}",
                assistant_mode_id="coding_debug",
                conversation_id="cnv_dbg_1",
                workspace_id="wrk_1",
                at=_dt(9, index),
            )
            await _create_message_at(
                messages,
                clock,
                message_id=f"msg_{index * 2 - 1}",
                conversation_id="cnv_dbg_1",
                role="user",
                seq=index * 2 - 1,
                text=f"user {index}",
                at=_dt(10, index * 2 - 1),
            )
            await _create_message_at(
                messages,
                clock,
                message_id=f"msg_{index * 2}",
                conversation_id="cnv_dbg_1",
                role="assistant",
                seq=index * 2,
                text=f"assistant {index}",
                at=_dt(10, index * 2),
            )
            await _create_event_at(
                events,
                event_id=f"ret_{index}",
                user_id="usr_1",
                conversation_id="cnv_dbg_1",
                request_message_id=f"msg_{index * 2 - 1}",
                response_message_id=f"msg_{index * 2}",
                assistant_mode_id="coding_debug",
                selected_memory_ids=[f"mem_{index}"],
                at=_dt(10, 10 + index),
            )

        await _create_feedback_at(feedback, clock, retrieval_event_id="ret_1", memory_id="mem_1", user_id="usr_1", feedback_type="irrelevant", at=_dt(10, 20))
        await _create_feedback_at(feedback, clock, retrieval_event_id="ret_2", memory_id="mem_2", user_id="usr_1", feedback_type="intrusive", at=_dt(10, 21))

        result = await computer.compute_ipr("usr_1", "coding_debug", "2026-03-31")

        assert result.value == pytest.approx(2 / 3)
        assert result.sample_count == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_slr_uses_explicit_scope_feedback_after_mode_softening() -> None:
    connection, clock, messages, memories, events, feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_auto",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Mode scoped memory",
            assistant_mode_id="general_qa",
            at=_dt(9, 0),
        )
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_explicit",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Conversation memory",
            assistant_mode_id="general_qa",
            conversation_id="cnv_qa_1",
            at=_dt(9, 1),
        )
        await _create_memory_at(
            memories,
            clock,
            memory_id="mem_ok",
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Allowed memory",
            assistant_mode_id="general_qa",
            conversation_id="cnv_qa_1",
            at=_dt(9, 2),
        )
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_qa_1", role="user", seq=1, text="Question", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_qa_1", role="assistant", seq=2, text="Answer", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_qa_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="general_qa",
            selected_memory_ids=["mem_auto", "mem_explicit", "mem_ok"],
            at=_dt(10, 2),
        )

        await _create_feedback_at(feedback, clock, retrieval_event_id="ret_1", memory_id="mem_auto", user_id="usr_1", feedback_type="wrong_scope", at=_dt(10, 3))
        await _create_feedback_at(feedback, clock, retrieval_event_id="ret_1", memory_id="mem_explicit", user_id="usr_1", feedback_type="wrong_scope", at=_dt(10, 4))

        result = await computer.compute_slr("usr_1", "general_qa", "2026-03-31")

        assert result.value == pytest.approx(1.0)
        assert result.sample_count == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_bder_detects_superseded_and_outdated_beliefs() -> None:
    connection, clock, messages, memories, events, _feedback, beliefs, computer = await _build_runtime()
    try:
        mem_superseded = await _create_memory_at(
            memories,
            clock,
            memory_id="mem_sup",
            user_id="usr_1",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Old debugging belief",
            assistant_mode_id="coding_debug",
            at=_dt(10, 0),
            status=MemoryStatus.SUPERSEDED,
        )
        mem_versioned = await _create_memory_at(
            memories,
            clock,
            memory_id="mem_ver",
            user_id="usr_1",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Versioned belief",
            assistant_mode_id="coding_debug",
            at=_dt(10, 1),
        )
        mem_current = await _create_memory_at(
            memories,
            clock,
            memory_id="mem_ok",
            user_id="usr_1",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Current belief",
            assistant_mode_id="coding_debug",
            at=_dt(10, 2),
        )
        await beliefs.create_first_version(
            belief_id=str(mem_superseded["id"]),
            claim_key="debug.preference",
            claim_value={"style": "old"},
            created_at=_dt(10, 0).isoformat(),
        )
        await beliefs.create_first_version(
            belief_id=str(mem_versioned["id"]),
            claim_key="debug.preference",
            claim_value={"style": "v1"},
            created_at=_dt(10, 1).isoformat(),
        )
        await beliefs.create_first_version(
            belief_id=str(mem_current["id"]),
            claim_key="debug.preference",
            claim_value={"style": "current"},
            created_at=_dt(10, 2).isoformat(),
        )
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Help", at=_dt(11, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Answer", at=_dt(11, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=["mem_sup", "mem_ver", "mem_ok"],
            at=_dt(11, 30),
        )
        await beliefs.create_new_version(
            belief_id="mem_ver",
            user_id="usr_1",
            version=2,
            claim_key="debug.preference",
            claim_value={"style": "v2"},
            condition=None,
            support_count=1,
            contradict_count=0,
            supersedes_version=1,
            created_at=_dt(12, 0).isoformat(),
        )

        result = await computer.compute_bder("usr_1", "coding_debug", "2026-03-31")

        assert result.value == pytest.approx(2 / 3)
        assert result.sample_count == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_ccr_llm_evaluation_produces_compliance_score() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need terse help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Short answer.", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            contract_block="[Interaction Contract]\n- brevity: high",
            at=_dt(10, 2),
        )
        provider = CCRProvider([{"compliance_score": 0.82, "reasoning": "Compliant."}])
        llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        )

        result = await computer.compute_ccr("usr_1", "coding_debug", "2026-03-31", llm_client)

        assert result.value == pytest.approx(0.82)
        assert result.sample_count == 1
        assert "<contract_block>" in provider.requests[0].messages[1].content
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_ccr_ignores_provider_extra_fields() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need terse help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Short answer.", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            contract_block="[Interaction Contract]\n- brevity: high",
            at=_dt(10, 2),
        )
        provider = CCRProvider(
            [
                {
                    "compliance_score": 0.82,
                    "reasoning": "Compliant.",
                    "provider_notes": "ignored",
                }
            ]
        )
        llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        )

        result = await computer.compute_ccr("usr_1", "coding_debug", "2026-03-31", llm_client)

        assert result.value == pytest.approx(0.82)
        assert result.sample_count == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_ccr_handles_llm_failure_gracefully() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need terse help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Short answer.", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            contract_block="[Interaction Contract]\n- brevity: high",
            at=_dt(10, 2),
        )

        result = await computer.compute_ccr(
            "usr_1",
            "coding_debug",
            "2026-03-31",
            LLMClient(provider_name="metrics-computer-tests", providers=[CCRProvider(fail=True)]),
        )

        assert result.value == 0.0
        assert result.sample_count == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_ccr_logs_structured_failure_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need terse help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Short answer.", at=_dt(10, 1))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            contract_block="[Interaction Contract]\n- brevity: high",
            at=_dt(10, 2),
        )
        provider = CCRProvider([{"compliance_score": 0.82}])
        llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        )

        with caplog.at_level(logging.WARNING, logger="atagia.memory.metrics_computer"):
            result = await computer.compute_ccr("usr_1", "coding_debug", "2026-03-31", llm_client)

        structured_records = [
            record
            for record in caplog.records
            if "CCR evaluation structured-output fallback" in record.getMessage()
        ]
        assert result.value == 0.0
        assert result.sample_count == 0
        assert len(structured_records) == 1
        assert structured_records[0].exc_info is None
        assert "Field required" in structured_records[0].getMessage()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_compute_system_metrics_computes_latency_counts_and_token_usage() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Answer", at=_dt(10, 1))
        await _create_message_at(messages, clock, message_id="msg_3", conversation_id="cnv_dbg_1", role="user", seq=3, text="More help", at=_dt(10, 10))
        await _create_message_at(messages, clock, message_id="msg_4", conversation_id="cnv_dbg_1", role="assistant", seq=4, text="More answer", at=_dt(10, 11))
        await _create_event_at(
            events,
            event_id="ret_1",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 0, 2),
            items_included=2,
            items_dropped=1,
            total_tokens_estimate=100,
            outcome={"cold_start": True, "zero_candidates": False},
            retrieval_duration_ms=20.0,
        )
        await _create_event_at(
            events,
            event_id="ret_2",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_3",
            response_message_id="msg_4",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 10, 5),
            items_included=4,
            items_dropped=0,
            total_tokens_estimate=200,
            outcome={"cold_start": False, "zero_candidates": True},
            retrieval_duration_ms=40.0,
        )

        metrics = await computer.compute_system_metrics("2026-03-31")

        # Both events measured their own retrieval stage, so the average is
        # mean(20.0, 40.0) and not the 3500 ms the message-timestamp gap would
        # have derived (2 s and 5 s after their request messages).
        assert metrics["retrieval_stage_latency_ms"].value == pytest.approx(30.0)
        assert metrics["retrieval_stage_latency_ms"].sample_count == 2
        # The wall-gap series exists but has no population here: every event
        # measured itself, so nothing fell back to the derivation. sample_count
        # 0 is "no data", not "zero milliseconds".
        assert metrics["request_to_event_wall_ms"].sample_count == 0
        assert metrics["request_to_event_wall_ms"].value == 0.0
        # The retired blended name is never written again, so a stored row still
        # carrying it is unambiguously pre-cutover.
        assert "retrieval_latency_ms" not in metrics
        assert metrics["avg_items_included"].value == pytest.approx(3.0)
        assert metrics["avg_items_dropped"].value == pytest.approx(0.5)
        assert metrics["avg_token_estimate"].value == pytest.approx(150.0)
        assert metrics["cold_start_rate"].value == pytest.approx(0.5)
        assert metrics["zero_candidate_rate"].value == pytest.approx(0.5)
        assert metrics["cold_start_rate"].sample_count == 2
    finally:
        await connection.close()


async def _seed_mixed_surface_events(
    messages: MessageRepository,
    clock: FrozenClock,
    events: RetrievalEventRepository,
) -> None:
    """Seed two chat turns and one retrieve-only context call for one user.

    The context event carries no response message because the host that called
    get_context owns the reply, which is exactly why its measurements are not
    comparable with a chat turn's.
    """
    await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need help", at=_dt(10, 0))
    await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Answer", at=_dt(10, 1))
    await _create_message_at(messages, clock, message_id="msg_3", conversation_id="cnv_dbg_1", role="user", seq=3, text="More help", at=_dt(10, 10))
    await _create_message_at(messages, clock, message_id="msg_4", conversation_id="cnv_dbg_1", role="assistant", seq=4, text="More answer", at=_dt(10, 11))
    await _create_message_at(messages, clock, message_id="msg_5", conversation_id="cnv_dbg_1", role="user", seq=5, text="Sidecar prompt", at=_dt(10, 20))
    await _create_event_at(
        events,
        event_id="ret_chat_1",
        user_id="usr_1",
        conversation_id="cnv_dbg_1",
        request_message_id="msg_1",
        response_message_id="msg_2",
        assistant_mode_id="coding_debug",
        selected_memory_ids=[],
        at=_dt(10, 0, 2),
        items_included=2,
        items_dropped=1,
        total_tokens_estimate=100,
        outcome={"cold_start": True, "zero_candidates": False},
        surface=TurnSurface.CHAT,
        retrieval_duration_ms=20.0,
    )
    await _create_event_at(
        events,
        event_id="ret_chat_2",
        user_id="usr_1",
        conversation_id="cnv_dbg_1",
        request_message_id="msg_3",
        response_message_id="msg_4",
        assistant_mode_id="coding_debug",
        selected_memory_ids=[],
        at=_dt(10, 10, 5),
        items_included=4,
        items_dropped=3,
        total_tokens_estimate=200,
        outcome={"cold_start": False, "zero_candidates": True},
        surface=TurnSurface.CHAT,
        retrieval_duration_ms=40.0,
    )
    await _create_event_at(
        events,
        event_id="ret_ctx_1",
        user_id="usr_1",
        conversation_id="cnv_dbg_1",
        request_message_id="msg_5",
        response_message_id=None,
        assistant_mode_id="coding_debug",
        selected_memory_ids=[],
        at=_dt(10, 20, 1),
        items_included=6,
        items_dropped=5,
        total_tokens_estimate=600,
        outcome={"cold_start": False, "zero_candidates": False},
        surface=TurnSurface.CONTEXT,
        retrieval_duration_ms=90.0,
    )


@pytest.mark.asyncio
async def test_summarize_retrieval_events_splits_chat_and_context_surfaces() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _seed_mixed_surface_events(messages, clock, events)

        summary = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=None,
        )

        assert summary.surface_filter is None
        assert summary.total_events == 3
        assert summary.cold_start_count == 1
        assert summary.zero_candidate_count == 1
        assert summary.avg_items_included == pytest.approx(4.0)
        assert summary.avg_items_dropped == pytest.approx(3.0)
        assert summary.avg_token_estimate == pytest.approx(300.0)
        assert summary.avg_retrieval_stage_latency_ms == pytest.approx(50.0)
        assert summary.retrieval_stage_latency_sample_count == 3
        assert summary.avg_request_to_event_wall_ms == 0.0
        assert summary.request_to_event_wall_sample_count == 0

        assert set(summary.by_surface) == {TurnSurface.CHAT, TurnSurface.CONTEXT}
        chat_stats = summary.by_surface[TurnSurface.CHAT]
        context_stats = summary.by_surface[TurnSurface.CONTEXT]
        assert chat_stats.total_events == 2
        assert chat_stats.cold_start_count == 1
        assert chat_stats.zero_candidate_count == 1
        assert chat_stats.avg_items_included == pytest.approx(3.0)
        assert chat_stats.avg_items_dropped == pytest.approx(2.0)
        assert chat_stats.avg_token_estimate == pytest.approx(150.0)
        assert chat_stats.avg_retrieval_stage_latency_ms == pytest.approx(30.0)
        assert chat_stats.retrieval_stage_latency_sample_count == 2
        assert context_stats.total_events == 1
        assert context_stats.cold_start_count == 0
        assert context_stats.zero_candidate_count == 0
        assert context_stats.avg_items_included == pytest.approx(6.0)
        assert context_stats.avg_token_estimate == pytest.approx(600.0)
        assert context_stats.avg_retrieval_stage_latency_ms == pytest.approx(90.0)
        assert context_stats.retrieval_stage_latency_sample_count == 1
        # Cross-surface totals are not any surface's number, which is the whole
        # point of shipping the breakdown alongside them.
        assert summary.avg_retrieval_stage_latency_ms != pytest.approx(
            chat_stats.avg_retrieval_stage_latency_ms
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summarize_retrieval_events_filters_to_a_single_surface() -> None:
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _seed_mixed_surface_events(messages, clock, events)

        chat_only = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=TurnSurface.CHAT,
        )
        context_only = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=TurnSurface.CONTEXT,
        )
        proxy_only = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=TurnSurface.PROXY_STREAM,
        )

        assert chat_only.surface_filter is TurnSurface.CHAT
        assert set(chat_only.by_surface) == {TurnSurface.CHAT}
        assert chat_only.total_events == 2
        assert chat_only.avg_items_included == pytest.approx(3.0)
        assert chat_only.avg_retrieval_stage_latency_ms == pytest.approx(30.0)

        assert context_only.surface_filter is TurnSurface.CONTEXT
        assert set(context_only.by_surface) == {TurnSurface.CONTEXT}
        assert context_only.total_events == 1
        assert context_only.avg_items_included == pytest.approx(6.0)
        assert context_only.avg_retrieval_stage_latency_ms == pytest.approx(90.0)

        # A surface with no events in the window reports emptiness instead of
        # inventing a zero-latency measurement for it.
        assert proxy_only.surface_filter is TurnSurface.PROXY_STREAM
        assert proxy_only.by_surface == {}
        assert proxy_only.total_events == 0
        assert proxy_only.avg_retrieval_stage_latency_ms == 0.0
        assert proxy_only.retrieval_stage_latency_sample_count == 0
        assert proxy_only.avg_request_to_event_wall_ms == 0.0
        assert proxy_only.request_to_event_wall_sample_count == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summarize_retrieval_events_reports_measured_and_derived_latency_apart() -> None:
    """The measured stage and the legacy wall gap are different quantities.

    A mean over both is neither: the derivation contains the reply generation,
    so blending it with a retrieval-stage measurement produces a figure that
    describes nothing. Each series is reported on its own, over its own
    population.
    """
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Answer", at=_dt(10, 1))
        await _create_message_at(messages, clock, message_id="msg_3", conversation_id="cnv_dbg_1", role="user", seq=3, text="More help", at=_dt(10, 10))
        await _create_message_at(messages, clock, message_id="msg_4", conversation_id="cnv_dbg_1", role="assistant", seq=4, text="More answer", at=_dt(10, 11))
        await _create_event_at(
            events,
            event_id="ret_measured",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 0, 2),
            retrieval_duration_ms=20.0,
        )
        await _create_event_at(
            events,
            event_id="ret_pre_migration",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_3",
            response_message_id="msg_4",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 10, 5),
        )
        await _strip_persisted_telemetry(connection, "ret_pre_migration", "usr_1")

        summary = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=None,
        )

        # ret_measured contributes its persisted 20 ms to the measured series and
        # nothing to the derived one. ret_pre_migration has a NULL column, so it
        # contributes only the 5 s gap between its request message (10:10:00)
        # and the event row (10:10:05) to the derived series.
        assert summary.total_events == 2
        assert summary.retrieval_stage_latency_sample_count == 1
        assert summary.avg_retrieval_stage_latency_ms == pytest.approx(20.0)
        assert summary.request_to_event_wall_sample_count == 1
        assert summary.avg_request_to_event_wall_ms == pytest.approx(5000.0, rel=1e-3)
        # The old blended average, mean(20 ms, 5000 ms), is exactly the number
        # neither series may report.
        assert summary.avg_retrieval_stage_latency_ms != pytest.approx(2510.0, rel=1e-3)
        assert summary.avg_request_to_event_wall_ms != pytest.approx(2510.0, rel=1e-3)
        chat_stats = summary.by_surface[TurnSurface.CHAT]
        assert chat_stats.total_events == 2
        assert chat_stats.retrieval_stage_latency_sample_count == 1
        assert chat_stats.avg_retrieval_stage_latency_ms == pytest.approx(20.0)
        assert chat_stats.request_to_event_wall_sample_count == 1
        assert chat_stats.avg_request_to_event_wall_ms == pytest.approx(5000.0, rel=1e-3)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_cross_surface_totals_weight_each_latency_series_by_its_own_population() -> None:
    """Folding surfaces back together must not weight by the event count.

    Each latency series covers a subset of the rows, so weighting a surface's
    average by ``total_events`` credits it with samples it never contributed.
    Here chat measured 1 of its 2 events and context measured its only one: the
    measured total is mean(20, 90) = 55, not the 43.33 an event-count weighting
    would produce.
    """
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(messages, clock, message_id="msg_1", conversation_id="cnv_dbg_1", role="user", seq=1, text="Need help", at=_dt(10, 0))
        await _create_message_at(messages, clock, message_id="msg_2", conversation_id="cnv_dbg_1", role="assistant", seq=2, text="Answer", at=_dt(10, 1))
        await _create_message_at(messages, clock, message_id="msg_3", conversation_id="cnv_dbg_1", role="user", seq=3, text="More help", at=_dt(10, 10))
        await _create_message_at(messages, clock, message_id="msg_4", conversation_id="cnv_dbg_1", role="assistant", seq=4, text="More answer", at=_dt(10, 11))
        await _create_message_at(messages, clock, message_id="msg_5", conversation_id="cnv_dbg_1", role="user", seq=5, text="Sidecar prompt", at=_dt(10, 20))
        await _create_event_at(
            events,
            event_id="ret_chat_measured",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id="msg_2",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 0, 2),
            surface=TurnSurface.CHAT,
            retrieval_duration_ms=20.0,
        )
        await _create_event_at(
            events,
            event_id="ret_chat_pre_migration",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_3",
            response_message_id="msg_4",
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 10, 5),
            surface=TurnSurface.CHAT,
        )
        await _strip_persisted_telemetry(connection, "ret_chat_pre_migration", "usr_1")
        await _create_event_at(
            events,
            event_id="ret_ctx_measured",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_5",
            response_message_id=None,
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 20, 1),
            surface=TurnSurface.CONTEXT,
            retrieval_duration_ms=90.0,
        )

        summary = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=None,
        )

        assert summary.total_events == 3
        assert summary.retrieval_stage_latency_sample_count == 2
        assert summary.avg_retrieval_stage_latency_ms == pytest.approx(55.0)
        assert summary.avg_retrieval_stage_latency_ms != pytest.approx(43.333, rel=1e-3)
        assert summary.request_to_event_wall_sample_count == 1
        assert summary.avg_request_to_event_wall_ms == pytest.approx(5000.0, rel=1e-3)
        # Only the chat surface holds a pre-migration row, so the derived series
        # is empty for context rather than reported as a zero measurement.
        context_stats = summary.by_surface[TurnSurface.CONTEXT]
        assert context_stats.request_to_event_wall_sample_count == 0
        assert context_stats.avg_request_to_event_wall_ms == 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_failed_open_retrieval_never_enters_the_measured_success_series() -> None:
    """A retrieval that FAILED does not describe what a retrieval costs.

    The proxy's fail-open row carries a real, deliberately-measured duration --
    the attempt was paid for -- but it timed an attempt that produced no memory
    context. Averaging it with the turns whose retrieval worked yields a figure
    that describes neither: here mean(200, 220, 52.87) = 157.62 instead of the
    210 two working retrievals actually cost. The spend stays visible in its own
    series instead of being deleted or blended.
    """
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        for seq, text in ((1, "Need help"), (2, "More help"), (3, "Third")):
            await _create_message_at(
                messages,
                clock,
                message_id=f"msg_{seq}",
                conversation_id="cnv_dbg_1",
                role="user",
                seq=seq,
                text=text,
                at=_dt(10, seq),
            )
        for event_id, request_message_id, duration in (
            ("ret_ok_1", "msg_1", 200.0),
            ("ret_ok_2", "msg_2", 220.0),
        ):
            await _create_event_at(
                events,
                event_id=event_id,
                user_id="usr_1",
                conversation_id="cnv_dbg_1",
                request_message_id=request_message_id,
                response_message_id=None,
                assistant_mode_id="coding_debug",
                selected_memory_ids=[],
                at=_dt(10, 30),
                surface=TurnSurface.PROXY_COMPLETION,
                retrieval_duration_ms=duration,
            )
        # Exactly the row _turn_telemetry builds when memory context failed open.
        await _create_event_at(
            events,
            event_id="ret_failed_open",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_3",
            response_message_id=None,
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 31),
            surface=TurnSurface.PROXY_COMPLETION,
            retrieval_duration_ms=52.87,
            outcome={"memory_context_available": False},
        )

        summary = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=None,
        )

        assert summary.total_events == 3
        assert summary.retrieval_stage_latency_sample_count == 2
        assert summary.avg_retrieval_stage_latency_ms == pytest.approx(210.0)
        assert summary.avg_retrieval_stage_latency_ms != pytest.approx(157.623, rel=1e-3)
        assert summary.failed_retrieval_stage_latency_sample_count == 1
        assert summary.avg_failed_retrieval_stage_latency_ms == pytest.approx(52.87)
        # Every row lands in exactly one series.
        assert summary.request_to_event_wall_sample_count == 0
        proxy_stats = summary.by_surface[TurnSurface.PROXY_COMPLETION]
        assert proxy_stats.avg_retrieval_stage_latency_ms == pytest.approx(210.0)
        assert proxy_stats.failed_retrieval_stage_latency_sample_count == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_a_slice_without_failed_retrievals_reports_an_empty_failure_series() -> None:
    """0.0 with a 0 count means "nothing contributed", not "it took no time"."""
    connection, clock, messages, _memories, events, _feedback, _beliefs, computer = await _build_runtime()
    try:
        await _create_message_at(
            messages,
            clock,
            message_id="msg_1",
            conversation_id="cnv_dbg_1",
            role="user",
            seq=1,
            text="Need help",
            at=_dt(10, 0),
        )
        await _create_event_at(
            events,
            event_id="ret_ok",
            user_id="usr_1",
            conversation_id="cnv_dbg_1",
            request_message_id="msg_1",
            response_message_id=None,
            assistant_mode_id="coding_debug",
            selected_memory_ids=[],
            at=_dt(10, 5),
            retrieval_duration_ms=33.0,
        )

        summary = await computer.summarize_retrieval_events(
            from_date="2026-03-31",
            to_date="2026-03-31",
            user_id="usr_1",
            assistant_mode_id=None,
            turn_surface=None,
        )

        assert summary.retrieval_stage_latency_sample_count == 1
        assert summary.avg_retrieval_stage_latency_ms == pytest.approx(33.0)
        assert summary.failed_retrieval_stage_latency_sample_count == 0
        assert summary.avg_failed_retrieval_stage_latency_ms == 0.0
    finally:
        await connection.close()
