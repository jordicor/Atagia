"""Tests for evaluation metric computation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository, WorkspaceRepository
from atagia.core.retrieval_event_repository import MemoryFeedbackRepository, RetrievalEventRepository
from atagia.memory.metrics_computer import MetricsComputer
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


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
    object,
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
) -> dict[str, object]:
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
        }
    )


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
async def test_compute_slr_combines_explicit_and_automatic_scope_issues() -> None:
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

        assert result.value == pytest.approx(2 / 3)
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
        llm_client = LLMClient(provider_name=provider.name, providers=[provider])

        result = await computer.compute_ccr("usr_1", "coding_debug", "2026-03-31", llm_client)

        assert result.value == pytest.approx(0.82)
        assert result.sample_count == 1
        assert "<contract_block>" in provider.requests[0].messages[1].content
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
        )

        metrics = await computer.compute_system_metrics("2026-03-31")

        assert metrics["retrieval_latency_ms"].value == pytest.approx(3500.0, rel=1e-3)
        assert metrics["avg_items_included"].value == pytest.approx(3.0)
        assert metrics["avg_items_dropped"].value == pytest.approx(0.5)
        assert metrics["avg_token_estimate"].value == pytest.approx(150.0)
        assert metrics["cold_start_rate"].value == pytest.approx(0.5)
        assert metrics["zero_candidate_rate"].value == pytest.approx(0.5)
        assert metrics["cold_start_rate"].sample_count == 2
    finally:
        await connection.close()
