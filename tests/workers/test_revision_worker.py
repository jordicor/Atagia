"""Tests for the revision worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import (
    JobEnvelope,
    JobType,
    REVISE_STREAM_NAME,
    RevisionJobPayload,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.workers.revision_worker import RevisionWorker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "revision-worker-tests"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.metadata.get("purpose") == "intent_classifier_explicit":
            message_text = request.messages[-1].content
            is_explicit = "I prefer" in message_text or "I also prefer" in message_text
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_explicit": is_explicit,
                        "reasoning": "Test classifier response.",
                    }
                ),
            )
        if request.metadata.get("purpose") == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"equivalent": True}),
            )
        if not self.outputs:
            raise AssertionError("No queued output left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in revision worker tests")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime(outputs: list[str]):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    beliefs = BeliefRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "One")
    await conversations.create_conversation("cnv_2", "usr_1", "wrk_1", "research_deep_dive", "Two")
    await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "I prefer terse debugging answers.",
        8,
        {},
    )
    await messages.create_message(
        "msg_2",
        "cnv_2",
        "user",
        1,
        "I also prefer terse debugging answers in research mode.",
        10,
        {},
    )
    await messages.create_message(
        "msg_3",
        "cnv_1",
        "user",
        2,
        "Please keep this debugging answer short.",
        8,
        {},
    )
    backend = InProcessBackend()
    provider = QueueProvider(outputs)
    worker = RevisionWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        settings=_settings(),
    )
    return connection, backend, memories, beliefs, worker


async def _seed_belief(
    memories: MemoryObjectRepository,
    beliefs: BeliefRepository,
    *,
    memory_id: str = "mem_belief",
    scope: MemoryScope = MemoryScope.CONVERSATION,
    assistant_mode_id: str | None = "coding_debug",
    workspace_id: str | None = "wrk_1",
    conversation_id: str | None = "cnv_1",
    source_message_ids: list[str] | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> dict[str, object]:
    created = await memories.create_memory_object(
        user_id="usr_1",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=MemoryObjectType.BELIEF,
        scope=scope,
        canonical_text="User prefers terse debugging answers.",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.8,
        stability=0.7,
        vitality=0.25,
        maya_score=1.0,
        privacy_level=1,
        status=status,
        payload={
            "claim_key": "response_style.debugging",
            "claim_value": "terse",
            "source_message_ids": source_message_ids or ["msg_0"],
        },
        memory_id=memory_id,
    )
    await beliefs.create_first_version(
        belief_id=str(created["id"]),
        claim_key="response_style.debugging",
        claim_value="terse",
        created_at=str(created["created_at"]),
    )
    return created


async def _seed_evidence(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    conversation_id: str,
    assistant_mode_id: str,
    source_message_id: str,
    claim_key: str | None = None,
) -> dict[str, object]:
    payload = {"source_message_ids": [source_message_id]}
    if claim_key is not None:
        payload["claim_key"] = claim_key
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id=conversation_id,
        assistant_mode_id=assistant_mode_id,
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=f"Evidence for {source_message_id}",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=1,
        payload=payload,
        memory_id=memory_id,
    )


def _revision_job(
    *,
    belief_id: str,
    evidence_memory_ids: list[str],
    source_message_id: str,
    scope: str,
    claim_key: str = "response_style.debugging",
    claim_value: str = "terse",
) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_revision_1",
        job_type=JobType.REVISE_BELIEFS,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[source_message_id],
        payload={
            "belief_id": belief_id,
            "claim_key": claim_key,
            "claim_value": json.dumps(claim_value),
            "evidence_memory_ids": evidence_memory_ids,
            "source_message_id": source_message_id,
            "user_id": "usr_1",
            "assistant_mode_id": "coding_debug",
            "workspace_id": "wrk_1",
            "conversation_id": "cnv_1",
            "scope": scope,
        },
        created_at=datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_revision_worker_processes_revision_job_end_to_end_and_releases_lock() -> None:
    connection, backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "REINFORCE", "explanation": "This evidence reinforces the belief."})]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        cursor = await connection.execute(
            "SELECT support_count FROM belief_versions WHERE belief_id = ? AND is_current = 1",
            (belief["id"],),
        )
        version = await cursor.fetchone()

        assert result.acked == 1
        assert result.failed == 0
        assert version["support_count"] == 2
        assert backend._locks == {}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_defers_contradictory_revision_below_tension_threshold() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime([])
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_defer",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )

        result = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_value="verbose",
            ).model_dump(mode="json")
        )

        assert result is not None
        assert result["status"] == "deferred_tension"
        assert result["signal_type"] == "contradictory"
        assert result["tension_score"] == pytest.approx(0.15)
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.15)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_defers_same_value_scope_exception_after_preview() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "This belief only holds in a narrower scope."})]
    )
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.ASSISTANT_MODE,
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
        )
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_scope_split",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )

        result = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_value="terse",
            ).model_dump(mode="json")
        )

        assert result is not None
        assert result["status"] == "deferred_tension"
        assert result["signal_type"] == "ambiguous"
        assert result["preview_action"] == "SPLIT_BY_SCOPE"
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.15)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_allows_same_value_narrower_scope_reinforcement_after_preview() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "REINFORCE", "explanation": "The narrower evidence still reinforces the broader belief."})]
    )
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.ASSISTANT_MODE,
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
        )
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_scope_reinforce",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )

        result = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_value="terse",
            ).model_dump(mode="json")
        )
        cursor = await connection.execute(
            "SELECT support_count FROM belief_versions WHERE belief_id = ? AND is_current = 1",
            (belief["id"],),
        )
        version = await cursor.fetchone()

        assert result is not None
        assert result["action"] == "REINFORCE"
        assert result["signal_type"] == "ambiguous"
        assert version["support_count"] == 2
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.0)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_resets_tension_and_revises_once_threshold_is_reached() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "WEAKEN", "explanation": "Contradiction reached the threshold."})]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        await beliefs.increment_tension(str(belief["id"]), 0.45, user_id="usr_1")
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_threshold",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )

        result = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_value="verbose",
            ).model_dump(mode="json")
        )

        assert result is not None
        assert result["action"] == "WEAKEN"
        assert result["signal_type"] == "contradictory"
        assert result["trigger_tension_score"] == pytest.approx(0.60)
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.0)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_reinforcing_evidence_reduces_tension_before_reinforce() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "REINFORCE", "explanation": "This evidence reinforces the belief."})]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        await beliefs.increment_tension(str(belief["id"]), 0.20, user_id="usr_1")
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_reinforce",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )

        result = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
            ).model_dump(mode="json")
        )

        assert result is not None
        assert result["action"] == "REINFORCE"
        assert result["signal_type"] == "ambiguous"
        assert result["tension_score"] == pytest.approx(0.15)
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.15)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_propagates_tension_decrement_failures_after_reinforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime([])
    try:
        belief = await _seed_belief(memories, beliefs)
        payload = RevisionJobPayload.model_validate(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=["mem_evidence_reinforce_fail"],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
            ).payload
        )

        async def fail_decrement(*args: object, **kwargs: object) -> float:
            del args, kwargs
            raise RuntimeError("decrement failed")

        monkeypatch.setattr(worker._belief_repository, "decrement_tension", fail_decrement)

        with pytest.raises(RuntimeError, match="decrement failed"):
            await worker._post_revision_tension_update(  # noqa: SLF001
                payload,
                {"action": "REINFORCE"},
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_replays_accumulated_contradictory_evidence_at_threshold() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "WEAKEN", "explanation": "Accumulated contradictions reached the threshold."})]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence_ids: list[str] = []
        for index in range(1, 5):
            evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_acc_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            evidence_ids.append(str(evidence["id"]))
            result = await worker.process_job(
                _revision_job(
                    belief_id=str(belief["id"]),
                    evidence_memory_ids=[str(evidence["id"])],
                    source_message_id="msg_1",
                    scope=MemoryScope.CONVERSATION.value,
                    claim_value="verbose",
                ).model_dump(mode="json")
            )
            assert result is not None

        cursor = await connection.execute(
            "SELECT contradict_count FROM belief_versions WHERE belief_id = ? AND is_current = 1",
            (belief["id"],),
        )
        version = await cursor.fetchone()
        belief_row = await memories.get_memory_object(str(belief["id"]), "usr_1")

        assert result["action"] == "WEAKEN"
        assert result["trigger_tension_score"] == pytest.approx(0.60)
        assert version["contradict_count"] == 4
        assert belief_row is not None
        assert belief_row["payload_json"].get("tension_evidence_memory_ids") is None
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.0)
        assert len(evidence_ids) == 4
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_clears_stale_contradiction_buffer_when_tension_returns_to_zero() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [
            json.dumps({"action": "REINFORCE", "explanation": "Reinforcing evidence."}),
            json.dumps({"action": "REINFORCE", "explanation": "Reinforcing evidence."}),
            json.dumps({"action": "REINFORCE", "explanation": "Reinforcing evidence."}),
            json.dumps({"action": "WEAKEN", "explanation": "Fresh contradictions reached the threshold."}),
        ]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        contradiction = await _seed_evidence(
            memories,
            memory_id="mem_evidence_old_contradiction",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )
        deferred = await worker.process_job(
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(contradiction["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_value="verbose",
            ).model_dump(mode="json")
        )
        assert deferred is not None
        assert deferred["status"] == "deferred_tension"

        for index in range(1, 4):
            reinforce_evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_reinforce_zero_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            reinforce = await worker.process_job(
                _revision_job(
                    belief_id=str(belief["id"]),
                    evidence_memory_ids=[str(reinforce_evidence["id"])],
                    source_message_id="msg_1",
                    scope=MemoryScope.CONVERSATION.value,
                ).model_dump(mode="json")
            )
            assert reinforce is not None
            assert reinforce["action"] == "REINFORCE"

        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.0)

        for index in range(1, 5):
            contradiction_evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_new_contradiction_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            result = await worker.process_job(
                _revision_job(
                    belief_id=str(belief["id"]),
                    evidence_memory_ids=[str(contradiction_evidence["id"])],
                    source_message_id="msg_1",
                    scope=MemoryScope.CONVERSATION.value,
                    claim_value="verbose",
                ).model_dump(mode="json")
            )
            assert result is not None

        cursor = await connection.execute(
            "SELECT contradict_count FROM belief_versions WHERE belief_id = ? AND is_current = 1",
            (belief["id"],),
        )
        version = await cursor.fetchone()

        assert result["action"] == "WEAKEN"
        assert version["contradict_count"] == 4
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_reuses_threshold_preview_for_ambiguous_same_value_cases() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 1."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 2."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 3."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Threshold trigger preview."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Threshold apply preview."}),
            json.dumps({"action": "REINFORCE", "explanation": "Should remain unused."}),
        ]
    )
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.ASSISTANT_MODE,
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
        )
        for index in range(1, 5):
            evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_scope_threshold_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            result = await worker.process_job(
                _revision_job(
                    belief_id=str(belief["id"]),
                    evidence_memory_ids=[str(evidence["id"])],
                    source_message_id="msg_1",
                    scope=MemoryScope.CONVERSATION.value,
                    claim_value="terse",
                ).model_dump(mode="json")
            )
            assert result is not None

        assert result["action"] == "SPLIT_BY_SCOPE"
        assert result["signal_type"] == "ambiguous"
        assert result["trigger_tension_score"] == pytest.approx(0.60)
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.0)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_preserves_contradiction_buffer_when_threshold_revision_fails() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(["not-json"])
    try:
        belief = await _seed_belief(memories, beliefs)
        for index in range(1, 5):
            evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_fail_contradiction_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            if index < 4:
                result = await worker.process_job(
                    _revision_job(
                        belief_id=str(belief["id"]),
                        evidence_memory_ids=[str(evidence["id"])],
                        source_message_id="msg_1",
                        scope=MemoryScope.CONVERSATION.value,
                        claim_value="verbose",
                    ).model_dump(mode="json")
                )
                assert result is not None
                assert result["status"] == "deferred_tension"
                continue

            with pytest.raises(Exception):
                await worker.process_job(
                    _revision_job(
                        belief_id=str(belief["id"]),
                        evidence_memory_ids=[str(evidence["id"])],
                        source_message_id="msg_1",
                        scope=MemoryScope.CONVERSATION.value,
                        claim_value="verbose",
                    ).model_dump(mode="json")
                )

        belief_row = await memories.get_memory_object(str(belief["id"]), "usr_1")

        assert belief_row is not None
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.60)
        assert belief_row["payload_json"]["tension_evidence_memory_ids"] == [
            "mem_evidence_fail_contradiction_1",
            "mem_evidence_fail_contradiction_2",
            "mem_evidence_fail_contradiction_3",
            "mem_evidence_fail_contradiction_4",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_preserves_ambiguous_buffer_when_threshold_preview_fails() -> None:
    connection, _backend, memories, beliefs, worker = await _build_runtime(
        [
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 1."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 2."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Preview 3."}),
            json.dumps({"action": "SPLIT_BY_SCOPE", "explanation": "Threshold trigger preview."}),
            "not-json",
        ]
    )
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.ASSISTANT_MODE,
            assistant_mode_id="coding_debug",
            workspace_id=None,
            conversation_id=None,
        )
        for index in range(1, 5):
            evidence = await _seed_evidence(
                memories,
                memory_id=f"mem_evidence_fail_ambiguous_{index}",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                source_message_id="msg_1",
            )
            if index < 4:
                result = await worker.process_job(
                    _revision_job(
                        belief_id=str(belief["id"]),
                        evidence_memory_ids=[str(evidence["id"])],
                        source_message_id="msg_1",
                        scope=MemoryScope.CONVERSATION.value,
                        claim_value="terse",
                    ).model_dump(mode="json")
                )
                assert result is not None
                assert result["status"] == "deferred_tension"
                continue

            with pytest.raises(Exception):
                await worker.process_job(
                    _revision_job(
                        belief_id=str(belief["id"]),
                        evidence_memory_ids=[str(evidence["id"])],
                        source_message_id="msg_1",
                        scope=MemoryScope.CONVERSATION.value,
                        claim_value="terse",
                    ).model_dump(mode="json")
                )

        belief_row = await memories.get_memory_object(str(belief["id"]), "usr_1")

        assert belief_row is not None
        assert await beliefs.get_tension(str(belief["id"]), user_id="usr_1") == pytest.approx(0.60)
        assert belief_row["payload_json"]["tension_evidence_memory_ids"] == [
            "mem_evidence_fail_ambiguous_1",
            "mem_evidence_fail_ambiguous_2",
            "mem_evidence_fail_ambiguous_3",
            "mem_evidence_fail_ambiguous_4",
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_dead_letters_after_max_failed_deliveries() -> None:
    connection, backend, memories, beliefs, worker = await _build_runtime(
        ["not-json", "not-json", "not-json"]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_fail",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )
        job = _revision_job(
            belief_id=str(belief["id"]),
            evidence_memory_ids=[str(evidence["id"])],
            source_message_id="msg_1",
            scope=MemoryScope.CONVERSATION.value,
        ).model_dump(mode="json")
        await backend.stream_add(REVISE_STREAM_NAME, job)

        first = await worker.run_once()
        second = await worker.run_once()
        third = await worker.run_once()
        dead_letter = await backend.dequeue_job(f"dead_letter:{REVISE_STREAM_NAME}", timeout_seconds=0)

        assert first.failed == 1
        assert second.failed == 1
        assert third.failed == 1
        assert third.dead_lettered == 1
        assert dead_letter is not None
        assert dead_letter["delivery_count"] == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_skips_inactive_belief_and_acks_job() -> None:
    connection, backend, memories, beliefs, worker = await _build_runtime([])
    try:
        belief = await _seed_belief(memories, beliefs, status=MemoryStatus.ARCHIVED)
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_skip",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()

        assert result.acked == 1
        assert result.failed == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_promotion_job_meeting_threshold_creates_belief() -> None:
    connection, backend, memories, _beliefs, worker = await _build_runtime([])
    try:
        evidence_one = await _seed_evidence(
            memories,
            memory_id="mem_evidence_a",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
            claim_key="response_style.debugging",
        )
        evidence_two = await _seed_evidence(
            memories,
            memory_id="mem_evidence_b",
            conversation_id="cnv_2",
            assistant_mode_id="research_deep_dive",
            source_message_id="msg_2",
            claim_key="response_style.debugging",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id="",
                evidence_memory_ids=[str(evidence_one["id"]), str(evidence_two["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        beliefs = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.BELIEF.value
        ]

        assert result.acked == 1
        assert len(beliefs) == 1
        assert beliefs[0]["scope"] == MemoryScope.WORKSPACE.value
        assert beliefs[0]["assistant_mode_id"] == "coding_debug"
        assert beliefs[0]["workspace_id"] == "wrk_1"
        assert beliefs[0]["conversation_id"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_promotion_below_threshold_does_not_create_belief() -> None:
    connection, backend, memories, _beliefs, worker = await _build_runtime([])
    try:
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_low",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_3",
            claim_key="response_style.debugging",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id="",
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_3",
                scope=MemoryScope.CONVERSATION.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        beliefs = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.BELIEF.value
        ]

        assert result.acked == 1
        assert beliefs == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_explicit_user_statement_without_evidence_does_not_promote_fast() -> None:
    connection, backend, memories, _beliefs, worker = await _build_runtime([])
    try:
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id="",
                evidence_memory_ids=[],
                source_message_id="msg_1",
                scope=MemoryScope.ASSISTANT_MODE.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        beliefs = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.BELIEF.value
        ]

        assert result.acked == 1
        assert beliefs == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_explicit_user_statement_with_evidence_promotes_fast() -> None:
    connection, backend, memories, _beliefs, worker = await _build_runtime([])
    try:
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_fast",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
            claim_key="response_style.debugging",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id="",
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.ASSISTANT_MODE.value,
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        beliefs = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.BELIEF.value
        ]

        assert result.acked == 1
        assert len(beliefs) == 1
        assert beliefs[0]["scope"] == MemoryScope.GLOBAL_USER.value
        assert beliefs[0]["assistant_mode_id"] is None
        assert beliefs[0]["workspace_id"] is None
        assert beliefs[0]["conversation_id"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revision_worker_acks_job_with_empty_claim_key() -> None:
    connection, backend, memories, beliefs, worker = await _build_runtime([])
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_invalid",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id=str(belief["id"]),
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_key="",
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()

        assert result.acked == 1
        assert result.failed == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_promotion_matches_semantically_equivalent_claim_key() -> None:
    connection, backend, memories, beliefs, worker = await _build_runtime(
        [json.dumps({"action": "REINFORCE", "explanation": "Equivalent claim key."})]
    )
    try:
        belief = await _seed_belief(memories, beliefs)
        await connection.execute(
            """
            UPDATE belief_versions
            SET claim_key = ?
            WHERE belief_id = ?
              AND is_current = 1
            """,
            ("response_style.debug_response", belief["id"]),
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET payload_json = json_set(payload_json, '$.claim_key', ?)
            WHERE id = ?
            """,
            ("response_style.debug_response", belief["id"]),
        )
        await connection.commit()
        evidence = await _seed_evidence(
            memories,
            memory_id="mem_evidence_equivalent",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            source_message_id="msg_1",
            claim_key="response_style.debugging",
        )
        await backend.stream_add(
            REVISE_STREAM_NAME,
            _revision_job(
                belief_id="",
                evidence_memory_ids=[str(evidence["id"])],
                source_message_id="msg_1",
                scope=MemoryScope.CONVERSATION.value,
                claim_key="response_style.debugging",
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        belief_rows = [
            row
            for row in await memories.list_for_user("usr_1")
            if row["object_type"] == MemoryObjectType.BELIEF.value
        ]
        cursor = await connection.execute(
            "SELECT support_count FROM belief_versions WHERE belief_id = ? AND is_current = 1",
            (belief["id"],),
        )
        version = await cursor.fetchone()

        assert result.acked == 1
        assert len(belief_rows) == 1
        assert version["support_count"] == 2
    finally:
        await connection.close()
