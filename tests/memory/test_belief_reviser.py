"""Tests for belief revision actions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.belief_repository import BeliefRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.belief_reviser import BeliefReviser, RevisionContext
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


class QueueProvider(LLMProvider):
    name = "belief-reviser-tests"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
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
        raise AssertionError("Embeddings are not used in reviser tests")


async def _build_runtime(action: str):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 1, 16, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    beliefs = BeliefRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "One")
    await conversations.create_conversation("cnv_2", "usr_1", "wrk_1", "research_deep_dive", "Two")
    provider = QueueProvider(
        [
            json.dumps(
                {
                    "action": action,
                    "explanation": f"{action} because the evidence points there.",
                    "successor_canonical_text": f"{action.lower()} successor belief",
                }
            )
        ]
    )
    reviser = BeliefReviser(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
    )
    return connection, memories, beliefs, reviser


async def _seed_belief(
    memories: MemoryObjectRepository,
    beliefs: BeliefRepository,
    *,
    memory_id: str = "mem_belief",
    scope: MemoryScope = MemoryScope.ASSISTANT_MODE,
    assistant_mode_id: str | None = "coding_debug",
    workspace_id: str | None = "wrk_1",
    conversation_id: str | None = None,
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
        status=MemoryStatus.ACTIVE,
        payload={
            "claim_key": "response_style.debugging",
            "claim_value": "terse",
            "source_message_ids": ["msg_0"],
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
    memory_id: str = "mem_evidence",
    conversation_id: str = "cnv_1",
    text: str = "The user asked for terse debugging answers again.",
) -> dict[str, object]:
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=1,
        payload={},
        memory_id=memory_id,
    )


async def _current_version(connection, belief_id: str):
    cursor = await connection.execute(
        "SELECT * FROM belief_versions WHERE belief_id = ? AND is_current = 1",
        (belief_id,),
    )
    return await cursor.fetchone()


async def _all_links(connection):
    cursor = await connection.execute(
        "SELECT * FROM memory_links ORDER BY created_at ASC, id ASC"
    )
    return await cursor.fetchall()


@pytest.mark.asyncio
async def test_reinforce_increments_support_without_new_belief() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("REINFORCE")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories)

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[evidence],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("terse"),
                source_message_id="msg_1",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        version = await _current_version(connection, str(belief["id"]))
        links = await _all_links(connection)
        assert result.action == "REINFORCE"
        assert version["support_count"] == 2
        reinforced = await memories.get_memory_object(str(belief["id"]), "usr_1")
        assert reinforced is not None
        assert reinforced["stability"] == pytest.approx(0.73)
        assert result.new_belief_ids == []
        assert links[0]["relation_type"] == "reinforces"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_weaken_increments_contradictions_and_decreases_confidence() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("WEAKEN")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories, text="The user now wants more explanation for debugging answers.")

        await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[evidence],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("verbose"),
                source_message_id="msg_2",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        version = await _current_version(connection, str(belief["id"]))
        weakened = await memories.get_memory_object(str(belief["id"]), "usr_1")
        links = await _all_links(connection)
        assert version["contradict_count"] == 1
        assert weakened is not None
        assert weakened["confidence"] < 0.8
        assert weakened["stability"] == pytest.approx(0.67)
        assert links[0]["relation_type"] == "weakens"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_supersede_creates_new_belief_marks_old_superseded_and_links() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("SUPERSEDE")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories, text="The user now prefers concise but not terse debugging answers.")

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[evidence],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("concise"),
                source_message_id="msg_3",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        old_belief = await memories.get_memory_object(str(belief["id"]), "usr_1")
        new_belief = await memories.get_memory_object(result.new_belief_ids[0], "usr_1")
        links = await _all_links(connection)
        assert old_belief is not None
        assert new_belief is not None
        assert old_belief["status"] == MemoryStatus.SUPERSEDED.value
        assert new_belief["status"] == MemoryStatus.ACTIVE.value
        assert new_belief["canonical_text"] == "supersede successor belief"
        assert {row["relation_type"] for row in links} == {"supersedes", "reinforces"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_split_by_mode_archives_original_and_creates_mode_scoped_belief() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("SPLIT_BY_MODE")
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.GLOBAL_USER,
            assistant_mode_id=None,
            workspace_id=None,
            conversation_id=None,
        )

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("terse"),
                source_message_id="msg_4",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        archived = await memories.get_memory_object(str(belief["id"]), "usr_1")
        child = await memories.get_memory_object(result.new_belief_ids[0], "usr_1")
        version = await _current_version(connection, result.new_belief_ids[0])
        links = await _all_links(connection)
        assert archived is not None
        assert child is not None
        assert archived["status"] == MemoryStatus.ARCHIVED.value
        assert child["scope"] == MemoryScope.ASSISTANT_MODE.value
        assert child["assistant_mode_id"] == "coding_debug"
        assert child["workspace_id"] is None
        assert child["conversation_id"] is None
        assert json.loads(version["condition_json"]) == {"mode": "coding_debug"}
        assert links[0]["relation_type"] == "derived_from"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_split_by_scope_archives_original_and_creates_narrower_belief() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("SPLIT_BY_SCOPE")
    try:
        belief = await _seed_belief(
            memories,
            beliefs,
            scope=MemoryScope.WORKSPACE,
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            conversation_id=None,
        )

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("terse"),
                source_message_id="msg_5",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        archived = await memories.get_memory_object(str(belief["id"]), "usr_1")
        child = await memories.get_memory_object(result.new_belief_ids[0], "usr_1")
        links = await _all_links(connection)
        assert archived is not None
        assert child is not None
        assert archived["status"] == MemoryStatus.ARCHIVED.value
        assert child["scope"] == MemoryScope.CONVERSATION.value
        assert links[0]["relation_type"] == "derived_from"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_split_by_time_sets_temporal_bounds_and_creates_successor() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("SPLIT_BY_TIME")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories, text="The user now wants more verbose debugging answers.")

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[evidence],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("verbose"),
                source_message_id="msg_6",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.ASSISTANT_MODE,
            ),
        )

        old_belief = await memories.get_memory_object(str(belief["id"]), "usr_1")
        new_belief = await memories.get_memory_object(result.new_belief_ids[0], "usr_1")
        links = await _all_links(connection)
        assert old_belief is not None
        assert new_belief is not None
        assert old_belief["valid_to"] is not None
        assert new_belief["valid_from"] is not None
        assert new_belief["assistant_mode_id"] == "coding_debug"
        assert new_belief["workspace_id"] is None
        assert new_belief["conversation_id"] is None
        assert links[0]["relation_type"] == "supersedes"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_mark_exception_creates_exception_belief_with_link() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("MARK_EXCEPTION")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories, text="In this conversation the user wants extra detail.")

        result = await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[evidence],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("detailed"),
                source_message_id="msg_7",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        exception = await memories.get_memory_object(result.new_belief_ids[0], "usr_1")
        version = await _current_version(connection, result.new_belief_ids[0])
        links = await _all_links(connection)
        assert exception is not None
        assert exception["canonical_text"] == "Exception: mark_exception successor belief"
        assert json.loads(version["condition_json"])["conversation_id"] == "cnv_1"
        assert links[0]["relation_type"] == "exception_to"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_archive_sets_belief_status_archived() -> None:
    connection, memories, beliefs, reviser = await _build_runtime("ARCHIVE")
    try:
        belief = await _seed_belief(memories, beliefs)

        await reviser.revise(
            belief_id=str(belief["id"]),
            new_evidence=[],
            context=RevisionContext(
                user_id="usr_1",
                claim_key="response_style.debugging",
                claim_value=json.dumps("terse"),
                source_message_id="msg_8",
                assistant_mode_id="coding_debug",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                scope=MemoryScope.CONVERSATION,
            ),
        )

        archived = await memories.get_memory_object(str(belief["id"]), "usr_1")
        assert archived is not None
        assert archived["status"] == MemoryStatus.ARCHIVED.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_revise_rolls_back_all_writes_when_apply_step_fails(monkeypatch) -> None:
    connection, memories, beliefs, reviser = await _build_runtime("REINFORCE")
    try:
        belief = await _seed_belief(memories, beliefs)
        evidence = await _seed_evidence(memories)

        async def fail_link(*args, **kwargs) -> None:
            raise RuntimeError("link failure")

        monkeypatch.setattr(reviser, "_link_evidence", fail_link)

        with pytest.raises(RuntimeError, match="link failure"):
            await reviser.revise(
                belief_id=str(belief["id"]),
                new_evidence=[evidence],
                context=RevisionContext(
                    user_id="usr_1",
                    claim_key="response_style.debugging",
                    claim_value=json.dumps("terse"),
                    source_message_id="msg_9",
                    assistant_mode_id="coding_debug",
                    workspace_id="wrk_1",
                    conversation_id="cnv_1",
                    scope=MemoryScope.CONVERSATION,
                ),
            )

        current = await _current_version(connection, str(belief["id"]))
        unchanged = await memories.get_memory_object(str(belief["id"]), "usr_1")
        assert current["support_count"] == 1
        assert unchanged is not None
        assert unchanged["confidence"] == 0.8
        assert unchanged["stability"] == 0.7
    finally:
        await connection.close()
