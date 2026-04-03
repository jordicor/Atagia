"""Tests for building consequence chains."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.consequence_builder import ConsequenceChainBuilder
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ConsequenceSignal,
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
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
    name = "consequence-builder-tests"

    def __init__(self, outputs: list[str], *, fail: bool = False) -> None:
        self.outputs = list(outputs)
        self.fail = fail
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("synthetic tendency failure")
        if not self.outputs:
            raise AssertionError("No queued output left for this test")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in consequence builder tests")


async def _build_runtime(*, outputs: list[str], fail: bool = False):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 2, 14, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    provider = QueueProvider(outputs, fail=fail)
    builder = ConsequenceChainBuilder(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
    )
    resolved_policy = PolicyResolver().resolve(manifest_loader.get("coding_debug"), None, None)
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_user_1",
        workspace_id="wrk_1",
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )
    return connection, memories, builder, resolved_policy, context


async def _seed_action_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str = "mem_action",
    message_id: str = "msg_assistant_1",
) -> dict[str, object]:
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Suggested a large refactor.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.83,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        payload={"source_message_ids": [message_id]},
        memory_id=memory_id,
    )


@pytest.mark.asyncio
async def test_builds_complete_chain_with_tendency_and_links() -> None:
    connection, memories, builder, resolved_policy, context = await _build_runtime(
        outputs=[json.dumps({"tendency_text": "This workspace is fragile to sweeping changes."})]
    )
    try:
        action = await _seed_action_memory(memories)
        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description="Suggested a large refactor.",
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.86,
                likely_action_message_id="msg_assistant_1",
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        assert result is not None
        assert result.action_memory_id == str(action["id"])
        chain_cursor = await connection.execute("SELECT * FROM consequence_chains WHERE id = ?", (result.chain_id,))
        chain_row = await chain_cursor.fetchone()
        assert chain_row["tendency_belief_id"] == result.tendency_belief_id
        tendency = await memories.get_memory_object(str(result.tendency_belief_id), "usr_1")
        assert tendency is not None
        assert tendency["object_type"] == MemoryObjectType.CONSEQUENCE_CHAIN.value
        assert tendency["canonical_text"] == "This workspace is fragile to sweeping changes."
        link_cursor = await connection.execute(
            "SELECT src_memory_id, dst_memory_id, relation_type FROM memory_links ORDER BY id ASC"
        )
        links = await link_cursor.fetchall()
        assert {row["relation_type"] for row in links} == {"led_to", "derived_from", "exception_to"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_builds_chain_without_tendency_when_llm_inference_fails() -> None:
    connection, memories, builder, resolved_policy, context = await _build_runtime(outputs=[], fail=True)
    try:
        action = await _seed_action_memory(memories)
        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description="Suggested a large refactor.",
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.72,
                likely_action_message_id="msg_assistant_1",
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        assert result is not None
        assert result.action_memory_id == str(action["id"])
        assert result.tendency_belief_id is None
        cursor = await connection.execute("SELECT tendency_belief_id FROM consequence_chains WHERE id = ?", (result.chain_id,))
        row = await cursor.fetchone()
        assert row["tendency_belief_id"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_builder_finds_existing_action_memory_instead_of_creating_new_one() -> None:
    connection, memories, builder, resolved_policy, context = await _build_runtime(
        outputs=[json.dumps({"tendency_text": "Use smaller patches in this workspace."})]
    )
    try:
        action = await _seed_action_memory(memories, memory_id="mem_existing_action")
        before = await memories.list_for_user("usr_1")

        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description="Suggested a large refactor.",
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.76,
                likely_action_message_id="msg_assistant_1",
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        after = await memories.list_for_user("usr_1")
        action_like_memories = [
            row for row in after if row["canonical_text"] == "Suggested a large refactor."
        ]
        assert result is not None
        assert result.action_memory_id == str(action["id"])
        assert len(after) == len(before) + 2
        assert len(action_like_memories) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_builder_creates_action_outcome_and_tendency_links() -> None:
    connection, memories, builder, resolved_policy, context = await _build_runtime(
        outputs=[json.dumps({"tendency_text": "This workflow prefers incremental changes."})]
    )
    try:
        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description="Suggested a large refactor.",
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.8,
                likely_action_message_id=None,
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        assert result is not None
        cursor = await connection.execute(
            "SELECT src_memory_id, dst_memory_id, relation_type FROM memory_links ORDER BY relation_type ASC"
        )
        links = await cursor.fetchall()
        relation_types = [row["relation_type"] for row in links]
        assert relation_types == ["derived_from", "exception_to", "led_to"]
        action = await memories.get_memory_object(result.action_memory_id, "usr_1")
        outcome = await memories.get_memory_object(result.outcome_memory_id, "usr_1")
        tendency = await memories.get_memory_object(str(result.tendency_belief_id), "usr_1")
        assert action is not None
        assert outcome is not None
        assert tendency is not None
        assert action["source_kind"] == MemorySourceKind.INFERRED.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_builder_returns_none_when_action_memory_cannot_be_found_or_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, _memories, builder, resolved_policy, context = await _build_runtime(
        outputs=[json.dumps({"tendency_text": "Unused tendency."})]
    )
    try:
        async def fail_create(*args, **kwargs):
            raise RuntimeError("cannot create action memory")

        monkeypatch.setattr(builder, "_create_inferred_action_memory", fail_create)

        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description="Suggested a large refactor.",
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.8,
                likely_action_message_id=None,
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        assert result is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_builder_safely_normalizes_punctuation_heavy_action_lookup() -> None:
    connection, memories, builder, resolved_policy, context = await _build_runtime(
        outputs=[json.dumps({"tendency_text": "Keep retry changes narrowly scoped."})]
    )
    try:
        action = await _seed_action_memory(memories, memory_id="mem_safe_action")

        result = await builder.build_chain(
            ConsequenceSignal(
                is_consequence=True,
                action_description='Suggested a "large: refactor" (retry-guard) OR websocket?',
                outcome_description="Regressions appeared afterwards.",
                outcome_sentiment="negative",
                confidence=0.8,
                likely_action_message_id=None,
            ),
            user_id="usr_1",
            conversation_context=context,
            resolved_policy=resolved_policy,
        )

        assert result is not None
        assert result.action_memory_id == str(action["id"])
    finally:
        await connection.close()
