"""Tests for conservative conversation rebuild purging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

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
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.admin_rebuild_service import AdminRebuildService
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


class NoOpProvider(LLMProvider):
    name = "admin-rebuild-service-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM completion is not expected in this test: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embedding is not expected in this test: {request.metadata}")


class ExtractorProvider(LLMProvider):
    name = "admin-rebuild-extractor"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"equivalent": true}',
            )
        if request.metadata.get("purpose") == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"is_explicit": true, "reasoning": "Test classifier response."}',
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embedding is not expected in this test: {request.metadata}")


class RebuildReplayProvider(LLMProvider):
    name = "admin-rebuild-replay"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata.get("purpose")
        if purpose == "memory_extraction":
            prompt = request.messages[1].content
            canonical_text = (
                "Assistant plan"
                if "Assistant plan" in prompt
                else "User fact"
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "evidences": [
                            {
                                "canonical_text": canonical_text,
                                "scope": "conversation",
                                "confidence": 0.92,
                                "source_kind": "extracted",
                                "privacy_level": 0,
                                "payload": {"kind": "fact"},
                            }
                        ],
                        "beliefs": [],
                        "contract_signals": [],
                        "state_updates": [],
                        "mode_guess": None,
                        "nothing_durable": False,
                    }
                ),
            )
        if purpose == "contract_projection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"signals": [], "nothing_durable": true}',
            )
        if purpose == "consequence_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_consequence": False,
                        "action_description": "",
                        "outcome_description": "",
                        "outcome_sentiment": "neutral",
                        "confidence": 0.0,
                        "likely_action_message_id": None,
                    }
                ),
            )
        if purpose == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"is_explicit": true, "reasoning": "Test classifier response."}',
            )
        if purpose == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"equivalent": true}',
            )
        if purpose == "workspace_rollup_synthesis":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text='{"summary_text": "Workspace summary", "cited_memory_ids": []}',
            )
        raise AssertionError(f"Unexpected completion purpose in rebuild replay test: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embedding is not expected in this test: {request.metadata}")


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


@pytest.mark.asyncio
async def test_conversation_rebuild_purge_keeps_shared_cross_conversation_memory() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await conversations.create_conversation("cnv_2", "usr_1", "wsp_1", "coding_debug", "Two")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "First conversation fact", 4, {})
        await messages.create_message("msg_2", "cnv_2", "user", 1, "Second conversation fact", 4, {})
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wsp_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.WORKSPACE,
            canonical_text="Shared workspace belief",
            payload={
                "claim_key": "workflow.testing.framework",
                "claim_value": "pytest",
                "source_message_ids": ["msg_1", "msg_2"],
            },
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.9,
            vitality=0.5,
            privacy_level=1,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_shared",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wsp_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Conversation-owned evidence",
            payload={"source_message_ids": ["msg_1"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            vitality=0.5,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_owned",
        )

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(provider_name=NoOpProvider.name, providers=[NoOpProvider()]),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        purgeable_ids = await service._memory_ids_for_conversation("usr_1", "cnv_1")
        await service._purge_conversation_state("usr_1", "cnv_1")

        assert purgeable_ids == ["mem_owned"]
        assert await memories.get_memory_object("mem_shared", "usr_1") is not None
        assert await memories.get_memory_object("mem_owned", "usr_1") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_extractor_shared_workspace_memory_survives_conversation_purge() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    resolved_policy = PolicyResolver().resolve(manifest_loader.load_all()["coding_debug"], None, None)
    provider = ExtractorProvider(
        {
            "evidences": [
                {
                    "canonical_text": "The workspace uses pytest for backend testing",
                    "scope": "workspace",
                    "confidence": 0.92,
                    "source_kind": "extracted",
                    "privacy_level": 0,
                    "payload": {"kind": "tooling"},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    extractor = MemoryExtractor(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=InProcessBackend(),
        settings=_settings(),
    )
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await conversations.create_conversation("cnv_2", "usr_1", "wsp_1", "coding_debug", "Two")
        first = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "The workspace uses pytest for backend testing.",
            7,
            {},
        )
        second = await messages.create_message(
            "msg_2",
            "cnv_2",
            "user",
            1,
            "The workspace uses pytest for backend testing.",
            7,
            {},
        )

        await extractor.extract(
            message_text=str(first["text"]),
            role="user",
            conversation_context=ExtractionConversationContext(
                user_id="usr_1",
                conversation_id="cnv_1",
                source_message_id="msg_1",
                workspace_id="wsp_1",
                assistant_mode_id="coding_debug",
                recent_messages=[],
            ),
            resolved_policy=resolved_policy,
        )
        await extractor.extract(
            message_text=str(second["text"]),
            role="user",
            conversation_context=ExtractionConversationContext(
                user_id="usr_1",
                conversation_id="cnv_2",
                source_message_id="msg_2",
                workspace_id="wsp_1",
                assistant_mode_id="coding_debug",
                recent_messages=[],
            ),
            resolved_policy=resolved_policy,
        )

        stored = await memories.list_for_user("usr_1")
        assert len(stored) == 1
        assert stored[0]["conversation_id"] is None
        assert stored[0]["payload_json"]["source_message_ids"] == ["msg_1", "msg_2"]

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(provider_name=NoOpProvider.name, providers=[NoOpProvider()]),
            embedding_index=None,
            clock=clock,
            manifest_loader=manifest_loader,
            settings=_settings(),
        )

        purgeable_ids = await service._memory_ids_for_conversation("usr_1", "cnv_1")
        await service._purge_conversation_state("usr_1", "cnv_1")

        assert purgeable_ids == []
        assert await memories.get_memory_object(str(stored[0]["id"]), "usr_1") is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_wipes_context_cache_for_user() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    cache_backend = InProcessBackend()
    try:
        await users.create_user("usr_1")
        await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "One")
        await cache_backend.set_context_view(
            "ctx:1",
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await cache_backend.set_context_view(
            "ctx:2",
            {"user_id": "usr_1", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )
        await cache_backend.set_context_view(
            "ctx:3",
            {"user_id": "usr_2", "conversation_id": "cnv_9"},
            ttl_seconds=60,
        )

        result = await AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(provider_name=NoOpProvider.name, providers=[NoOpProvider()]),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
            storage_backend=cache_backend,
        ).rebuild_conversation("usr_1", "cnv_1")

        assert result.conversation_ids == ["cnv_1"]
        assert await cache_backend.get_context_view("ctx:1") is None
        assert await cache_backend.get_context_view("ctx:2") is None
        assert await cache_backend.get_context_view("ctx:3") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_9",
        }
    finally:
        await cache_backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_user_wipes_context_cache_for_user() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    cache_backend = InProcessBackend()
    try:
        await users.create_user("usr_1")
        await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "One")
        await conversations.create_conversation("cnv_2", "usr_1", None, "coding_debug", "Two")
        await cache_backend.set_context_view(
            "ctx:1",
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await cache_backend.set_context_view(
            "ctx:2",
            {"user_id": "usr_1", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )
        await cache_backend.set_context_view(
            "ctx:3",
            {"user_id": "usr_2", "conversation_id": "cnv_9"},
            ttl_seconds=60,
        )

        result = await AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(provider_name=NoOpProvider.name, providers=[NoOpProvider()]),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
            storage_backend=cache_backend,
        ).rebuild_user("usr_1")

        assert result.conversation_ids == ["cnv_1", "cnv_2"]
        assert await cache_backend.get_context_view("ctx:1") is None
        assert await cache_backend.get_context_view("ctx:2") is None
        assert await cache_backend.get_context_view("ctx:3") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_9",
        }
    finally:
        await cache_backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_replays_assistant_messages_for_extraction() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")
        await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Assistant plan", 2, {}, "2023-05-08T14:10:00")
        provider = RebuildReplayProvider()

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[provider],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation("usr_1", "cnv_1")

        stored = await memories.list_for_user("usr_1")
        source_ids = {
            source_id
            for row in stored
            for source_id in row["payload_json"].get("source_message_ids", [])
        }

        assert result.processed_messages == 2
        assert result.extract_jobs_processed == 2
        assert result.contract_jobs_processed == 1
        assert {"msg_1", "msg_2"} <= source_ids
        extraction_prompts = [
            request.messages[1].content
            for request in provider.requests
            if request.metadata.get("purpose") == "memory_extraction"
        ]
        contract_prompts = [
            request.messages[1].content
            for request in provider.requests
            if request.metadata.get("purpose") == "contract_projection"
        ]
        assert "<message_timestamp>2023-05-08T13:56:00</message_timestamp>" in contract_prompts[0]
        assert any("<message_timestamp>2023-05-08T13:56:00</message_timestamp>" in prompt for prompt in extraction_prompts)
        assert any("<message_timestamp>2023-05-08T14:10:00</message_timestamp>" in prompt for prompt in extraction_prompts)
    finally:
        await connection.close()
