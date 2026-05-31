"""Tests for conservative conversation rebuild purging."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

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
from atagia.core.summary_repository import SummaryRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    SummaryViewKind,
)
import atagia.services.admin_rebuild_service as admin_rebuild_module
from atagia.services.admin_rebuild_service import AdminRebuildService
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.llm_client import (
    LLMError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMRunGuardError,
    OutputLimitExceededError,
    StructuredOutputError,
)
from atagia.services.llm_run_guard import LLMRunGuardDecision
from tests.extraction_payload_support import rich_extraction_payload_to_lean

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
        self.payload = rich_extraction_payload_to_lean(payload)

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
                    rich_extraction_payload_to_lean(
                        {
                            "evidences": [
                                {
                                    "canonical_text": canonical_text,
                                    "index_text": f"Context for {canonical_text.lower()} from replayed conversation history.",
                                    "scope": "conversation",
                                    "confidence": 0.92,
                                    "source_kind": "extracted",
                                    "privacy_level": 0,
                                    "language_codes": ["en"],
                                    "payload": {"kind": "fact"},
                                }
                            ],
                            "beliefs": [],
                            "contract_signals": [],
                            "state_updates": [],
                            "mode_guess": None,
                            "nothing_durable": False,
                        }
                    )
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
        if purpose == "summary_chunk_segmentation":
            prompt = request.messages[1].content
            message_sequences = [int(item) for item in re.findall(r'<message seq="(\d+)"', prompt)]
            if not message_sequences:
                raise AssertionError("Expected message sequences in chunk segmentation prompt")
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "episodes": [
                            {
                                "start_seq": min(message_sequences),
                                "end_seq": max(message_sequences),
                                "summary_text": "Replay chunk summary.",
                            }
                        ]
                    }
                ),
            )
        if purpose == "episode_synthesis":
            prompt = request.messages[1].content
            chunk_ids = re.findall(r'<conversation_chunk id="([^"]+)"', prompt)
            if not chunk_ids:
                raise AssertionError("Expected conversation chunk IDs in episode synthesis prompt")
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_key": "replay",
                                "summary_text": "Replay episode summary.",
                            }
                        ],
                        "chunk_episode_keys": ["replay"] * len(chunk_ids),
                    }
                ),
            )
        if purpose == "thematic_profile_synthesis":
            prompt = request.messages[1].content
            episode_ids = re.findall(r'<episode id="([^"]+)"', prompt)
            if not episode_ids:
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text='{"profiles": []}',
                )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "profiles": [
                            {
                                "source_memory_ids": [episode_ids[0]],
                                "summary_text": "Replay thematic profile.",
                            }
                        ]
                    }
                ),
            )
        raise AssertionError(f"Unexpected completion purpose in rebuild replay test: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embedding is not expected in this test: {request.metadata}")


class EpisodeOutputLimitReplayProvider(RebuildReplayProvider):
    def __init__(self) -> None:
        super().__init__()
        self._remaining_episode_output_limit_failures = 1

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = request.metadata.get("purpose")
        if purpose == "summary_chunk_segmentation":
            self.requests.append(request)
            prompt = request.messages[1].content
            message_sequences = [
                int(item) for item in re.findall(r'<message seq="(\d+)"', prompt)
            ]
            if not message_sequences:
                raise AssertionError("Expected message sequences in chunk segmentation prompt")
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "episodes": [
                            {
                                "start_seq": sequence,
                                "end_seq": sequence,
                                "summary_text": f"Chunk summary {sequence}.",
                            }
                            for sequence in message_sequences
                        ]
                    }
                ),
            )
        if (
            purpose == "episode_synthesis"
            and self._remaining_episode_output_limit_failures
        ):
            self.requests.append(request)
            self._remaining_episode_output_limit_failures -= 1
            raise OutputLimitExceededError(
                "openai stopped because it reached max output tokens",
                provider="openai",
                finish_reason="length",
                max_output_tokens=8192,
                partial_output_excerpt='"collaboration_and_growth","collaboration_and_growth"',
            )
        return await super().complete(request)


class RecordingEmbeddingIndex(EmbeddingIndex):
    @property
    def vector_limit(self) -> int:
        return 1

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        return None

    async def search(self, query: str, user_id: str, top_k: int):
        return []

    async def delete(self, memory_id: str) -> None:
        return None


def _settings(**overrides: object) -> Settings:
    settings = Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        topic_working_set_enabled=False,
    )
    return Settings(**{**asdict(settings), **overrides})


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
async def test_conversation_rebuild_purge_removes_hierarchy_summaries_and_mirrors() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="User prefers patch-first debugging.",
            payload={"claim_key": "workflow.debugging.style", "claim_value": "patch_first"},
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_belief",
        )
        await connection.execute(
            """
            INSERT INTO summary_views(
                id, user_id, conversation_id, workspace_id, source_message_start_seq, source_message_end_seq,
                summary_kind, summary_text, source_object_ids_json, maya_score, model, hierarchy_level, created_at
            )
            VALUES
                ('sum_episode', 'usr_1', NULL, NULL, NULL, NULL, 'episode', 'Episode summary', '["mem_belief"]', 1.5, 'score', 1, '2026-04-10T12:00:00+00:00'),
                ('sum_theme', 'usr_1', NULL, NULL, NULL, NULL, 'thematic_profile', 'Thematic profile', '["mem_belief"]', 1.5, 'score', 2, '2026-04-10T12:00:01+00:00')
            """
        )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Episode summary",
            payload={"summary_view_id": "sum_episode", "summary_kind": "episode", "hierarchy_level": 1, "source_object_ids": ["mem_belief"]},
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.7,
            privacy_level=0,
            memory_id="sum_mem_sum_episode",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Thematic profile",
            payload={"summary_view_id": "sum_theme", "summary_kind": "thematic_profile", "hierarchy_level": 2, "source_object_ids": ["mem_belief"]},
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.7,
            privacy_level=0,
            memory_id="sum_mem_sum_theme",
        )

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(provider_name=NoOpProvider.name, providers=[NoOpProvider()]),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        await service._purge_conversation_state("usr_1", "cnv_1")

        cursor = await connection.execute("SELECT id FROM summary_views ORDER BY id ASC")
        summary_rows = await cursor.fetchall()
        assert [row["id"] for row in summary_rows] == []
        assert await memories.get_memory_object("sum_mem_sum_episode", "usr_1") is None
        assert await memories.get_memory_object("sum_mem_sum_theme", "usr_1") is None
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
        extracted_rows = [row for row in stored if row["object_type"] == MemoryObjectType.EVIDENCE.value]

        assert result.processed_messages == 2
        assert result.extract_jobs_processed == 2
        assert result.contract_jobs_processed == 1
        assert result.initial_context_package_refresh_jobs_processed >= 1
        assert {"msg_1", "msg_2"} <= source_ids
        assert extracted_rows
        assert all(row["index_text"] for row in extracted_rows)
        pending_cursor = await connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM worker_job_runs
            WHERE status IN ('queued', 'running', 'retrying')
            """
        )
        pending_row = await pending_cursor.fetchone()
        assert int(pending_row["count"]) == 0
        cursor = await connection.execute(
            """
            SELECT edge.support_kind, span.span_role, span.quote_text, span.message_id
            FROM memory_support_edges AS edge
            JOIN memory_evidence_spans AS span
              ON span.support_edge_id = edge.id
            WHERE edge.memory_id IN ({placeholders})
            ORDER BY span.message_id ASC, span.span_role ASC
            """.format(
                placeholders=", ".join("?" for _ in extracted_rows)
            ),
            tuple(str(row["id"]) for row in extracted_rows),
        )
        packets = [dict(row) for row in await cursor.fetchall()]
        source_packets = [row for row in packets if row["span_role"] == "source"]
        assert source_packets
        assert {row["message_id"] for row in source_packets} == {"msg_1", "msg_2"}
        assert any(row["quote_text"] == "User fact" for row in source_packets)
        assert any(row["quote_text"] == "Assistant plan" for row in source_packets)
        assert {row["support_kind"] for row in source_packets} == {"direct"}
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


@pytest.mark.asyncio
async def test_rebuild_conversation_paginates_and_uses_placeholder_recent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admin_rebuild_module, "REBUILD_MESSAGE_PAGE_SIZE", 2)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "One")
        await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "large biography segment " * 800,
            None,
            {},
        )
        await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Assistant plan", 2, {})
        await messages.create_message("msg_3", "cnv_1", "user", 3, "User fact", 2, {})
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

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        extraction_prompts = [
            request.messages[1].content
            for request in provider.requests
            if request.metadata.get("purpose") == "memory_extraction"
        ]
        assistant_prompt = next(prompt for prompt in extraction_prompts if "Assistant plan" in prompt)
        assert result.processed_messages == 3
        assert result.extract_jobs_processed == 3
        assert "[Skipped message | id=msg_1 seq=1 role=user" in assistant_prompt
        assert "policy=mechanical_size_threshold" in assistant_prompt
        assert "large biography segment" not in assistant_prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_recreates_hierarchy_for_short_conversation() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_old_episode",
                "conversation_id": None,
                "workspace_id": None,
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "summary_text": "Old episode.",
                "source_object_ids_json": [],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-10T11:00:00+00:00",
            }
        )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Old episode.",
            payload={"summary_view_id": "sum_old_episode", "summary_kind": "episode", "hierarchy_level": 1, "source_object_ids": []},
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.7,
            privacy_level=0,
            memory_id="sum_mem_sum_old_episode",
        )

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation("usr_1", "cnv_1")

        chunk_rows = await summaries.list_conversation_chunks("usr_1", "cnv_1", limit=10)
        episode_rows = await summaries.list_summaries_by_kind("usr_1", SummaryViewKind.EPISODE)
        thematic_rows = await summaries.list_summaries_by_kind("usr_1", SummaryViewKind.THEMATIC_PROFILE)

        assert result.conversation_compaction_jobs_processed == 1
        assert result.episode_compaction_jobs_processed == 1
        assert result.thematic_profile_jobs_processed == 1
        assert len(chunk_rows) == 1
        assert len(episode_rows) == 1
        assert len(thematic_rows) == 1
        assert await memories.get_memory_object("sum_mem_sum_old_episode", "usr_1") is None
        assert await memories.get_memory_object(f"sum_mem_{episode_rows[0]['id']}", "usr_1") is not None
        assert await memories.get_memory_object(f"sum_mem_{thematic_rows[0]['id']}", "usr_1") is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_recovers_episode_synthesis_output_limit() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    provider = EpisodeOutputLimitReplayProvider()
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        for sequence in range(1, 5):
            await messages.create_message(
                f"msg_{sequence}",
                "cnv_1",
                "user" if sequence % 2 else "assistant",
                sequence,
                f"Message {sequence}",
                sequence,
                {},
                f"2026-04-10T12:0{sequence}:00+00:00",
            )

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=EpisodeOutputLimitReplayProvider.name,
                providers=[provider],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation("usr_1", "cnv_1")

        episode_rows = await summaries.list_summaries_by_kind("usr_1", SummaryViewKind.EPISODE)
        episode_chunk_counts = [
            request.metadata.get("chunk_count")
            for request in provider.requests
            if request.metadata.get("purpose") == "episode_synthesis"
        ]

        assert result.status == "rebuilt"
        assert result.recoverable_job_failures == 0
        assert result.conversation_compaction_jobs_processed == 1
        assert result.episode_compaction_jobs_processed == 1
        assert len(episode_rows) == 2
        assert episode_chunk_counts == [4, 2, 2]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_can_skip_final_compaction() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    provider = RebuildReplayProvider()
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")

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

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        assert result.processed_messages == 1
        assert result.conversation_compaction_jobs_processed == 0
        assert result.episode_compaction_jobs_processed == 0
        assert result.thematic_profile_jobs_processed == 0
        assert result.workspace_rollup_jobs_processed == 0
        assert await summaries.list_conversation_chunks("usr_1", "cnv_1", limit=10) == []
        assert await summaries.list_summaries_by_kind("usr_1", SummaryViewKind.EPISODE) == []
        assert await summaries.list_summaries_by_kind("usr_1", SummaryViewKind.THEMATIC_PROFILE) == []
        assert not any(
            request.metadata.get("purpose")
            in {
                "summary_chunk_segmentation",
                "episode_synthesis",
                "thematic_profile_synthesis",
                "workspace_rollup_synthesis",
            }
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_purges_character_rollups_when_final_compaction_is_skipped() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Character chat",
            user_persona_id="persona_writer",
            character_id="char_debug",
            platform_id="web",
        )
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_old_character_rollup",
                "conversation_id": None,
                "workspace_id": None,
                "user_persona_id": "persona_writer",
                "platform_id": "web",
                "character_id": "char_debug",
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "character_rollup",
                "hierarchy_level": 0,
                "summary_text": "Old character rollup.",
                "source_object_ids_json": ["mem_old"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-09T12:00:00+00:00",
            },
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_old_character_rollup",
            summary_kind=SummaryViewKind.CHARACTER_ROLLUP,
            hierarchy_level=0,
            summary_text="Old character rollup.",
            source_object_ids=["mem_old"],
            created_at="2026-04-09T12:00:00+00:00",
            scope=MemoryScope.WORKSPACE,
            user_persona_id="persona_writer",
            character_id="char_debug",
            scope_canonical=MemoryScope.CHARACTER.value,
        )
        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        assert result.workspace_rollup_jobs_processed == 0
        assert await summaries.list_character_rollups("usr_1", "char_debug", limit=10) == []
        assert await memories.get_memory_object("sum_mem_sum_old_character_rollup", "usr_1") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_user_processes_workspace_rollups_synchronously() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_user("usr_1")
        workspace_rollups = await summaries.list_character_rollups("usr_1", "wsp_1", limit=10)

        assert result.workspace_rollup_jobs_processed == 1
        assert len(workspace_rollups) == 1
        assert workspace_rollups[0]["summary_text"] == "Workspace summary"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_user_processes_character_rollups_without_workspace() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    try:
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Character chat",
            character_id="char_debug",
            platform_id="web",
        )
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {}, "2023-05-08T13:56:00")

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_user("usr_1")
        character_rollups = await summaries.list_character_rollups("usr_1", "char_debug", limit=10)

        assert result.workspace_ids == []
        assert result.workspace_rollup_jobs_processed == 1
        assert len(character_rollups) == 1
        assert character_rollups[0]["workspace_id"] is None
        assert character_rollups[0]["character_id"] == "char_debug"
        assert character_rollups[0]["summary_text"] == "Workspace summary"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_admin_rebuild_service_passes_embedding_index_to_revision_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    captured: dict[str, object] = {}
    embedding_index = RecordingEmbeddingIndex()
    try:
        await users.create_user("usr_1")

        class FakeRevisionWorker:
            def __init__(self, *, embedding_index, **kwargs) -> None:
                captured["embedding_index"] = embedding_index

            async def process_job(self, payload):
                return None

        monkeypatch.setattr("atagia.services.admin_rebuild_service.RevisionWorker", FakeRevisionWorker)

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=NoOpProvider.name,
                providers=[NoOpProvider()],
            ),
            embedding_index=embedding_index,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        await service.rebuild_user("usr_1")

        assert captured["embedding_index"] is embedding_index
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_skips_recoverable_contract_projection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)

    class FailingContractWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise StructuredOutputError(
                "Provider returned invalid structured output",
                details=("contract projection validation failed",),
            )

    monkeypatch.setattr(admin_rebuild_module, "ContractWorker", FailingContractWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        assert result.processed_messages == 1
        assert result.status == "rebuilt_partial"
        assert result.extract_jobs_processed == 1
        assert result.contract_jobs_processed == 0
        assert result.recoverable_job_failures == 1
        assert result.recoverable_contract_job_failures == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_skips_recoverable_graph_projection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)

    class FailingGraphWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise StructuredOutputError(
                "Provider returned invalid structured output",
                details=("graph projection validation failed",),
            )

    monkeypatch.setattr(admin_rebuild_module, "GraphSyncWorker", FailingGraphWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(graph_projection_enabled=True),
        )

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        stored = await memories.list_for_user("usr_1")
        assert result.processed_messages == 1
        assert result.status == "rebuilt_partial"
        assert result.extract_jobs_processed == 1
        assert result.graph_jobs_processed == 0
        assert result.recoverable_job_failures == 1
        assert result.recoverable_graph_job_failures == 1
        assert [row["canonical_text"] for row in stored] == ["User fact"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_skips_recoverable_output_limit_extract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)

    class TruncatingIngestWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise OutputLimitExceededError(
                "openrouter stopped because it reached max output tokens "
                "(finish_reason=length)"
            )

    monkeypatch.setattr(admin_rebuild_module, "IngestWorker", TruncatingIngestWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        assert result.processed_messages == 1
        assert result.status == "rebuilt_partial"
        assert result.extract_jobs_processed == 0
        assert result.recoverable_job_failures >= 1
        assert result.recoverable_extract_job_failures == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_skips_generic_llm_extract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)

    class FailingIngestWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise LLMError("JSON error injected into SSE stream")

    monkeypatch.setattr(admin_rebuild_module, "IngestWorker", FailingIngestWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        result = await service.rebuild_conversation(
            "usr_1",
            "cnv_1",
            skip_final_compaction=True,
        )

        assert result.processed_messages == 1
        assert result.status == "rebuilt_partial"
        assert result.extract_jobs_processed == 0
        assert result.recoverable_job_failures == 1
        assert result.recoverable_extract_job_failures == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_propagates_llm_run_guard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)

    class GuardedIngestWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise LLMRunGuardError(
                LLMRunGuardDecision(
                    healthy=False,
                    should_block=True,
                    violations=("total failed LLM calls exceeded 1: 2",),
                    snapshot={"status": "failed", "failed_calls": 2},
                )
            )

    monkeypatch.setattr(admin_rebuild_module, "IngestWorker", GuardedIngestWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        with pytest.raises(LLMRunGuardError):
            await service.rebuild_conversation(
                "usr_1",
                "cnv_1",
                skip_final_compaction=True,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rebuild_conversation_propagates_non_llm_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)

    class BrokenContractWorker:
        def __init__(self, **kwargs) -> None:
            pass

        async def process_job(self, payload):
            raise RuntimeError("database invariant failed")

    monkeypatch.setattr(admin_rebuild_module, "ContractWorker", BrokenContractWorker)

    try:
        await users.create_user("usr_1")
        await workspaces.create_workspace("wsp_1", "usr_1", "Workspace")
        await conversations.create_conversation("cnv_1", "usr_1", "wsp_1", "coding_debug", "One")
        await messages.create_message("msg_1", "cnv_1", "user", 1, "User fact", 2, {})

        service = AdminRebuildService(
            connection=connection,
            llm_client=LLMClient(
                provider_name=RebuildReplayProvider.name,
                providers=[RebuildReplayProvider()],
            ),
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=_settings(),
        )

        with pytest.raises(RuntimeError, match="database invariant failed"):
            await service.rebuild_conversation(
                "usr_1",
                "cnv_1",
                skip_final_compaction=True,
            )
    finally:
        await connection.close()
