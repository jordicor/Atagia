"""Tests for retrieval replay service."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind
from atagia.models.schemas_replay import AblationConfig
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.replay_service import ReplayService
from atagia.services.retrieval_pipeline import RetrievalPipeline

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class ReplayProvider(LLMProvider):
    name = "replay-service-tests"

    def __init__(self, score_map: dict[str, float]) -> None:
        self.score_map = dict(score_map)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "needs": [],
                        "temporal_range": None,
                        "sub_queries": ["retry loop"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "retry loop",
                                "fts_phrase": "retry loop",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            payload = [
                {"memory_id": memory_id, "llm_applicability": self.score_map.get(memory_id, 0.5)}
                for memory_id in memory_ids
            ]
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text=json.dumps(payload))
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in replay service tests")


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
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other Chat")
    await messages.create_message("msg_1", "cnv_1", "user", 1, "Please help me debug this retry loop.", 7, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Try the previous workaround.", 5, {})
    await messages.create_message("msg_3", "cnv_1", "user", 3, "The retry loop still fails.", 6, {})
    await messages.create_message("msg_4", "cnv_1", "assistant", 4, "Let's narrow the scope.", 5, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="please help debug retry loop direct fix",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id="mem_1",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="please help debug retry loop previous workaround with extra context",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=1,
        memory_id="mem_2",
    )
    await events.create_event(
        {
            "id": "ret_1",
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry loop"]},
            "selected_memory_ids_json": ["mem_2"],
            "context_view_json": {
                "selected_memory_ids": ["mem_2"],
                "contract_block": "",
                "workspace_block": "",
                "memory_block": "memories",
                "state_block": "",
                "total_tokens_estimate": 80,
            },
            "outcome_json": {
                "scored_candidates": [
                    {"memory_id": "mem_1", "final_score": 0.3},
                    {"memory_id": "mem_2", "final_score": 0.95},
                ]
            },
            "created_at": "2026-04-05T15:00:00+00:00",
        }
    )
    await events.create_event(
        {
            "id": "ret_2",
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_3",
            "response_message_id": "msg_4",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry loop fails"]},
            "selected_memory_ids_json": ["mem_2"],
            "context_view_json": {
                "selected_memory_ids": ["mem_2"],
                "contract_block": "",
                "workspace_block": "",
                "memory_block": "memories",
                "state_block": "",
                "total_tokens_estimate": 90,
            },
            "outcome_json": {
                "scored_candidates": [
                    {"memory_id": "mem_1", "final_score": 0.2},
                    {"memory_id": "mem_2", "final_score": 0.9},
                ]
            },
            "created_at": "2026-04-05T15:10:00+00:00",
        }
    )
    provider = ReplayProvider({"mem_1": 0.3, "mem_2": 0.95})
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=_settings(),
    )
    replay_service = ReplayService(
        connection=connection,
        retrieval_pipeline=pipeline,
        clock=clock,
        settings=_settings(),
    )
    return connection, replay_service


@pytest.mark.asyncio
async def test_replay_single_event_returns_pipeline_result_and_comparison() -> None:
    connection, replay_service = await _build_runtime()
    try:
        result = await replay_service.replay_retrieval_event("ret_1", "usr_1")

        assert result.original_event_id == "ret_1"
        assert result.replay_pipeline_result.retrieval_plan.conversation_id == "cnv_1"
        assert result.comparison.replay_items_count >= 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_replay_event_with_ablation_changes_result() -> None:
    connection, replay_service = await _build_runtime()
    try:
        result = await replay_service.replay_retrieval_event(
            "ret_1",
            "usr_1",
            ablation=AblationConfig(
                skip_applicability_scoring=True,
                override_retrieval_params={"final_context_items": 1, "privacy_ceiling": 0},
            ),
        )

        assert result.comparison.memories_only_original == ["mem_2"]
        assert result.comparison.memories_only_replay == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_replay_conversation_returns_one_result_per_user_message() -> None:
    connection, replay_service = await _build_runtime()
    try:
        results = await replay_service.replay_conversation("cnv_1", "usr_1")

        assert [result.original_event_id for result in results] == ["ret_1", "ret_2"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_replay_verifies_user_id_ownership_and_missing_event() -> None:
    connection, replay_service = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="Retrieval event not found for user"):
            await replay_service.replay_retrieval_event("ret_1", "usr_2")
        with pytest.raises(ValueError, match="Retrieval event not found for user"):
            await replay_service.replay_retrieval_event("ret_missing", "usr_1")
    finally:
        await connection.close()
