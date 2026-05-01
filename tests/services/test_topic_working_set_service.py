"""Tests for async Topic Working Set refresh orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.topic_working_set_service import TopicWorkingSetRefreshService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class TopicProvider(LLMProvider):
    name = "topic-service"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("No topic payload queued")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payloads.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in Topic Working Set service tests")


def _settings(**overrides: object) -> Settings:
    base = Settings(
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
        llm_forced_global_model="openai/reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        topic_working_set_refresh_message_lag=4,
        topic_working_set_stale_message_lag=10,
        topic_working_set_refresh_token_lag=2000,
        topic_working_set_stale_token_lag=5000,
        topic_working_set_refresh_batch_messages=8,
    )
    return Settings(**{**asdict(base), **overrides})


async def _runtime(payloads: list[dict[str, object]], *, message_count: int = 4):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 1, 4, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
    created_messages = []
    for seq in range(1, message_count + 1):
        created_messages.append(
            await messages.create_message(
                f"msg_{seq}",
                "cnv_1",
                "user" if seq % 2 else "assistant",
                seq,
                f"Planning message {seq}",
                12,
                {},
            )
        )
    provider = TopicProvider(payloads)
    service = TopicWorkingSetRefreshService(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        settings=_settings(),
    )
    return connection, service, provider, created_messages


@pytest.mark.asyncio
async def test_topic_refresh_waits_until_missing_topic_reaches_message_threshold() -> None:
    connection, service, provider, messages = await _runtime([], message_count=3)
    try:
        result = await service.maybe_refresh_after_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_id=str(messages[-1]["id"]),
        )

        assert result.refreshed is False
        assert result.reason == "missing"
        assert provider.requests == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_refresh_creates_topic_and_records_processed_seq_bounds() -> None:
    payload = {
        "actions": [
            {
                "action": "create",
                "title": "Trip planning",
                "summary": "Keep the current travel-planning thread oriented.",
                "active_goal": "Decide the next booking step.",
                "source_message_ids": ["msg_1", "msg_4"],
                "confidence": 0.82,
            }
        ],
        "nothing_to_update": False,
    }
    connection, service, provider, messages = await _runtime([payload], message_count=4)
    try:
        result = await service.maybe_refresh_after_message(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_id=str(messages[-1]["id"]),
        )
        topics = TopicRepository(connection, service.clock)
        snapshot = await topics.get_topic_snapshot(user_id="usr_1", conversation_id="cnv_1")

        assert result.refreshed is True
        assert result.processed_message_count == 4
        assert len(provider.requests) == 1
        assert provider.requests[0].metadata["purpose"] == "topic_working_set_update"
        assert snapshot["active_topics"][0]["title"] == "Trip planning"
        assert snapshot["active_topics"][0]["source_message_start_seq"] == 1
        assert snapshot["active_topics"][0]["source_message_end_seq"] == 4
        assert snapshot["active_topics"][0]["last_touched_seq"] == 4
        assert snapshot["freshness"]["status"] == "fresh"
        assert snapshot["freshness"]["last_processed_seq"] == 4
    finally:
        await connection.close()
