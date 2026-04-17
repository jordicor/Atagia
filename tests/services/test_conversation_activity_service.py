"""Tests for conversation activity aggregation and warm-up."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.conversation_activity_repository import ConversationActivityRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.services.conversation_activity_service import ConversationActivityService
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


class NoopProvider(LLMProvider):
    name = "noop-activity-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM should not be called in activity tests: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings should not be called in activity tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-activity.db"),
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
        llm_extraction_model="test-model",
        llm_scoring_model="test-model",
        llm_classifier_model="test-model",
        llm_chat_model="test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppRuntime:
    provider = NoopProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    runtime.clock = FrozenClock(datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc))
    return runtime


async def _seed_user_conversation(
    runtime: AppRuntime,
    *,
    user_id: str,
    conversation_id: str,
    workspace_id: str | None = None,
    title: str = "Chat",
) -> None:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        workspaces = WorkspaceRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)

        if await users.get_user(user_id) is None:
            await users.create_user(user_id)
        if workspace_id is not None:
            if await workspaces.get_workspace(workspace_id, user_id) is None:
                await workspaces.create_workspace(
                    workspace_id,
                    user_id,
                    "Workspace",
                    {"timezone": "UTC"},
                )
        await conversations.create_conversation(
            conversation_id,
            user_id,
            workspace_id,
            "coding_debug",
            title,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_activity_stats_aggregate_messages_retrieval_and_histograms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(
            runtime,
            user_id="usr_1",
            conversation_id="cnv_1",
            workspace_id="wrk_1",
        )
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)

            runtime.clock = FrozenClock(datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "First weekly check-in",
                5,
                {},
                "2026-03-02T09:00:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 2, 9, 1, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_2",
                "cnv_1",
                "assistant",
                2,
                "Noted.",
                2,
                {},
                "2026-03-02T09:01:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_3",
                "cnv_1",
                "user",
                3,
                "Second weekly check-in",
                5,
                {},
                "2026-03-09T09:00:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 5, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_4",
                "cnv_1",
                "assistant",
                4,
                "Still on track.",
                3,
                {},
                "2026-03-09T09:05:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 6, tzinfo=timezone.utc))
            await events.create_event(
                {
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_3",
                    "response_message_id": "msg_4",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["weekly"]},
                    "selected_memory_ids_json": [],
                    "context_view_json": {},
                    "outcome_json": {},
                }
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc))

            service = ConversationActivityService(runtime)
            snapshot = await service.get_activity_snapshot(
                connection,
                "usr_1",
                conversation_id="cnv_1",
            )
            stats = snapshot["conversations"][0]

            assert stats["message_count"] == 4
            assert stats["user_message_count"] == 2
            assert stats["assistant_message_count"] == 2
            assert stats["retrieval_count"] == 1
            assert stats["active_day_count"] == 3
            assert stats["recent_1d_message_count"] == 0
            assert stats["recent_7d_message_count"] == 2
            assert stats["recent_30d_message_count"] == 4
            assert len(stats["weekday_histogram_json"]) == 7
            assert len(stats["hour_histogram_json"]) == 24
            assert len(stats["hour_of_week_histogram_json"]) == 168
            assert stats["median_return_interval_minutes"] == 5850.0
            assert stats["p90_return_interval_minutes"] == 10080.0
            assert stats["schedule_pattern_kind"] == "weekly"
            assert stats["main_thread_score"] > 0.0
            assert 0.0 <= stats["likely_soon_score"] <= 1.0

            stored = await ConversationActivityRepository(connection, runtime.clock).get_activity_stats(
                user_id="usr_1",
                conversation_id="cnv_1",
            )
            assert stored is not None
            assert stored["conversation_id"] == "cnv_1"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_hot_ranking_prefers_recent_recurrent_conversations_and_respects_user_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(runtime, user_id="usr_a", conversation_id="cnv_hot")
        await _seed_user_conversation(runtime, user_id="usr_a", conversation_id="cnv_cold")
        await _seed_user_conversation(runtime, user_id="usr_b", conversation_id="cnv_other")

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)

            await messages.create_message(
                "msg_hot_1",
                "cnv_hot",
                "user",
                1,
                "Recent ping",
                2,
                {},
                "2026-03-10T11:00:00+00:00",
            )
            await messages.create_message(
                "msg_hot_2",
                "cnv_hot",
                "assistant",
                2,
                "Recent pong",
                2,
                {},
                "2026-03-10T11:05:00+00:00",
            )
            for index, day in enumerate([1, 8, 15, 22], start=1):
                await messages.create_message(
                    f"msg_cold_{index}",
                    "cnv_cold",
                    "user" if index % 2 else "assistant",
                    index,
                    f"Older message {index}",
                    2,
                    {},
                    f"2026-02-{day:02d}T09:00:00+00:00",
                )
            await messages.create_message(
                "msg_other_1",
                "cnv_other",
                "user",
                1,
                "Different user recent ping",
                2,
                {},
                "2026-03-10T11:10:00+00:00",
            )

            service = ConversationActivityService(runtime)
            hot = await service.list_hot_conversations(connection, "usr_a", limit=10)
            assert [row["conversation_id"] for row in hot][:2] == ["cnv_hot", "cnv_cold"]
            assert all(row["user_id"] == "usr_a" for row in hot)
            assert "cnv_other" not in {row["conversation_id"] for row in hot}

            other_user = await service.list_hot_conversations(connection, "usr_b", limit=10)
            assert [row["conversation_id"] for row in other_user] == ["cnv_other"]
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_warmup_primes_recent_window_and_recommended_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            for seq in range(1, 7):
                await messages.create_message(
                    f"msg_{seq}",
                    "cnv_1",
                    "user" if seq % 2 else "assistant",
                    seq,
                    f"Message {seq}",
                    2,
                    {},
                    f"2026-03-10T11:{seq:02d}:00+00:00",
                )

            service = ConversationActivityService(runtime)
            single = await service.warmup_conversation(
                connection,
                "usr_1",
                "cnv_1",
                max_messages=3,
            )
            assert single["recent_window_key"] == "usr_1:cnv_1"
            assert single["recent_message_count"] == 3
            assert single["recent_message_ids"] == ["msg_4", "msg_5", "msg_6"]
            assert single["cached_context_available"] is False
            assert await runtime.storage_backend.get_recent_window("usr_1:cnv_1") == single["recent_messages"]

            recommended = await service.warmup_recommended_conversations(
                connection,
                "usr_1",
                limit=1,
                total_message_budget=2,
                per_conversation_message_budget=2,
            )
            assert recommended["warmed_conversation_count"] == 1
            assert recommended["warmed_message_count"] == 2
            assert recommended["hot_conversations"][0]["conversation_id"] == "cnv_1"

            bounded = await service.warmup_conversation(
                connection,
                "usr_1",
                "cnv_1",
                max_messages=-1,
            )
            assert bounded["recent_message_count"] == 1
            assert bounded["recent_message_ids"] == ["msg_6"]

            no_budget = await service.warmup_recommended_conversations(
                connection,
                "usr_1",
                limit=-10,
                total_message_budget=-1,
            )
            assert no_budget["requested_limit"] == 1
            assert no_budget["warmed_conversation_count"] == 0
            assert no_budget["warmed_message_count"] == 0
        finally:
            await connection.close()
    finally:
        await runtime.close()
