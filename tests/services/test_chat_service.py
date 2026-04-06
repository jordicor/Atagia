"""Tests for the ChatService orchestration flow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import ConversationRepository, MessageRepository, UserRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.models.schemas_jobs import CONTRACT_STREAM_NAME, EXTRACT_STREAM_NAME, WORKER_GROUP_NAME
from atagia.services.chat_service import ChatService
from atagia.services.errors import ConversationNotFoundError, LLMUnavailableError
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class ChatServiceProvider(LLMProvider):
    name = "chat-service-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps([]),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    [
                        {"memory_id": memory_id, "llm_applicability": 0.5}
                        for memory_id in memory_ids
                    ]
                ),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in chat service tests: {request.model}")


class FailingChatServiceProvider(ChatServiceProvider):
    def __init__(self, fail_purpose: str) -> None:
        super().__init__()
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == self._fail_purpose:
            raise LLMError(f"Injected failure for {self._fail_purpose}")
        return await super().complete(request)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-chat-service.db"),
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


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AppRuntime, ChatServiceProvider]:
    provider = ChatServiceProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    return runtime, provider


async def _seed_conversation(
    runtime: AppRuntime,
    *,
    user_id: str,
    conversation_id: str,
    assistant_mode_id: str = "coding_debug",
) -> None:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        await users.create_user(user_id)
        await conversations.create_conversation(
            conversation_id,
            user_id,
            None,
            assistant_mode_id,
            "Chat",
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_chat_reply_basic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
        )

        assert result.response_text == "Check the retry guard first."
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [message["role"] for message in stored_messages[-2:]] == ["user", "assistant"]
            assert stored_messages[-1]["text"] == "Check the retry guard first."
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_does_not_raise_when_post_commit_recent_window_update_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    original_set_recent_window = runtime.storage_backend.set_recent_window

    async def failing_set_recent_window(key: str, messages: list[dict[str, object]]) -> None:
        raise RuntimeError("Injected recent window failure")

    monkeypatch.setattr(runtime.storage_backend, "set_recent_window", failing_set_recent_window)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
            debug=True,
        )

        assert result.response_text == "Check the retry guard first."
        assert result.debug is not None
        assert "recent_window_failed" in result.debug["post_commit_errors"]

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [message["role"] for message in stored_messages[-2:]] == ["user", "assistant"]
        finally:
            await connection.close()
    finally:
        monkeypatch.setattr(runtime.storage_backend, "set_recent_window", original_set_recent_window)
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_marks_background_tasks_false_when_enqueue_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)

    async def failing_enqueue_message_jobs(*, storage_backend: object, jobs: list[object]) -> list[str]:
        del storage_backend
        del jobs
        raise RuntimeError("Injected enqueue failure")

    monkeypatch.setattr("atagia.services.chat_service.enqueue_message_jobs", failing_enqueue_message_jobs)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
            debug=True,
        )

        assert result.debug is not None
        assert result.debug["enqueued_job_ids"] == []
        assert "job_enqueue_failed" in result.debug["post_commit_errors"]

        connection = await runtime.open_connection()
        try:
            event = await RetrievalEventRepository(connection, runtime.clock).get_event(
                result.retrieval_event_id,
                "usr_1",
            )
            assert event is not None
            assert event["outcome_json"]["background_tasks_enqueued"] is False
            assert "job_enqueue_failed" in event["outcome_json"]["post_commit_errors"]
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_user_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        await _seed_conversation(runtime, user_id="usr_2", conversation_id="cnv_2")

        with pytest.raises(ConversationNotFoundError):
            await ChatService(runtime).chat_reply(
                user_id="usr_1",
                conversation_id="cnv_2",
                message_text="Please help me debug this retry loop.",
                assistant_mode_id="coding_debug",
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_enqueues_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
        )

        extract_messages = await runtime.storage_backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test-consumer-extract",
            count=2,
            block_ms=0,
        )
        contract_messages = await runtime.storage_backend.stream_read(
            CONTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test-consumer-contract",
            count=2,
            block_ms=0,
        )

        assert len(extract_messages) == 2
        assert len(contract_messages) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_applies_message_occurred_at_only_to_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc))
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
            message_occurred_at="2023-05-08T13:56:00",
        )

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert stored_messages[0]["occurred_at"] == "2023-05-08T13:56:00"
            assert stored_messages[1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_rolls_back_messages_when_llm_reply_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingChatServiceProvider("chat_reply")
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")

        with pytest.raises(LLMUnavailableError):
            await ChatService(runtime).chat_reply(
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
                assistant_mode_id="coding_debug",
            )

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            assert await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0) == []
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_cache_hit_creates_retrieval_event_and_debug_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ChatService(runtime)

        first = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
            debug=True,
        )
        need_count_before = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "need_detection"
        )
        chat_count_before = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
        )

        second = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="continue",
            assistant_mode_id="coding_debug",
            debug=True,
        )

        need_count_after = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "need_detection"
        )
        chat_count_after = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "chat_reply"
        )
        assert first.retrieval_event_id is not None
        assert second.retrieval_event_id is not None
        assert second.debug is not None
        assert second.debug["cache"]["from_cache"] is True
        assert second.debug["cache"]["need_detection_skipped"] is True
        assert need_count_after == need_count_before
        assert chat_count_after == chat_count_before + 1

        connection = await runtime.open_connection()
        try:
            events = RetrievalEventRepository(connection, runtime.clock)
            listed = await events.list_events("usr_1", "cnv_1", limit=10)
            latest = listed[0]
            assert latest["id"] == second.retrieval_event_id
            assert latest["outcome_json"]["from_cache"] is True
            assert latest["outcome_json"]["need_detection_skipped"] is True
            assert latest["outcome_json"]["detected_needs"] == []
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_uses_windowed_transcript_and_records_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            summaries = SummaryRepository(connection, runtime.clock)
            for seq in range(1, 11):
                await messages.create_message(
                    f"msg_{seq}",
                    "cnv_1",
                    "user" if seq % 2 else "assistant",
                    seq,
                    (f"old-{seq}-" + ("x" * 6000)) if seq <= 6 else f"recent-{seq}",
                    None,
                    {},
                )
            await summaries.create_summary(
                "usr_1",
                {
                    "id": "sum_old",
                    "conversation_id": "cnv_1",
                    "workspace_id": None,
                    "source_message_start_seq": 1,
                    "source_message_end_seq": 6,
                    "summary_kind": "conversation_chunk",
                    "summary_text": "Earlier retry-loop investigation and rollback plan.",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "classify-test-model",
                    "created_at": "2026-04-03T10:00:00+00:00",
                }
            )
        finally:
            await connection.close()

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What should I check next?",
            assistant_mode_id="coding_debug",
        )

        chat_request = next(
            request
            for request in reversed(provider.requests)
            if request.metadata.get("purpose") == "chat_reply"
        )
        transcript_messages = chat_request.messages[1:]

        assert result.response_text == "Check the retry guard first."
        assert "[Conversation summary | historical context only | ...]" in chat_request.messages[0].content
        assert transcript_messages[0].role == "assistant"
        assert transcript_messages[0].content.startswith(
            "[Conversation summary | historical context only | turns 1-6]"
        )
        assert [message.content for message in transcript_messages[1:5]] == [
            "recent-7",
            "recent-8",
            "recent-9",
            "recent-10",
        ]
        assert transcript_messages[-1].role == "user"
        assert transcript_messages[-1].content == "What should I check next?"

        connection = await runtime.open_connection()
        try:
            event = await RetrievalEventRepository(connection, runtime.clock).get_event(
                result.retrieval_event_id,
                "usr_1",
            )
            assert event is not None
            assert event["outcome_json"]["transcript_window"]["raw_message_seqs"] == [7, 8, 9, 10]
            assert event["outcome_json"]["transcript_window"]["chunk_ids"] == ["sum_old"]
            assert event["outcome_json"]["transcript_window"]["budget_tokens"] == 8000
            assert event["outcome_json"]["transcript_window"]["budget_used_tokens"] > 0
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_fetches_full_uncovered_tail_when_chunks_lag_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            for seq in range(1, 521):
                await messages.create_message(
                    f"msg_tail_{seq}",
                    "cnv_1",
                    "user" if seq % 2 else "assistant",
                    seq,
                    f"tail-{seq}",
                    None,
                    {},
                )
        finally:
            await connection.close()

        await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Continue from the start.",
            assistant_mode_id="coding_debug",
        )

        chat_request = next(
            request
            for request in reversed(provider.requests)
            if request.metadata.get("purpose") == "chat_reply"
        )
        transcript_messages = chat_request.messages[1:]

        assert transcript_messages[0].content == "tail-1"
        assert transcript_messages[-1].content == "Continue from the start."
    finally:
        await runtime.close()
