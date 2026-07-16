"""Tests for the ChatService orchestration flow."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.summary_repository import SummaryRepository
from atagia.core.space_repository import SpaceRepository
from atagia.core.storage_backend import build_recent_window_key
from atagia.models.schemas_jobs import JobType
from atagia.models.schemas_memory import SpaceBoundaryMode
from atagia.services.chat_service import ChatService
from atagia.services.errors import (
    ConversationNotActiveError,
    ConversationNotFoundError,
    LLMUnavailableError,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _is_need_detection_card_purpose(purpose: object) -> bool:
    value = str(purpose)
    return value.startswith("need_detection_") and value.endswith("_card")


def _need_detection_card_count(provider: "ChatServiceProvider") -> int:
    return sum(
        1
        for request in provider.requests
        if _is_need_detection_card_purpose(request.metadata.get("purpose"))
    )


class ChatServiceProvider(LLMProvider):
    name = "chat-service-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if _is_need_detection_card_purpose(purpose):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "retry loop",
                "need_detection_search_words_other_language_card": "none",
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=outputs[purpose],
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": True,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} useful" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "consent_confirmation_intent":
            prompt = request.messages[1].content.lower()
            if "<user_reply>\nno" in prompt:
                intent = "deny"
            elif "<user_reply>\nyes" in prompt:
                intent = "confirm"
            else:
                intent = "ambiguous"
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": intent}),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        if purpose == "answer_postcondition_verification":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "readable": True,
                        "is_abstention": False,
                        "contains_concrete_claims": True,
                        "unsupported_concrete_claims": False,
                        "covers_requested_facets": True,
                        "requires_abstention": False,
                        "pass_postcondition": True,
                        "failure_reasons": [],
                        "explanation": "Supported for the test.",
                    }
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(
            f"Embeddings are not used in chat service tests: {request.model}"
        )


class FailingChatServiceProvider(ChatServiceProvider):
    def __init__(self, fail_purpose: str) -> None:
        super().__init__()
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = request.metadata.get("purpose")
        if purpose == self._fail_purpose or (
            self._fail_purpose == "need_detection"
            and _is_need_detection_card_purpose(purpose)
        ):
            raise LLMError(f"Injected failure for {self._fail_purpose}")
        return await super().complete(request)


def test_redacted_debug_payload_omits_raw_context() -> None:
    payload = {
        "cold_start": False,
        "detected_needs": ["exact_recall"],
        "selected_memory_ids": ["mem_secret"],
        "context_view": {"memory_block": "secret text"},
        "retrieval_plan": {"fts_queries": ["secret"]},
        "cache": {"from_cache": False, "staleness": 1.0},
        "post_commit_errors": [],
        "answer_postcondition_guard": {
            "status": "passed",
            "failure_reasons": [],
            "retry_count": 0,
            "output_limit_seen": False,
            "verdict": {"contains": "raw"},
        },
        "authority": {"sensitive_trace": True},
    }

    redacted = ChatService._redact_debug_payload(payload)

    assert "context_view" not in redacted
    assert "retrieval_plan" not in redacted
    assert "selected_memory_ids" not in redacted
    assert redacted["selected_memory_count"] == 1
    assert redacted["authority"]["sensitive_trace"] is False


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-chat-service.db"),
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
        small_corpus_token_threshold_ratio=0.0,
        # Pinned so transcript-window math below stays fixture-sized and
        # independent of the production envelope default.
        context_envelope_budget_tokens=4096,
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
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert [message["role"] for message in stored_messages[-2:]] == [
                "user",
                "assistant",
            ]
            assert stored_messages[-1]["text"] == "Check the retry guard first."
            event = await RetrievalEventRepository(connection, runtime.clock).get_event(
                result.retrieval_event_id,
                "usr_1",
            )
            assert event is not None
            assert event["outcome_json"]["retrieval_custody_v2_status"] == "fresh"
            assert event["outcome_json"]["retrieval_custody_v2"] == []
            assert event["outcome_json"]["sufficiency_diagnostics_v1_status"] == "fresh"
            assert event["outcome_json"]["sufficiency_diagnostics_v1"]["state"] == (
                "insufficient_no_candidates"
            )
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_rejects_namespace_change_while_provider_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    provider_reached = asyncio.Event()
    resume_provider = asyncio.Event()
    original_complete = provider.complete

    async def blocking_complete(
        request: LLMCompletionRequest,
    ) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "chat_reply":
            provider_reached.set()
            await asyncio.wait_for(resume_provider.wait(), timeout=5.0)
        return await original_complete(request)

    monkeypatch.setattr(provider, "complete", blocking_complete)
    try:
        await _seed_conversation(
            runtime,
            user_id="usr_namespace_race",
            conversation_id="cnv_namespace_race",
        )
        setup_connection = await runtime.open_connection()
        try:
            spaces = SpaceRepository(setup_connection, runtime.clock)
            for space_id in ("space_old", "space_new"):
                await spaces.resolve_space(
                    owner_user_id="usr_namespace_race",
                    space_id=space_id,
                    boundary_mode=SpaceBoundaryMode.FOCUS,
                    display_name=space_id,
                    source_kind="explicit",
                    source_id=space_id,
                )
            await setup_connection.execute(
                """
                UPDATE conversations
                SET active_space_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    "space_old",
                    runtime.clock.now().isoformat(),
                    "cnv_namespace_race",
                    "usr_namespace_race",
                ),
            )
            await setup_connection.commit()
        finally:
            await setup_connection.close()

        chat_task = asyncio.create_task(
            ChatService(runtime).chat_reply(
                user_id="usr_namespace_race",
                conversation_id="cnv_namespace_race",
                message_text="Do not persist this turn under stale coordinates.",
                assistant_mode_id="coding_debug",
            )
        )
        await asyncio.wait_for(provider_reached.wait(), timeout=5.0)
        mutation_connection = await runtime.open_connection()
        try:
            await mutation_connection.execute(
                """
                UPDATE conversations
                SET active_space_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    "space_new",
                    runtime.clock.now().isoformat(),
                    "cnv_namespace_race",
                    "usr_namespace_race",
                ),
            )
            await mutation_connection.commit()
        finally:
            await mutation_connection.close()
            resume_provider.set()

        with pytest.raises(ConversationNotActiveError):
            await chat_task

        verification = await runtime.open_connection()
        try:
            cursor = await verification.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE conversation_id = ?
                """,
                ("cnv_namespace_race",),
            )
            assert int((await cursor.fetchone())["count"]) == 0
        finally:
            await verification.close()
    finally:
        resume_provider.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_publishes_committed_window_with_exact_conversation_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    original_publish = runtime.storage_backend.set_recent_window_for_lifecycle
    captured: dict[str, object] = {}

    async def capture_recent_window(
        key: str,
        messages: list[dict[str, object]],
        **identity: object,
    ) -> bool:
        captured["key"] = key
        captured["messages"] = messages
        captured.update(identity)
        return await original_publish(key, messages, **identity)

    monkeypatch.setattr(
        runtime.storage_backend,
        "set_recent_window_for_lifecycle",
        capture_recent_window,
    )
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
        )

        assert result.response_text == "Check the retry guard first."
        expected_window = [
            {
                "role": "user",
                "content": "Please help me debug this retry loop.",
            },
            {
                "role": "assistant",
                "content": "Check the retry guard first.",
            },
        ]
        recent_window_key = build_recent_window_key("usr_1", "cnv_1")
        assert captured["key"] == recent_window_key
        assert captured["messages"] == expected_window
        assert (
            await runtime.storage_backend.get_recent_window(recent_window_key)
            == expected_window
        )

        connection = await runtime.open_connection()
        try:
            stored_messages = await MessageRepository(
                connection,
                runtime.clock,
            ).get_recent_messages("cnv_1", "usr_1", limit=20)
            conversation_identity = await ConversationLifecycleRepository(
                connection,
                runtime.clock,
            ).get_active_identity(
                user_id="usr_1",
                conversation_id="cnv_1",
            )
        finally:
            await connection.close()
        assert conversation_identity is not None
        assert captured["messages"] == [
            {"role": row["role"], "content": row["text"]} for row in stored_messages
        ]
        assert captured["conversation_lifecycle_epoch"] == (
            conversation_identity.lifecycle_epoch
        )
        assert captured["conversation_source_revision"] == (
            conversation_identity.source_revision
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_records_answer_postcondition_report_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ChatServiceProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(
        replace(_settings(tmp_path), answer_postcondition_guard_enabled=True)
    )
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
        assert result.debug["answer_postcondition_guard"]["status"] == "passed"
        connection = await runtime.open_connection()
        try:
            event = await RetrievalEventRepository(connection, runtime.clock).get_event(
                result.retrieval_event_id,
                "usr_1",
            )
            assert event is not None
            assert (
                event["outcome_json"]["answer_postcondition_guard"]["status"]
                == "passed"
            )
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
    original_set_recent_window = runtime.storage_backend.set_recent_window_for_lifecycle

    async def failing_set_recent_window(
        key: str,
        messages: list[dict[str, object]],
        **_kwargs: object,
    ) -> bool:
        raise RuntimeError("Injected recent window failure")

    monkeypatch.setattr(
        runtime.storage_backend,
        "set_recent_window_for_lifecycle",
        failing_set_recent_window,
    )
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
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert [message["role"] for message in stored_messages[-2:]] == [
                "user",
                "assistant",
            ]
        finally:
            await connection.close()
    finally:
        monkeypatch.setattr(
            runtime.storage_backend,
            "set_recent_window_for_lifecycle",
            original_set_recent_window,
        )
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_rolls_back_turn_when_durable_job_insert_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)

    async def failing_enqueue_message_jobs(
        *, storage_backend: object, jobs: list[object], **_kwargs: object
    ) -> list[str]:
        del storage_backend
        del jobs
        raise RuntimeError("Injected enqueue failure")

    monkeypatch.setattr(
        "atagia.services.chat_service.enqueue_message_jobs",
        failing_enqueue_message_jobs,
    )
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")

        with pytest.raises(RuntimeError, match="Injected enqueue failure"):
            await ChatService(runtime).chat_reply(
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
                assistant_mode_id="coding_debug",
                debug=True,
            )

        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).get_messages("cnv_1", "usr_1", limit=10, offset=0)
            turn_job_count = await connection.execute_fetchall(
                """
                SELECT COUNT(*) AS count
                FROM worker_job_runs
                WHERE user_id = ?
                  AND job_type <> ?
                """,
                ("usr_1", JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value),
            )
            assert messages == []
            assert int(turn_job_count[0]["count"]) == 0
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

        connection = await runtime.open_connection()
        try:
            rows = await connection.execute_fetchall(
                """
                SELECT job_type, COUNT(*) AS count
                FROM worker_job_runs
                WHERE user_id = ?
                  AND status = 'queued'
                  AND recovery_envelope_json IS NOT NULL
                GROUP BY job_type
                """,
                ("usr_1",),
            )
        finally:
            await connection.close()

        counts = {str(row["job_type"]): int(row["count"]) for row in rows}
        assert counts["extract_memory_candidates"] == 2
        assert counts["project_contract"] == 1
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
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
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
            assert (
                await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0) == []
            )
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_degrades_when_need_detection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Wave 1-B: need detector is a counselor. A failure of the need
    # detection LLM must not break retrieval — the base search still runs
    # and the chat reply is produced in degraded mode.
    provider = FailingChatServiceProvider("need_detection")
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")

        reply = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
        )
        assert reply is not None

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert len(stored_messages) >= 1
            assert stored_messages[0]["text"] == "Please help me debug this retry loop."
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_reply_raises_llm_unavailable_when_cache_signal_detection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ChatService(runtime)
        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Please help me debug this retry loop.",
            assistant_mode_id="coding_debug",
        )
        provider = FailingChatServiceProvider("context_cache_signal_detection")
        runtime.llm_client = LLMClient(
            provider_name=provider.name, providers=[provider]
        )
        service = ChatService(runtime)
        with pytest.raises(LLMUnavailableError):
            await service.chat_reply(
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
                assistant_mode_id="coding_debug",
            )

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert len(stored_messages) == 2
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
        need_count_before = _need_detection_card_count(provider)
        chat_count_before = sum(
            1
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        )

        second = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="continue",
            assistant_mode_id="coding_debug",
            debug=True,
        )

        need_count_after = _need_detection_card_count(provider)
        chat_count_after = sum(
            1
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
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
            first_event = next(
                event for event in listed if event["id"] == first.retrieval_event_id
            )
            assert latest["id"] == second.retrieval_event_id
            assert latest["outcome_json"]["from_cache"] is True
            assert latest["outcome_json"]["need_detection_skipped"] is True
            assert latest["outcome_json"]["detected_needs"] == []
            assert (
                latest["outcome_json"]["retrieval_custody_v2_status"]
                == "cache_hit_no_candidate_custody"
            )
            assert latest["outcome_json"]["retrieval_custody_v2"] == []
            assert latest["outcome_json"]["sufficiency_diagnostics_v1_status"] == (
                "cache_hit_no_sufficiency_diagnostics"
            )
            assert latest["outcome_json"]["sufficiency_diagnostics_v1"] is None
            cache_key = first_event["outcome_json"]["cache_key"]
            cache_entry = await runtime.storage_backend.get_context_view(cache_key)
            assert cache_entry is not None
            assert "sufficiency_diagnostics_v1" not in cache_entry
            assert "retrieval_sufficiency" not in cache_entry
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
                },
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
        assert (
            "[Conversation summary | historical context only | ...]"
            in chat_request.messages[0].content
        )
        assert transcript_messages[0].role == "user"
        assert transcript_messages[0].content.startswith(
            "Historical transcript summary data, not an assistant answer"
        )
        assert (
            "[Conversation summary | historical context only | turns 1-6]"
            in transcript_messages[0].content
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
            assert event["outcome_json"]["transcript_window"][
                "transcript_message_seqs"
            ] == [7, 8, 9, 10]
            assert event["outcome_json"]["transcript_window"]["chunk_ids"] == [
                "sum_old"
            ]
            # The recent-transcript budget is derived from the unified context
            # envelope (context_envelope_budget_tokens pinned to 4096 in
            # _settings * the recent_transcript ratio 0.20 = floor(819.2) =
            # 819), not from the manifest's transcript_budget_tokens field.
            assert event["outcome_json"]["transcript_window"]["budget_tokens"] == 819
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

        # No summary chunks exist, so the whole transcript window is the raw
        # uncovered tail. It is bounded by the recent-transcript budget derived
        # from the unified context envelope (4096 pinned in _settings * 0.20
        # ratio = 819 tokens), which fits the most recent 409 short messages, so the window
        # starts at tail-112 (520 - 409 + 1) rather than tail-1. The tail is
        # contiguous through the latest stored message and ends with the new
        # user turn.
        assert transcript_messages[0].content == "tail-112"
        transcript_tail = [message.content for message in transcript_messages[:-1]]
        assert transcript_tail == [f"tail-{seq}" for seq in range(112, 521)]
        assert transcript_messages[-1].content == "Continue from the start."
    finally:
        await runtime.close()
