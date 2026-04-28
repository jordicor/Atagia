"""End-to-end tests for the consent confirmation UX flow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.consent_repository import (
    MemoryConsentProfileRepository,
    PendingMemoryConfirmationRepository,
)
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.models.schemas_jobs import EXTRACT_STREAM_NAME, WORKER_GROUP_NAME
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.chat_service import ChatService
from atagia.services.errors import LLMUnavailableError
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.workers.ingest_worker import IngestWorker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')
_NO_DURABLE_OUTPUT = json.dumps(
    {
        "evidences": [],
        "beliefs": [],
        "contract_signals": [],
        "state_updates": [],
        "mode_guess": None,
        "nothing_durable": True,
    }
)


class ConfirmationFlowProvider(LLMProvider):
    name = "confirmation-flow-tests"

    def __init__(self, extraction_outputs: list[str]) -> None:
        self.extraction_outputs = list(extraction_outputs)
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
                        "sub_queries": ["weather forecast"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "weather forecast",
                                "fts_phrase": "weather forecast",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
        if purpose == "context_cache_signal_detection":
            prompt = request.messages[1].content.lower()
            short_followup = any(reply in prompt for reply in ("<new_user_message>\nyes", "<new_user_message>\nno"))
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": short_followup,
                        "ambiguous_wording": False,
                    }
                ),
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
        if purpose == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"is_explicit": False, "reasoning": "No beliefs in this test."}),
            )
        if purpose == "intent_classifier_claim_key_equivalence":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"equivalent": False}),
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
        if purpose == "consequence_tendency_inference":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"tendency_text": ""}),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Forecast looks clear.",
            )
        if purpose == "consent_confirmation_intent":
            prompt = request.messages[1].content.lower()
            if "<user_reply>\nyes" in prompt:
                intent = "confirm"
            elif "<user_reply>\nno" in prompt:
                intent = "deny"
            else:
                intent = "ambiguous"
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": intent}),
            )
        if not self.extraction_outputs:
            raise AssertionError(f"No extraction output left for purpose={purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.extraction_outputs.pop(0),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in confirmation flow tests: {request.model}")


class FailingConfirmationFlowProvider(ConfirmationFlowProvider):
    def __init__(self, extraction_outputs: list[str], fail_purpose: str) -> None:
        super().__init__(extraction_outputs)
        self._fail_purpose = fail_purpose

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == self._fail_purpose:
            raise LLMError(f"Injected failure for {self._fail_purpose}")
        return await super().complete(request)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-chat-confirmation.db"),
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
        llm_chat_model="openai/reply-test-model",
        llm_ingest_model="openai/extract-test-model",
        llm_retrieval_model="openai/score-test-model",
        llm_component_models={"intent_classifier": "openai/classify-test-model"},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        lifecycle_lazy_enabled=False,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extraction_outputs: list[str],
    assistant_mode_id: str = "personal_assistant",
    provider: ConfirmationFlowProvider | None = None,
) -> tuple[AppRuntime, ConfirmationFlowProvider, IngestWorker]:
    resolved_provider = provider or ConfirmationFlowProvider(extraction_outputs)
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=resolved_provider.name, providers=[resolved_provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    runtime.clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            assistant_mode_id,
            "Consent Chat",
        )
    finally:
        await connection.close()
    ingest_connection = await runtime.open_connection()
    ingest_worker = IngestWorker(
        storage_backend=runtime.storage_backend,
        connection=ingest_connection,
        llm_client=runtime.llm_client,
        clock=runtime.clock,
        manifest_loader=runtime.manifest_loader,
        embedding_index=runtime.embedding_index,
        settings=runtime.settings,
    )
    return runtime, resolved_provider, ingest_worker


async def _close_runtime(runtime: AppRuntime, ingest_worker: IngestWorker) -> None:
    await ingest_worker._connection.close()
    await runtime.close()


def _pin_output(*, canonical_text: str, index_text: str) -> str:
    return json.dumps(
        {
            "evidences": [
                {
                    "canonical_text": canonical_text,
                    "index_text": index_text,
                    "scope": "global_user",
                    "confidence": 0.97,
                    "source_kind": "extracted",
                    "privacy_level": 3,
                    "memory_category": "pin_or_password",
                    "preserve_verbatim": True,
                    "language_codes": ["en"],
                    "payload": {},
                }
            ],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )


async def _drain_user_extract_jobs(runtime: AppRuntime, ingest_worker: IngestWorker) -> None:
    while True:
        messages = await runtime.storage_backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "confirmation-flow-consumer",
            count=20,
            block_ms=0,
        )
        if not messages:
            return
        for message in messages:
            payload = message.payload
            role = str(payload.get("payload", {}).get("role", ""))
            if role == "user":
                await ingest_worker.process_job(payload)
            await runtime.storage_backend.stream_ack(
                EXTRACT_STREAM_NAME,
                WORKER_GROUP_NAME,
                message.message_id,
            )


async def _list_all_memories(runtime: AppRuntime) -> list[dict[str, object]]:
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        return await memories.list_for_user("usr_1", statuses=None)
    finally:
        await connection.close()


async def _get_profile(runtime: AppRuntime) -> dict[str, object] | None:
    connection = await runtime.open_connection()
    try:
        repository = MemoryConsentProfileRepository(connection, runtime.clock)
        return await repository.get_profile("usr_1", MemoryCategory.PIN_OR_PASSWORD)
    finally:
        await connection.close()


async def _get_marker(runtime: AppRuntime, memory_id: str) -> dict[str, object] | None:
    connection = await runtime.open_connection()
    try:
        repository = PendingMemoryConfirmationRepository(connection, runtime.clock)
        return await repository.get_marker_for_memory("usr_1", memory_id)
    finally:
        await connection.close()


async def _stored_assistant_text(runtime: AppRuntime) -> str:
    connection = await runtime.open_connection()
    try:
        messages = MessageRepository(connection, runtime.clock)
        stored = await messages.get_messages("cnv_1", "usr_1", limit=50, offset=0)
        return str(stored[-1]["text"])
    finally:
        await connection.close()


async def _seed_pending_memory(
    runtime: AppRuntime,
    *,
    memory_id: str,
    category: MemoryCategory,
    index_text: str,
    asked_at: str | None = None,
    confirmation_asked_once: bool = False,
) -> None:
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        confirmations = PendingMemoryConfirmationRepository(connection, runtime.clock)
        created = await memories.create_memory_object(
            memory_id=memory_id,
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="personal_assistant",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text=f"Secret value for {memory_id}",
            index_text=index_text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.97,
            privacy_level=3,
            memory_category=category,
            preserve_verbatim=True,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            commit=False,
        )
        await confirmations.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id=str(created["id"]),
            category=category,
            created_at=str(created["created_at"]),
            commit=False,
        )
        if asked_at is not None:
            await confirmations.mark_markers_asked(
                "usr_1",
                [memory_id],
                asked_at=asked_at,
                commit=False,
            )
        if confirmation_asked_once:
            await confirmations.reset_after_ambiguous("usr_1", [memory_id], commit=False)
            if asked_at is not None:
                await confirmations.mark_markers_asked(
                    "usr_1",
                    [memory_id],
                    asked_at=asked_at,
                    commit=False,
                )
        await connection.commit()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_confirmation_classifier_failure_raises_llm_unavailable_and_keeps_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[],
        provider=FailingConfirmationFlowProvider([], "consent_confirmation_intent"),
    )
    try:
        await _seed_pending_memory(
            runtime,
            memory_id="mem_pending_1",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="bank card PIN",
            asked_at=runtime.clock.now().isoformat(),
        )

        with pytest.raises(LLMUnavailableError):
            await ChatService(runtime).chat_reply(
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="yes",
                assistant_mode_id="personal_assistant",
            )

        marker = await _get_marker(runtime, "mem_pending_1")
        assert marker is not None
        assert marker["asked_at"] is not None

        memories = await _list_all_memories(runtime)
        assert len(memories) == 1
        assert memories[0]["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            assert await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0) == []
        finally:
            await connection.close()
    finally:
        await ingest_worker._connection.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_walk_a_first_sensitive_share_becomes_pending_then_confirms_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Banking card PIN: 4512", index_text="bank card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
        ],
    )
    try:
        service = ChatService(runtime)

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My banking card PIN is 4512.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        pending = await _list_all_memories(runtime)
        assert len(pending) == 1
        assert pending[0]["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert await _get_marker(runtime, str(pending[0]["id"])) is not None

        prompted = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert prompted.response_text.startswith("Before I answer, I noted bank card PIN earlier.")
        assert await _stored_assistant_text(runtime) == "Forecast looks clear."
        asked_marker = await _get_marker(runtime, str(pending[0]["id"]))
        assert asked_marker is not None
        assert asked_marker["asked_at"] is not None

        confirmed = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes please",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        updated = await _list_all_memories(runtime)
        profile = await _get_profile(runtime)
        assert confirmed.response_text == "Forecast looks clear."
        assert updated[0]["status"] == MemoryStatus.ACTIVE.value
        assert await _get_marker(runtime, str(updated[0]["id"])) is None
        assert profile is not None
        assert profile["confirmed_count"] == 1
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_walk_b_second_sensitive_share_still_requires_confirmation_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Gym membership card PIN: 9988", index_text="gym card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
        ],
    )
    try:
        connection = await runtime.open_connection()
        try:
            profiles = MemoryConsentProfileRepository(connection, runtime.clock)
            await profiles.upsert_profile(
                user_id="usr_1",
                category=MemoryCategory.PIN_OR_PASSWORD,
                confirmed_count=1,
                declined_count=0,
            )
        finally:
            await connection.close()

        service = ChatService(runtime)
        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My gym membership card PIN is 9988.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        pending = await _list_all_memories(runtime)
        assert pending[0]["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value

        prompted = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's next?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompted.response_text.startswith("Before I answer, I noted gym card PIN earlier.")

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        updated = await _list_all_memories(runtime)
        profile = await _get_profile(runtime)
        assert updated[0]["status"] == MemoryStatus.ACTIVE.value
        assert profile is not None
        assert profile["confirmed_count"] == 2
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_walk_c_threshold_met_stores_sensitive_memory_silently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Locker code: 1234", index_text="locker PIN"),
            _NO_DURABLE_OUTPUT,
        ],
    )
    try:
        connection = await runtime.open_connection()
        try:
            profiles = MemoryConsentProfileRepository(connection, runtime.clock)
            await profiles.upsert_profile(
                user_id="usr_1",
                category=MemoryCategory.PIN_OR_PASSWORD,
                confirmed_count=2,
                declined_count=0,
            )
        finally:
            await connection.close()

        service = ChatService(runtime)
        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My locker code is 1234.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        stored = await _list_all_memories(runtime)
        assert stored[0]["status"] == MemoryStatus.ACTIVE.value
        assert await _get_marker(runtime, str(stored[0]["id"])) is None

        follow_up = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert follow_up.response_text == "Forecast looks clear."
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_walk_d_declines_twice_then_suppresses_future_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Company card PIN: 7000", index_text="company card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
            _pin_output(canonical_text="Gym membership card PIN: 9988", index_text="gym card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
            _pin_output(canonical_text="Locker code: 1234", index_text="locker PIN"),
        ],
    )
    try:
        service = ChatService(runtime)

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My company card PIN is 7000.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        prompt_one = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompt_one.response_text.startswith("Before I answer, I noted company card PIN earlier.")

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="no, don't save work stuff",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        after_first_decline = await _list_all_memories(runtime)
        profile = await _get_profile(runtime)
        assert after_first_decline[0]["status"] == MemoryStatus.DECLINED.value
        assert profile is not None
        assert profile["declined_count"] == 1

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My gym membership card PIN is 9988.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        prompt_two = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What time is it?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompt_two.response_text.startswith("Before I answer, I noted gym card PIN earlier.")

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="no thanks",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        declined_twice = await _list_all_memories(runtime)
        profile = await _get_profile(runtime)
        assert [row["status"] for row in declined_twice] == [
            MemoryStatus.DECLINED.value,
            MemoryStatus.DECLINED.value,
        ]
        assert profile is not None
        assert profile["declined_count"] == 2

        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My locker code is 1234.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        suppressed = await _list_all_memories(runtime)
        assert len(suppressed) == 2
        assert await _get_profile(runtime) is not None
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_chat_reply_batches_same_category_confirmations_and_keeps_prompt_ephemeral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[_NO_DURABLE_OUTPUT],
    )
    try:
        await _seed_pending_memory(
            runtime,
            memory_id="mem_pin_1",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="bank card PIN",
        )
        await _seed_pending_memory(
            runtime,
            memory_id="mem_pin_2",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="gym card PIN",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert "bank card PIN and gym card PIN" in result.response_text
        assert "Want me to keep your PINs or passwords for next time?" in result.response_text
        assert await _stored_assistant_text(runtime) == "Forecast looks clear."
        assert result.debug is not None
        chat_request = next(
            request
            for request in reversed(provider.requests)
            if request.metadata.get("purpose") == "chat_reply"
        )
        assert all("Before I answer" not in message.content for message in chat_request.messages)
        stored_cache = await runtime.storage_backend.get_context_view(str(result.debug["cache"]["cache_key"]))
        assert stored_cache is not None
        assert stored_cache["last_user_message_text"] == "What's the weather?"
        marker_one = await _get_marker(runtime, "mem_pin_1")
        marker_two = await _get_marker(runtime, "mem_pin_2")
        assert marker_one is not None and marker_one["asked_at"] is not None
        assert marker_two is not None and marker_two["asked_at"] is not None
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_ambiguous_confirmation_reasks_once_then_implicitly_declines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[_NO_DURABLE_OUTPUT, _NO_DURABLE_OUTPUT, _NO_DURABLE_OUTPUT],
    )
    try:
        await _seed_pending_memory(
            runtime,
            memory_id="mem_pending_1",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="bank card PIN",
            asked_at="2026-04-06T12:00:00+00:00",
        )
        service = ChatService(runtime)

        first_ambiguous = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        marker = await _get_marker(runtime, "mem_pending_1")
        memories = await _list_all_memories(runtime)
        assert first_ambiguous.response_text == "Forecast looks clear."
        assert marker is not None
        assert marker["asked_at"] is None
        assert marker["confirmation_asked_once"] == 1
        assert memories[0]["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value

        prompted_again = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Any updates?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompted_again.response_text.startswith("Before I answer, I noted bank card PIN earlier.")

        final_ambiguous = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Still wondering about the weather.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        profile = await _get_profile(runtime)
        memories = await _list_all_memories(runtime)
        assert final_ambiguous.response_text == "Forecast looks clear."
        assert await _get_marker(runtime, "mem_pending_1") is None
        assert memories[0]["status"] == MemoryStatus.DECLINED.value
        assert profile is not None
        assert profile["declined_count"] == 1
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_confirm_turn_invalidates_cache_entry_and_forces_fresh_follow_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Banking card PIN: 4512", index_text="bank card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
        ],
    )
    try:
        service = ChatService(runtime)
        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My banking card PIN is 4512.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        prompted = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompted.debug is not None
        cache_key = str(prompted.debug["cache"]["cache_key"])
        assert await runtime.storage_backend.get_context_view(cache_key) is not None

        confirmed = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert confirmed.debug is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None

        follow_up = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="continue",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert follow_up.debug is not None
        assert follow_up.debug["cache"]["from_cache"] is False
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_mixed_batch_only_implicitly_declines_items_already_reasked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[_NO_DURABLE_OUTPUT],
    )
    try:
        await _seed_pending_memory(
            runtime,
            memory_id="mem_reask",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="bank card PIN",
            asked_at="2026-04-06T12:00:00+00:00",
            confirmation_asked_once=True,
        )
        await _seed_pending_memory(
            runtime,
            memory_id="mem_first_ask",
            category=MemoryCategory.PIN_OR_PASSWORD,
            index_text="gym card PIN",
            asked_at="2026-04-06T12:00:00+00:00",
            confirmation_asked_once=False,
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        profile = await _get_profile(runtime)
        reask_memory = next(row for row in await _list_all_memories(runtime) if row["id"] == "mem_reask")
        first_ask_memory = next(
            row for row in await _list_all_memories(runtime) if row["id"] == "mem_first_ask"
        )
        first_ask_marker = await _get_marker(runtime, "mem_first_ask")

        assert result.response_text == "Forecast looks clear."
        assert reask_memory["status"] == MemoryStatus.DECLINED.value
        assert first_ask_memory["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert first_ask_marker is not None
        assert first_ask_marker["asked_at"] is None
        assert first_ask_marker["confirmation_asked_once"] == 1
        assert profile is not None
        assert profile["declined_count"] == 1
    finally:
        await _close_runtime(runtime, ingest_worker)


@pytest.mark.asyncio
async def test_decline_turn_invalidates_cache_entry_and_forces_fresh_follow_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider, ingest_worker = await _build_runtime(
        tmp_path,
        monkeypatch,
        extraction_outputs=[
            _pin_output(canonical_text="Banking card PIN: 4512", index_text="bank card PIN"),
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
            _NO_DURABLE_OUTPUT,
        ],
    )
    try:
        service = ChatService(runtime)
        await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="My banking card PIN is 4512.",
            assistant_mode_id="personal_assistant",
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        prompted = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What's the weather?",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)
        assert prompted.debug is not None
        cache_key = str(prompted.debug["cache"]["cache_key"])
        assert await runtime.storage_backend.get_context_view(cache_key) is not None

        declined = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="no thanks",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert declined.debug is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None

        follow_up = await service.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="continue",
            assistant_mode_id="personal_assistant",
            debug=True,
        )
        await _drain_user_extract_jobs(runtime, ingest_worker)

        assert follow_up.debug is not None
        assert follow_up.debug["cache"]["from_cache"] is False
    finally:
        await _close_runtime(runtime, ingest_worker)
