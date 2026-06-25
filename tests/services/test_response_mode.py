"""Tests for F4 response_mode (fast / smart_fast) context assembly."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    ResponseMode,
)
from atagia.services.chat_service import ChatService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.sidecar_service import SidecarService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)

_RETRIEVAL_PIPELINE_PURPOSES = frozenset(
    {
        "need_detection_needs_card",
        "need_detection_language_card",
        "need_detection_memory_card",
        "need_detection_exact_card",
        "need_detection_shape_card",
        "need_detection_facets_card",
        "need_detection_callback_card",
        "need_detection_search_words_card",
        "need_detection_search_words_other_language_card",
        "applicability_relevance_card",
        "applicability_date_card",
        "context_cache_signal_detection",
    }
)


class RecordingProvider(LLMProvider):
    """LLM provider that records request purposes and answers chat/retrieval."""

    name = "response-mode-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose.startswith("need_detection_") and purpose.endswith("_card"):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "budget",
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
                    f"{score_key} exact" for _memory_id, score_key in candidate_keys
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
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Here is the answer.",
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used here: {request.model}")

    def retrieval_pipeline_purpose_count(self) -> int:
        return sum(
            1
            for request in self.requests
            if str(request.metadata.get("purpose")) in _RETRIEVAL_PIPELINE_PURPOSES
        )


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = dict(
        sqlite_path=str(tmp_path / "atagia-response-mode.db"),
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
    )
    base.update(overrides)
    return Settings(**base)


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **settings_overrides: object,
) -> tuple[AppRuntime, RecordingProvider]:
    provider = RecordingProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path, **settings_overrides))
    return runtime, provider


async def _seed_conversation(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    conversation_id: str = "cnv_1",
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


async def _seed_user_memory(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    memory_id: str,
    canonical_text: str,
) -> None:
    connection = await runtime.open_connection()
    try:
        await MemoryObjectRepository(connection, runtime.clock).create_memory_object(
            user_id=user_id,
            workspace_id=None,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text=canonical_text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id=memory_id,
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
        )
    finally:
        await connection.close()


async def _drain_background_tasks(runtime: AppRuntime) -> None:
    for _ in range(50):
        tasks = [task for task in runtime._background_tasks if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)
    raise AssertionError("Background tasks did not settle")


# ---------------------------------------------------------------------------
# F4.3 — Fast mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_mode_chat_skips_retrieval_pipeline_and_records_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What is the budget?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.FAST,
            debug=True,
        )

        # Retrieval pipeline never runs: no need_detection / applicability /
        # cache-signal LLM calls. Only the chat_reply call happens.
        assert provider.retrieval_pipeline_purpose_count() == 0
        chat_calls = sum(
            1
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        )
        assert chat_calls == 1
        assert result.response_text == "Here is the answer."
        assert result.debug is not None
        assert result.debug["response_mode"] == "fast"

        # Persistence still happens (user + assistant messages stored), and the
        # retrieval_event records the fast mode.
        from atagia.core.retrieval_event_repository import RetrievalEventRepository
        from atagia.core.repositories import MessageRepository

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored = await messages.get_messages("cnv_1", "usr_1", limit=10, offset=0)
            assert [m["role"] for m in stored[-2:]] == ["user", "assistant"]
            event = await RetrievalEventRepository(connection, runtime.clock).get_event(
                result.retrieval_event_id,
                "usr_1",
            )
            assert event is not None
            assert event["outcome_json"]["response_mode"] == "fast"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fast_mode_sidecar_context_has_contract_and_no_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)

        sidecar = SidecarService(runtime)
        # First turn establishes a prior message in history.
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Remember I like concise answers.",
            mode="coding_debug",
            response_mode="fast",
            message_id="host-user-1",
        )
        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Understood.",
            message_id="host-assistant-1",
        )
        # Second turn: the prior turn shows up in the recent transcript and the
        # cheap contract read still renders, all without retrieval.
        context = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What did I ask for?",
            mode="coding_debug",
            response_mode="fast",
            message_id="host-user-2",
        )

        assert context.response_mode == "fast"
        # No retrieval pipeline LLM calls at all for fast-mode get_context.
        assert provider.retrieval_pipeline_purpose_count() == 0
        # The contract block (cheap SQL read) is present in the prompt; manifest
        # default dimensions render the [Interaction Contract] section.
        assert "[Interaction Contract]" in context.system_prompt
        # The prior turn participates in the recent transcript window.
        transcript_texts = {entry.text for entry in context.recent_transcript}
        assert "Remember I like concise answers." in transcript_texts
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fast_mode_sidecar_still_enqueues_jobs_on_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please remember that I prefer concise answers.",
            mode="coding_debug",
            response_mode="fast",
            message_id="host-user-1",
        )
        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Got it.",
            message_id="host-assistant-1",
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM worker_job_runs"
            )
            row = await cursor.fetchone()
            assert int(row["count"]) > 0
        finally:
            await connection.close()
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# F4.4 — Smart Fast mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smart_fast_warms_dedicated_keyspace_and_next_turn_reads_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        sidecar = SidecarService(runtime)
        cache_service = ContextCacheService(runtime)

        # First smart_fast turn: immediate context has no warmed entry yet.
        first = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
            response_mode="smart_fast",
            message_id="host-user-1",
        )
        assert first.response_mode == "smart_fast"

        # The background warm runs after the turn and writes a smart_fast entry.
        await _drain_background_tasks(runtime)

        snapshot = await _seed_conversation_snapshot(runtime)
        smart_fast_key = cache_service.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=snapshot.get("workspace_id"),
            active_presence_id=snapshot.get("active_presence_id"),
            active_space_id=snapshot.get("active_space_id"),
            active_mind_id=snapshot.get("active_mind_id"),
            mind_topology=snapshot.get("mind_topology"),
            active_embodiment_id=snapshot.get("active_embodiment_id"),
            active_realm_id=snapshot.get("active_realm_id"),
            operational_profile_token=_profile_token(runtime),
            response_mode=ResponseMode.SMART_FAST,
        )
        normal_key = cache_service.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=snapshot.get("workspace_id"),
            active_presence_id=snapshot.get("active_presence_id"),
            active_space_id=snapshot.get("active_space_id"),
            active_mind_id=snapshot.get("active_mind_id"),
            mind_topology=snapshot.get("mind_topology"),
            active_embodiment_id=snapshot.get("active_embodiment_id"),
            active_realm_id=snapshot.get("active_realm_id"),
            operational_profile_token=_profile_token(runtime),
        )
        assert smart_fast_key != normal_key

        warmed = await runtime.storage_backend.get_context_view(smart_fast_key)
        assert warmed is not None, "smart_fast background task must warm its keyspace"
        # Normal mode never reads/writes the smart_fast key.
        assert await runtime.storage_backend.get_context_view(normal_key) is None

        # Second smart_fast turn folds in the warmed retrieved memory.
        second = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Remind me of the budget.",
            mode="coding_debug",
            response_mode="smart_fast",
            message_id="host-user-2",
        )
        warmed_memory_ids = {m.memory_id for m in second.memories}
        assert "mem_budget" in warmed_memory_ids
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_normal_turn_after_smart_fast_reads_only_normal_keyspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        sidecar = SidecarService(runtime)
        cache_service = ContextCacheService(runtime)

        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
            response_mode="smart_fast",
            message_id="host-user-1",
        )
        await _drain_background_tasks(runtime)

        snapshot = await _seed_conversation_snapshot(runtime)
        normal_key = cache_service.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=snapshot.get("workspace_id"),
            active_presence_id=snapshot.get("active_presence_id"),
            active_space_id=snapshot.get("active_space_id"),
            active_mind_id=snapshot.get("active_mind_id"),
            mind_topology=snapshot.get("mind_topology"),
            active_embodiment_id=snapshot.get("active_embodiment_id"),
            active_realm_id=snapshot.get("active_realm_id"),
            operational_profile_token=_profile_token(runtime),
        )
        # No normal-keyspace entry exists yet (only smart_fast warmed).
        assert await runtime.storage_backend.get_context_view(normal_key) is None

        normal = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
            message_id="host-user-2",
        )
        assert normal.response_mode == "normal"
        # A normal turn now publishes into the normal keyspace.
        assert await runtime.storage_backend.get_context_view(normal_key) is not None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_background_failure_does_not_break_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)

        async def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Injected warm failure")

        monkeypatch.setattr(
            ContextCacheService, "resolve_with_connection", boom
        )

        context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
            response_mode="smart_fast",
        )
        # The served response is unaffected by the background failure.
        assert context.response_mode == "smart_fast"
        assert context.system_prompt

        # The background task runs and logs (warning) without crashing.
        await _drain_background_tasks(runtime)
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# F4 acceptance — default normal mode is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_mode_is_normal_and_runs_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
        )
        assert context.response_mode == "normal"
        # Default mode runs the retrieval pipeline (need_detection at minimum).
        assert provider.retrieval_pipeline_purpose_count() > 0
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _seed_conversation_snapshot(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    conversation_id: str = "cnv_1",
) -> dict[str, object]:
    connection = await runtime.open_connection()
    try:
        conversation = await ConversationRepository(
            connection, runtime.clock
        ).get_conversation(conversation_id, user_id)
        assert conversation is not None
        return conversation
    finally:
        await connection.close()


def _profile_token(runtime: AppRuntime) -> str:
    from atagia.services.chat_support import default_operational_profile_snapshot

    return default_operational_profile_snapshot(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
    ).token
