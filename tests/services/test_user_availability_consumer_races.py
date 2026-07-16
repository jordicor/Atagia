"""Cross-runtime fences for consumers of canonical transcript sources."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.models.schemas_api import (
    ReplaceSelectedTranscriptRequest,
    SelectedTranscriptMessage,
)
from atagia.models.schemas_openai_proxy import OpenAIChatCompletionRequest
from atagia.services.chat_service import ChatService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import TranscriptRebuildInProgressError
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMStreamEvent,
)
from atagia.services.openai_proxy_service import OpenAIProxyService
from atagia.services.selected_transcript_service import SelectedTranscriptService
from atagia.services.sidecar_service import SidecarService

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)

USER_ID = "usr_consumer_race"
CONVERSATION_ID = "cnv_consumer_race"
PLATFORM_ID = "openclaw"
KEEP_USER_ID = "msg_keep_user"
KEEP_ASSISTANT_ID = "msg_keep_assistant"
KEEP_USER_TIME = "2026-07-13T10:00:00+00:00"
KEEP_ASSISTANT_TIME = "2026-07-13T10:01:00+00:00"


class _BlockingReplyProvider(LLMProvider):
    name = "consumer-race"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        assert request.metadata.get("purpose") == "chat_reply"
        self.entered.set()
        await self.release.wait()
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="A response from the stale source snapshot.",
            finish_reason="stop",
        )

    async def stream(
        self,
        request: LLMCompletionRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        assert request.metadata.get("purpose") == "chat_reply"
        self.entered.set()
        await self.release.wait()
        yield LLMStreamEvent(type="text", content="stale stream output")
        yield LLMStreamEvent(
            type="done",
            payload={"finish_reason": "stop"},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in this test: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "consumer-races.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="consumer-race-model",
        llm_forced_global_model="openai/consumer-race-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        lifecycle_worker_enabled=False,
        lifecycle_lazy_enabled=False,
        initial_context_package_read_enabled=False,
        initial_context_package_refresh_enabled=False,
        assistant_guidance_enabled=False,
        adaptive_retrieval=False,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtimes(tmp_path: Path) -> tuple[AppRuntime, AppRuntime]:
    settings = _settings(tmp_path)
    runtime_a = await initialize_runtime(settings)
    runtime_b = await initialize_runtime(settings)
    # The service contract requires durable workers. Tests exercise the durable
    # hand-off without starting worker tasks, matching selected-transcript tests.
    runtime_b.settings = replace(runtime_b.settings, workers_enabled=True)
    return runtime_a, runtime_b


async def _seed_canonical_turn(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        await UserRepository(connection, runtime.clock).create_user(USER_ID)
        await ConversationRepository(
            connection,
            runtime.clock,
        ).create_conversation(
            CONVERSATION_ID,
            USER_ID,
            None,
            "general_qa",
            "Consumer race",
            platform_id=PLATFORM_ID,
        )
        messages = MessageRepository(connection, runtime.clock)
        await messages.create_message(
            KEEP_USER_ID,
            CONVERSATION_ID,
            "user",
            1,
            "Retained user message.",
            occurred_at=KEEP_USER_TIME,
        )
        await messages.create_message(
            KEEP_ASSISTANT_ID,
            CONVERSATION_ID,
            "assistant",
            2,
            "Retained assistant message.",
            occurred_at=KEEP_ASSISTANT_TIME,
        )
    finally:
        await connection.close()


def _replacement_request(operation_id: str) -> ReplaceSelectedTranscriptRequest:
    return ReplaceSelectedTranscriptRequest(
        user_id=USER_ID,
        platform_id=PLATFORM_ID,
        operation_id=operation_id,
        selection_epoch=1,
        mutation_kind="regeneration",
        retained_cutoff_message_id=KEEP_ASSISTANT_ID,
        messages=[
            SelectedTranscriptMessage(
                message_id=KEEP_USER_ID,
                host_message_id="host_keep_user",
                generation_id="generation_keep",
                source_namespace="openclaw:selected",
                source_seq=1,
                role="user",
                text="Retained user message.",
                occurred_at=KEEP_USER_TIME,
            ),
            SelectedTranscriptMessage(
                message_id=KEEP_ASSISTANT_ID,
                host_message_id="host_keep_assistant",
                generation_id="generation_keep",
                source_namespace="openclaw:selected",
                source_seq=2,
                role="assistant",
                text="Retained assistant message.",
                occurred_at=KEEP_ASSISTANT_TIME,
            ),
        ],
    )


async def _replace_selected_transcript(
    runtime: AppRuntime,
    *,
    operation_id: str,
) -> None:
    response = await SelectedTranscriptService(runtime).replace(
        conversation_id=CONVERSATION_ID,
        request=_replacement_request(operation_id),
    )
    assert response.status == "rebuilding"
    assert response.stage == "preparing"


async def _stored_message_ids(runtime: AppRuntime) -> list[str]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC
            """,
            (CONVERSATION_ID,),
        )
        return [str(row["id"]) for row in await cursor.fetchall()]
    finally:
        await connection.close()


async def _proxy_run_count(runtime: AppRuntime) -> int:
    connection = await runtime.open_connection()
    try:
        row = await (
            await connection.execute(
                "SELECT COUNT(*) AS total FROM proxy_turn_runs WHERE user_id = ?",
                (USER_ID,),
            )
        ).fetchone()
        return int(row["total"])
    finally:
        await connection.close()


async def _close_runtimes(*runtimes: AppRuntime) -> None:
    for runtime in runtimes:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_final_write_rejects_cross_runtime_transcript_replacement(
    tmp_path: Path,
) -> None:
    runtime_a, runtime_b = await _build_runtimes(tmp_path)
    provider = _BlockingReplyProvider()
    runtime_a.llm_client = LLMClient(provider.name, [provider])
    task: asyncio.Task[Any] | None = None
    try:
        await _seed_canonical_turn(runtime_a)
        task = asyncio.create_task(
            ChatService(runtime_a).chat_reply(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                message_text="This response must not survive source replacement.",
                assistant_mode_id="general_qa",
                platform_id=PLATFORM_ID,
                privacy_enforcement="off",
                response_mode="fast",
            )
        )
        await asyncio.wait_for(provider.entered.wait(), timeout=5.0)

        await _replace_selected_transcript(
            runtime_b,
            operation_id="consumer-race/chat",
        )
        provider.release.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(task, timeout=5.0)
        task = None
        assert await _stored_message_ids(runtime_a) == [
            KEEP_USER_ID,
            KEEP_ASSISTANT_ID,
        ]
    finally:
        provider.release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _close_runtimes(runtime_a, runtime_b)


@pytest.mark.asyncio
async def test_sidecar_context_final_write_rejects_cross_runtime_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a, runtime_b = await _build_runtimes(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_resolve = ContextCacheService.resolve_fast_with_connection

    async def blocking_resolve(
        self: ContextCacheService,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolution = await original_resolve(self, *args, **kwargs)
        entered.set()
        await release.wait()
        return resolution

    monkeypatch.setattr(
        ContextCacheService,
        "resolve_fast_with_connection",
        blocking_resolve,
    )
    task: asyncio.Task[Any] | None = None
    try:
        await _seed_canonical_turn(runtime_a)
        task = asyncio.create_task(
            SidecarService(runtime_a).get_context(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                message="Context request from the old transcript.",
                mode="general_qa",
                platform_id=PLATFORM_ID,
                message_id="msg_stale_context",
                privacy_enforcement="off",
                response_mode="fast",
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        await _replace_selected_transcript(
            runtime_b,
            operation_id="consumer-race/context",
        )
        release.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(task, timeout=5.0)
        task = None
        assert "msg_stale_context" not in await _stored_message_ids(runtime_a)
    finally:
        release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _close_runtimes(runtime_a, runtime_b)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ingest", "add_response"])
async def test_sidecar_message_write_rejects_cross_runtime_replacement(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a, runtime_b = await _build_runtimes(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_recent = SidecarService._recent_messages_for_write

    async def blocking_recent(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        recent = await original_recent(*args, **kwargs)
        entered.set()
        await release.wait()
        return recent

    monkeypatch.setattr(
        SidecarService,
        "_recent_messages_for_write",
        staticmethod(blocking_recent),
    )
    stale_message_id = f"msg_stale_{operation}"
    task: asyncio.Task[Any] | None = None
    try:
        await _seed_canonical_turn(runtime_a)
        sidecar = SidecarService(runtime_a)
        if operation == "ingest":
            task = asyncio.create_task(
                sidecar.ingest_message(
                    user_id=USER_ID,
                    conversation_id=CONVERSATION_ID,
                    role="user",
                    text="Stale ingest.",
                    mode="general_qa",
                    platform_id=PLATFORM_ID,
                    message_id=stale_message_id,
                    privacy_enforcement="off",
                )
            )
        else:
            task = asyncio.create_task(
                sidecar.add_response(
                    user_id=USER_ID,
                    conversation_id=CONVERSATION_ID,
                    text="Stale response.",
                    mode="general_qa",
                    platform_id=PLATFORM_ID,
                    message_id=stale_message_id,
                    privacy_enforcement="off",
                )
            )
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        await _replace_selected_transcript(
            runtime_b,
            operation_id=f"consumer-race/{operation}",
        )
        release.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(task, timeout=5.0)
        task = None
        assert stale_message_id not in await _stored_message_ids(runtime_a)
    finally:
        release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _close_runtimes(runtime_a, runtime_b)


def _proxy_request(*, stream: bool) -> OpenAIChatCompletionRequest:
    return OpenAIChatCompletionRequest(
        model="atagia-memory-proxy",
        messages=[{"role": "user", "content": "Proxy turn from old sources."}],
        stream=stream,
    )


def _proxy_call_kwargs(*, suffix: str) -> dict[str, str]:
    return {
        "claimed_user_id": USER_ID,
        "conversation_id_header": CONVERSATION_ID,
        "assistant_mode_header": "general_qa",
        "platform_id_header": PLATFORM_ID,
        "message_id_header": f"msg_proxy_request_{suffix}",
        "response_message_id_header": f"msg_proxy_response_{suffix}",
    }


@pytest.mark.asyncio
async def test_proxy_nonstream_finalization_rejects_cross_runtime_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: SidecarService, **kwargs: Any) -> None:
        del self, kwargs

    monkeypatch.setattr(SidecarService, "get_context", no_context)
    runtime_a, runtime_b = await _build_runtimes(tmp_path)
    provider = _BlockingReplyProvider()
    runtime_a.llm_client = LLMClient(provider.name, [provider])
    task: asyncio.Task[Any] | None = None
    try:
        await _seed_canonical_turn(runtime_a)
        task = asyncio.create_task(
            OpenAIProxyService(runtime_a).complete(
                _proxy_request(stream=False),
                **_proxy_call_kwargs(suffix="nonstream"),
            )
        )
        await asyncio.wait_for(provider.entered.wait(), timeout=5.0)

        await _replace_selected_transcript(
            runtime_b,
            operation_id="consumer-race/proxy-nonstream",
        )
        provider.release.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(task, timeout=5.0)
        task = None
        assert await _stored_message_ids(runtime_a) == [
            KEEP_USER_ID,
            KEEP_ASSISTANT_ID,
        ]
        assert await _proxy_run_count(runtime_a) == 0
    finally:
        provider.release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _close_runtimes(runtime_a, runtime_b)


@pytest.mark.asyncio
async def test_proxy_stream_blocks_before_first_sse_after_cross_runtime_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: SidecarService, **kwargs: Any) -> None:
        del self, kwargs

    monkeypatch.setattr(SidecarService, "get_context", no_context)
    runtime_a, runtime_b = await _build_runtimes(tmp_path)
    provider = _BlockingReplyProvider()
    runtime_a.llm_client = LLMClient(provider.name, [provider])
    task: asyncio.Task[Any] | None = None
    try:
        await _seed_canonical_turn(runtime_a)
        task = asyncio.create_task(
            OpenAIProxyService(runtime_a).stream(
                _proxy_request(stream=True),
                **_proxy_call_kwargs(suffix="stream"),
            )
        )
        await asyncio.wait_for(provider.entered.wait(), timeout=5.0)

        await _replace_selected_transcript(
            runtime_b,
            operation_id="consumer-race/proxy-stream",
        )
        provider.release.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(task, timeout=5.0)
        task = None
        assert await _stored_message_ids(runtime_a) == [
            KEEP_USER_ID,
            KEEP_ASSISTANT_ID,
        ]
        assert await _proxy_run_count(runtime_a) == 0
    finally:
        provider.release.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _close_runtimes(runtime_a, runtime_b)
