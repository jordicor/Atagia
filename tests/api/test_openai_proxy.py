from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import json
import re
import sqlite3
from typing import Any

import httpx
import pytest

from atagia.app import create_app
from atagia.core.config import Settings
from atagia.core.mind_repository import MindRepository
from atagia.core.repositories import UserRepository
from atagia.models.schemas_memory import MindKind
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.models.schemas_openai_proxy import OpenAIChatCompletionRequest
from atagia.services.llm_client import (
    LLMError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMStreamEvent,
    TransientLLMError,
)
from atagia.services.llm_run_guard import LLMRunGuard, LLMRunGuardConfig
from atagia.services.openai_proxy_service import OpenAIProxyService

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


class ProxyProvider(LLMProvider):
    name = "proxy-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.raise_after_first_stream_event = False
        self.raise_before_first_stream_event = False
        self.raise_after_non_output_stream_event = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if _is_need_detection_card_purpose(purpose):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_query_language_card": "en",
                "need_detection_answer_language_card": "en",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "remember proxy",
                "need_detection_search_words_other_language_card": "none",
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=outputs[purpose],
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
                        "short_followup": False,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "chat_reply" and request.tools:
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                tool_calls=[
                    {
                        "id": "call_lookup",
                        "type": "function",
                        "name": "lookup",
                        "arguments": json.dumps({"query": "atagia"}),
                    }
                ],
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Proxy reply.",
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def stream(self, request: LLMCompletionRequest):
        self.requests.append(request)
        assert request.metadata.get("purpose") == "chat_reply"
        if self.raise_before_first_stream_event:
            raise LLMError("preflight failed")
        if self.raise_after_non_output_stream_event:
            yield LLMStreamEvent(type="done", payload={})
            raise LLMError("blocked after metadata")
        if request.tools:
            yield LLMStreamEvent(
                type="tool_call",
                payload={
                    "id": "call_stream_lookup",
                    "type": "function",
                    "name": "lookup",
                    "arguments": json.dumps({"query": "stream"}),
                },
            )
            yield LLMStreamEvent(type="done", payload={})
            return
        yield LLMStreamEvent(type="text", content="Proxy ")
        if self.raise_after_first_stream_event:
            raise TransientLLMError("stream failed")
        yield LLMStreamEvent(type="text", content="stream.")
        yield LLMStreamEvent(type="done", payload={})

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in proxy tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-openai-proxy.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="proxy-reply-model",
        llm_forced_global_model="openai/proxy-reply-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
        allow_insecure_http=False,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _ensure_capture_namespace(service, identity):
    """Create only the namespace while a test replaces the real context path."""

    from atagia.core.conversation_namespace import (
        capture_conversation_namespace_snapshot,
    )
    from atagia.services.sidecar_service import SidecarService

    connection = await service.runtime.open_connection()
    try:
        sidecar = SidecarService(service.runtime)
        await sidecar.ensure_user_exists(connection, identity.user_id)
        await sidecar.ensure_conversation(
            connection,
            user_id=identity.user_id,
            conversation_id=identity.conversation_id,
            workspace_id=None,
            assistant_mode_id=identity.mode or identity.assistant_mode_id,
            platform_id=identity.platform_id,
            mode=identity.mode or identity.assistant_mode_id,
            incognito=identity.incognito,
            cross_chat_memory=identity.cross_chat_memory,
        )
        snapshot = await capture_conversation_namespace_snapshot(
            connection,
            service.runtime.clock,
            user_id=identity.user_id,
            conversation_id=identity.conversation_id,
        )
        assert snapshot is not None
        return snapshot
    finally:
        await connection.close()


async def _precreate_proxy_conversations(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    conversation_ids: tuple[str, ...],
) -> None:
    """Create each conversation with its own sequential first turn.

    Creating a conversation is a user-scoped source mutation: it bumps the
    user's ``derivation_revision``, which invalidates the source snapshot every
    other in-flight turn for that user captured. Concurrent FIRST turns in two
    brand-new conversations therefore race, and the loser is refused with the
    documented retry -- see
    ``test_concurrent_first_turns_in_new_conversations_use_the_retry_contract``,
    which is where that behavior belongs. A test about something else must not
    inherit the race, so it creates the conversations through the same proxy
    endpoint first, one at a time.
    """
    for conversation_id in conversation_ids:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer service-key",
                "X-Atagia-User-Id": user_id,
                "X-Atagia-Platform-Id": "proxy_desktop",
                "X-Atagia-Conversation-Id": conversation_id,
            },
            json={
                "model": "atagia-memory-proxy",
                "messages": [{"role": "user", "content": "Open the conversation."}],
            },
        )
        assert response.status_code == 200, (conversation_id, response.text)


@pytest.mark.asyncio
async def test_openai_proxy_models_and_non_streaming_completion(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            models = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer service-key"},
            )
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-User-Persona-Id": "persona_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Character-Id": "char_proxy",
                    "X-Atagia-Conversation-Id": "cnv_proxy",
                    "X-Atagia-Mode": "general_qa",
                    "X-Atagia-Incognito": "false",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "system", "content": "Base system"},
                        {"role": "user", "content": "Remember the proxy path."},
                    ],
                },
            )

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "atagia-memory-proxy"
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Proxy reply."
    chat_requests = [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests
    assert chat_requests[-1].metadata["user_persona_id"] == "persona_proxy"
    assert chat_requests[-1].metadata["platform_id"] == "proxy_desktop"
    assert chat_requests[-1].metadata["character_id"] == "char_proxy"
    assert chat_requests[-1].metadata["mode"] == "general_qa"
    assert chat_requests[-1].metadata["incognito"] is False
    assert "[ATAGIA MEMORY CONTEXT" not in chat_requests[-1].messages[0].content
    assert "You are the Atagia assistant" not in chat_requests[-1].messages[0].content
    assert chat_requests[-1].messages[-1].content == "Remember the proxy path."


@pytest.mark.asyncio
async def test_openai_proxy_requires_explicit_conversation_id(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy_a",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-Conversation-Id" in response.json()["error"]["message"]
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_requires_explicit_platform_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Conversation-Id": "cnv_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-Platform-Id" in response.json()["error"]["message"]
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
@pytest.mark.parametrize("mind_case", ("missing", "cross_owner"))
async def test_openai_proxy_rejects_unowned_mind_before_provider(
    tmp_path: Path,
    mind_case: str,
    stream: bool,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    mind_id = f"mind_{mind_case}"
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        if mind_case == "cross_owner":
            connection = await app.state.runtime.open_connection()
            try:
                await UserRepository(
                    connection,
                    app.state.runtime.clock,
                ).create_user("usr_mind_owner")
                await MindRepository(
                    connection,
                    app.state.runtime.clock,
                ).resolve_mind(
                    owner_user_id="usr_mind_owner",
                    mind_id=mind_id,
                    kind=MindKind.OWNED_AI,
                    display_name="Foreign Mind",
                    source_kind="test",
                    source_id=mind_id,
                )
            finally:
                await connection.close()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_mind_requester",
                    "X-Atagia-Conversation-Id": f"cnv_{mind_case}_{stream}",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Mind-Id": mind_id,
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Use the requested mind."}
                    ],
                    "stream": stream,
                },
            )

        verification = await app.state.runtime.open_connection()
        try:
            message_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM messages"
                        )
                    ).fetchone()
                )["count"]
            )
            run_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await verification.close()

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "mind_not_found"
    assert provider.requests == []
    assert message_count == 0
    assert run_count == 0


@pytest.mark.asyncio
async def test_openai_proxy_reads_redesign_metadata_identity(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello metadata"}],
                    "metadata": {
                        "atagia_conversation_id": "cnv_proxy_metadata",
                        "atagia_user_persona_id": "persona_meta",
                        "atagia_platform_id": "platform_meta",
                        "atagia_character_id": "char_meta",
                        "atagia_mode": "general_qa",
                        "atagia_incognito": True,
                    },
                },
            )

    assert response.status_code == 200
    chat_requests = [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests[-1].metadata["conversation_id"] == "cnv_proxy_metadata"
    assert chat_requests[-1].metadata["user_persona_id"] == "persona_meta"
    assert chat_requests[-1].metadata["platform_id"] == "platform_meta"
    assert chat_requests[-1].metadata["character_id"] == "char_meta"
    assert chat_requests[-1].metadata["mode"] == "general_qa"
    assert chat_requests[-1].metadata["incognito"] is True
    assert chat_requests[-1].metadata["cross_chat_memory"] is False


@pytest.mark.asyncio
async def test_openai_proxy_propagates_sidecar_control_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        capture_context,
    )
    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.OpenAIProxyService._ensure_proxy_namespace",
        _ensure_capture_namespace,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_controls",
                    "X-Atagia-Active-Presence-Id": "presence_header",
                    "X-Atagia-Mind-Id": "mind_header",
                    "X-Atagia-Mind-Topology": "ojocentauri",
                    "X-Atagia-Embodiment-Id": "body_header",
                    "X-Atagia-Realm-Id": "realm_header",
                    "X-Atagia-Space-Id": "space_header",
                    "X-Atagia-Message-Id": "host-user-1",
                    "X-Atagia-Source-Seq": "7",
                    "X-Atagia-Response-Message-Id": "host-assistant-1",
                    "X-Atagia-Response-Source-Seq": "8",
                    "X-Atagia-Ingest-Origin": "live_turn",
                    "X-Atagia-Confirmation-Strategy": "live_prompt_allowed",
                    "X-Atagia-Memory-Privacy-Mode": "trusted_private",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello controls"}],
                },
            )
        connection = await app.state.runtime.open_connection()
        try:
            stored_response = await (
                await connection.execute(
                    "SELECT id, seq FROM messages WHERE id = ?",
                    ("host-assistant-1",),
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 200
    assert captured["context"]["message_id"] == "host-user-1"
    assert captured["context"]["active_presence_id"] == "presence_header"
    assert captured["context"]["mind_id"] == "mind_header"
    assert captured["context"]["mind_topology"] == "ojocentauri"
    assert captured["context"]["embodiment_id"] == "body_header"
    assert captured["context"]["realm_id"] == "realm_header"
    assert captured["context"]["space_id"] == "space_header"
    assert captured["context"]["source_seq"] == 7
    assert captured["context"]["ingest_origin"] == "live_turn"
    assert captured["context"]["confirmation_strategy"] == "live_prompt_allowed"
    assert captured["context"]["memory_privacy_mode"] == "trusted_private"
    assert stored_response is not None
    assert (stored_response["id"], stored_response["seq"]) == (
        "host-assistant-1",
        8,
    )
    chat_requests = [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]
    assert chat_requests[-1].metadata["message_id"] == "host-user-1"
    assert chat_requests[-1].metadata["active_presence_id"] == "presence_header"
    assert chat_requests[-1].metadata["mind_id"] == "mind_header"
    assert chat_requests[-1].metadata["mind_topology"] == "ojocentauri"
    assert chat_requests[-1].metadata["embodiment_id"] == "body_header"
    assert chat_requests[-1].metadata["realm_id"] == "realm_header"
    assert chat_requests[-1].metadata["space_id"] == "space_header"
    assert chat_requests[-1].metadata["response_message_id"] == "host-assistant-1"
    assert chat_requests[-1].metadata["memory_privacy_mode"] == "trusted_private"


@pytest.mark.asyncio
async def test_openai_proxy_accepts_control_fields_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        capture_context,
    )
    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.OpenAIProxyService._ensure_proxy_namespace",
        _ensure_capture_namespace,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Hello metadata controls"}
                    ],
                    "metadata": {
                        "conversation_id": "cnv_proxy_metadata_controls",
                        "platform_id": "proxy_metadata",
                        "active_presence_id": "presence_meta",
                        "mind_id": "mind_meta",
                        "mind_topology": "ojocentauri",
                        "embodiment_id": "body_meta",
                        "realm_id": "realm_meta",
                        "space_id": "space_meta",
                        "message_id": "meta-user-1",
                        "source_seq": 3,
                        "response_message_id": "meta-assistant-1",
                        "response_source_seq": "4",
                        "ingest_origin": "live_turn",
                        "confirmation_strategy": "live_prompt_allowed",
                        "memory_privacy_mode": "balanced",
                    },
                },
            )
        connection = await app.state.runtime.open_connection()
        try:
            stored_response = await (
                await connection.execute(
                    "SELECT id, seq FROM messages WHERE id = ?",
                    ("meta-assistant-1",),
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 200
    assert captured["context"]["message_id"] == "meta-user-1"
    assert captured["context"]["active_presence_id"] == "presence_meta"
    assert captured["context"]["mind_id"] == "mind_meta"
    assert captured["context"]["mind_topology"] == "ojocentauri"
    assert captured["context"]["embodiment_id"] == "body_meta"
    assert captured["context"]["realm_id"] == "realm_meta"
    assert captured["context"]["space_id"] == "space_meta"
    assert captured["context"]["source_seq"] == 3
    assert stored_response is not None
    assert (stored_response["id"], stored_response["seq"]) == (
        "meta-assistant-1",
        4,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "metadata"),
    [
        ({"X-Atagia-Response-Mode": "turbo_nonsense"}, None),
        ({"X-Atagia-Response-Mode": ""}, None),
        (None, {"atagia_response_mode": "turbo_nonsense"}),
        (None, {"response_mode": 7}),
    ],
)
async def test_openai_proxy_invalid_response_mode_is_rejected_before_provider(
    tmp_path: Path,
    headers: dict[str, str] | None,
    metadata: dict[str, Any] | None,
) -> None:
    """An invalid explicit response_mode fails with a stable 400, never a fallback."""

    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            payload: dict[str, Any] = {
                "model": "atagia-memory-proxy",
                "messages": [{"role": "user", "content": "Reject bad modes."}],
            }
            if metadata is not None:
                payload["metadata"] = metadata
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_bad_mode",
                    **(headers or {}),
                },
                json=payload,
            )
        connection = await app.state.runtime.open_connection()
        try:
            stored_messages = await (
                await connection.execute(
                    "SELECT COUNT(*) AS message_count FROM messages"
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_response_mode"
    assert error["param"] == "response_mode"
    assert not provider.requests
    assert stored_messages["message_count"] == 0


@pytest.mark.asyncio
async def test_openai_proxy_conflicting_response_mode_claims_are_rejected(
    tmp_path: Path,
) -> None:
    """Two valid but different explicit response_mode claims fail with 400."""

    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_conflict_mode",
                    "X-Atagia-Response-Mode": "fast",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Conflicting modes."}],
                    "metadata": {"atagia_response_mode": "smart_fast"},
                    "response_mode": "fast",
                },
            )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "conflicting_response_mode"
    assert error["param"] == "response_mode"
    assert not provider.requests


@pytest.mark.asyncio
async def test_openai_proxy_agreeing_response_mode_claims_reach_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same explicit response_mode across every source stays valid."""

    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        capture_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_agree_mode",
                    "X-Atagia-Response-Mode": "smart_fast",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Matching modes."}],
                    "metadata": {"atagia_response_mode": "smart_fast"},
                    "response_mode": "smart_fast",
                },
            )

    assert response.status_code == 200
    assert captured["context"]["response_mode"] == "smart_fast"


@pytest.mark.asyncio
async def test_openai_proxy_valid_response_mode_header_passes_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        capture_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_good_mode",
                    "X-Atagia-Response-Mode": "smart_fast",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello mode"}],
                },
            )

    assert response.status_code == 200
    assert captured["context"]["response_mode"] == "smart_fast"


@pytest.mark.asyncio
async def test_openai_proxy_adaptive_retrieval_header_reaches_context(
    tmp_path: Path,
) -> None:
    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "atagia.services.openai_proxy_service.SidecarService.get_context",
            capture_context,
        )
        app = create_app(_settings(tmp_path))
        provider = ProxyProvider()
        async with app.router.lifespan_context(app):
            app.state.runtime.llm_client = LLMClient(
                provider_name=provider.name,
                providers=[provider],
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer service-key",
                        "X-Atagia-User-Id": "usr_proxy",
                        "X-Atagia-Platform-Id": "proxy_desktop",
                        "X-Atagia-Conversation-Id": "cnv_proxy_adaptive_header",
                        "X-Atagia-Adaptive-Retrieval": "true",
                    },
                    json={
                        "model": "atagia-memory-proxy",
                        "messages": [{"role": "user", "content": "Hello gate"}],
                    },
                )

    assert response.status_code == 200
    assert captured["context"]["adaptive_retrieval"] is True


@pytest.mark.asyncio
async def test_openai_proxy_adaptive_retrieval_metadata_reaches_context(
    tmp_path: Path,
) -> None:
    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "atagia.services.openai_proxy_service.SidecarService.get_context",
            capture_context,
        )
        app = create_app(_settings(tmp_path))
        provider = ProxyProvider()
        async with app.router.lifespan_context(app):
            app.state.runtime.llm_client = LLMClient(
                provider_name=provider.name,
                providers=[provider],
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer service-key",
                        "X-Atagia-User-Id": "usr_proxy",
                    },
                    json={
                        "model": "atagia-memory-proxy",
                        "messages": [
                            {"role": "user", "content": "Hello metadata gate"}
                        ],
                        "metadata": {
                            "atagia_conversation_id": "cnv_proxy_adaptive_metadata",
                            "atagia_platform_id": "proxy_desktop",
                            "atagia_adaptive_retrieval": True,
                        },
                    },
                )

    assert response.status_code == 200
    assert captured["context"]["adaptive_retrieval"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "metadata"),
    [
        ({"X-Atagia-Adaptive-Retrieval": "maybe"}, None),
        ({"X-Atagia-Adaptive-Retrieval": ""}, None),
        (None, {"atagia_adaptive_retrieval": "sometimes"}),
        (None, {"adaptive_retrieval": 7}),
    ],
)
async def test_openai_proxy_invalid_adaptive_retrieval_is_rejected_before_provider(
    tmp_path: Path,
    headers: dict[str, str] | None,
    metadata: dict[str, Any] | None,
) -> None:
    """An invalid explicit adaptive_retrieval fails with a stable 400, never a fallback."""

    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            payload: dict[str, Any] = {
                "model": "atagia-memory-proxy",
                "messages": [{"role": "user", "content": "Reject bad gate flags."}],
            }
            if metadata is not None:
                payload["metadata"] = metadata
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_adaptive_bad",
                    **(headers or {}),
                },
                json=payload,
            )
        connection = await app.state.runtime.open_connection()
        try:
            stored_messages = await (
                await connection.execute(
                    "SELECT COUNT(*) AS message_count FROM messages"
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_adaptive_retrieval"
    assert error["param"] == "adaptive_retrieval"
    assert not provider.requests
    assert stored_messages["message_count"] == 0


@pytest.mark.asyncio
async def test_openai_proxy_conflicting_adaptive_retrieval_claims_are_rejected(
    tmp_path: Path,
) -> None:
    """Two valid but different explicit adaptive_retrieval claims fail with 400."""

    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_adaptive_conflict",
                    "X-Atagia-Adaptive-Retrieval": "true",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Conflicting flags."}],
                    "metadata": {"atagia_adaptive_retrieval": False},
                    "adaptive_retrieval": True,
                },
            )
        connection = await app.state.runtime.open_connection()
        try:
            stored_messages = await (
                await connection.execute(
                    "SELECT COUNT(*) AS message_count FROM messages"
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "conflicting_adaptive_retrieval"
    assert error["param"] == "adaptive_retrieval"
    assert not provider.requests
    assert stored_messages["message_count"] == 0


@pytest.mark.asyncio
async def test_openai_proxy_agreeing_adaptive_retrieval_claims_reach_context(
    tmp_path: Path,
) -> None:
    """The same explicit adaptive_retrieval across every source stays valid."""

    captured: dict[str, dict] = {}

    async def capture_context(self, **kwargs):
        captured["context"] = kwargs
        return None

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "atagia.services.openai_proxy_service.SidecarService.get_context",
            capture_context,
        )
        app = create_app(_settings(tmp_path))
        provider = ProxyProvider()
        async with app.router.lifespan_context(app):
            app.state.runtime.llm_client = LLMClient(
                provider_name=provider.name,
                providers=[provider],
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer service-key",
                        "X-Atagia-User-Id": "usr_proxy",
                        "X-Atagia-Platform-Id": "proxy_desktop",
                        "X-Atagia-Conversation-Id": "cnv_proxy_adaptive_agree",
                        "X-Atagia-Adaptive-Retrieval": "false",
                    },
                    json={
                        "model": "atagia-memory-proxy",
                        "messages": [{"role": "user", "content": "Matching flags."}],
                        "metadata": {"atagia_adaptive_retrieval": False},
                        "adaptive_retrieval": False,
                    },
                )

    assert response.status_code == 200
    # "false" everywhere must override the engine default (adaptive gate ON).
    assert captured["context"]["adaptive_retrieval"] is False


@pytest.mark.asyncio
async def test_openai_proxy_streaming_completion(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Stream with memory."},
                    ],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert "chat.completion.chunk" in text
    assert "Proxy " in text
    assert "stream." in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_concurrent_burst_streams_renew_by_time_not_per_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """800 streamed events across two live streams must renew on TIME, not per
    event.

    Both conversations are created before the burst: the subject here is
    renewal cadence, and concurrent conversation CREATION is a separate,
    separately-tested behavior that would otherwise decide this test's outcome
    (it did -- the burst ran on only one stream most runs, which is exactly when
    ``renew_calls == 0`` proves the least).
    """

    class BurstProvider(ProxyProvider):
        async def stream(self, request: LLMCompletionRequest):
            self.requests.append(request)
            assert request.metadata.get("purpose") == "chat_reply"
            for _index in range(400):
                yield LLMStreamEvent(type="text", content="x")
            yield LLMStreamEvent(type="done", payload={})

    renew_calls = 0

    async def count_renewals(
        _service: OpenAIProxyService,
        claim: Any,
    ) -> bool:
        nonlocal renew_calls
        assert claim is not None
        renew_calls += 1
        return True

    @asynccontextmanager
    async def no_heartbeat(
        _service: OpenAIProxyService,
        claim: Any,
    ):
        assert claim is not None
        yield asyncio.Event()

    monkeypatch.setattr(OpenAIProxyService, "_renew_claim", count_renewals)
    monkeypatch.setattr(OpenAIProxyService, "_renewing_claim", no_heartbeat)
    app = create_app(_settings(tmp_path))
    provider = BurstProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:

            async def run_stream(conversation_id: str) -> httpx.Response:
                return await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer service-key",
                        "X-Atagia-User-Id": "usr_proxy",
                        "X-Atagia-Platform-Id": "proxy_desktop",
                        "X-Atagia-Conversation-Id": conversation_id,
                    },
                    json={
                        "model": "atagia-memory-proxy",
                        "stream": True,
                        "messages": [{"role": "user", "content": "Burst."}],
                    },
                )

            await _precreate_proxy_conversations(
                client,
                user_id="usr_proxy",
                conversation_ids=("cnv_proxy_burst_a", "cnv_proxy_burst_b"),
            )
            responses = await asyncio.gather(
                run_stream("cnv_proxy_burst_a"),
                run_stream("cnv_proxy_burst_b"),
            )

    assert all(response.status_code == 200 for response in responses)
    assert all("data: [DONE]" in response.text for response in responses)
    # Counted across the setup turns too: no elapsed time, so no renewal is due
    # anywhere in the test, and the 800 burst events cannot have triggered one.
    assert renew_calls == 0


@pytest.mark.asyncio
async def test_concurrent_first_turns_in_new_conversations_use_the_retry_contract(
    tmp_path: Path,
) -> None:
    """A turn is either fully served or refused with the documented retry.

    Creating a conversation bumps the user's ``derivation_revision``
    (``SidecarService.ensure_conversation``), and every turn revalidates the
    source snapshot it captured before returning. Two FIRST turns for one user
    therefore race: whichever creates its conversation second invalidates the
    other's snapshot, and that turn fails CLOSED rather than answering from
    sources that moved underneath it.

    The engine does not serialize turns across a user's conversations -- once
    both exist, concurrent streams all succeed, which
    ``test_concurrent_burst_streams_renew_by_time_not_per_event`` exercises. So
    this covers exactly the creation window, and what must hold is the SHAPE of
    the outcome: a served turn streams to completion, and a refused one is a 409
    carrying ``Retry-After`` and the documented code -- never a 500 the client
    cannot act on.

    Whether a refusal happens is timing-dependent (measured 6 refusals in 20
    runs), so the refusal branch is not guaranteed to execute on every run. The
    invariant asserted here holds either way, and
    ``test_openai_proxy_rebuild_context_error_keeps_its_own_contract`` pins the
    409/503 payload deterministically. What only this test can catch is the
    outcome of a REAL race: any third outcome -- a 500, a hang, or two refusals
    with nothing served -- fails it.
    """
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:

            async def first_turn(conversation_id: str) -> httpx.Response:
                return await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer service-key",
                        "X-Atagia-User-Id": "usr_proxy",
                        "X-Atagia-Platform-Id": "proxy_desktop",
                        "X-Atagia-Conversation-Id": conversation_id,
                    },
                    json={
                        "model": "atagia-memory-proxy",
                        "stream": True,
                        "messages": [{"role": "user", "content": "First turn."}],
                    },
                )

            responses = await asyncio.gather(
                first_turn("cnv_proxy_first_a"),
                first_turn("cnv_proxy_first_b"),
            )

    served = [response for response in responses if response.status_code == 200]
    assert served, [
        (response.status_code, response.text[:200]) for response in responses
    ]
    assert all("data: [DONE]" in response.text for response in served)
    for response in responses:
        if response.status_code == 200:
            continue
        assert response.status_code == 409, response.text
        assert response.headers["Retry-After"] == "1"
        error = response.json()["error"]
        assert error["type"] == "conflict_error"
        assert error["code"] == "selected_transcript_rebuild_in_progress"


@pytest.mark.asyncio
async def test_claim_heartbeat_marks_lost_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = OpenAIProxyService(runtime=None)
    ownership_lost = asyncio.Event()

    async def no_delay(_seconds: float) -> None:
        return None

    async def lose_claim(
        _service: OpenAIProxyService,
        _claim: Any,
    ) -> bool:
        return False

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.asyncio.sleep",
        no_delay,
    )
    monkeypatch.setattr(OpenAIProxyService, "_renew_claim", lose_claim)

    await service._claim_heartbeat(object(), ownership_lost)

    assert ownership_lost.is_set()


@pytest.mark.asyncio
async def test_openai_proxy_streaming_does_not_fabricate_missing_usage(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_usage",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert '"choices": []' not in text
    assert '"usage":' not in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_openai_proxy_rejects_unknown_model_id(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    error = response.json()["error"]
    assert "Unknown model" in error["message"]
    assert error["param"] == "model"
    assert error["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_openai_proxy_rejects_tool_results_without_persisted_parent(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_tools",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Use the lookup tool."},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_previous",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"previous"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_previous",
                            "content": '{"result":"ok"}',
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Lookup memory",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "proxy_tool_parent_conflict"
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_honors_tool_choice_none_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_tools_none",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Do not call tools."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "none",
                },
            )

    assert response.status_code == 200
    assert "tool_calls" not in response.json()["choices"][0]["message"]
    chat_request = [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ][-1]
    assert chat_request.tools == []
    assert chat_request.metadata["openai_tool_choice"] == "none"


@pytest.mark.asyncio
async def test_openai_proxy_streams_tool_calls(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_tools",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Use lookup."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert '"tool_calls"' in text
    assert "call_stream_lookup" in text
    assert '"finish_reason": "tool_calls"' in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_streaming_openai_proxy_preflight_errors_become_503(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_before_first_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_preflight",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["message"] == "LLM service unavailable"
    assert error["code"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_streaming_openai_proxy_non_output_preflight_errors_become_503(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_non_output_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_preflight_metadata",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_streaming_openai_proxy_emits_sse_error_after_partial_failure(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_first_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_stream_error",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as response:
                body = await response.aread()

    assert response.status_code == 200
    text = body.decode("utf-8")
    assert "Proxy " in text
    assert "atagia_upstream_stream_error" in text
    assert "data: [DONE]" not in text


# ---------------------------------------------------------------------------
# A client that goes away must not leave the round-trip it paid for suspended.
#
# Starlette never closes a response body it abandons; `_ClosingStreamingResponse`
# does, and that close is the ONLY teardown the proxy gets. These two tests take
# the two moments a disconnect can land before the SSE loop begins: before the
# body generator has started at all, and while it is parked on the synthetic
# assistant-role chunk it emits first. Both used to leave the provider stream
# suspended until the event loop's async-generator finalizer collected it, in a
# task outside the request and after the turn's accounting had closed.
# ---------------------------------------------------------------------------


class _TeardownTrackingProvider(ProxyProvider):
    """Records the moment the provider round-trip is really torn down."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def stream(self, request: LLMCompletionRequest):
        try:
            async for event in super().stream(request):
                yield event
        finally:
            self.events.append("provider_stream_closed")


class _TeardownTrackingProxyService(OpenAIProxyService):
    """Records claim resolution, so its order against the close is observable."""

    def __init__(self, runtime: Any, events: list[str]) -> None:
        super().__init__(runtime)
        self.events = events

    async def _mark_ambiguous_best_effort(
        self,
        claim: Any,
        exc: BaseException,
    ) -> None:
        await super()._mark_ambiguous_best_effort(claim, exc)
        self.events.append("claim_marked_ambiguous")


async def _abandon_proxy_stream(
    app: Any,
    *,
    events: list[str],
    conversation_id: str,
    chunks_before_disconnect: int,
) -> None:
    """Start a streamed proxy turn and drop it the way a disconnect does."""
    service = _TeardownTrackingProxyService(app.state.runtime, events)
    body = await service.stream(
        OpenAIChatCompletionRequest.model_validate(
            {
                "model": "atagia-memory-proxy",
                "stream": True,
                "messages": [{"role": "user", "content": "Stream with memory."}],
            }
        ),
        claimed_user_id="usr_proxy",
        conversation_id_header=conversation_id,
        platform_id_header="proxy_desktop",
    )
    pulled = 0
    try:
        while pulled < chunks_before_disconnect:
            await body.__anext__()
            pulled += 1
    finally:
        # Exactly what `_ClosingStreamingResponse.__call__` does in its finally.
        await body.aclose()


async def _proxy_turn_states(app: Any, conversation_id: str) -> list[str]:
    connection = await app.state.runtime.open_connection()
    try:
        cursor = await connection.execute(
            "SELECT state FROM proxy_turn_runs WHERE conversation_id = ?",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    finally:
        await connection.close()
    return [str(row[0]) for row in rows]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks_before_disconnect", [0, 1])
async def test_an_abandoned_proxy_stream_closes_its_round_trip_inside_the_request(
    tmp_path: Path,
    chunks_before_disconnect: int,
) -> None:
    """Whether or not the body ever ran, the teardown happens in one pass here.

    ``chunks_before_disconnect=0`` is the wider window: an async generator that
    was never iterated runs no code when it is closed, so nothing inside it can
    own anything. ``1`` is the narrow one: the body is parked on the role chunk,
    which is the first byte written after ``http.response.start`` and therefore
    exactly where a client that is already gone lands.
    """
    conversation_id = f"cnv_proxy_abandon_{chunks_before_disconnect}"
    app = create_app(_settings(tmp_path))
    events: list[str] = []
    provider = _TeardownTrackingProvider(events)
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            llm_run_guard=LLMRunGuard(LLMRunGuardConfig()),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await _precreate_proxy_conversations(
                client,
                user_id="usr_proxy",
                conversation_ids=(conversation_id,),
            )
        # The turn that created the conversation is non-streaming, so the only
        # provider stream in this test is the one abandoned below.
        assert events == []

        await _abandon_proxy_stream(
            app,
            events=events,
            conversation_id=conversation_id,
            chunks_before_disconnect=chunks_before_disconnect,
        )

        # The round-trip is closed FIRST and the claim resolved after, both
        # before `aclose()` returns -- not deferred to a finalizer.
        #
        # The provider close appears EXACTLY ONCE either way. The claim
        # resolution appears twice when the body had started, because the body
        # resolves the claim on its way out and `_ProxyStreamBody.aclose` then
        # runs the abandon path unconditionally -- it cannot tell a body that
        # finished its teardown from one that was cancelled mid-teardown, so it
        # always tries, and the second attempt is a fenced no-op. See
        # `_ProxyStreamBody` for why that question has no answer from out there.
        assert events == (
            ["provider_stream_closed", "claim_marked_ambiguous"]
            if chunks_before_disconnect == 0
            else [
                "provider_stream_closed",
                "claim_marked_ambiguous",
                "claim_marked_ambiguous",
            ]
        )
        snapshot = app.state.runtime.llm_client.llm_run_guard_snapshot()
        assert snapshot is not None
        assert snapshot["cancelled_calls"] == 1
        assert await _proxy_turn_states(app, conversation_id) == [
            "completed",
            "ambiguous_exposed",
        ]


class _CloseRaisingProvider(_TeardownTrackingProvider):
    """A provider whose stream raises while the abandon path is closing it."""

    async def stream(self, request: LLMCompletionRequest):
        try:
            async for event in ProxyProvider.stream(self, request):
                yield event
        finally:
            self.events.append("provider_stream_closed")
            raise RuntimeError("provider close exploded")


@pytest.mark.asyncio
async def test_a_provider_that_raises_on_close_does_not_escape_the_abandon_path(
    tmp_path: Path,
) -> None:
    """A noisy provider teardown must not cost the turn its claim resolution.

    ``chunks_before_disconnect=0`` is the shape where the abandon path itself
    closes the provider round-trip, so a close that raises lands squarely between
    the two steps. Unhandled, it propagated out of
    ``_ClosingStreamingResponse.__call__``'s ``finally`` into the ASGI task error
    log AND took the claim resolution queued behind it -- the claim being the
    part that is durable and the log line the part that is not.
    """
    conversation_id = "cnv_proxy_close_raises"
    app = create_app(_settings(tmp_path))
    events: list[str] = []
    provider = _CloseRaisingProvider(events)
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            llm_run_guard=LLMRunGuard(LLMRunGuardConfig()),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await _precreate_proxy_conversations(
                client,
                user_id="usr_proxy",
                conversation_ids=(conversation_id,),
            )
        assert events == []

        # No `pytest.raises`: nothing may reach the caller of `aclose`.
        await _abandon_proxy_stream(
            app,
            events=events,
            conversation_id=conversation_id,
            chunks_before_disconnect=0,
        )

        # The resolution runs AFTER the raising close, which is the point.
        assert events == ["provider_stream_closed", "claim_marked_ambiguous"]
        snapshot = app.state.runtime.llm_client.llm_run_guard_snapshot()
        assert snapshot is not None
        assert snapshot["cancelled_calls"] == 1
        assert await _proxy_turn_states(app, conversation_id) == [
            "completed",
            "ambiguous_exposed",
        ]


class _GatedStreamProvider(ProxyProvider):
    """A provider whose stream parks between events, so a disconnect can be aimed.

    The park is what makes the two cancellation SHAPES reachable on demand. The
    first event is consumed by the setup coroutine's preflight, so the body sees
    it via the prepending wrapper; parking before the second one leaves the body
    suspended inside ``__anext__`` awaiting the provider, which is where a real
    streamed turn spends nearly all of its wall time.
    """

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.parked = asyncio.Event()
        self.resume = asyncio.Event()

    async def stream(self, request: LLMCompletionRequest):
        self.requests.append(request)
        try:
            yield LLMStreamEvent(type="text", content="Proxy ")
            self.parked.set()
            await self.resume.wait()
            yield LLMStreamEvent(type="text", content="stream.")
            yield LLMStreamEvent(type="done", payload={})
        finally:
            self.events.append("provider_stream_closed")


async def _disconnect_mid_stream(
    app: Any,
    *,
    conversation_id: str,
    provider: _GatedStreamProvider,
    park_in_send_at: int | None,
) -> int:
    """Run a streamed turn over raw ASGI and cut it off with ``http.disconnect``.

    Deliberately NOT ``body.aclose()``. A manual close is a cooperative teardown
    of a suspended generator and exercises none of this: real cancellation is
    delivered by Starlette, which -- for any scope advertising an ASGI
    ``spec_version`` below 2.4, which is what uvicorn sends -- races
    ``stream_response`` against ``listen_for_disconnect`` in a task group and
    cancels the former when the latter returns. Where the response task happens
    to be parked at that moment decides whether the body ends up SUSPENDED or
    TERMINATED, and those two lead to opposite durable outcomes.

    ``park_in_send_at`` holds the ASGI ``send`` on the given body message, which
    parks the body at a ``yield``. Leaving it ``None`` lets the body run on until
    it parks inside ``__anext__`` awaiting ``provider``. Returns the number of
    body messages the response got out before the cut.
    """
    payload = json.dumps(
        {
            "model": "atagia-memory-proxy",
            "stream": True,
            "messages": [{"role": "user", "content": "Stream with memory."}],
        }
    ).encode()
    scope = {
        "type": "http",
        # Below 2.4 on purpose: this is the branch uvicorn drives, and the only
        # one that cancels the response task on disconnect.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "client": ("127.0.0.1", 54321),
        "server": ("testserver", 80),
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer service-key"),
            (b"x-atagia-user-id", b"usr_proxy"),
            (b"x-atagia-platform-id", b"proxy_desktop"),
            (b"x-atagia-conversation-id", conversation_id.encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
    }

    disconnected = asyncio.Event()
    parked_in_send = asyncio.Event()
    request_delivered = False
    body_messages = 0

    async def receive() -> dict[str, Any]:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal body_messages
        if message["type"] != "http.response.body":
            return
        body_messages += 1
        if park_in_send_at is not None and body_messages == park_in_send_at:
            parked_in_send.set()
            # Never returns. The cancellation arrives here, which leaves the
            # body suspended at the yield that produced this chunk.
            await asyncio.Event().wait()

    task = asyncio.create_task(app(scope, receive, send))
    waiter = parked_in_send if park_in_send_at is not None else provider.parked
    await asyncio.wait_for(waiter.wait(), timeout=10)
    disconnected.set()
    await asyncio.wait_for(task, timeout=10)
    return body_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "park_in_send_at", "expected_body_messages"),
    [
        ("cancel_at_yield", 1, 1),
        ("cancel_inside_anext", None, 2),
    ],
)
async def test_a_disconnected_proxy_stream_resolves_its_claim_in_both_cancel_shapes(
    tmp_path: Path,
    case: str,
    park_in_send_at: int | None,
    expected_body_messages: int,
) -> None:
    """Neither cancellation shape may leave the turn claiming to be emitting.

    ``cancel_at_yield`` is the survivable one: the body is parked at a ``yield``
    when the cancel lands, so it stays SUSPENDED and the route's later close can
    revive it to run its own teardown.

    ``cancel_inside_anext`` is the one that used to escape. The cancel lands
    while the body's frame is running, which TERMINATES it -- and the teardown it
    attempts on the way out runs inside the cancelled scope, where every await is
    re-cancelled, so the provider close and the claim resolution are both entered
    and interrupted. From the route's ``finally`` the wreckage is
    indistinguishable from an exhausted generator, which is why the abandon path
    can no longer be conditional on it. This is also the WIDE case: it needs
    nothing but a disconnect during an inter-token gap.

    Both must end ``ambiguous_exposed``: emission was marked started before the
    body existed, so the turn cannot prove nothing reached the client. Leaving
    the row at ``emission_started`` is permanent -- the 30s lease expires but
    nothing reaps it -- and it makes a host using the same-message-ID
    idempotency contract retry into ``request_in_progress`` instead of
    ``stream_retry_requires_new_ids``.
    """
    conversation_id = f"cnv_proxy_disconnect_{case}"
    app = create_app(_settings(tmp_path))
    events: list[str] = []
    provider = _GatedStreamProvider(events)
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            llm_run_guard=LLMRunGuard(LLMRunGuardConfig()),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await _precreate_proxy_conversations(
                client,
                user_id="usr_proxy",
                conversation_ids=(conversation_id,),
            )
        # The conversation-opening turn is non-streaming, so the only provider
        # stream in this test is the one cut off below.
        assert events == []

        body_messages = await _disconnect_mid_stream(
            app,
            conversation_id=conversation_id,
            provider=provider,
            park_in_send_at=park_in_send_at,
        )

        assert body_messages == expected_body_messages
        # Torn down inside the request, not left to a finalizer in another task.
        assert events.count("provider_stream_closed") == 1
        snapshot = app.state.runtime.llm_client.llm_run_guard_snapshot()
        assert snapshot is not None
        assert snapshot["cancelled_calls"] == 1
        assert await _proxy_turn_states(app, conversation_id) == [
            "completed",
            "ambiguous_exposed",
        ]


@pytest.mark.asyncio
async def test_a_completed_proxy_stream_is_unaffected_by_the_abandon_path(
    tmp_path: Path,
) -> None:
    """The unconditional abandon must be invisible on the success path.

    It runs on every streamed turn, after the last byte, so it has to match
    nothing here: the provider stream is exhausted and the claim reached
    ``completed``, which the fenced ``emission_started`` update cannot touch.
    """
    conversation_id = "cnv_proxy_stream_completed"
    app = create_app(_settings(tmp_path))
    events: list[str] = []
    provider = _TeardownTrackingProvider(events)
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
            llm_run_guard=LLMRunGuard(LLMRunGuardConfig()),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await _precreate_proxy_conversations(
                client,
                user_id="usr_proxy",
                conversation_ids=(conversation_id,),
            )
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": conversation_id,
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Stream with memory."}],
                },
            ) as response:
                text = (await response.aread()).decode("utf-8")

        assert response.status_code == 200
        # Natural exhaustion, in order, before the terminal commit's [DONE].
        assert 0 < text.index("Proxy ") < text.index("stream.")
        assert text.rstrip().endswith("data: [DONE]")
        assert events == ["provider_stream_closed"]
        snapshot = app.state.runtime.llm_client.llm_run_guard_snapshot()
        assert snapshot is not None
        assert snapshot["cancelled_calls"] == 0
        assert await _proxy_turn_states(app, conversation_id) == [
            "completed",
            "completed",
        ]


@pytest.mark.asyncio
async def test_openai_proxy_terminal_persistence_failure_is_not_fail_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_finalize(*args, **kwargs):
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.finalize_proxy_turn",
        fail_finalize,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_fail_open",
                    "X-Atagia-Message-Id": "msg_failed_terminal_request",
                    "X-Atagia-Response-Message-Id": "msg_failed_terminal_response",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        connection = await app.state.runtime.open_connection()
        try:
            assistant = await (
                await connection.execute(
                    "SELECT id FROM messages WHERE id = ?",
                    ("msg_failed_terminal_response",),
                )
            ).fetchone()
            run = await (
                await connection.execute(
                    "SELECT state FROM proxy_turn_runs WHERE request_message_id = ?",
                    ("msg_failed_terminal_request",),
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 500
    assert assistant is None
    assert run is not None and run["state"] == "generating"


@pytest.mark.asyncio
async def test_openai_proxy_conversation_id_collision_returns_404(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_a",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "shared_conversation_id",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            second = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_b",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "shared_conversation_id",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert first.status_code == 200
    assert second.status_code == 404
    assert second.json()["error"]["message"] == "Conversation not found for user"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            TranscriptRebuildInProgressError("Memory sources changed"),
            409,
            "selected_transcript_rebuild_in_progress",
        ),
        (
            TranscriptRebuildRemediationRequiredError("Rebuild needs remediation"),
            503,
            "selected_transcript_remediation_required",
        ),
    ],
)
@pytest.mark.asyncio
async def test_openai_proxy_rebuild_context_error_keeps_its_own_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    """A fail-closed source error keeps its contract wherever it is raised.

    Both errors already produce these responses when raised while preparing the
    turn. Raised from ``get_context`` they used to fall through to the generic
    branch and surface as a 500 ``memory_context_internal_error`` instead, so
    the same condition reported a retryable conflict or a server fault depending
    only on which stage noticed it.
    """

    async def fail_context(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_rebuild_contract",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    # The turn is refused, not answered without memory.
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_unexpected_context_runtime_error_is_internal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_context(*args, **kwargs):
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_context_fail_open",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "memory_context_internal_error"
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_unexpected_context_value_error_is_internal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_context(*args, **kwargs):
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": "cnv_proxy_value_error_fail_open",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "memory_context_internal_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_proxy_context_integrity_error_fails_closed_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    async def fail_context(*args, **kwargs):
        raise sqlite3.IntegrityError("deterministic context invariant failure")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    conversation_id = f"cnv_proxy_integrity_{stream}"
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": conversation_id,
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        connection = await app.state.runtime.open_connection()
        try:
            assistant_count = int(
                (
                    await (
                        await connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM messages
                            WHERE conversation_id = ? AND role = 'assistant'
                            """,
                            (conversation_id,),
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "memory_context_internal_error"
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]
    assert assistant_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_proxy_busy_context_store_remains_fail_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    async def fail_context(*args, **kwargs):
        error = sqlite3.OperationalError("database is busy")
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY
        raise error

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        fail_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                    "X-Atagia-Platform-Id": "proxy_desktop",
                    "X-Atagia-Conversation-Id": f"cnv_proxy_busy_{stream}",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert (
        len(
            [
                request
                for request in provider.requests
                if request.metadata.get("purpose") == "chat_reply"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_openai_proxy_workspace_mismatch_is_not_fail_open(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_proxy",
        "X-Atagia-Platform-Id": "proxy_desktop",
    }
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            await client.post(
                "/v1/workspaces",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "workspace_id": "wrk_proxy_a",
                    "name": "Workspace A",
                    "metadata": {},
                },
            )
            await client.post(
                "/v1/workspaces",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "workspace_id": "wrk_proxy_b",
                    "name": "Workspace B",
                    "metadata": {},
                },
            )
            await client.post(
                "/v1/conversations",
                headers=headers,
                json={
                    "user_id": "usr_proxy",
                    "conversation_id": "cnv_proxy_workspace",
                    "assistant_mode_id": "general_qa",
                    "workspace_id": "wrk_proxy_a",
                    "platform_id": "proxy_desktop",
                    "title": None,
                    "metadata": {},
                },
            )
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    **headers,
                    "X-Atagia-Conversation-Id": "cnv_proxy_workspace",
                    "X-Atagia-Workspace-Id": "wrk_proxy_b",
                },
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 409
    assert (
        response.json()["error"]["message"]
        == "Requested workspace does not match the existing conversation workspace"
    )
    assert not [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_openai_proxy_validation_errors_use_openai_shape(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer service-key",
                    "X-Atagia-User-Id": "usr_proxy",
                },
                json={
                    "model": "atagia-memory-proxy",
                },
            )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "validation_error"
    assert error["param"] == "messages"


@pytest.mark.asyncio
async def test_openai_proxy_requires_resolved_user_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer service-key"},
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-User-Id" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_openai_proxy_requires_resolved_user_id(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer service-key"},
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 400
    assert "require X-Atagia-User-Id" in response.json()["error"]["message"]
