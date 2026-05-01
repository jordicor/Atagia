"""Tests for the transport-agnostic Atagia client facade."""

from __future__ import annotations

from pathlib import Path
import json
import re

import httpx
import pytest

from atagia.app import create_app
from atagia.client import HttpAtagiaClient, LocalAtagiaClient, connect_atagia
from atagia.core.config import Settings
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class ClientProvider(LLMProvider):
    """Purpose-aware provider stub for local and HTTP client smoke tests."""

    name = "client-tests"

    def __init__(self) -> None:
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
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "scores": [
                            {"memory_id": memory_id, "llm_applicability": 0.5}
                            for memory_id in memory_ids
                        ],
                    }
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
        if purpose == "consent_confirmation_intent":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": "confirm"}),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Check the retry guard first.",
            )
        if purpose == "memory_extraction":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "evidences": [],
                        "beliefs": [],
                        "contract_signals": [],
                        "state_updates": [],
                        "nothing_durable": True,
                    }
                ),
            )
        if purpose == "contract_projection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"signals": [], "nothing_durable": True}),
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
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in client tests: {request.model}")


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: ClientProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


def _settings(tmp_path: Path, *, workers_enabled: bool = False) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-client-api.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="client-reply-model",
        llm_forced_global_model="openai/client-reply-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=workers_enabled,
        debug=False,
        allow_insecure_http=False,
        small_corpus_token_threshold_ratio=0.0,
    )


@pytest.mark.asyncio
async def test_connect_atagia_local_sidecar_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ClientProvider()
    _install_stub_client(monkeypatch, provider)
    client = await connect_atagia(
        transport="local",
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    try:
        assert isinstance(client, LocalAtagiaClient)
        await client.create_user("usr_1")
        await client.create_workspace("usr_1", "wrk_1", "Workspace")
        conversation_id = await client.create_conversation(
            "usr_1",
            "cnv_1",
            workspace_id="wrk_1",
            assistant_mode_id="coding_debug",
        )

        context = await client.get_context(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Please remember the retry guard.",
            mode="coding_debug",
        )
        await client.add_response(
            user_id="usr_1",
            conversation_id=conversation_id,
            text="I will keep the retry guard in mind.",
        )
        await client.ingest_message(
            user_id="usr_1",
            conversation_id=conversation_id,
            role="user",
            text="Historical note: retry loops need a guard.",
            mode="coding_debug",
        )
        chat_result = await client.chat(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Why is the retry loop failing?",
            mode="coding_debug",
        )

        assert context.system_prompt
        assert chat_result.response_text == "Check the retry guard first."
        assert await client.flush(timeout_seconds=0.1) is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_connect_atagia_http_sidecar_round_trip(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    provider = ClientProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            client = await connect_atagia(
                transport="http",
                base_url="http://testserver",
                api_key="service-key",
                http_client=http_client,
            )

            assert isinstance(client, HttpAtagiaClient)
            await client.create_user("usr_1")
            await client.create_workspace("usr_1", "wrk_1", "Workspace")
            conversation_id = await client.create_conversation(
                "usr_1",
                "cnv_1",
                workspace_id="wrk_1",
                assistant_mode_id="coding_debug",
            )
            context = await client.get_context(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Please remember the retry guard.",
                mode="coding_debug",
            )
            await client.add_response(
                user_id="usr_1",
                conversation_id=conversation_id,
                text="I will keep the retry guard in mind.",
            )
            await client.ingest_message(
                user_id="usr_1",
                conversation_id=conversation_id,
                role="user",
                text="Historical note: retry loops need a guard.",
                mode="coding_debug",
            )
            chat_result = await client.chat(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Why is the retry loop failing?",
            )

            assert context.system_prompt
            assert chat_result.response_text == "Check the retry guard first."
            assert await client.flush(timeout_seconds=0.1) is False


@pytest.mark.asyncio
async def test_http_sidecar_routes_require_claimed_user_header(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            response = await http_client.post(
                "/v1/conversations/cnv_1/context",
                headers={"Authorization": "Bearer service-key"},
                json={
                    "user_id": "usr_1",
                    "message_text": "What did we decide?",
                    "assistant_mode_id": "coding_debug",
                },
            )

            assert response.status_code == 401
            assert response.json()["detail"] == "X-Atagia-User-Id header is required in service mode"


@pytest.mark.asyncio
async def test_connect_atagia_auto_uses_http_when_base_url_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BASE_URL", "http://example.test")
    monkeypatch.setenv("ATAGIA_SERVICE_API_KEY", "service-key")

    client = await connect_atagia(transport="auto")

    assert isinstance(client, HttpAtagiaClient)
    await client.close()
