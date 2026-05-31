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
from atagia.transport_ids import decode_path_id, encode_path_id

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


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
        if purpose == "need_detection_unknown_only_contract_review":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_exact_value_lookup": False,
                        "exact_facets": [],
                        "must_keep_terms": [],
                        "quoted_phrases": [],
                    }
                ),
            )
        if purpose == "need_detection_multi_facet_exact_review":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "has_multiple_obligations": False,
                        "sub_queries": [],
                    }
                ),
            )
        if purpose == "applicability_scoring":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "scores": [
                            {"score_key": score_key, "llm_applicability": 0.5}
                            for _memory_id, score_key in candidate_keys
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
        raise AssertionError(
            f"Embeddings are not used in client tests: {request.model}"
        )


def _install_stub_client(
    monkeypatch: pytest.MonkeyPatch, provider: ClientProvider
) -> None:
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
        lifecycle_lazy_enabled=False,
        debug=False,
        allow_insecure_http=False,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _conversation_and_message_embodiments(
    runtime: object,
    *,
    user_id: str,
    conversation_id: str,
) -> tuple[str | None, list[str | None]]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT active_embodiment_id
            FROM conversations
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, conversation_id),
        )
        conversation = await cursor.fetchone()
        if conversation is None:
            raise AssertionError("Conversation should exist")
        cursor = await connection.execute(
            """
            SELECT active_embodiment_id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC, id ASC
            """,
            (conversation_id,),
        )
        messages = await cursor.fetchall()
        return conversation["active_embodiment_id"], [
            message["active_embodiment_id"] for message in messages
        ]
    finally:
        await connection.close()


async def _conversation_and_message_realms(
    runtime: object,
    *,
    user_id: str,
    conversation_id: str,
) -> tuple[str | None, list[str | None]]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT active_realm_id
            FROM conversations
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, conversation_id),
        )
        conversation = await cursor.fetchone()
        if conversation is None:
            raise AssertionError("Conversation should exist")
        cursor = await connection.execute(
            """
            SELECT active_realm_id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq ASC, id ASC
            """,
            (conversation_id,),
        )
        messages = await cursor.fetchall()
        return conversation["active_realm_id"], [
            message["active_realm_id"] for message in messages
        ]
    finally:
        await connection.close()


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
            workspace_id="wrk_1",
        )
        await client.add_response(
            user_id="usr_1",
            conversation_id=conversation_id,
            text="I will keep the retry guard in mind.",
            character_id="wrk_1",
        )
        await client.ingest_message(
            user_id="usr_1",
            conversation_id=conversation_id,
            role="user",
            text="Historical note: retry loops need a guard.",
            mode="coding_debug",
            workspace_id="wrk_1",
        )
        chat_result = await client.chat(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Why is the retry loop failing?",
            mode="coding_debug",
            workspace_id="wrk_1",
            privacy_enforcement="off",
            authenticated_user_privilege_level="atagia_master",
            authenticated_user_is_atagia_master=True,
        )

        assert context.system_prompt
        assert chat_result.response_text == "Check the retry guard first."
        chat_request = next(
            request
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        )
        assert chat_request.metadata["privacy_enforcement"] == "off"
        assert chat_request.metadata["effective_privacy_enforcement"] == "off"
        assert chat_request.metadata["authenticated_privilege_level"] == "atagia_master"
        assert chat_request.metadata["authenticated_atagia_master"] is True
        assert await client.flush(timeout_seconds=0.1) is True
        status = await client.get_processing_status("usr_1", conversation_id)
        assert status.workers_enabled is True
        assert status.processing is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_local_client_propagates_embodiment_id_to_engine_rows(
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
        conversation_id = await client.create_conversation(
            "usr_1",
            "cnv_body",
            assistant_mode_id="coding_debug",
            platform_id="client_local",
            embodiment_id="body_local",
        )
        await client.get_context(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="client_local",
            embodiment_id="body_local",
        )
        await client.add_response(
            user_id="usr_1",
            conversation_id=conversation_id,
            text="I will keep the retry guard in mind.",
            platform_id="client_local",
            embodiment_id="body_local",
        )
        await client.ingest_message(
            user_id="usr_1",
            conversation_id=conversation_id,
            role="user",
            text="Historical note: retry loops need a guard.",
            mode="coding_debug",
            platform_id="client_local",
            embodiment_id="body_local",
        )
        await client.chat(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Why is the retry loop failing?",
            mode="coding_debug",
            platform_id="client_local",
            embodiment_id="body_local",
        )

        runtime = client._engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        (
            conversation_embodiment,
            message_embodiments,
        ) = await _conversation_and_message_embodiments(
            runtime,
            user_id="usr_1",
            conversation_id=conversation_id,
        )

        assert conversation_embodiment == "body_local"
        assert len(message_embodiments) >= 5
        assert set(message_embodiments) == {"body_local"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_local_client_propagates_realm_id_to_engine_rows(
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
        conversation_id = await client.create_conversation(
            "usr_1",
            "cnv_realm",
            assistant_mode_id="coding_debug",
            platform_id="client_local",
            realm_id="realm_local",
        )
        await client.get_context(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Please remember the retry guard.",
            mode="coding_debug",
            platform_id="client_local",
            realm_id="realm_local",
        )
        await client.add_response(
            user_id="usr_1",
            conversation_id=conversation_id,
            text="I will keep the retry guard in mind.",
            platform_id="client_local",
            realm_id="realm_local",
        )
        await client.ingest_message(
            user_id="usr_1",
            conversation_id=conversation_id,
            role="user",
            text="Historical note: retry loops need a guard.",
            mode="coding_debug",
            platform_id="client_local",
            realm_id="realm_local",
        )
        await client.chat(
            user_id="usr_1",
            conversation_id=conversation_id,
            message="Why is the retry loop failing?",
            mode="coding_debug",
            platform_id="client_local",
            realm_id="realm_local",
        )

        runtime = client._engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        conversation_realm, message_realms = await _conversation_and_message_realms(
            runtime,
            user_id="usr_1",
            conversation_id=conversation_id,
        )

        assert conversation_realm == "realm_local"
        assert len(message_realms) >= 5
        assert set(message_realms) == {"realm_local"}
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
                platform_id="client_http",
            )
            context = await client.get_context(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Please remember the retry guard.",
                mode="coding_debug",
                workspace_id="wrk_1",
                platform_id="client_http",
            )
            await client.add_response(
                user_id="usr_1",
                conversation_id=conversation_id,
                text="I will keep the retry guard in mind.",
                platform_id="client_http",
                character_id="wrk_1",
            )
            await client.ingest_message(
                user_id="usr_1",
                conversation_id=conversation_id,
                role="user",
                text="Historical note: retry loops need a guard.",
                mode="coding_debug",
                workspace_id="wrk_1",
                platform_id="client_http",
            )
            chat_result = await client.chat(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Why is the retry loop failing?",
                workspace_id="wrk_1",
                platform_id="client_http",
                privacy_enforcement="off",
                authenticated_user_privilege_level="atagia_master",
                authenticated_user_is_atagia_master=True,
            )

            assert context.system_prompt
            assert chat_result.response_text == "Check the retry guard first."
            chat_request = next(
                request
                for request in provider.requests
                if request.metadata.get("purpose") == "chat_reply"
            )
            assert chat_request.metadata["privacy_enforcement"] == "off"
            assert chat_request.metadata["effective_privacy_enforcement"] == "off"
            assert (
                chat_request.metadata["authenticated_privilege_level"]
                == "atagia_master"
            )
            assert chat_request.metadata["authenticated_atagia_master"] is True
            assert await client.flush(timeout_seconds=0.1) is False
            status = await client.get_processing_status("usr_1", conversation_id)
            assert status.workers_enabled is False
            assert status.status == "blocked"
            assert status.pending_jobs > 0


@pytest.mark.asyncio
async def test_http_client_propagates_embodiment_id_to_api_rows(tmp_path: Path) -> None:
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
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                http_client=http_client,
            )
            await client.create_user("usr_1")
            conversation_id = await client.create_conversation(
                "usr_1",
                "cnv_body",
                assistant_mode_id="coding_debug",
                platform_id="client_http",
                embodiment_id="body_http",
            )
            await client.get_context(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Please remember the retry guard.",
                mode="coding_debug",
                platform_id="client_http",
                embodiment_id="body_http",
            )
            await client.add_response(
                user_id="usr_1",
                conversation_id=conversation_id,
                text="I will keep the retry guard in mind.",
                platform_id="client_http",
                embodiment_id="body_http",
            )
            await client.ingest_message(
                user_id="usr_1",
                conversation_id=conversation_id,
                role="user",
                text="Historical note: retry loops need a guard.",
                mode="coding_debug",
                platform_id="client_http",
                embodiment_id="body_http",
            )
            await client.chat(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Why is the retry loop failing?",
                mode="coding_debug",
                platform_id="client_http",
                embodiment_id="body_http",
            )

        (
            conversation_embodiment,
            message_embodiments,
        ) = await _conversation_and_message_embodiments(
            app.state.runtime,
            user_id="usr_1",
            conversation_id=conversation_id,
        )

    assert conversation_embodiment == "body_http"
    assert len(message_embodiments) >= 5
    assert set(message_embodiments) == {"body_http"}


@pytest.mark.asyncio
async def test_http_client_propagates_realm_id_to_api_rows(tmp_path: Path) -> None:
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
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                http_client=http_client,
            )
            await client.create_user("usr_1")
            conversation_id = await client.create_conversation(
                "usr_1",
                "cnv_realm",
                assistant_mode_id="coding_debug",
                platform_id="client_http",
                realm_id="realm_http",
            )
            await client.get_context(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Please remember the retry guard.",
                mode="coding_debug",
                platform_id="client_http",
                realm_id="realm_http",
            )
            await client.add_response(
                user_id="usr_1",
                conversation_id=conversation_id,
                text="I will keep the retry guard in mind.",
                platform_id="client_http",
                realm_id="realm_http",
            )
            await client.ingest_message(
                user_id="usr_1",
                conversation_id=conversation_id,
                role="user",
                text="Historical note: retry loops need a guard.",
                mode="coding_debug",
                platform_id="client_http",
                realm_id="realm_http",
            )
            await client.chat(
                user_id="usr_1",
                conversation_id=conversation_id,
                message="Why is the retry loop failing?",
                mode="coding_debug",
                platform_id="client_http",
                realm_id="realm_http",
            )

        conversation_realm, message_realms = await _conversation_and_message_realms(
            app.state.runtime,
            user_id="usr_1",
            conversation_id=conversation_id,
        )

    assert conversation_realm == "realm_http"
    assert len(message_realms) >= 5
    assert set(message_realms) == {"realm_http"}


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_conversation_id", ["chat/with spaces?#", ".", ".."])
async def test_http_client_encodes_unsafe_conversation_ids_for_asgi_routes(
    tmp_path: Path,
    raw_conversation_id: str,
) -> None:
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
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                http_client=http_client,
            )
            await client.create_user("usr_1")
            await client.create_workspace("usr_1", "wrk_1", "Workspace")
            created_id = await client.create_conversation(
                "usr_1",
                raw_conversation_id,
                workspace_id="wrk_1",
                assistant_mode_id="coding_debug",
                platform_id="client_http",
            )
            context = await client.get_context(
                user_id="usr_1",
                conversation_id=raw_conversation_id,
                message="Please remember the retry guard.",
                mode="coding_debug",
                workspace_id="wrk_1",
                platform_id="client_http",
            )
            await client.add_response(
                user_id="usr_1",
                conversation_id=raw_conversation_id,
                text="I will keep the retry guard in mind.",
                platform_id="client_http",
                character_id="wrk_1",
            )
            await client.ingest_message(
                user_id="usr_1",
                conversation_id=raw_conversation_id,
                role="user",
                text="Historical note: retry loops need a guard.",
                mode="coding_debug",
                workspace_id="wrk_1",
                platform_id="client_http",
            )
            chat_result = await client.chat(
                user_id="usr_1",
                conversation_id=raw_conversation_id,
                message="Why is the retry loop failing?",
                workspace_id="wrk_1",
                platform_id="client_http",
            )
            status = await client.get_processing_status("usr_1", raw_conversation_id)
        connection = await app.state.runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT id FROM conversations WHERE user_id = ?",
                ("usr_1",),
            )
            stored_ids = {str(row[0]) for row in await cursor.fetchall()}
        finally:
            await connection.close()

    assert created_id == raw_conversation_id
    assert context.system_prompt
    assert chat_result.conversation_id == raw_conversation_id
    assert chat_result.response_text == "Check the retry guard first."
    assert status.status == "blocked"
    assert raw_conversation_id in stored_ids
    assert encode_path_id(raw_conversation_id) not in stored_ids


def test_transport_path_ids_are_reversible_and_dot_safe() -> None:
    for raw_id in ("chat/with spaces?#", ".", "..", "__atagia_b64_existing"):
        encoded = encode_path_id(raw_id)
        assert "/" not in encoded
        assert encoded not in {".", ".."}
        assert decode_path_id(encoded) == raw_id


@pytest.mark.asyncio
async def test_http_client_encodes_unsafe_user_id_for_user_status_route(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    raw_user_id = "usr/with spaces?#"
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                http_client=http_client,
            )
            await client.create_user(raw_user_id)
            status = await client.get_processing_status(raw_user_id)

    assert status.status == "idle"


@pytest.mark.asyncio
async def test_http_client_worker_control_uses_admin_key(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as http_client:
            client = HttpAtagiaClient(
                base_url="http://testserver",
                api_key="service-key",
                admin_api_key="admin-key",
                http_client=http_client,
            )

            default_state = await client.get_worker_control()
            paused_state = await client.set_worker_control(
                "hard_pause",
                reason="restore",
            )

    assert default_state.mode == "active"
    assert paused_state.mode == "hard_pause"
    assert paused_state.worker_claims_allowed is False


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
            assert (
                response.json()["detail"]
                == "X-Atagia-User-Id header is required in service mode"
            )


@pytest.mark.asyncio
async def test_connect_atagia_auto_uses_http_when_base_url_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_BASE_URL", "http://example.test")
    monkeypatch.setenv("ATAGIA_SERVICE_API_KEY", "service-key")

    client = await connect_atagia(transport="auto")

    assert isinstance(client, HttpAtagiaClient)
    await client.close()
