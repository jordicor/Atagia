"""API tests for verbatim pin routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, UserRepository
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
    name = "noop-verbatim-pin-api-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM should not be called in verbatim pin API tests: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings should not be called in verbatim pin API tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-verbatim-pin-api.db"),
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
        llm_chat_model="openai/test-model",
        llm_ingest_model="openai/test-model",
        llm_retrieval_model="openai/test-model",
        llm_component_models={"intent_classifier": "openai/test-model"},
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
    )


def test_verbatim_pin_routes_enforce_user_claims_and_crud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NoopProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    app = create_app(_settings(tmp_path))
    headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_1",
    }
    wrong_headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_2",
    }

    with TestClient(app) as client:
        runtime = client.app.state.runtime
        connection = client.portal.call(runtime.open_connection)
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Pinned chat",
            )
        finally:
            client.portal.call(connection.close)

        created = client.post(
            "/v1/verbatim-pins",
            json={
                "user_id": "usr_1",
                "scope": "conversation",
                "target_kind": "message",
                "target_id": "msg_1",
                "conversation_id": "cnv_1",
                "canonical_text": "alpha beta gamma",
                "index_text": "alpha beta gamma",
                "privacy_level": 0,
                "reason": "test pin",
                "created_by": "usr_1",
            },
            headers=headers,
        )
        assert created.status_code == 200
        pin_id = created.json()["id"]
        assert created.json()["status"] == "active"

        forbidden = client.post(
            "/v1/verbatim-pins",
            json={
                "user_id": "usr_1",
                "scope": "conversation",
                "target_kind": "message",
                "target_id": "msg_2",
                "canonical_text": "other text",
                "index_text": "other text",
                "privacy_level": 0,
                "created_by": "usr_1",
            },
            headers=wrong_headers,
        )
        assert forbidden.status_code == 403

        listing = client.get(
            "/v1/verbatim-pins",
            params={"user_id": "usr_1"},
            headers=headers,
        )
        assert listing.status_code == 200
        assert [pin["id"] for pin in listing.json()] == [pin_id]

        wrong_user_listing = client.get(
            "/v1/verbatim-pins",
            params={"user_id": "usr_1"},
            headers=wrong_headers,
        )
        assert wrong_user_listing.status_code == 403

        fetched = client.get(
            f"/v1/verbatim-pins/{pin_id}",
            params={"user_id": "usr_1"},
            headers=headers,
        )
        assert fetched.status_code == 200
        assert fetched.json()["canonical_text"] == "alpha beta gamma"

        updated = client.patch(
            f"/v1/verbatim-pins/{pin_id}",
            params={"user_id": "usr_1"},
            json={"status": "archived", "reason": "done"},
            headers=headers,
        )
        assert updated.status_code == 200
        assert updated.json()["status"] == "archived"
        assert updated.json()["reason"] == "done"

        deleted = client.delete(
            f"/v1/verbatim-pins/{pin_id}",
            params={"user_id": "usr_1"},
            headers=headers,
        )
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "deleted"

        reloaded = client.get(
            "/v1/verbatim-pins",
            params={"user_id": "usr_1", "status": "deleted"},
            headers=headers,
        )
        assert reloaded.status_code == 200
        assert reloaded.json()[0]["deleted_at"] is not None
