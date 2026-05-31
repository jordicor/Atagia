"""Tests for post-ingest user communication language profiles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository, WorkspaceRepository
from atagia.memory.language_profile import UserCommunicationProfileService
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import ExtractionConversationContext, MemoryScope
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


class LanguageProfileProvider(LLMProvider):
    name = "language-profile-tests"

    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.metadata.get("purpose") != "user_language_profile_update":
            raise AssertionError(f"Unexpected purpose {request.metadata.get('purpose')}")
        if not self.outputs:
            raise AssertionError("No language profile output configured")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.outputs.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in language profile tests")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
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
    )


async def _seed_service(outputs: list[dict[str, object]]):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await WorkspaceRepository(connection, clock).create_workspace("wrk_1", "usr_1", "Workspace")
    await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Chat",
        platform_id="mac",
        active_presence_id="assistant_presence",
    )
    repository = CommunicationProfileRepository(connection, clock)
    provider = LanguageProfileProvider(outputs)
    service = UserCommunicationProfileService(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        profile_repository=repository,
        settings=_settings(),
    )
    return connection, repository, provider, service


def _context(
    *,
    source_message_id: str = "msg_1",
    incognito: bool = False,
    remember_across_chats: bool = True,
) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id=source_message_id,
        workspace_id="wrk_1",
        assistant_mode_id="coding_debug",
        platform_id="mac",
        character_id="wrk_1",
        active_presence_id="assistant_presence",
        source_presence_id="user_presence",
        incognito=incognito,
        remember_across_chats=remember_across_chats,
    )


@pytest.mark.asyncio
async def test_language_profile_records_observed_user_language_without_fluency_claim() -> None:
    connection, _repository, provider, service = await _seed_service(
        [
            {
                "observed_user_languages": [
                    {"language_code": "ES", "confidence": 0.91}
                ],
                "explicit_language_preferences": [],
                "explicit_language_abilities": [],
                "contextual_norms": [],
                "external_content_language_codes": [],
            }
        ]
    )
    try:
        profile = await service.update_from_message(
            message_text="Me va mejor si seguimos en espanol.",
            role="user",
            conversation_context=_context(),
            occurred_at="2026-05-20T12:03:00+00:00",
        )

        assert profile is not None
        assert profile.subject_presence_id == "user_presence"
        assert profile.observed_user_languages[0].language_code == "es"
        assert profile.observed_user_languages[0].message_count == 1
        assert profile.observed_user_languages[0].source_refs[0].source_message_id == "msg_1"
        assert profile.explicit_language_abilities == []
        assert provider.requests[0].metadata["purpose"] == "user_language_profile_update"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_language_profile_merges_preferences_abilities_and_observations() -> None:
    connection, repository, _provider, service = await _seed_service(
        [
            {
                "observed_user_languages": [
                    {"language_code": "es", "confidence": 0.9}
                ],
                "explicit_language_preferences": [
                    {
                        "language_code": "es",
                        "preference_kind": "default_answer_language",
                        "context_label": "ordinary_chat",
                        "confidence": 0.94,
                    }
                ],
                "explicit_language_abilities": [
                    {
                        "language_code": "en",
                        "ability_kind": "understands",
                        "confidence": 0.92,
                    }
                ],
                "contextual_norms": [
                    {
                        "language_code": "en",
                        "norm_kind": "comfortable_for_terms_or_code",
                        "context_label": "technical_terms",
                        "confidence": 0.84,
                    }
                ],
                "external_content_language_codes": [],
            },
            {
                "observed_user_languages": [
                    {"language_code": "es", "confidence": 0.88}
                ],
                "explicit_language_preferences": [],
                "explicit_language_abilities": [],
                "contextual_norms": [],
                "external_content_language_codes": [],
            },
        ]
    )
    try:
        await service.update_from_message(
            message_text="Prefiero respuestas en espanol, pero entiendo ingles.",
            role="user",
            conversation_context=_context(source_message_id="msg_1"),
            occurred_at="2026-05-20T12:03:00+00:00",
        )
        await service.update_from_message(
            message_text="Gracias, sigamos asi.",
            role="user",
            conversation_context=_context(source_message_id="msg_2"),
            occurred_at="2026-05-20T12:05:00+00:00",
        )

        loaded = await repository.get_exact_user_language_profile(
            _context(),
            scope=MemoryScope.CHARACTER,
        )
        assert loaded is not None
        assert loaded.observed_user_languages[0].message_count == 2
        assert loaded.observed_user_languages[0].last_seen_at == "2026-05-20T12:05:00+00:00"
        assert len(loaded.observed_user_languages[0].source_refs) == 2
        assert loaded.explicit_language_preferences[0].context_label == "ordinary_chat"
        assert loaded.explicit_language_abilities[0].language_code == "en"
        assert loaded.contextual_norms[0].context_label == "technical_terms"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_language_profile_does_not_merge_from_stale_profile_rows() -> None:
    connection, repository, _provider, service = await _seed_service(
        [
            {
                "observed_user_languages": [
                    {"language_code": "es", "confidence": 0.9}
                ],
                "explicit_language_preferences": [],
                "explicit_language_abilities": [],
                "contextual_norms": [],
                "external_content_language_codes": [],
            },
            {
                "observed_user_languages": [
                    {"language_code": "ca", "confidence": 0.88}
                ],
                "explicit_language_preferences": [],
                "explicit_language_abilities": [],
                "contextual_norms": [],
                "external_content_language_codes": [],
            },
        ]
    )
    try:
        await service.update_from_message(
            message_text="Me va mejor seguir en espanol.",
            role="user",
            conversation_context=_context(source_message_id="msg_old"),
            occurred_at="2026-05-20T12:03:00+00:00",
        )
        changed = await repository.mark_stale_for_source_message(
            user_id="usr_1",
            source_message_id="msg_old",
            reason="source_message_deleted",
        )
        assert changed == 1
        assert await repository.get_exact_user_language_profile(
            _context(),
            scope=MemoryScope.CHARACTER,
        ) is None

        await service.update_from_message(
            message_text="Ara em va be continuar en catala.",
            role="user",
            conversation_context=_context(source_message_id="msg_new"),
            occurred_at="2026-05-20T12:05:00+00:00",
        )

        loaded = await repository.get_exact_user_language_profile(
            _context(),
            scope=MemoryScope.CHARACTER,
        )
        assert loaded is not None
        assert [row.language_code for row in loaded.observed_user_languages] == ["ca"]
        assert loaded.observed_user_languages[0].message_count == 1
        assert loaded.observed_user_languages[0].source_refs[0].source_message_id == "msg_new"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_language_profile_drops_invalid_language_codes_without_losing_valid_rows() -> None:
    connection, _repository, _provider, service = await _seed_service(
        [
            {
                "observed_user_languages": [
                    {"language_code": "jp", "confidence": 0.95},
                    {"language_code": "ca", "confidence": 0.88},
                ],
                "explicit_language_preferences": [
                    {
                        "language_code": "spanish",
                        "preference_kind": "default_answer_language",
                        "context_label": "ordinary_chat",
                        "confidence": 0.9,
                    }
                ],
                "explicit_language_abilities": [
                    {
                        "language_code": "EN",
                        "ability_kind": "understands",
                        "confidence": 0.84,
                    }
                ],
                "contextual_norms": [],
                "external_content_language_codes": ["zz", "zh"],
            }
        ]
    )
    try:
        profile = await service.update_from_message(
            message_text="Puc seguir en catala i entenc angles.",
            role="user",
            conversation_context=_context(),
            occurred_at="2026-05-20T12:03:00+00:00",
        )

        assert profile is not None
        assert [row.language_code for row in profile.observed_user_languages] == ["ca"]
        assert profile.explicit_language_preferences == []
        assert profile.explicit_language_abilities[0].language_code == "en"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_external_content_language_does_not_become_user_language() -> None:
    connection, _repository, _provider, service = await _seed_service(
        [
            {
                "observed_user_languages": [
                    {"language_code": "es", "confidence": 0.9}
                ],
                "explicit_language_preferences": [],
                "explicit_language_abilities": [],
                "contextual_norms": [],
                "external_content_language_codes": ["zh"],
            }
        ]
    )
    try:
        profile = await service.update_from_message(
            message_text="Traduceme este manual chino, por favor.",
            role="user",
            conversation_context=_context(),
            occurred_at="2026-05-20T12:03:00+00:00",
        )
        assert profile is not None
        assert [row.language_code for row in profile.observed_user_languages] == ["es"]
        assert profile.explicit_language_abilities == []
        assert profile.external_content_languages_excluded is True
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_language_profile_skips_non_user_and_incognito_messages() -> None:
    connection, repository, provider, service = await _seed_service([])
    try:
        assert await service.update_from_message(
            message_text="I can answer in English.",
            role="assistant",
            conversation_context=_context(),
        ) is None
        assert await service.update_from_message(
            message_text="Esto no deberia guardarse.",
            role="user",
            conversation_context=_context(incognito=True),
        ) is None
        assert provider.requests == []
        assert await repository.get_user_language_profile_for_context(_context()) is None
    finally:
        await connection.close()
