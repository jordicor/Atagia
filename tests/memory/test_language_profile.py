"""Tests for post-ingest user communication language profiles."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.communication_profile_repository import CommunicationProfileRepository
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, UserRepository, WorkspaceRepository
from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.language_profile import (
    _CARD_NAMES,
    _card_prompt,
    UserCommunicationProfileService,
)
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
from tests.memory.card_leak_guard import assert_prompt_has_no_benchmark_leak

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"

_LANGUAGE_PROFILE_CARD_PURPOSES = {
    "user_language_profile_observed_card",
    "user_language_profile_preference_card",
    "user_language_profile_ability_card",
    "user_language_profile_norm_card",
}

_EMPTY_CARD_OUTPUTS = {
    "user_language_profile_observed_card": "none",
    "user_language_profile_preference_card": "none",
    "user_language_profile_ability_card": "none",
    "user_language_profile_norm_card": "none",
}


class LanguageProfileProvider(LLMProvider):
    name = "language-profile-tests"

    def __init__(self, outputs: list[dict[str, str]]) -> None:
        self.outputs = list(outputs)
        self._active_outputs: dict[str, str] | None = None
        self._active_consumed: set[str] = set()
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose not in _LANGUAGE_PROFILE_CARD_PURPOSES:
            raise AssertionError(f"Unexpected purpose {purpose}")
        if self._active_outputs is None:
            if not self.outputs:
                raise AssertionError("No language profile card output configured")
            self._active_outputs = {**_EMPTY_CARD_OUTPUTS, **self.outputs.pop(0)}
            self._active_consumed = set()
        output_text = self._active_outputs[purpose]
        self._active_consumed.add(purpose)
        if self._active_consumed == _LANGUAGE_PROFILE_CARD_PURPOSES:
            self._active_outputs = None
            self._active_consumed = set()
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
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


async def _seed_service(
    outputs: list[dict[str, object]],
    *,
    settings: Settings | None = None,
):
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
        settings=settings or _settings(),
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
                "user_language_profile_observed_card": "ES",
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
        assert {
            request.metadata["purpose"]
            for request in provider.requests
        } == _LANGUAGE_PROFILE_CARD_PURPOSES
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_language_profile_merges_preferences_abilities_and_observations() -> None:
    connection, repository, _provider, service = await _seed_service(
        [
            {
                "user_language_profile_observed_card": "es",
                "user_language_profile_preference_card": (
                    "default_answer_language es ordinary_chat"
                ),
                "user_language_profile_ability_card": "understands en",
                "user_language_profile_norm_card": (
                    "comfortable_for_terms_or_code en technical_terms"
                ),
            },
            {
                "user_language_profile_observed_card": "es",
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
                "user_language_profile_observed_card": "es",
            },
            {
                "user_language_profile_observed_card": "ca",
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
                "user_language_profile_observed_card": "jp\nca",
                "user_language_profile_preference_card": (
                    "default_answer_language spanish ordinary_chat"
                ),
                "user_language_profile_ability_card": "understands EN",
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
                "user_language_profile_observed_card": "es",
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


def _preference_card_prompt(provider: LanguageProfileProvider) -> str:
    for request in provider.requests:
        if (
            request.metadata["purpose"]
            == "user_language_profile_preference_card"
        ):
            return request.messages[-1].content
    raise AssertionError("preference card request not found")


@pytest.mark.asyncio
async def test_card_prompt_includes_examples_by_default() -> None:
    connection, _repository, provider, service = await _seed_service(
        [
            {
                "user_language_profile_observed_card": "es",
            }
        ]
    )
    try:
        await service.update_from_message(
            message_text="Me va mejor si seguimos en espanol.",
            role="user",
            conversation_context=_context(),
            occurred_at="2026-05-20T12:03:00+00:00",
        )
        prompt = _preference_card_prompt(provider)
        assert "Examples:" in prompt
        assert (
            "I prefer Dutch. -> default_answer_language nl default" in prompt
        )
        assert "Format examples:" not in prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_card_prompt_omits_examples_when_disabled() -> None:
    settings = replace(
        _settings(),
        card_examples_enabled=False,
    )
    connection, _repository, provider, service = await _seed_service(
        [
            {
                "user_language_profile_observed_card": "es",
            }
        ],
        settings=settings,
    )
    try:
        await service.update_from_message(
            message_text="Me va mejor si seguimos en espanol.",
            role="user",
            conversation_context=_context(),
            occurred_at="2026-05-20T12:03:00+00:00",
        )
        prompt = _preference_card_prompt(provider)
        # Instruction half is still present.
        assert "Allowed preference_kind values:" in prompt
        # Gated demonstration block is omitted.
        assert "Examples:" not in prompt
        assert (
            "I prefer Dutch. -> default_answer_language nl default" not in prompt
        )
    finally:
        await connection.close()


def test_language_card_prompts_do_not_leak_shadow_benchmark_content() -> None:
    # The language-profile cards have their own shadow benchmark; their few-shot
    # examples must not reuse a benchmark case message or distinctive answer token,
    # so the benchmark keeps measuring generalization rather than recall of the key.
    combined_prompt = "\n".join(
        compose_card_prompt(
            *_card_prompt(card_name=card_name, message_text="", role="user"),
            include_examples=True,
        )
        for card_name in _CARD_NAMES
    )
    assert_prompt_has_no_benchmark_leak(
        combined_prompt, "benchmarks/language_profile_cards/cases.jsonl"
    )
