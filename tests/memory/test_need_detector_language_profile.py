"""Language-profile coverage for parallel-card need detection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExplicitLanguagePreference,
    ExtractionContextMessage,
    ExtractionConversationContext,
    LanguageProfileSourceRef,
    ObservedUserLanguage,
    UserCommunicationProfile,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class CannedCardProvider(LLMProvider):
    name = "canned-cards"

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose") or "")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.get(purpose, "none"),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by need detector tests")


def _resolved_policy(mode_id: str = "coding_debug"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id="wrk_1",
        assistant_mode_id="coding_debug",
        recent_messages=[
            ExtractionContextMessage(
                role="assistant",
                content="I suggested checking websocket middleware.",
            ),
            ExtractionContextMessage(role="user", content="That still did not fix it."),
        ],
    )


def _clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


def _outputs(language: str = "es\nfr") -> dict[str, str]:
    return {
        "need_detection_needs_card": "none",
        "need_detection_language_card": language,
        "need_detection_memory_card": "personal",
        "need_detection_exact_card": "yes",
        "need_detection_shape_card": "slot",
        "need_detection_facets_card": "location",
        "need_detection_callback_card": "no",
        "need_detection_search_words_card": "Claire\nadresse",
    }


@pytest.mark.asyncio
async def test_language_card_result_controls_query_and_answer_language() -> None:
    provider = CannedCardProvider(_outputs("es\nfr"))
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Responde en frances: cual es mi direccion?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.query_language == "es"
    assert result.answer_language == "fr"


@pytest.mark.asyncio
async def test_anchor_card_aliases_are_materialized_for_saved_memory_language() -> None:
    outputs = _outputs("es\nes")
    outputs["need_detection_search_words_card"] = "ibuprofeno"
    outputs["need_detection_search_words_other_language_card"] = (
        "ibuprofeno => ibuprofen"
    )
    provider = CannedCardProvider(outputs)
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Cual es mi dosis actual de ibuprofeno?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 42,
                "last_seen_at": "2026-04-01T00:00:00+00:00",
            }
        ],
    )

    assert [anchor.original_surface for anchor in result.anchors] == ["ibuprofeno"]
    assert result.sparse_query_hints[0].must_keep_terms == ["ibuprofeno"]
    assert {
        request.metadata["purpose"] for request in provider.requests
    } >= {
        "need_detection_search_words_card",
        "need_detection_search_words_other_language_card",
    }
    alias = result.anchors[0].aliases[0]
    assert alias.surface == "ibuprofen"
    assert alias.alias_language == "en"
    assert alias.alias_kind == "translation"
    assert alias.non_evidential is True
    assert alias.derivation["target_content_languages"] == ["en"]


@pytest.mark.asyncio
async def test_other_language_search_words_card_is_not_called_when_languages_match() -> None:
    provider = CannedCardProvider(_outputs("es\nes"))
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="Cual es mi direccion?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "es",
                "memory_count": 42,
                "last_seen_at": "2026-04-01T00:00:00+00:00",
            }
        ],
    )

    purposes = {request.metadata["purpose"] for request in provider.requests}
    assert "need_detection_search_words_card" in purposes
    assert "need_detection_search_words_other_language_card" not in purposes


@pytest.mark.asyncio
async def test_other_language_search_words_card_is_not_called_without_search_words() -> None:
    outputs = _outputs("es\nes")
    outputs["need_detection_search_words_card"] = "none"
    provider = CannedCardProvider(outputs)
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="Cual es?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 42,
                "last_seen_at": "2026-04-01T00:00:00+00:00",
            }
        ],
    )

    purposes = {request.metadata["purpose"] for request in provider.requests}
    assert "need_detection_search_words_card" in purposes
    assert "need_detection_search_words_other_language_card" not in purposes


@pytest.mark.asyncio
async def test_content_language_profile_is_rendered_for_each_card() -> None:
    provider = CannedCardProvider(_outputs())
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="Quelle adresse Claire a-t-elle donnee?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 12,
                "last_seen_at": "2026-04-01T00:00:00+00:00",
            },
            {
                "language_code": "unknown",
                "memory_count": 3,
                "last_seen_at": "2026-04-02T00:00:00+00:00",
            },
        ],
    )

    for request in provider.requests:
        prompt = request.messages[1].content
        assert "Saved memory languages:" in prompt
        assert "en: 12 memories" in prompt
        assert "unknown: 3 memories" in prompt


@pytest.mark.asyncio
async def test_user_communication_profile_is_rendered_as_control_plane_only() -> None:
    provider = CannedCardProvider(_outputs("fr\nfr"))
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )
    source = LanguageProfileSourceRef(
        source_kind="source_message",
        source_message_id="msg_profile",
        conversation_id="cnv_1",
    )
    profile = UserCommunicationProfile(
        observed_user_languages=[
                ObservedUserLanguage(
                    language_code="fr",
                    message_count=4,
                    source_refs=[source],
                    confidence=0.9,
                )
            ],
            explicit_language_preferences=[
                ExplicitLanguagePreference(
                    language_code="fr",
                    preference_kind="contextual_answer_language",
                    context_label="coding_debug",
                    source_refs=[source],
                    confidence=0.9,
                )
            ],
    )

    await detector.detect(
        message_text="Quelle adresse Claire a-t-elle donnee?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
        user_communication_profile=profile,
    )

    prompt = provider.requests[0].messages[1].content
    assert "control_plane_only=true" in prompt
    assert "not_factual_answer_evidence=true" in prompt
    assert "observed_user_languages: fr: 4 observed user-authored messages" in prompt
    assert (
        "explicit_language_preferences: fr/contextual_answer_language/coding_debug"
        in prompt
    )
