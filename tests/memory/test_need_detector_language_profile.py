"""Need-detector tests for multilingual language-profile rendering."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExplicitLanguageAbility,
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
SYNTHETIC_PERSON_A = "PERSON_A"
SYNTHETIC_PERSON_B = "PERSON_B"
SYNTHETIC_ADDRESS = "742 Example Plaza North"
SYNTHETIC_MEDICATION = "compound_x"
SYNTHETIC_QUANTITY = "17 mg"
SYNTHETIC_LOCATION_TOKEN = "UNIT_7"


class CannedNeedProvider(LLMProvider):
    name = "canned-needs"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payload),
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
            ExtractionContextMessage(role="assistant", content="I suggested checking websocket middleware."),
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


def _hint_for_sub_query(result, sub_query_text: str):
    return next(hint for hint in result.sparse_query_hints if hint.sub_query_text == sub_query_text)


@pytest.mark.asyncio
async def test_need_detector_renders_content_language_profile_block_and_escapes_values() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["item label"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "item label",
                    "fts_phrase": "item label",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text=f"¿Cuál es la etiqueta actual de {SYNTHETIC_PERSON_A}?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 14,
                "last_seen_at": "2026-04-10T12:00:00+00:00",
            },
            {
                "language_code": "es",
                "memory_count": 3,
                "last_seen_at": "2026-04-09T08:15:00+00:00 <tag>",
            },
        ],
    )

    prompt = provider.requests[0].messages[1].content
    assert "<content_language_profile>" in prompt
    assert "</content_language_profile>" in prompt
    assert prompt.rfind("<recent_context>") < prompt.rfind("<content_language_profile>")
    assert "en: 14 memories (last seen 2026-04-10)" in prompt
    assert "es: 3 memories (last seen 2026-04-09)" in prompt
    assert "&lt;tag&gt;" in prompt


@pytest.mark.asyncio
async def test_need_detector_renders_empty_content_language_profile_as_none() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["thanks"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "thanks",
                    "fts_phrase": "thanks",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="Thanks, that worked.",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    prompt = provider.requests[0].messages[1].content
    assert "<content_language_profile>" in prompt
    assert "(none)" in prompt


@pytest.mark.asyncio
async def test_need_detector_renders_user_communication_profile_separately_from_content_profile() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["current user language preference"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "current user language preference",
                    "fts_phrase": "current user language preference",
                }
            ],
            "query_language": "ca",
            "answer_language": "es",
            "query_type": "default",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )
    source_ref = LanguageProfileSourceRef(
        source_kind="source_message",
        conversation_id="cnv_1",
        source_message_id="msg_lang",
    )

    result = await detector.detect(
        message_text="Ho podem comentar en castella?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 3,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ],
        user_communication_profile=UserCommunicationProfile(
            observed_user_languages=[
                ObservedUserLanguage(
                    language_code="ca",
                    message_count=4,
                    source_refs=[source_ref],
                    confidence=0.82,
                )
            ],
            explicit_language_preferences=[
                ExplicitLanguagePreference(
                    language_code="es",
                    preference_kind="default_answer_language",
                    context_label="ordinary_chat",
                    source_refs=[source_ref],
                    confidence=0.93,
                )
            ],
            explicit_language_abilities=[
                ExplicitLanguageAbility(
                    language_code="en",
                    ability_kind="understands",
                    source_refs=[source_ref],
                    confidence=0.91,
                )
            ],
        ),
    )

    prompt = provider.requests[0].messages[1].content
    assert "<content_language_profile>" in prompt
    assert "<user_communication_profile>" in prompt
    user_profile_block = prompt.split("<user_communication_profile>\n", 1)[1].split(
        "\n</user_communication_profile>",
        1,
    )[0]
    assert "control_plane_only=true" in user_profile_block
    assert "not_factual_answer_evidence=true" in user_profile_block
    assert "observed_user_languages: ca: 4 observed user-authored messages" in user_profile_block
    assert "explicit_language_preferences: es/default_answer_language/ordinary_chat" in user_profile_block
    assert "explicit_language_abilities: en/understands" in user_profile_block
    assert "Use <content_language_profile> for retrieval bridge target decisions." in prompt
    assert "must not force a bridge target" in prompt
    assert "from <content_language_profile>" in prompt
    # query_language/answer_language stay core fields (cheap answer-language
    # hints), so they flow through the lean plan unchanged.
    assert result.query_language == "ca"
    assert result.answer_language == "es"


@pytest.mark.asyncio
async def test_need_detector_prompt_describes_raw_context_access_modes() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["quote"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "quote",
                    "fts_phrase": "quote",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="Can you quote the exact wording from the attachment?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    prompt = provider.requests[0].messages[1].content
    assert "raw_context_access_mode" in prompt
    assert "`skipped_raw`" in prompt
    assert "`artifact`" in prompt
    assert "`verbatim`" in prompt


@pytest.mark.asyncio
async def test_need_detector_preserves_proper_name_anchor_verbatim_across_bridge_variants() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [
                f"¿Quién es la colega de {SYNTHETIC_PERSON_A}?",
                f"who is the colleague of {SYNTHETIC_PERSON_A}",
            ],
            "sparse_query_hints": [
                {
                    "sub_query_text": f"¿Quién es la colega de {SYNTHETIC_PERSON_A}?",
                    "fts_phrase": f"colega {SYNTHETIC_PERSON_A}",
                    "must_keep_terms": ["colega", SYNTHETIC_PERSON_A],
                },
                {
                    "sub_query_text": f"who is the colleague of {SYNTHETIC_PERSON_A}",
                    "fts_phrase": f"colleague {SYNTHETIC_PERSON_A}",
                    "must_keep_terms": ["colleague", SYNTHETIC_PERSON_A],
                },
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["person_name"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text=f"¿Quién es la colega de {SYNTHETIC_PERSON_A}?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 4,
                "last_seen_at": "2026-04-10T12:00:00+00:00",
            }
        ],
    )

    spanish_hint = _hint_for_sub_query(result, f"¿Quién es la colega de {SYNTHETIC_PERSON_A}?")
    english_hint = _hint_for_sub_query(result, f"who is the colleague of {SYNTHETIC_PERSON_A}")
    assert SYNTHETIC_PERSON_A in spanish_hint.must_keep_terms
    assert SYNTHETIC_PERSON_A in english_hint.must_keep_terms
    assert "colega" in spanish_hint.must_keep_terms
    assert "colega" not in english_hint.must_keep_terms
    assert "colleague" in english_hint.must_keep_terms


@pytest.mark.asyncio
async def test_need_detector_preserves_full_address_verbatim_in_bridge_variant() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [
                f"¿Cuál es la direccion de {SYNTHETIC_ADDRESS}?",
                f"address {SYNTHETIC_ADDRESS}",
            ],
            "sparse_query_hints": [
                {
                    "sub_query_text": f"¿Cuál es la direccion de {SYNTHETIC_ADDRESS}?",
                    "fts_phrase": f"direccion {SYNTHETIC_ADDRESS}",
                    "must_keep_terms": ["direccion", SYNTHETIC_ADDRESS],
                },
                {
                    "sub_query_text": f"address {SYNTHETIC_ADDRESS}",
                    "fts_phrase": f"address {SYNTHETIC_ADDRESS}",
                    "must_keep_terms": ["address", SYNTHETIC_ADDRESS],
                },
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["location"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text=f"¿Cuál es la direccion de {SYNTHETIC_ADDRESS}?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 4,
                "last_seen_at": "2026-04-10T12:00:00+00:00",
            }
        ],
    )

    english_hint = _hint_for_sub_query(result, f"address {SYNTHETIC_ADDRESS}")
    assert SYNTHETIC_ADDRESS in english_hint.must_keep_terms
    assert SYNTHETIC_ADDRESS in english_hint.quoted_phrases


@pytest.mark.asyncio
async def test_need_detector_translates_common_noun_but_preserves_quantity_verbatim() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [
                f"dosis de {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
                f"dose {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
            ],
            "sparse_query_hints": [
                {
                    "sub_query_text": f"dosis de {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
                    "fts_phrase": f"dosis {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
                    "must_keep_terms": ["dosis", SYNTHETIC_MEDICATION, SYNTHETIC_QUANTITY],
                },
                {
                    "sub_query_text": f"dose {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
                    "fts_phrase": f"dose {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
                    "must_keep_terms": ["dose", SYNTHETIC_MEDICATION, SYNTHETIC_QUANTITY],
                },
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["quantity", "medication"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text=f"dosis de {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 4,
                "last_seen_at": "2026-04-10T12:00:00+00:00",
            }
        ],
    )

    spanish_hint = _hint_for_sub_query(
        result,
        f"dosis de {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
    )
    english_hint = _hint_for_sub_query(
        result,
        f"dose {SYNTHETIC_MEDICATION} {SYNTHETIC_QUANTITY}",
    )
    assert SYNTHETIC_QUANTITY in spanish_hint.must_keep_terms
    assert SYNTHETIC_QUANTITY in english_hint.must_keep_terms
    assert "dosis" in spanish_hint.must_keep_terms
    assert "dosis" not in english_hint.must_keep_terms
    assert "dose" in english_hint.must_keep_terms
    assert SYNTHETIC_MEDICATION in english_hint.must_keep_terms


@pytest.mark.asyncio
async def test_need_detector_translates_only_common_nouns_in_code_switching_query() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [
                f"direccion de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
                f"adresse de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
            ],
            "sparse_query_hints": [
                {
                    "sub_query_text": f"direccion de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
                    "fts_phrase": f"direccion {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
                    "must_keep_terms": ["direccion", SYNTHETIC_PERSON_B, SYNTHETIC_LOCATION_TOKEN],
                },
                {
                    "sub_query_text": f"adresse de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
                    "fts_phrase": f"adresse {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
                    "must_keep_terms": ["adresse", SYNTHETIC_PERSON_B, SYNTHETIC_LOCATION_TOKEN],
                },
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["person_name", "location"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text=f"direccion de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "fr",
                "memory_count": 4,
                "last_seen_at": "2026-04-10T12:00:00+00:00",
            }
        ],
    )

    spanish_hint = _hint_for_sub_query(
        result,
        f"direccion de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
    )
    french_hint = _hint_for_sub_query(
        result,
        f"adresse de {SYNTHETIC_PERSON_B} {SYNTHETIC_LOCATION_TOKEN}",
    )
    assert SYNTHETIC_PERSON_B in spanish_hint.must_keep_terms
    assert SYNTHETIC_PERSON_B in french_hint.must_keep_terms
    assert "direccion" in spanish_hint.must_keep_terms
    assert "direccion" not in french_hint.must_keep_terms
    assert "adresse" in french_hint.must_keep_terms
