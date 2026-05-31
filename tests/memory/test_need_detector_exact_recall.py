"""Exact recall routing tests for the need detector (Wave 1 batch 2, 1-D).

These tests confirm the need detector surfaces exact recall signals and
that the prompt is language-agnostic: it must work for Spanish, English,
or any language without relying on English keyword lists.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import (
    ANCHOR_REVIEW_PROMPT_TEMPLATE,
    MULTI_FACET_EXACT_RECALL_REVIEW_PROMPT_TEMPLATE,
    NEED_DETECTOR_PROMPT_TEMPLATE,
    MultiFacetExactRecallReview,
    NeedDetector,
    UnknownOnlyExactValueReview,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExactFacet,
    ExtractionContextMessage,
    ExtractionConversationContext,
    QueryIntelligenceResult,
    RuntimeAnchor,
    RuntimeAnchorAlias,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    StructuredOutputError,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class CannedNeedProvider(LLMProvider):
    name = "canned-needs"

    def __init__(
        self,
        payload: dict[str, object] | list[dict[str, object]],
    ) -> None:
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        payload_index = min(len(self.requests) - 1, len(self.payloads) - 1)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payloads[payload_index]),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by need detector tests")


def _resolved_policy(mode_id: str = "general_qa"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id=None,
        assistant_mode_id="general_qa",
        recent_messages=[
            ExtractionContextMessage(role="user", content="Previous turn"),
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


@pytest.mark.asyncio
async def test_need_detector_returns_exact_recall_signal() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["birthday year"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "birthday year",
                    "fts_phrase": "birthday year",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["date", "person_name"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="In what year was I born?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.DATE, ExactFacet.PERSON_NAME]


@pytest.mark.asyncio
async def test_need_detector_exact_recall_fires_for_spanish_query() -> None:
    """Language-agnostic: Spanish wording with no English keywords."""
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["cumpleaños fecha"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "cumpleaños fecha",
                    "fts_phrase": "cumpleaños fecha",
                }
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["date"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="¿Recuerdas en qué fecha cumplo años?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.DATE]


@pytest.mark.asyncio
async def test_need_detector_promotes_person_anchored_broad_list_to_exact_recall() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Which cities has Jon visited?"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Which cities has Jon visited?",
                    "fts_phrase": "Jon visited cities",
                    "must_keep_terms": ["Jon", "visited", "cities"],
                }
            ],
            "anchors": [
                {
                    "sub_query_text": "Which cities has Jon visited?",
                    "anchor_type": "person",
                    "original_surface": "Jon",
                    "preserve_verbatim": True,
                    "confidence": 1.0,
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0, 1],
            "exact_recall_needed": False,
            "exact_facets": [],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Which cities has Jon visited?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]
    assert any(
        event.mechanism == "anchored_broad_list_exact_recall_fallback"
        for event in result.temporary_scaffolding
    )


@pytest.mark.asyncio
async def test_need_detector_promotes_unknown_verbatim_anchor_broad_list_to_exact_recall() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["Which places has the named person visited?"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "Which places has the named person visited?",
                    "fts_phrase": "named person visited places",
                    "must_keep_terms": ["named person", "visited", "places"],
                }
            ],
            "anchors": [
                {
                    "sub_query_text": "Which places has the named person visited?",
                    "anchor_type": "unknown",
                    "original_surface": "named person",
                    "preserve_verbatim": True,
                    "confidence": 0.92,
                }
            ],
            "query_type": "broad_list",
            "retrieval_levels": [0, 1],
            "exact_recall_needed": False,
            "exact_facets": [],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Which places has the named person visited?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.query_type == "broad_list"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]


def test_need_detector_prompt_describes_language_agnostic_rules_without_real_examples() -> None:
    """The prompt should stay abstract, multilingual, and benchmark-agnostic."""
    template = NEED_DETECTOR_PROMPT_TEMPLATE
    lowered = template.lower()

    assert "The user message may be written in any language." in template
    assert "Do not rely on English keywords." in template
    assert "specific language or phrasing" in lowered
    assert "<recent_context>" in template
    assert "<person_name>" in template
    assert "<street_address>" in template
    assert "<quantity_with_unit>" in template
    assert "It is not a bridge target" in template
    assert "unknown-only" in template
    assert "Do not guess a bridge language" in template
    assert "current or last-known value/setting/amount" in template
    assert "unknown-only only prevents guessing a bridge" in lowered
    assert "must not downgrade that classification" in lowered
    assert "do not require evidence that the" in lowered
    assert "expected answer shape is one stored value" in lowered
    assert "exact_recall_needed" in lowered
    assert "exact_facets" in lowered
    assert "multiple concrete named items" in lowered
    assert "how a concrete person described" in lowered
    assert "compact remembered description" in lowered
    # Structured-anchor guidance now lives in the anchor review prompt, not the
    # lean base prompt. The base prompt must NOT ask the model for structured
    # anchors / aliases anymore (the bridge stays via must_keep_terms, and the
    # cheap query_language/answer_language hints remain core fields).
    assert "phase 4 structured anchors" not in lowered
    assert "`anchors`" not in template


def test_anchor_review_prompt_is_generic_and_carries_structured_anchor_rules() -> None:
    template = ANCHOR_REVIEW_PROMPT_TEMPLATE
    lowered = template.lower()

    assert "structured retrieval anchors" in lowered
    assert "non-evidential" in lowered
    assert "Do not rely on English keywords." in template
    assert "Do not use regexes" in template
    assert "benchmark-specific" in lowered
    assert "<sub_queries>" in template
    assert "preserve_verbatim" in template
    assert "aliases" in lowered
    assert "are not bridge target" in lowered


def test_multi_facet_exact_recall_review_prompt_is_generic() -> None:
    template = MULTI_FACET_EXACT_RECALL_REVIEW_PROMPT_TEMPLATE
    lowered = template.lower()

    assert "separate retrieval obligations" in lowered
    assert "two or more distinct exact facts" in lowered
    assert "Do not rely on English keywords." in template
    assert "Do not use regexes" in template
    assert "benchmark-specific" in lowered
    assert "unknown-only and cold-start profiles" in lowered
    assert "Allowed exact facets" in template


@pytest.mark.asyncio
async def test_need_detector_splits_single_exact_query_into_obligations() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["original budget and final payment for the shared lease"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "original budget and final payment for the shared lease",
                        "fts_phrase": "shared lease budget final payment",
                        "must_keep_terms": ["shared lease", "budget", "payment"],
                    }
                ],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
                "exact_recall_needed": True,
                "exact_facets": ["quantity", "person_name"],
            },
            {"anchors": []},
            {
                "has_multiple_obligations": True,
                "sub_queries": [
                    {
                        "sub_query_text": "original budget for the shared lease",
                        "fts_phrase": "shared lease original budget",
                        "must_keep_terms": ["shared lease", "original budget"],
                        "exact_facets": ["quantity"],
                    },
                    {
                        "sub_query_text": "final amount paid for the shared lease",
                        "fts_phrase": "shared lease final amount paid",
                        "must_keep_terms": ["shared lease", "final amount paid"],
                        "exact_facets": ["quantity"],
                    },
                ],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text=(
            "What was the original budget and final amount paid for the shared lease?"
        ),
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert len(provider.requests) == 3
    assert provider.requests[1].metadata["purpose"] == "need_detection_anchor_review"
    assert provider.requests[2].metadata["purpose"] == (
        "need_detection_multi_facet_exact_review"
    )
    review_prompt = provider.requests[2].messages[1].content
    assert "<initial_query_intelligence_json>" in review_prompt
    assert "<user_message>" in review_prompt
    assert "<content_language_profile>" in review_prompt
    assert result.query_type == "broad_list"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.QUANTITY]
    assert result.sub_queries == [
        "original budget for the shared lease",
        "final amount paid for the shared lease",
    ]
    assert [
        hint.fts_phrase for hint in result.sparse_query_hints
    ] == [
        "shared lease original budget",
        "shared lease final amount paid",
    ]
    assert result.anchors == []


@pytest.mark.asyncio
async def test_need_detector_keeps_single_slot_fill_when_review_says_one_obligation() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["birth year for the user"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "birth year for the user",
                        "fts_phrase": "birth year user",
                        "must_keep_terms": ["birth year", "user"],
                    }
                ],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
                "exact_recall_needed": True,
                "exact_facets": ["date", "person_name"],
            },
            {"anchors": []},
            {
                "has_multiple_obligations": False,
                "sub_queries": [],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="In what year was I born?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert len(provider.requests) == 3
    assert provider.requests[1].metadata["purpose"] == "need_detection_anchor_review"
    assert provider.requests[2].metadata["purpose"] == (
        "need_detection_multi_facet_exact_review"
    )
    assert result.query_type == "slot_fill"
    assert result.sub_queries == ["birth year for the user"]
    assert result.exact_facets == [ExactFacet.DATE, ExactFacet.PERSON_NAME]


def test_multi_facet_exact_recall_review_infers_positive_payload_without_flag() -> None:
    review = MultiFacetExactRecallReview.model_validate(
        {
            "sub_queries": [
                {
                    "sub_query_text": "first requested amount",
                    "fts_phrase": "first amount",
                    "exact_facets": ["quantity"],
                },
                {
                    "sub_query_text": "second requested amount",
                    "fts_phrase": "second amount",
                    "exact_facets": ["quantity"],
                },
            ]
        }
    )

    assert review.has_multiple_obligations is True
    assert review.sub_queries[0].must_keep_terms == ["first amount"]
    assert review.sub_queries[1].must_keep_terms == ["second amount"]


@pytest.mark.asyncio
async def test_need_detector_anchor_review_delivers_structured_anchors_and_aliases() -> None:
    """Structured anchors now come from the conditional anchor review, not the
    lean primary plan. The review output is merged into the rich result."""
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["what pipeline does Ben's team use"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "what pipeline does Ben's team use",
                        "fts_phrase": "ci cd pipeline ben team",
                        "must_keep_terms": ["Ben", "CI/CD", "pipeline"],
                    }
                ],
                "query_language": "EN",
                "answer_language": "en",
                "query_type": "slot_fill",
                "retrieval_levels": [0],
                "exact_recall_needed": True,
                "exact_facets": ["other_verbatim"],
            },
            {
                "anchors": [
                    {
                        "sub_query_text": "what pipeline does Ben's team use",
                        "anchor_type": "proper_name",
                        "original_surface": "Ben",
                        "preserve_verbatim": True,
                        "confidence": 0.95,
                    },
                    {
                        "sub_query_text": "what pipeline does Ben's team use",
                        "anchor_type": "concept",
                        "original_surface": "CI/CD pipeline",
                        "normalized_surface": "ci/cd pipeline",
                        "aliases": [
                            {
                                "surface": "continuous integration continuous delivery pipeline",
                                "alias_language": "en",
                                "alias_kind": "acronym_expansion",
                                "confidence": 0.72,
                                "derivation": {"source": "fake_llm"},
                            }
                        ],
                        "confidence": 0.88,
                    },
                ],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="What CI/CD pipeline does Ben's team use?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == "need_detection_anchor_review"
    anchor_review_prompt = provider.requests[1].messages[1].content
    assert "<sub_queries>" in anchor_review_prompt
    assert "structured retrieval anchors" in anchor_review_prompt
    # Cheap answer-language hints flow from the lean primary plan.
    assert result.query_language == "en"
    assert result.answer_language == "en"
    assert [anchor.anchor_type for anchor in result.anchors] == [
        "proper_name",
        "concept",
    ]
    assert result.anchors[0].preserve_verbatim is True
    assert result.anchors[0].non_evidential is True
    assert result.anchors[1].aliases[0].alias_kind == "acronym_expansion"
    assert result.anchors[1].aliases[0].non_evidential is True


@pytest.mark.asyncio
async def test_need_detector_unknown_only_accepts_current_value_slot_fill_contract() -> None:
    """Unknown-only profiles must not downgrade exact value/current-setting plans."""
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["dosis actual del suplemento nocturno"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "dosis actual del suplemento nocturno",
                        "fts_phrase": "dosis suplemento nocturno",
                        "must_keep_terms": ["suplemento", "dosis"],
                    }
                ],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
                "exact_recall_needed": True,
                "exact_facets": ["quantity"],
            },
            {
                "anchors": [
                    {
                        "sub_query_text": "dosis actual del suplemento nocturno",
                        "anchor_type": "concept",
                        "original_surface": "suplemento nocturno",
                        "confidence": 0.82,
                    },
                    {
                        "sub_query_text": "dosis actual del suplemento nocturno",
                        "anchor_type": "attribute",
                        "original_surface": "dosis actual",
                        "confidence": 0.86,
                    },
                ],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="¿Cuál es la dosis actual del suplemento nocturno?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.QUANTITY]
    assert result.sparse_query_hints[0].must_keep_terms == ["suplemento", "dosis"]
    assert [anchor.anchor_type for anchor in result.anchors] == ["concept", "attribute"]
    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == "need_detection_anchor_review"
    prompt = provider.requests[0].messages[1].content
    assert "unknown-only only prevents guessing a bridge" in prompt.lower()
    assert "must not downgrade that classification" in prompt.lower()


@pytest.mark.asyncio
async def test_need_detector_unknown_only_reviews_default_current_value_output() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["¿Cuál es la dosis actual del suplemento nocturno?"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "¿Cuál es la dosis actual del suplemento nocturno?",
                        "fts_phrase": "dosis actual suplemento nocturno",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "dosis actual del suplemento nocturno",
                "fts_phrase": "dosis suplemento nocturno",
                "must_keep_terms": ["suplemento", "dosis"],
                "exact_facets": ["quantity"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="¿Cuál es la dosis actual del suplemento nocturno?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    review_prompt = provider.requests[1].messages[1].content
    assert "<initial_query_intelligence_json>" in review_prompt
    assert "<user_message>" in review_prompt
    assert "<content_language_profile>" in review_prompt
    assert "not bridge\n  target languages" in review_prompt.lower()
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.QUANTITY]
    assert result.sparse_query_hints[0].must_keep_terms == ["suplemento", "dosis"]


@pytest.mark.asyncio
async def test_need_detector_unknown_only_reviews_default_interaction_preference_output() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["Comment l'assistant doit-il aider quand je panique ?"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": (
                            "Comment l'assistant doit-il aider quand je panique ?"
                        ),
                        "fts_phrase": "assistant aider panique",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "instruction d'aide quand je panique",
                "fts_phrase": "panique prochaine étape liste",
                "must_keep_terms": ["panique", "prochaine étape", "liste"],
                "exact_facets": ["other_verbatim"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Comment l'assistant doit-il aider quand je panique ?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    review_prompt = provider.requests[1].messages[1].content
    assert "stored interaction instruction" in review_prompt
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "panique",
        "prochaine étape",
        "liste",
    ]


@pytest.mark.asyncio
async def test_need_detector_cold_start_reviews_default_current_code_output() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["Quel est le code actuel du casier partagé ?"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "Quel est le code actuel du casier partagé ?",
                        "fts_phrase": "code actuel casier partagé",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "code actuel du casier partagé",
                "fts_phrase": "code casier partagé",
                "must_keep_terms": ["code", "casier partagé"],
                "exact_facets": ["code"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Quel est le code actuel du casier partagé ?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    review_prompt = provider.requests[1].messages[1].content
    assert "empty/cold-start" in review_prompt
    assert "(none)" in review_prompt
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.CODE]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "code",
        "casier partagé",
    ]


@pytest.mark.asyncio
async def test_need_detector_cold_start_structured_error_reviews_exact_value() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": [],
                "sparse_query_hints": [],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "current lab door code",
                "fts_phrase": "lab door code",
                "must_keep_terms": ["lab door", "code"],
                "exact_facets": ["code"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="What is the current lab door code?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    review_schema = provider.requests[1].response_schema or {}
    review_properties = review_schema.get("properties") or {}
    assert "is_exact_value_lookup" in review_properties
    assert "sub_queries" not in review_properties
    assert result.sub_queries == ["current lab door code"]
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.CODE]


@pytest.mark.asyncio
async def test_need_detector_unknown_only_review_request_uses_review_schema() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["valor actual del ajuste de cabina"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "valor actual del ajuste de cabina",
                        "fts_phrase": "ajuste cabina",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": False,
                "exact_facets": [],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    await detector.detect(
        message_text="¿Cuál es el valor actual del ajuste de cabina?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 3,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    review_schema = provider.requests[1].response_schema or {}
    review_properties = review_schema.get("properties") or {}
    review_required = set(review_schema.get("required") or [])
    assert "is_exact_value_lookup" in review_properties
    assert "sub_query_text" in review_properties
    assert "sub_queries" not in review_properties
    assert "sub_queries" not in review_required
    assert "needs" not in review_properties


@pytest.mark.asyncio
async def test_need_detector_unknown_only_reviews_structured_error_current_dose() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": [],
                "sparse_query_hints": [],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "dosis actual del medicamento matutino",
                "fts_phrase": "dosis medicamento matutino",
                "must_keep_terms": ["dosis", "medicamento", "matutino"],
                "exact_facets": ["quantity", "medication"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="¿Cuál es la dosis actual del medicamento matutino?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.QUANTITY, ExactFacet.MEDICATION]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "dosis",
        "medicamento",
        "matutino",
    ]


@pytest.mark.asyncio
async def test_need_detector_unknown_only_reviews_missing_sparse_hint_current_setting() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["valor actual del ajuste de cabina"],
                "sparse_query_hints": [],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "valor actual del ajuste de cabina",
                "fts_phrase": "ajuste cabina",
                "must_keep_terms": ["ajuste", "cabina"],
                "exact_facets": ["other_verbatim"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="¿Cuál es el valor actual del ajuste de cabina?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 3,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_unknown_only_contract_review"
    )
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]
    assert result.sparse_query_hints[0].must_keep_terms == ["ajuste", "cabina"]


def test_unknown_only_exact_value_review_infers_positive_payload_without_flag() -> None:
    review = UnknownOnlyExactValueReview.model_validate(
        {
            "sub_query_text": "current stored amount",
            "fts_phrase": "stored amount",
            "must_keep_terms": ["stored", "amount"],
            "exact_facets": ["quantity"],
        }
    )

    assert review.is_exact_value_lookup is True
    assert review.must_keep_terms == ["stored", "amount"]
    assert review.exact_facets == ["quantity"]


@pytest.mark.asyncio
async def test_need_detector_unknown_only_review_can_keep_non_exact_default() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["explica la idea general del proyecto"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "explica la idea general del proyecto",
                        "fts_phrase": "idea general proyecto",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": False,
                "exact_facets": [],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Explica la idea general del proyecto.",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert result.query_type == "default"
    assert result.exact_recall_needed is False
    assert result.exact_facets == []


@pytest.mark.asyncio
async def test_need_detector_unknown_only_review_normalizes_unsupported_exact_facet() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["current stored setting"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "current stored setting",
                        "fts_phrase": "stored setting",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "current stored setting",
                "fts_phrase": "stored setting",
                "exact_facets": ["unsupported_facet"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="What is the current stored setting?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]
    assert result.sparse_query_hints[0].must_keep_terms == ["stored setting"]


@pytest.mark.asyncio
async def test_need_detector_non_unknown_default_output_does_not_run_unknown_review() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["explica la idea general del proyecto"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "explica la idea general del proyecto",
                        "fts_phrase": "idea general proyecto",
                    }
                ],
                "query_type": "default",
                "retrieval_levels": [0],
                "exact_recall_needed": False,
                "exact_facets": [],
            },
            {
                "is_exact_value_lookup": False,
                "exact_facets": [],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Explica la idea general del proyecto.",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "es",
                "memory_count": 6,
                "last_seen_at": "2026-05-13T06:01:40.171117+00:00",
            }
        ],
    )

    assert len(provider.requests) == 1
    assert result.query_type == "default"
    assert result.exact_recall_needed is False


@pytest.mark.asyncio
async def test_need_detector_accepts_current_setting_slot_fill_contract() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["current calibration setting for the studio monitor"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "current calibration setting for the studio monitor",
                    "fts_phrase": "studio monitor calibration setting",
                    "must_keep_terms": ["studio monitor", "calibration setting"],
                }
            ],
            "query_language": "en",
            "answer_language": "en",
            "anchors": [
                {
                    "sub_query_text": "current calibration setting for the studio monitor",
                    "anchor_type": "concept",
                    "original_surface": "studio monitor",
                    "confidence": 0.8,
                },
                {
                    "sub_query_text": "current calibration setting for the studio monitor",
                    "anchor_type": "attribute",
                    "original_surface": "calibration setting",
                    "confidence": 0.85,
                },
            ],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["other_verbatim"],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="What is the current calibration setting for the studio monitor?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.OTHER_VERBATIM]


@pytest.mark.asyncio
async def test_need_detector_prompt_renders_unknown_only_language_profile_as_non_target() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": ["dirección del nuevo apartamento de Ben"],
                "sparse_query_hints": [
                    {
                        "sub_query_text": "dirección del nuevo apartamento de Ben",
                        "fts_phrase": "Ben apartamento dirección",
                        "must_keep_terms": ["Ben", "apartamento", "dirección"],
                    }
                ],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
                "exact_recall_needed": True,
                "exact_facets": ["location"],
            },
            {
                "anchors": [
                    {
                        "sub_query_text": "dirección del nuevo apartamento de Ben",
                        "anchor_type": "person",
                        "original_surface": "Ben",
                        "preserve_verbatim": True,
                        "confidence": 0.9,
                    }
                ],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Cual es la direccion del nuevo apartamento de Ben?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 14,
                "last_seen_at": "2026-05-13T06:01:13.997859+00:00",
            }
        ],
    )

    assert result.sub_queries == ["dirección del nuevo apartamento de Ben"]
    assert len(provider.requests) == 2
    prompt = provider.requests[0].messages[1].content
    profile_block = prompt.split("<content_language_profile>\n", 1)[1].split(
        "\n</content_language_profile>",
        1,
    )[0]
    assert profile_block == "unknown: 14 memories (last seen 2026-05-13)"
    assert "It is not a bridge target" in prompt
    assert "do not guess a\n    bridge language" in prompt
    assert "Unknown-only\n    is not an error state" in prompt


@pytest.mark.asyncio
async def test_need_detector_unknown_only_structured_error_returns_original_language_plan() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [],
            "sparse_query_hints": [],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Cual es la direccion del nuevo apartamento de Ben?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 14,
                "last_seen_at": "2026-05-13T06:01:13.997859+00:00",
            }
        ],
    )

    assert result.sub_queries == [
        "Cual es la direccion del nuevo apartamento de Ben?"
    ]
    assert result.sparse_query_hints[0].sub_query_text == result.sub_queries[0]
    assert result.sparse_query_hints[0].fts_phrase == result.sub_queries[0]
    assert result.query_type == "default"
    assert result.query_language is None
    assert result.answer_language is None
    assert result.anchors == []
    assert result.exact_recall_needed is False
    assert result.exact_facets == []


@pytest.mark.asyncio
async def test_need_detector_unknown_only_missing_sparse_hint_returns_original_language_plan() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["dirección del nuevo apartamento de Ben"],
            "sparse_query_hints": [],
            "query_type": "default",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="Cual es la direccion del nuevo apartamento de Ben?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "unknown",
                "memory_count": 14,
                "last_seen_at": "2026-05-13T06:01:13.997859+00:00",
            }
        ],
    )

    assert result.sub_queries == [
        "Cual es la direccion del nuevo apartamento de Ben?"
    ]
    assert result.sparse_query_hints[0].fts_phrase == result.sub_queries[0]
    assert result.anchors == []


@pytest.mark.asyncio
async def test_need_detector_non_unknown_profile_keeps_structured_errors_visible() -> None:
    provider = CannedNeedProvider(
        {
            "needs": [],
            "temporal_range": None,
            "sub_queries": [],
            "sparse_query_hints": [],
            "query_type": "slot_fill",
            "retrieval_levels": [0],
        }
    )
    detector = NeedDetector(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        clock=_clock(),
        settings=_settings(),
    )

    with pytest.raises(StructuredOutputError):
        await detector.detect(
            message_text="Cual es la direccion del nuevo apartamento de Ben?",
            role="user",
            conversation_context=_context(),
            resolved_policy=_resolved_policy(),
            content_language_profile=[
                {
                    "language_code": "en",
                    "memory_count": 14,
                    "last_seen_at": "2026-05-13T06:01:13.997859+00:00",
                }
            ],
        )


@pytest.mark.asyncio
async def test_need_detector_non_unknown_structured_error_reviews_exact_value() -> None:
    provider = CannedNeedProvider(
        [
            {
                "needs": [],
                "temporal_range": None,
                "sub_queries": [],
                "sparse_query_hints": [],
                "query_type": "slot_fill",
                "retrieval_levels": [0],
            },
            {
                "is_exact_value_lookup": True,
                "sub_query_text": "country of Caroline's grandma",
                "fts_phrase": "Caroline grandma country",
                "must_keep_terms": ["Caroline", "grandma", "country"],
                "exact_facets": ["location"],
            },
        ]
    )
    detector = NeedDetector(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        clock=_clock(),
        settings=_settings(),
    )

    result = await detector.detect(
        message_text="What country is Caroline's grandma from?",
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[
            {
                "language_code": "en",
                "memory_count": 14,
                "last_seen_at": "2026-05-13T06:01:13.997859+00:00",
            }
        ],
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["purpose"] == (
        "need_detection_degraded_exact_contract_review"
    )
    assert result.query_type == "slot_fill"
    assert result.sub_queries == ["country of Caroline's grandma"]
    assert result.sparse_query_hints[0].fts_phrase == "Caroline grandma country"
    assert result.sparse_query_hints[0].must_keep_terms == [
        "Caroline",
        "grandma",
        "country",
    ]
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.LOCATION]


def test_query_intelligence_requires_verbatim_preservation_for_literal_anchors() -> None:
    with pytest.raises(ValidationError, match="preserve_verbatim=true"):
        QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["Caroline owner"],
            sparse_query_hints=[
                {
                    "sub_query_text": "Caroline owner",
                    "fts_phrase": "caroline owner",
                    "must_keep_terms": ["Caroline"],
                }
            ],
            anchors=[
                {
                    "sub_query_text": "Caroline owner",
                    "anchor_type": "proper_name",
                    "original_surface": "Caroline",
                    "preserve_verbatim": False,
                }
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
        )


def test_runtime_anchor_alias_rejects_blank_required_surface() -> None:
    with pytest.raises(ValidationError, match="surface must be non-empty"):
        RuntimeAnchorAlias(surface="  ", alias_kind="translation")


def test_query_intelligence_rejects_blank_anchor_original_surface() -> None:
    with pytest.raises(ValidationError, match="original_surface must be non-empty"):
        QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["dosage"],
            sparse_query_hints=[],
            anchors=[
                {
                    "sub_query_text": "dosage",
                    "anchor_type": "concept",
                    "original_surface": "  ",
                }
            ],
            query_type="default",
            retrieval_levels=[0],
        )


def test_runtime_anchor_rejects_blank_required_sub_query_text() -> None:
    with pytest.raises(ValidationError, match="sub_query_text must be non-empty"):
        RuntimeAnchor(
            sub_query_text="  ",
            anchor_type="concept",
            original_surface="dosage",
        )


def test_runtime_anchor_optional_text_fields_can_collapse_to_none() -> None:
    alias = RuntimeAnchorAlias(
        surface="amlodipine",
        alias_language="  ",
        alias_kind="translation",
    )
    anchor = RuntimeAnchor(
        sub_query_text="dosage",
        anchor_type="concept",
        original_surface="amlodipino",
        normalized_surface="  ",
    )

    assert alias.alias_language is None
    assert anchor.normalized_surface is None


def test_runtime_anchors_and_aliases_reject_evidential_flags() -> None:
    with pytest.raises(ValidationError, match="runtime anchors must be non_evidential"):
        RuntimeAnchor(
            sub_query_text="Caroline owner",
            anchor_type="proper_name",
            original_surface="Caroline",
            preserve_verbatim=True,
            non_evidential=False,
        )

    with pytest.raises(ValidationError, match="runtime aliases must be non_evidential"):
        RuntimeAnchorAlias(
            surface="Carolina",
            alias_kind="translation",
            non_evidential=False,
        )
