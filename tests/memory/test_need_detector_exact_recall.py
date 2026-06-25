"""Exact-recall routing tests for parallel-card need detection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExactFacet,
    ExtractionContextMessage,
    ExtractionConversationContext,
    MemoryDependence,
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
            output_text=self.outputs[purpose],
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


async def _detect(message: str, outputs: dict[str, str]):
    provider = CannedCardProvider(outputs)
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )
    result = await detector.detect(
        message_text=message,
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )
    return result, provider


@pytest.mark.asyncio
async def test_exact_slot_fill_for_spanish_address_query() -> None:
    result, provider = await _detect(
        "Cual es la direccion del nuevo apartamento de Ben?",
        {
            "need_detection_needs_card": "none",
            "need_detection_language_card": "es\nes",
            "need_detection_memory_card": "personal",
            "need_detection_exact_card": "yes",
            "need_detection_shape_card": "slot",
            "need_detection_facets_card": "location",
            "need_detection_callback_card": "no",
            "need_detection_search_words_card": "Ben\napartamento\ndireccion",
            "need_detection_search_words_other_language_card": "none",
        },
    )

    assert len(provider.requests) == 8
    assert result.query_language == "es"
    assert result.answer_language == "es"
    assert result.memory_dependence is MemoryDependence.PERSONAL
    assert result.query_type == "slot_fill"
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.LOCATION]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "Ben",
        "apartamento",
        "direccion",
    ]


@pytest.mark.asyncio
async def test_exact_facets_cover_multiple_saved_detail_types() -> None:
    result, _provider = await _detect(
        "What specific API rate limit configuration does Ben need?",
        {
            "need_detection_needs_card": "none",
            "need_detection_language_card": "en\nen",
            "need_detection_memory_card": "personal",
            "need_detection_exact_card": "yes",
            "need_detection_shape_card": "slot",
            "need_detection_facets_card": "quantity\ncode\nwording",
            "need_detection_callback_card": "no",
            "need_detection_search_words_card": "Ben\nAPI\nrate limit",
            "need_detection_search_words_other_language_card": "none",
        },
    )

    assert result.exact_recall_needed is True
    assert result.query_type == "slot_fill"
    assert result.exact_facets == [
        ExactFacet.QUANTITY,
        ExactFacet.CODE,
        ExactFacet.OTHER_VERBATIM,
    ]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "Ben",
        "API",
        "rate limit",
    ]


@pytest.mark.asyncio
async def test_broad_list_exact_recall_keeps_original_query_as_single_sub_query() -> None:
    result, _provider = await _detect(
        "What concrete things was Ben planning for Sarah's birthday trip?",
        {
            "need_detection_needs_card": "none",
            "need_detection_language_card": "en\nen",
            "need_detection_memory_card": "personal",
            "need_detection_exact_card": "yes",
            "need_detection_shape_card": "list",
            "need_detection_facets_card": "location\nquantity\nwording",
            "need_detection_callback_card": "no",
            "need_detection_search_words_card": "Ben\nSarah\nbirthday\ntrip",
            "need_detection_search_words_other_language_card": "none",
        },
    )

    assert result.query_type == "broad_list"
    assert result.exact_recall_needed is True
    assert result.sub_queries == [
        "What concrete things was Ben planning for Sarah's birthday trip?"
    ]
    assert result.sparse_query_hints[0].must_keep_terms == [
        "Ben",
        "Sarah",
        "birthday",
        "trip",
    ]


@pytest.mark.asyncio
async def test_public_exact_sounding_world_question_does_not_trigger_memory_exact() -> None:
    result, _provider = await _detect(
        "Who wrote Romeo and Juliet?",
        {
            "need_detection_needs_card": "none",
            "need_detection_language_card": "en\nen",
            "need_detection_memory_card": "world",
            "need_detection_exact_card": "no",
            "need_detection_shape_card": "default",
            "need_detection_facets_card": "none",
            "need_detection_callback_card": "no",
            "need_detection_search_words_card": "Romeo and Juliet",
            "need_detection_search_words_other_language_card": "none",
        },
    )

    assert result.memory_dependence is MemoryDependence.WORLD
    assert result.exact_recall_needed is False
    assert result.query_type == "default"
    assert result.exact_facets == []
