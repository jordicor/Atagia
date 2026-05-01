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

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import NEED_DETECTOR_PROMPT_TEMPLATE, NeedDetector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExactFacet,
    ExtractionContextMessage,
    ExtractionConversationContext,
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
        user_language_profile=[],
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
        user_language_profile=[],
    )

    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.DATE]


def test_need_detector_prompt_describes_language_agnostic_rules_without_real_examples() -> None:
    """The prompt should stay abstract, multilingual, and benchmark-agnostic."""
    template = NEED_DETECTOR_PROMPT_TEMPLATE
    lowered = template.lower()

    assert "The user message may be written in any language." in template
    assert "Do not rely on English keywords." in template
    assert "specific language or phrasing" in lowered
    assert "<person_name>" in template
    assert "<street_address>" in template
    assert "<quantity_with_unit>" in template
    assert "exact_recall_needed" in lowered
    assert "exact_facets" in lowered
    assert "multiple concrete named items" in lowered
