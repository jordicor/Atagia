"""Tests for need-detection retrieval shaping: search words and retrieval levels."""

from __future__ import annotations

import pytest

from atagia.memory.need_detector import NeedDetector
from atagia.services.llm_client import LLMClient
from tests.memory.test_need_detector import (
    CannedCardProvider,
    _clock,
    _context,
    _resolved_policy,
    _settings,
)


def _outputs(*, shape: str, search_words: str) -> dict[str, str]:
    return {
        "need_detection_needs_card": "none",
        "need_detection_query_language_card": "en",
        "need_detection_answer_language_card": "en",
        "need_detection_memory_card": "personal",
        "need_detection_exact_card": "yes",
        "need_detection_shape_card": shape,
        "need_detection_facets_card": "none",
        "need_detection_callback_card": "no",
        "need_detection_search_words_card": search_words,
    }


async def _detect(outputs: dict[str, str], message_text: str):
    provider = CannedCardProvider(outputs)
    detector = NeedDetector(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=_clock(),
        settings=_settings(),
    )
    return await detector.detect(
        message_text=message_text,
        role="user",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )


@pytest.mark.asyncio
async def test_broad_list_queries_include_episode_and_theme_levels() -> None:
    detected = await _detect(
        _outputs(shape="list", search_words="pottery\nstudio"),
        "What are all the pottery studios I have mentioned?",
    )

    assert detected.query_type == "broad_list"
    assert detected.retrieval_levels == [0, 1, 2]


@pytest.mark.asyncio
async def test_narrow_queries_stay_on_fact_level() -> None:
    detected = await _detect(
        _outputs(shape="slot", search_words="locker\ncode"),
        "What was the locker code you recommended?",
    )

    assert detected.query_type == "slot_fill"
    assert detected.retrieval_levels == [0]


@pytest.mark.asyncio
async def test_search_words_become_the_primary_fts_clause() -> None:
    detected = await _detect(
        _outputs(shape="slot", search_words="locker\ncode"),
        "What was the locker code you recommended?",
    )

    hint = detected.sparse_query_hints[0]
    assert hint.fts_phrase == "locker code"
    assert hint.must_keep_terms == ["locker", "code"]


@pytest.mark.asyncio
async def test_fts_phrase_falls_back_to_question_without_search_words() -> None:
    message_text = "What was the locker code you recommended?"
    detected = await _detect(
        _outputs(shape="slot", search_words="none"),
        message_text,
    )

    assert detected.sparse_query_hints[0].fts_phrase == message_text
