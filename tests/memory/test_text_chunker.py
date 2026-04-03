"""Tests for oversized-text chunk planning."""

from __future__ import annotations

import json
import re

import pytest

from atagia.memory.chunking_config import LEVEL1_MAX_CHUNK_TOKENS
from atagia.memory.text_chunker import TextChunker
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)


class ChunkerProvider(LLMProvider):
    name = "chunker-tests"

    def __init__(self, responder) -> None:
        self._responder = responder
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self._responder(request),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in chunker tests: {request.model}")


def _chunker(responder) -> tuple[TextChunker, ChunkerProvider]:
    provider = ChunkerProvider(responder)
    chunker = TextChunker(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        model="chunker-test-model",
    )
    return chunker, provider


def test_split_by_natural_separators_respects_transcript_boundaries() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))
    first = "Speaker: " + ("alpha " * 120)
    second = "[00:01:00.000 --> 00:01:05.000]\n" + ("beta " * 120)
    text = f"{first}\n\n{second}"

    segments = chunker.split_by_natural_separators(text)

    assert len(segments) == 2
    assert segments[0].startswith("Speaker:")
    assert segments[1].startswith("[00:01:00.000 --> 00:01:05.000]")


def test_split_by_natural_separators_recognizes_case_insensitive_annotations() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))
    text = "[Music]\n" + ("alpha " * 120) + "\n\n[Inaudible]\n" + ("beta " * 120)

    segments = chunker.split_by_natural_separators(text)

    assert len(segments) == 2
    assert segments[0].startswith("[Music]")
    assert segments[1].startswith("[Inaudible]")


def test_split_by_natural_separators_returns_empty_for_blank_text() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))

    assert chunker.split_by_natural_separators("   \n\n ") == []


def test_split_by_natural_separators_keeps_single_block_without_separators() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))
    text = "alpha " * 200

    segments = chunker.split_by_natural_separators(text)

    assert segments == [text.strip()]


def test_split_by_natural_separators_merges_small_segments() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))
    text = "A:\nshort\n\nB:\nsmall\n\nC:\n" + ("large " * 150)

    segments = chunker.split_by_natural_separators(text)

    assert len(segments) == 1
    assert "short" in segments[0]
    assert "small" in segments[0]
    assert "large" in segments[0]


@pytest.mark.asyncio
async def test_plan_chunks_marks_level0_segments_when_threshold_exceeded() -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps({"cut_markers": []}))
    text = (
        "Speaker: " + ("alpha " * 120)
        + "\n\n"
        + "Responder: " + ("beta " * 120)
    )

    plan = await chunker.plan_chunks(text, threshold_tokens=20)

    assert plan.chunked is True
    assert plan.fallback_count == 0
    assert len(plan.chunks) == 2
    assert [chunk.chunking_strategy for chunk in plan.chunks] == ["level0", "level0"]
    assert [chunk.chunk_index for chunk in plan.chunks] == [1, 2]
    assert all(chunk.chunk_count == 2 for chunk in plan.chunks)


@pytest.mark.asyncio
async def test_chunk_with_ai_level1_returns_valid_chunks() -> None:
    def responder(request: LLMCompletionRequest) -> str:
        marked_text = request.messages[1].content.split("<marked_text>\n", 1)[1].split(
            "\n</marked_text>",
            1,
        )[0]
        markers = re.findall(r"<<<BM_[A-Z0-9]{8}_\d+>>>", marked_text)
        return json.dumps({"cut_markers": [markers[4]]})

    chunker, provider = _chunker(responder)
    text = ("topic alpha " * 1800) + ("topic beta " * 1800)

    chunks = await chunker.chunk_with_ai_level1(text)

    assert len(chunks) == 2
    assert 4000 <= chunker.estimate_tokens(chunks[0]) <= LEVEL1_MAX_CHUNK_TOKENS
    assert 4000 <= chunker.estimate_tokens(chunks[1]) <= LEVEL1_MAX_CHUNK_TOKENS
    assert provider.requests[0].metadata["purpose"] == "text_chunking_level1"


@pytest.mark.asyncio
async def test_chunk_with_ai_level1_merges_soft_min_chunk_instead_of_rejecting() -> None:
    def responder(request: LLMCompletionRequest) -> str:
        marked_text = request.messages[1].content.split("<marked_text>\n", 1)[1].split(
            "\n</marked_text>",
            1,
        )[0]
        markers = re.findall(r"<<<BM_[A-Z0-9]{8}_\d+>>>", marked_text)
        return json.dumps({"cut_markers": [markers[2], markers[10]]})

    chunker, _provider = _chunker(responder)
    text = ("topic alpha " * 2500) + ("topic beta " * 2500) + ("topic gamma " * 2500)

    chunks = await chunker.chunk_with_ai_level1(text)

    assert len(chunks) == 2
    assert all(chunker.estimate_tokens(chunk) <= LEVEL1_MAX_CHUNK_TOKENS for chunk in chunks)
    assert all(chunker.estimate_tokens(chunk) >= 4000 for chunk in chunks)


@pytest.mark.asyncio
async def test_plan_chunks_falls_back_to_deterministic_splits_when_level1_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunker, _provider = _chunker(lambda _request: "not-json")
    text = "narrative " * (LEVEL1_MAX_CHUNK_TOKENS + 2000)

    with caplog.at_level("WARNING"):
        plan = await chunker.plan_chunks(text)

    assert plan.chunked is True
    assert plan.fallback_count == 1
    assert len(plan.chunks) >= 2
    assert all(chunk.chunking_strategy == "deterministic_fallback" for chunk in plan.chunks)
    assert all(chunk.level1_failure_reason == "malformed_json" for chunk in plan.chunks)
    assert all(chunker.estimate_tokens(chunk.text) <= LEVEL1_MAX_CHUNK_TOKENS for chunk in plan.chunks)
    assert "Level 1 chunking fallback activated" in caplog.text
    assert "segment_index=1" in caplog.text


@pytest.mark.asyncio
async def test_plan_chunks_falls_back_when_level1_returns_invalid_json_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunker, _provider = _chunker(lambda _request: json.dumps(["not", "a", "dict"]))
    text = "narrative " * (LEVEL1_MAX_CHUNK_TOKENS + 2000)

    with caplog.at_level("WARNING"):
        plan = await chunker.plan_chunks(text)

    assert plan.chunked is True
    assert plan.fallback_count == 1
    assert all(chunk.chunking_strategy == "deterministic_fallback" for chunk in plan.chunks)
    assert all(chunk.level1_failure_reason == "invalid_payload_type" for chunk in plan.chunks)


@pytest.mark.asyncio
async def test_chunk_with_ai_level1_regenerates_marker_prefix_on_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(request: LLMCompletionRequest) -> str:
        marked_text = request.messages[1].content.split("<marked_text>\n", 1)[1].split(
            "\n</marked_text>",
            1,
        )[0]
        assert "<<<BM_DEADBEEF_" not in marked_text
        assert "<<<LITERAL_BM_DEADBEEF_" in marked_text
        markers = re.findall(r"<<<BM_[A-Z0-9]{8}_\d+>>>", marked_text)
        assert markers
        return json.dumps({"cut_markers": [markers[4]]})

    chunker, _provider = _chunker(responder)
    token_values = iter(["deadbeef", "cafebabe"])
    monkeypatch.setattr("atagia.memory.text_chunker.secrets.token_hex", lambda _n: next(token_values))
    text = "<<<BM_DEADBEEF_1>>> " + (("topic alpha " * 1800) + ("topic beta " * 1800))

    chunks = await chunker.chunk_with_ai_level1(text)

    assert len(chunks) == 2
