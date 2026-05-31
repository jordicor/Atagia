"""Tests for the benchmark judge scorer."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from benchmarks.scorer import LLMJudgeScorer
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)


class JudgeProvider(LLMProvider):
    name = "judge-tests"

    def __init__(self, outputs: Iterator[str]) -> None:
        self._outputs = iter(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=next(self._outputs),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in scorer tests")


def test_extract_json_payload_accepts_valid_json() -> None:
    payload = LLMJudgeScorer._extract_json_payload('{"verdict": 1, "reasoning": "Matches."}')

    assert payload == {"verdict": 1, "reasoning": "Matches."}


def test_extract_json_payload_accepts_markdown_fenced_json() -> None:
    payload = LLMJudgeScorer._extract_json_payload(
        '```json\n{"verdict": 1, "reasoning": "Matches."}\n```'
    )

    assert payload == {"verdict": 1, "reasoning": "Matches."}


def test_extract_json_payload_returns_none_for_garbage_input() -> None:
    assert LLMJudgeScorer._extract_json_payload("not-json") is None


@pytest.mark.asyncio
async def test_judge_correct() -> None:
    scorer = LLMJudgeScorer(
        LLMClient(provider_name="judge-tests", providers=[JudgeProvider(iter(['{"verdict": 1, "reasoning": "Matches."}']))]),
        judge_model="judge-model",
    )

    result = await scorer.score("Question?", "Prediction", "Ground truth")

    assert result.score == 1
    assert result.reasoning == "Matches."
    assert result.judge_model == "judge-model"


@pytest.mark.asyncio
async def test_judge_incorrect() -> None:
    scorer = LLMJudgeScorer(
        LLMClient(provider_name="judge-tests", providers=[JudgeProvider(iter(['{"verdict": 0, "reasoning": "Wrong fact."}']))]),
        judge_model="judge-model",
    )

    result = await scorer.score("Question?", "Prediction", "Ground truth")

    assert result.score == 0
    assert result.reasoning == "Wrong fact."


@pytest.mark.asyncio
async def test_judge_includes_official_source_evidence_when_available() -> None:
    provider = JudgeProvider(
        iter(['{"verdict": 1, "reasoning": "Source-supported year."}'])
    )
    scorer = LLMJudgeScorer(
        LLMClient(provider_name="judge-tests", providers=[provider]),
        judge_model="judge-model",
    )

    result = await scorer.score(
        question="Is Elena expecting a baby, and if so, when?",
        prediction="Elena is expecting a baby in May 2026.",
        ground_truth="Elena is expecting a baby in May.",
        source_evidence=[
            {
                "turn_id": "rosa-04-t07",
                "timestamp": "2025-12-02T11:07:00",
                "speaker": "user",
                "text": "She is expecting a baby. She is due in May.",
            }
        ],
    )

    assert result.score == 1
    assert provider.requests[0].metadata["source_evidence_used"] is True
    system_prompt = provider.requests[0].messages[0].content
    assert "official source evidence" in system_prompt.lower()
    assert "Temporal specificity rule" in system_prompt
    assert "next applicable future occurrence" in system_prompt
    assert "timestamp=2025-12-02T11:07:00" in provider.requests[0].messages[1].content
    assert "May 2026" in provider.requests[0].messages[1].content


@pytest.mark.asyncio
async def test_judge_accepts_string_and_boolean_verdicts() -> None:
    scorer = LLMJudgeScorer(
        LLMClient(
            provider_name="judge-tests",
            providers=[
                JudgeProvider(
                    iter(
                        [
                            '{"verdict": "1", "reasoning": "String verdict."}',
                            '{"verdict": true, "reasoning": "Boolean verdict."}',
                        ]
                    )
                )
            ],
        ),
        judge_model="judge-model",
    )

    first = await scorer.score("Question?", "Prediction", "Ground truth")
    second = await scorer.score("Question?", "Prediction", "Ground truth")

    assert first.score == 1
    assert second.score == 1


@pytest.mark.asyncio
async def test_judge_malformed_response() -> None:
    scorer = LLMJudgeScorer(
        LLMClient(provider_name="judge-tests", providers=[JudgeProvider(iter(["not-json"]))]),
        judge_model="judge-model",
    )

    result = await scorer.score("Question?", "Prediction", "Ground truth")

    assert result.score == 0
    assert "could not be parsed" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_judge_malformed_response_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    scorer = LLMJudgeScorer(
        LLMClient(provider_name="judge-tests", providers=[JudgeProvider(iter(["not-json"]))]),
        judge_model="judge-model",
    )

    with caplog.at_level(logging.WARNING):
        result = await scorer.score("Question?", "Prediction", "Ground truth")

    assert result.score == 0
    assert "Judge response could not be parsed: not-json" in caplog.text
