"""Tests for the benchmark judge scorer."""

from __future__ import annotations

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

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=next(self._outputs),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in scorer tests")


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
