"""Tests for the three benchmark judge protocols."""

from __future__ import annotations

import json
from typing import Any

import pytest

from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.scorer import (
    FAILURE_REASONS,
    JudgeProtocol,
    JudgeVerdict,
    LLMJudgeScorer,
)


class _FakeLLMClient:
    """Capture the judge request and return a scripted response."""

    def __init__(self, payload: dict[str, Any], model: str = "kimi-k2.7-code-highspeed") -> None:
        self._payload = payload
        self._model = model
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider="kimi",
            model=self._model,
            output_text=json.dumps(self._payload),
        )


_EVIDENCE = [
    {
        "turn_id": "SYNJ:1",
        "session_id": "session_1",
        "timestamp": "2025-09-23T10:00:00",
        "speaker": "Mira",
        "text": "The observatory reopened this morning.",
    }
]


@pytest.mark.asyncio
async def test_source_aware_strict_prompt_and_parse() -> None:
    client = _FakeLLMClient({"verdict": 1, "reasoning": "matches", "failure_reason": None})
    scorer = LLMJudgeScorer(client, "kimi/kimi-k2.7-code-highspeed")
    verdict = await scorer.evaluate(
        question="On which date did the observatory reopen?",
        prediction="The observatory reopened on 23 September 2025.",
        ground_truth="23 September 2025",
        source_evidence=_EVIDENCE,
    )
    assert verdict.protocol == JudgeProtocol.SOURCE_AWARE_STRICT.value
    assert verdict.score == 1
    assert verdict.missing_info is None  # decomposition only for memory_quality
    system_prompt = client.requests[0].messages[0].content
    # The strict verdict criteria stay unchanged.
    assert "Reject unsupported or contradictory added facts" in system_prompt
    assert client.requests[0].metadata["judge_protocol"] == "source_aware_strict"


@pytest.mark.asyncio
async def test_gold_only_lenient_ignores_evidence() -> None:
    client = _FakeLLMClient({"verdict": 1, "reasoning": "contains gold"})
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.GOLD_ONLY_LENIENT
    )
    verdict = await scorer.evaluate(
        question="On which date did the observatory reopen?",
        prediction="The observatory reopened on 23 September 2025, plus other details.",
        ground_truth="23 September 2025",
        source_evidence=_EVIDENCE,
    )
    assert verdict.score == 1
    system_prompt = client.requests[0].messages[0].content
    user_prompt = client.requests[0].messages[1].content
    assert "lenient gold-only protocol" in system_prompt
    assert "Do not penalize additional" in system_prompt
    # Gold-only never shows the source evidence.
    assert "observatory reopened this morning" not in user_prompt


@pytest.mark.asyncio
async def test_memory_quality_requires_transcript() -> None:
    client = _FakeLLMClient({"verdict": 1})
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.MEMORY_QUALITY
    )
    with pytest.raises(ValueError, match="conversation_transcript"):
        await scorer.evaluate(
            question="When?",
            prediction="x",
            ground_truth="y",
            source_evidence=_EVIDENCE,
        )


@pytest.mark.asyncio
async def test_memory_quality_decomposition_parsed() -> None:
    client = _FakeLLMClient(
        {
            "verdict": 0,
            "missing_info": False,
            "false_addition": True,
            "misattribution": False,
            "true_addition_only": False,
            "failure_reason": "unsupported_addition",
            "reasoning": "adds a fabricated fact",
        }
    )
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.MEMORY_QUALITY
    )
    verdict = await scorer.evaluate(
        question="On which date did the observatory reopen?",
        prediction="It reopened on 23 September 2025 and the curator moved to Lisbon.",
        ground_truth="23 September 2025",
        source_evidence=_EVIDENCE,
        conversation_transcript=(
            "[session_1] Mira: The observatory reopened on 23 September 2025."
        ),
    )
    assert verdict.score == 0
    assert verdict.missing_info is False
    assert verdict.false_addition is True
    assert verdict.misattribution is False
    assert verdict.true_addition_only is False
    assert verdict.failure_reason == "unsupported_addition"
    system_prompt = client.requests[0].messages[0].content
    user_prompt = client.requests[0].messages[1].content
    assert "MEMORY QUALITY" in system_prompt
    # The transcript is placed first so it forms a cache-reusable prefix.
    assert user_prompt.startswith("Full source conversation:")


@pytest.mark.asyncio
async def test_unparseable_response_fails_closed() -> None:
    client = _FakeLLMClient({}, model="kimi-k2.7-code-highspeed")
    client._payload = {}  # type: ignore[attr-defined]

    class _BadClient(_FakeLLMClient):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            return LLMCompletionResponse(
                provider="kimi", model="kimi-k2.7-code-highspeed", output_text="not json"
            )

    bad = _BadClient({})
    scorer = LLMJudgeScorer(bad, "kimi/kimi-k2.7-code-highspeed")
    verdict = await scorer.evaluate(
        question="q", prediction="p", ground_truth="g", source_evidence=_EVIDENCE
    )
    assert verdict.score == 0
    assert verdict.failure_reason == "other"


def test_failure_reason_normalization() -> None:
    verdict = JudgeVerdict(
        score=0,
        reasoning="r",
        judge_model="m",
        protocol="source_aware_strict",
        failure_reason="omission",
    )
    assert verdict.failure_reason in FAILURE_REASONS


@pytest.mark.asyncio
async def test_score_downcasts_and_records_protocol() -> None:
    client = _FakeLLMClient({"verdict": 1, "reasoning": "ok"})
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.GOLD_ONLY_LENIENT
    )
    result = await scorer.score("q", "p", "g", source_evidence=_EVIDENCE)
    assert result.score == 1
    assert result.protocol == "gold_only_lenient"
