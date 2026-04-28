"""Tests for compaction summary privacy validation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from atagia.memory.summary_privacy_judge import (
    SummaryPrivacyJudge,
    SummaryPrivacyVerdict,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)


class PrivacyJudgeProvider(LLMProvider):
    name = "summary-privacy-judge-tests"

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("No queued privacy judge output left")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.outputs.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in summary privacy judge tests")


def _judge(provider: PrivacyJudgeProvider) -> SummaryPrivacyJudge:
    return SummaryPrivacyJudge(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        judge_model="judge-model",
        refiner_model="refiner-model",
        timeout_seconds=1.0,
        max_source_chars=1000,
    )


@pytest.mark.asyncio
async def test_summary_privacy_verdict_ignores_provider_extra_fields() -> None:
    provider = PrivacyJudgeProvider(
        [
            {
                "is_safe_to_publish": True,
                "reasoning": "Safe non-sensitive anchors only.",
                "unsafe_detail_categories": [],
                "required_changes": [],
                "provider_notes": "ignored",
            }
        ]
    )

    verdict = await _judge(provider).judge(
        user_id="usr_1",
        summary_kind="episode",
        summary_text="The user prefers concise implementation notes.",
        retrieval_constraints=[],
        index_text=None,
        source_texts=["The user prefers concise implementation notes."],
        source_privacy_max=0,
    )

    assert verdict.is_safe_to_publish is True
    assert verdict.reasoning == "Safe non-sensitive anchors only."
    assert not hasattr(verdict, "provider_notes")


@pytest.mark.asyncio
async def test_summary_privacy_refinement_ignores_provider_extra_fields() -> None:
    provider = PrivacyJudgeProvider(
        [
            {
                "summary_text": (
                    "The user has a private access detail that should not be surfaced."
                ),
                "retrieval_constraints": ["Do not surface raw private access details."],
                "reasoning": "Removed the sensitive literal.",
                "removed_or_changed": ["Removed raw access detail."],
                "source_evidence": ["ignored"],
            }
        ]
    )
    verdict = SummaryPrivacyVerdict(
        is_safe_to_publish=False,
        reasoning="Contains a sensitive literal.",
        unsafe_detail_categories=["secret"],
        required_changes=["Remove the literal."],
    )

    refinement = await _judge(provider).refine(
        user_id="usr_1",
        summary_kind="episode",
        summary_text="The user's access code is 1234.",
        retrieval_constraints=[],
        source_texts=["The user's access code is 1234."],
        source_privacy_max=3,
        verdict=verdict,
    )

    assert (
        refinement.summary_text
        == "The user has a private access detail that should not be surfaced."
    )
    assert refinement.retrieval_constraints == ["Do not surface raw private access details."]
    assert not hasattr(refinement, "source_evidence")
