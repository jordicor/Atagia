"""Focused contract tests for the reference-only benchmark challengers."""

from __future__ import annotations

import pytest

from atagia.memory.extraction_cards import CandidateDraft
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.source_quote_selection.selector import (
    build_reference_only_request,
    parse_reference_only_output,
    select_references,
)


class CannedClient:
    def __init__(self, choices: list[dict[str, str]]) -> None:
        self.choices = list(choices)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        selected = self.choices.pop(0)
        return LLMCompletionResponse(
            provider="canned",
            model=request.model,
            output_text="ignored",
            choice_answers={
                key: ChoiceAnswer(type="choice", choice=value, probabilities={value: 1.0}, confidence=1.0)
                for key, value in selected.items()
            },
        )


def _candidate(candidate_id: str = "cand_001") -> CandidateDraft:
    return CandidateDraft(candidate_id=candidate_id, canonical_text="The second blue token is relevant")


@pytest.mark.asyncio
async def test_one_native_call_selects_repeated_literal_second_occurrence() -> None:
    source = "blue red blue"
    client = CannedClient([{"cand_001.start": "r3", "cand_001.end": "r3"}])
    result = await select_references(client, model="typesafe/jev-latest", source_text=source, candidates=(_candidate(),))  # type: ignore[arg-type]
    assert len(client.requests) == 1
    assert result["cand_001"].quote(source) == "blue"
    assert result["cand_001"].char_start == 9
    assert client.requests[0].choice_questions is not None
    assert all(len(question.criteria) <= 255 for question in client.requests[0].choice_questions.values())


@pytest.mark.asyncio
async def test_none_requires_both_independent_choices() -> None:
    client = CannedClient([{"cand_001.start": "none", "cand_001.end": "none"}])
    assert await select_references(client, model="typesafe/jev-latest", source_text="blue", candidates=(_candidate(),)) == {"cand_001": None}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_long_source_uses_two_calls_and_cross_block_span() -> None:
    source = " ".join(f"word{index}" for index in range(255))
    client = CannedClient(
        [
            {"cand_001.start": "b1", "cand_001.end": "b2"},
            {"cand_001.start": "r254", "cand_001.end": "r255"},
        ]
    )
    result = await select_references(client, model="typesafe/jev-latest", source_text=source, candidates=(_candidate(),))  # type: ignore[arg-type]
    assert len(client.requests) == 2
    assert all(
        len(question.criteria) <= 255
        for request in client.requests
        for question in (request.choice_questions or {}).values()
    )
    assert "<block_boundary id=\"b2\"" in client.requests[0].messages[1].content
    assert "[r255]" in client.requests[1].messages[1].content
    assert result["cand_001"].quote(source) == "word253 word254"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"cand_001.start": "none", "cand_001.end": "r1"},
        {"cand_001.start": "r2", "cand_001.end": "r1"},
        {"cand_001.start": "r999", "cand_001.end": "r1"},
        {"cand_001.start": "r1"},
    ],
)
async def test_native_invalid_answers_fail(answer: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        await select_references(CannedClient([answer]), model="typesafe/jev-latest", source_text="blue red", candidates=(_candidate(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output",
    ["cand_001 | r2 r1", "cand_001 | r999 r999", "cand_001 | r1", "cand_001 | none\ncand_001 | none"],
)
def test_generative_parser_rejects_invalid_ranges_and_membership(output: str) -> None:
    with pytest.raises(ValueError):
        parse_reference_only_output(output, source_text="blue red", candidates=(_candidate(),))


def test_generative_request_and_parser_use_production_anchors() -> None:
    request = build_reference_only_request(model="openai/test", source_text="blue red blue", candidates=(_candidate(),))
    assert "[r3]blue" in request.messages[1].content
    result = parse_reference_only_output("cand_001 | r3 r3", source_text="blue red blue", candidates=(_candidate(),))
    assert result["cand_001"].char_start == 9
