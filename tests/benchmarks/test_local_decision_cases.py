"""Offline capture and parsing checks for frozen local decision cases."""

from __future__ import annotations

from collections import Counter
import html

import pytest

from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.services.llm_client import LLMCompletionResponse
from benchmarks.local_decision_cards.cases import (
    EVALUATION_CASES,
    ROBUSTNESS_CASES,
    ROBUSTNESS_CASES_FROZEN_ON_UTC,
    SMOKE_CASES,
    DecisionCase,
    _context,
    capture_request,
    score_response,
)


def _response(request_model: str, *, text: str = "", choice: str | None = None, question: str | None = None) -> LLMCompletionResponse:
    answers = {}
    if choice is not None and question is not None:
        answers[question] = ChoiceAnswer(
            type="choice", choice=choice, probabilities={choice: 1.0}, confidence=1.0,
        )
    return LLMCompletionResponse(
        provider="offline", model=request_model, output_text=text, choice_answers=answers,
    )


def test_frozen_case_counts_and_labels() -> None:
    all_cases = (*SMOKE_CASES, *EVALUATION_CASES)
    assert len(SMOKE_CASES) == 6
    assert len(EVALUATION_CASES) == 36
    assert len({case.case_id for case in all_cases}) == 42
    assert Counter(case.card for case in SMOKE_CASES) == {
        "context_reuse": 2, "need_detector_exact": 2, "consequence_sentiment": 2,
    }
    assert Counter(case.card for case in EVALUATION_CASES) == {
        "context_reuse": 12, "need_detector_exact": 12, "consequence_sentiment": 12,
    }
    for card in ("context_reuse", "need_detector_exact", "consequence_sentiment"):
        assert Counter(case.language for case in EVALUATION_CASES if case.card == card) == {
            "en": 4, "es": 4, "ca": 4,
        }
    assert all(case.rationale and case.message for case in all_cases)
    assert all(case.previous_message for case in all_cases if case.card != "need_detector_exact")


def test_robustness_cases_were_predeclared_and_are_distinct() -> None:
    assert ROBUSTNESS_CASES_FROZEN_ON_UTC == "2026-09-25"
    assert len(ROBUSTNESS_CASES) == 12
    assert Counter(case.card for case in ROBUSTNESS_CASES) == {
        "context_reuse": 4, "need_detector_exact": 4, "consequence_sentiment": 4,
    }
    assert sum(case.language in {"es", "ca"} for case in ROBUSTNESS_CASES) == 10
    assert not {case.case_id for case in ROBUSTNESS_CASES} & {
        case.case_id for case in (*SMOKE_CASES, *EVALUATION_CASES)
    }
    long_cases = [case for case in ROBUSTNESS_CASES if case.case_id.endswith("-long")]
    assert len(long_cases) == 3
    assert {case.card for case in long_cases} == {
        "context_reuse", "need_detector_exact", "consequence_sentiment",
    }
    assert all(1200 <= len(case.message.split()) <= 1800 for case in long_cases)


@pytest.mark.parametrize("case", [case for case in ROBUSTNESS_CASES if case.case_id.endswith("-long")], ids=lambda case: case.case_id)
async def test_long_robustness_case_reaches_production_builder(case: DecisionCase) -> None:
    request = await capture_request(case, "typesafe/jev-latest")
    assert html.escape(case.message) in request.messages[-1].content
    question = next(iter(request.choice_questions or {}))
    assert score_response(
        case, request, _response(request.model, choice=case.expected, question=question),
    ) == (case.expected, True, True)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("model_spec", ["typesafe/jev-latest", "openrouter/test/model", "local/test-endpoint/test-model"])
async def test_capture_uses_one_real_card_request(case: DecisionCase, model_spec: str) -> None:
    if case.card == "context_reuse" and model_spec.startswith("openrouter/"):
        with pytest.raises(ValueError, match="does not support"):
            await capture_request(case, model_spec)
        return
    request = await capture_request(case, model_spec)
    assert request.model == model_spec
    assert html.escape(case.message) in request.messages[-1].content
    if case.previous_message:
        assert html.escape(case.previous_message) in request.messages[-1].content
    assert request.metadata["user_id"] == "local_decision_user"
    if case.card == "context_reuse":
        assert set(request.choice_questions or {}) == {"context_reuse"}
        assert set(request.choice_questions["context_reuse"].criteria) == {"reuse", "refresh"}
    elif case.card == "need_detector_exact":
        assert request.metadata["need_detection_card"] == "exact"
        assert (request.choice_questions is not None) == model_spec.startswith("typesafe/")
    else:
        assert request.metadata["consequence_detection_card"] == "sentiment"
        assert (request.choice_questions is not None) == model_spec.startswith("typesafe/")


@pytest.mark.parametrize("case", (*SMOKE_CASES, *EVALUATION_CASES), ids=lambda case: case.case_id)
async def test_score_valid_and_invalid_answers(case: DecisionCase) -> None:
    request = await capture_request(case, "typesafe/jev-latest")
    question = next(iter(request.choice_questions or {}))
    decision, valid, matches = score_response(
        case, request,
        _response(request.model, choice=case.expected, question=question),
    )
    assert (decision, valid, matches) == (case.expected, True, True)
    decision, valid, matches = score_response(
        case, request,
        _response(request.model, choice="unsupported", question=question),
    )
    assert (decision, valid, matches) == (None, False, False)


@pytest.mark.parametrize("case", [case for case in SMOKE_CASES if case.card != "context_reuse"], ids=lambda case: case.case_id)
async def test_generative_card_outputs_use_production_parsers(case: DecisionCase) -> None:
    request = await capture_request(case, "openrouter/test/model")
    text = {
        "yes": "yes", "no": "no", "positive": "good", "negative": "bad", "neutral": "mixed",
    }[case.expected]
    assert score_response(case, request, _response(request.model, text=text)) == (
        case.expected, True, True,
    )
    assert score_response(case, request, _response(request.model, text="gibberish")) == (
        None, False, False,
    )


@pytest.mark.parametrize(
    ("case", "expected_role", "prompt_marker"),
    [
        (next(case for case in EVALUATION_CASES if case.case_id == "exact-en-1"), "user", "user: I named the new studio Hazel Room."),
        (next(case for case in EVALUATION_CASES if case.case_id == "sentiment-en-1"), "assistant", '<assistant_message id="assistant_1">Try the smaller image file.</assistant_message>'),
    ],
)
async def test_prior_message_authorship_is_preserved(
    case: DecisionCase, expected_role: str, prompt_marker: str,
) -> None:
    context = _context(case)
    assert [(message.role, message.content) for message in context.recent_messages] == [
        (expected_role, case.previous_message),
    ]
    request = await capture_request(case, "typesafe/jev-latest")
    prompt = request.messages[-1].content
    assert prompt_marker in prompt
    assert prompt.count(case.previous_message) == 1
