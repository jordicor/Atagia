"""Five-level memory confidence Score contract, entirely offline."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from atagia.diagnostics.recorder import DiagnosticRecorder
from atagia.models.schemas_decisions import ChoiceQuestion, ScoreQuestion
from atagia.services.llm_client import (
    ConfigurationError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMMessage,
    LLMProvider,
    RetryPolicy,
)
from atagia.services.providers.typesafe import TypeSafeProvider


RUBRIC = (
    "The source does not support or contradicts the candidate.",
    "There is a weak signal but essential information is missing.",
    "The source partly supports the candidate with material ambiguity.",
    "The source supports the candidate with only minor uncertainty.",
    "The candidate faithfully represents explicit, unambiguous source content.",
)


def _questions() -> dict[str, ScoreQuestion]:
    return {
        "candidate_a": ScoreQuestion(
            instructions="How strongly does the source support candidate A as worded, including attribution and conditions?",
            criteria=RUBRIC,
        ),
        "candidate_b": ScoreQuestion(
            instructions="How strongly does the source support candidate B as worded, including attribution and conditions?",
            criteria=RUBRIC,
        ),
    }


def _payload() -> dict[str, object]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "candidate_a": {
                "type": "score",
                "score": 2.6,
                "legend": {str(index): level for index, level in enumerate(RUBRIC)},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.4, "3": 0.6, "4": 0.0},
                "confidence": 0.22,
            }
        },
        "usage": {"input_tokens": 74, "output_tokens": 13},
    }


async def test_typed_score_retains_weighted_value_and_provider_certainty(tmp_path) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_payload())

    recorder = DiagnosticRecorder(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
            diagnostic_recorder=recorder,
        )
        result = await client.complete_score_questions(
            model="typesafe/jev-1.13.0",
            messages=[LLMMessage(role="user", content="Synthetic source and candidate A")],
            questions={"candidate_a": _questions()["candidate_a"]},
            metadata={"purpose": "memory_extraction_confidence_card", "user_id": "test-user"},
        )
    recorder.close()

    assert len(seen) == 1
    assert seen[0]["model"] == "jev-1.13.0"
    assert seen[0]["questions"] == {
        "candidate_a": _questions()["candidate_a"].model_dump(mode="json")
    }
    assert "temperature" not in seen[0] and "max_tokens" not in seen[0]
    answer = result["candidate_a"]
    assert answer.normalized_score == pytest.approx(0.65)
    assert answer.typed_answer.score == pytest.approx(2.6)
    assert answer.typed_answer.confidence == pytest.approx(0.22)
    assert answer.typed_answer.probabilities["3"] == pytest.approx(0.6)
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_text().splitlines()]
    attempts = [event for event in events if event["kind"] == "provider_attempt"]
    assert len(attempts) == 2
    assert attempts[0]["phase"] == "start" and attempts[1]["status"] == "success"
    assert attempts[1]["data"]["usage"] == {"input_tokens": 74, "output_tokens": 13}


async def test_prepared_scores_share_one_native_state_and_attempt() -> None:
    payload = _payload()
    payload["answers"]["candidate_b"] = {
        **payload["answers"]["candidate_a"],
        "score": 1.0,
        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0},
    }
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert set(json.loads(request.content)["questions"]) == {"candidate_a", "candidate_b"}
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        result = await client.complete_score_questions(
            model="typesafe/jev-1.13.0",
            messages=[LLMMessage(role="user", content="One synthetic source for both candidates")],
            questions=_questions(),
            metadata={"purpose": "memory_extraction_confidence_card"},
        )
    assert calls == 1
    assert result["candidate_a"].normalized_score == pytest.approx(0.65)
    assert result["candidate_b"].normalized_score == pytest.approx(0.25)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda answer: answer["legend"].update({"4": "Wrong level"}),
        lambda answer: answer["legend"].pop("4"),
        lambda answer: answer["probabilities"].pop("4"),
        lambda answer: answer["probabilities"].update({"2": 0.2}),
        lambda answer: answer.update({"score": 3.4}),
        lambda answer: answer.update({"score": 4.1}),
        lambda answer: answer.update({"score": "2.6"}),
        lambda answer: answer["probabilities"].update({"3": "0.6"}),
        lambda answer: answer.update({"type": "choice"}),
    ],
)
async def test_invalid_score_payload_fails_without_repair(mutate) -> None:
    payload = _payload()
    mutate(payload["answers"]["candidate_a"])
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=3),
        )
        with pytest.raises(LLMError):
            await client.complete_score_questions(
                model="typesafe/jev-1.13.0",
                messages=[LLMMessage(role="user", content="Synthetic state")],
                questions={"candidate_a": _questions()["candidate_a"]},
                metadata={"purpose": "memory_extraction_confidence_card"},
            )
    assert calls == 1


async def test_mixed_typed_question_kinds_are_rejected_before_transport() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("No mixed request should reach transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = TypeSafeProvider("test-key", client=http)
        with pytest.raises(ConfigurationError, match="either choice_questions or score_questions"):
            await provider.complete(
                LLMCompletionRequest(
                    model="jev-1.13.0",
                    messages=[LLMMessage(role="user", content="Synthetic state")],
                    choice_questions={
                        "kind": ChoiceQuestion(
                            instructions="Choose kind.", criteria={"a": None, "b": None}
                        )
                    },
                    score_questions={"confidence": _questions()["candidate_a"]},
                )
            )


class ScalarProvider(LLMProvider):
    name = "openai"

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs[request.metadata["stage"]],
        )


async def test_llm_score_is_continuous_and_uses_the_same_five_anchors() -> None:
    provider = ScalarProvider({"candidate_a": "0.73"})
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    result = await client.complete_score_questions(
        model="openai/test-model",
        messages=[LLMMessage(role="user", content="Synthetic source and candidate A")],
        questions={"candidate_a": _questions()["candidate_a"]},
        metadata={"purpose": "memory_extraction_confidence_card"},
    )
    assert result["candidate_a"].normalized_score == pytest.approx(0.73)
    assert result["candidate_a"].typed_answer is None
    assert provider.requests[0].max_output_tokens == 8192
    assert provider.requests[0].score_questions is None
    assert provider.requests[0].messages[0].content.count("Intermediate values are allowed") == 1
    assert all(level in provider.requests[0].messages[0].content for level in RUBRIC)


@pytest.mark.parametrize("output", ["", "0.5 extra", "NaN", "-0.1", "1.1"])
async def test_invalid_llm_scalar_fails_without_repair(output: str) -> None:
    provider = ScalarProvider({"candidate_a": output})
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    with pytest.raises(LLMError):
        await client.complete_score_questions(
            model="openai/test-model",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions={"candidate_a": _questions()["candidate_a"]},
            metadata={"purpose": "memory_extraction_confidence_card"},
        )
    assert len(provider.requests) == 1


async def test_score_requests_reject_a_choice_only_provider_before_dispatch() -> None:
    class ChoiceOnlyProvider(LLMProvider):
        name = "openai"
        supports_choices = True

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            raise AssertionError("Score dispatch must fail before this call")

    client = LLMClient(providers=[ChoiceOnlyProvider()])
    with pytest.raises(ConfigurationError, match="does not support typed scores"):
        await client.complete(
            LLMCompletionRequest(
                model="openai/test-model",
                messages=[LLMMessage(role="user", content="Synthetic state")],
                score_questions={"candidate_a": _questions()["candidate_a"]},
            )
        )


async def test_failed_scalar_sibling_is_cancelled_and_joined() -> None:
    cancelled = asyncio.Event()

    class FailingProvider(LLMProvider):
        name = "openai"

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            if request.metadata["stage"] == "candidate_a":
                await asyncio.sleep(0)
                raise LLMError("synthetic score failure")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("The sibling should have been cancelled")

    client = LLMClient(providers=[FailingProvider()], retry_policy=RetryPolicy(attempts=1))
    with pytest.raises(LLMError, match="synthetic score failure"):
        await client.complete_score_questions(
            model="openai/test-model",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions=_questions(),
            metadata={"purpose": "memory_extraction_confidence_card"},
        )
    assert cancelled.is_set()


def test_score_rubric_requires_exactly_five_levels() -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Rate support.", criteria=RUBRIC[:4])
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Rate support.", criteria=(*RUBRIC, "sixth"))
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Rate support.", criteria=(*RUBRIC[:4], " "))
