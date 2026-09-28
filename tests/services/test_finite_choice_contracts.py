"""Shared finite-choice execution through the normal LLM client."""

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
    LLMPolicyBlockedError,
    LLMMessage,
    LLMProvider,
    RetryPolicy,
)
from atagia.services.providers.typesafe import TypeSafeProvider
from tests.services.providers.test_typesafe import _answer_payload


class CannedProvider(LLMProvider):
    name = "openai"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        question_id = request.metadata["stage"]
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.answers[question_id],
        )


def _questions() -> dict[str, ChoiceQuestion]:
    return {
        "candidate_a": ChoiceQuestion(
            instructions="Which kind describes candidate A in the supplied source?",
            criteria={"evidence": "A directly stated fact", "belief": "A stated belief"},
        ),
        "candidate_b": ChoiceQuestion(
            instructions="Which kind describes candidate B in the supplied source?",
            criteria={"evidence": "A directly stated fact", "belief": "A stated belief"},
        ),
    }


async def test_prepared_questions_use_the_same_options_for_llm_and_typesafe() -> None:
    questions = _questions()
    state = [LLMMessage(role="user", content="Synthetic source and candidates A and B")]
    metadata = {"purpose": "memory_extraction_kind_card", "user_id": "test-user"}
    provider = CannedProvider({"candidate_a": "belief", "candidate_b": "evidence"})
    llm = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    llm_answers = await llm.complete_choice_questions(
        model="openai/test-model", messages=state, questions=questions, metadata=metadata,
    )

    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_answer_payload(llm_answers, ["evidence", "belief"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        typed = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        typed_answers = await typed.complete_choice_questions(
            model="typesafe/jev-latest", messages=state, questions=questions, metadata=metadata,
        )

    assert typed_answers == llm_answers
    assert len(sent) == 1
    assert sent[0]["questions"] == {
        key: question.model_dump(mode="json") for key, question in questions.items()
    }
    assert sent[0]["state"] == [{"role": "user", "content": state[0].content}]
    assert all(request.choice_questions is None for request in provider.requests)
    assert {request.metadata["stage"] for request in provider.requests} == set(questions)
    for request in provider.requests:
        question = questions[request.metadata["stage"]]
        assert question.instructions in request.messages[0].content
        assert request.messages[1:] == state
        assert all(option in request.messages[0].content for option in question.criteria)


@pytest.mark.parametrize("answer", ["unknown", '"evidence" "belief"', "ev!idence", '"evidence'])
async def test_malformed_llm_choice_fails_without_technical_repair(answer: str) -> None:
    provider = CannedProvider({"candidate_a": answer})
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    with pytest.raises(LLMError, match="unknown option"):
        await client.complete_choice_questions(
            model="openai/test-model",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions={"candidate_a": _questions()["candidate_a"]},
            metadata={"purpose": "memory_extraction_kind_card"},
        )
    assert len(provider.requests) == 1


@pytest.mark.parametrize("answer", ['"evidence"', "\u201cevidence\u201d", "**`evidence`**", "```text\nevidence\n```"])
async def test_llm_choice_accepts_outer_formatting_without_another_call(answer: str) -> None:
    provider = CannedProvider({"candidate_a": answer})
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    assert await client.complete_choice_questions(
        model="openai/test-model",
        messages=[LLMMessage(role="user", content="Synthetic state")],
        questions={"candidate_a": _questions()["candidate_a"]},
        metadata={"purpose": "memory_extraction_kind_card"},
    ) == {"candidate_a": "evidence"}
    assert len(provider.requests) == 1


@pytest.mark.parametrize("answer,valid", [('"0.8"', True), ('"-0.8"', False), ("1. 0.99", False), ('"NaN"', False)])
async def test_llm_score_keeps_numeric_validation_after_unwrapping(answer: str, valid: bool) -> None:
    provider = CannedProvider({"candidate_a": answer})
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
    call = client.complete_score_questions(
        model="openai/test-model",
        messages=[LLMMessage(role="user", content="Synthetic state")],
        questions={"candidate_a": ScoreQuestion(
            instructions="Rate source support.",
            criteria=("Absent", "Weak", "Partial", "Strong", "Explicit"),
        )},
        metadata={"purpose": "memory_extraction_confidence_card"},
    )
    if valid:
        assert (await call)["candidate_a"].normalized_score == 0.8
    else:
        with pytest.raises(LLMError):
            await call
    assert len(provider.requests) == 1


async def test_finite_llm_failure_does_not_use_configured_fallback() -> None:
    seen: list[str] = []

    class FailingProvider(LLMProvider):
        name = "openai"

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            seen.append(request.model)
            raise LLMPolicyBlockedError("synthetic provider policy block")

    client = LLMClient(
        providers=[FailingProvider()],
        retry_policy=RetryPolicy(attempts=1),
        intimacy_fallback_models={"extraction_kind": "openai/backup"},
    )
    with pytest.raises(LLMPolicyBlockedError, match="synthetic provider policy block"):
        await client.complete_choice_questions(
            model="openai/primary",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions={"candidate_a": _questions()["candidate_a"]},
            metadata={
                "purpose": "memory_extraction_kind_card",
                "atagia_intimacy_context": True,
            },
        )
    assert seen == ["primary"]


async def test_finite_llm_keeps_output_floor_and_proactive_intimacy_route() -> None:
    ordinary = CannedProvider({"candidate_a": "belief"})
    client = LLMClient(providers=[ordinary], retry_policy=RetryPolicy(attempts=1))
    await client.complete(
        LLMCompletionRequest(
            model="openai/primary",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            max_output_tokens=64,
            metadata={"purpose": "memory_extraction_kind_card", "stage": "candidate_a"},
        )
    )
    ordinary_limit = ordinary.requests[0].max_output_tokens
    ordinary.requests.clear()
    await client.complete_choice_questions(
        model="openai/primary",
        messages=[LLMMessage(role="user", content="Synthetic state")],
        questions={"candidate_a": _questions()["candidate_a"]},
        metadata={"purpose": "memory_extraction_kind_card"},
        max_output_tokens=64,
    )
    assert ordinary.requests[0].max_output_tokens == ordinary_limit == 8192

    proactive = CannedProvider({"candidate_a": "belief"})
    routed = LLMClient(
        providers=[proactive],
        retry_policy=RetryPolicy(attempts=1),
        intimacy_fallback_models={"extraction_kind": "openai/proactive"},
        intimacy_proactive_routing_enabled=True,
    )
    await routed.complete_choice_questions(
        model="openai/primary",
        messages=[LLMMessage(role="user", content="Synthetic state")],
        questions={"candidate_a": _questions()["candidate_a"]},
        metadata={
            "purpose": "memory_extraction_kind_card",
            "source_intimacy_boundary": "romantic_private",
        },
    )
    assert [request.model for request in proactive.requests] == ["proactive"]
    assert proactive.requests[0].metadata["atagia_intimacy_proactive_route"] is True


async def test_native_choice_stays_typed_when_proactive_route_is_configured() -> None:
    ordinary = CannedProvider({"candidate_a": "belief"})

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_payload({"candidate_a": "belief"}, ["evidence", "belief"]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[ordinary, TypeSafeProvider("test-key", client=http)],
            intimacy_fallback_models={"extraction_kind": "openai/proactive"},
            intimacy_proactive_routing_enabled=True,
        )
        result = await client.complete_choice_questions(
            model="typesafe/jev-latest",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions={"candidate_a": _questions()["candidate_a"]},
            metadata={
                "purpose": "memory_extraction_kind_card",
                "source_intimacy_boundary": "romantic_private",
            },
        )
    assert result == {"candidate_a": "belief"}
    assert ordinary.requests == []


async def test_failed_llm_sibling_is_cancelled_and_joined() -> None:
    cancelled = asyncio.Event()

    class FailingProvider(LLMProvider):
        name = "openai"

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            if request.metadata["stage"] == "candidate_a":
                await asyncio.sleep(0)
                raise LLMError("synthetic failure")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("The sibling should have been cancelled")

    client = LLMClient(providers=[FailingProvider()], retry_policy=RetryPolicy(attempts=1))
    with pytest.raises(LLMError, match="synthetic failure"):
        await client.complete_choice_questions(
            model="openai/test-model",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions=_questions(),
            metadata={"purpose": "memory_extraction_kind_card"},
        )
    assert cancelled.is_set()


async def test_sibling_cards_share_one_dispatch_limit_per_request() -> None:
    active = 0
    peak = 0

    class SlowProvider(LLMProvider):
        name = "openai"

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text="evidence"
            )

    client = LLMClient(providers=[SlowProvider()], retry_policy=RetryPolicy(attempts=1))
    shared = asyncio.Semaphore(1)

    async def run_card() -> dict[str, str]:
        return await client.complete_choice_questions(
            model="openai/test-model",
            messages=[LLMMessage(role="user", content="Synthetic state")],
            questions=_questions(),
            metadata={"purpose": "memory_extraction_kind_card"},
            dispatch_semaphore=shared,
        )

    first, second = await asyncio.gather(run_card(), run_card())
    assert first == second == {key: "evidence" for key in _questions()}
    assert peak == 1


async def test_native_question_packing_keeps_each_batch_as_one_attempt(tmp_path) -> None:
    questions = {
        f"candidate_{index}": ChoiceQuestion(
            instructions=f"Decide candidate {index}. " + ("x" * 12_000),
            criteria={"yes": None, "no": None},
        )
        for index in range(6)
    }
    seen: list[set[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = set(json.loads(request.content)["questions"])
        seen.append(ids)
        return httpx.Response(200, json=_answer_payload({key: "yes" for key in ids}, ["yes", "no"]))

    recorder = DiagnosticRecorder(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
            diagnostic_recorder=recorder,
        )
        result = await client.complete_choice_questions(
            model="typesafe/jev-1.13.0",
            messages=[LLMMessage(role="user", content="Synthetic shared source")],
            questions=questions,
            metadata={"purpose": "memory_extraction_kind_card"},
        )
    recorder.close()
    assert len(seen) == 2
    assert seen[0].isdisjoint(seen[1])
    assert set.union(*seen) == set(questions)
    assert result == {key: "yes" for key in questions}
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "provider_attempt" and event["phase"] == "start"]) == 2


async def test_oversized_native_state_fails_before_transport() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Oversized state must not be sent")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(ConfigurationError, match="safe 32k-token bound"):
            await client.complete_choice_questions(
                model="typesafe/jev-1.13.0",
                messages=[LLMMessage(role="user", content="x" * 30_000)],
                questions={"candidate_a": _questions()["candidate_a"]},
                metadata={"purpose": "memory_extraction_kind_card"},
            )


def test_choice_limit_matches_typesafe_contract() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            instructions="Choose one.",
            criteria={str(index): None for index in range(256)},
        )
