"""Closed-choice intent decisions retain fail-fast provider behavior."""

import json

import httpx
import pytest

from atagia.memory.intent_classifier import (
    are_claim_key_pairs_equivalent_batch,
    are_claim_keys_equivalent,
    is_explicit_user_statement,
)
from atagia.services.llm_client import LLMClient, LLMRequestError
from atagia.services.providers.typesafe import TypeSafeProvider
from tests.services.providers.test_typesafe import _answer_payload


@pytest.mark.parametrize("choice, expected", [("yes", True), ("no", False)])
async def test_classifiers_use_separate_typed_boolean_questions(choice: str, expected: bool) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert len(payload["questions"]) == 1
        key = next(iter(payload["questions"]))
        seen.append(key)
        assert "Schema:" not in json.dumps(payload)
        assert "Return JSON only" not in json.dumps(payload)
        return httpx.Response(200, json=_answer_payload({key: choice}, ["yes", "no"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        assert await is_explicit_user_statement(client, "typesafe/jev-latest", "I prefer concise answers.") is expected
        assert await are_claim_keys_equivalent(client, "typesafe/jev-latest", "answer.length", "response.verbosity") is expected
        assert await are_claim_keys_equivalent(client, "typesafe/jev-latest", "answer.length", "answer.length") is True
    assert seen == ["is_explicit", "equivalent"]


async def test_provider_failure_is_not_converted_to_a_negative_classification() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401))) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(LLMRequestError, match="401"):
            await is_explicit_user_statement(client, "typesafe/jev-latest", "I prefer concise answers.")
        with pytest.raises(LLMRequestError, match="401"):
            await are_claim_keys_equivalent(client, "typesafe/jev-latest", "one", "two")


async def test_batched_equivalence_uses_independent_native_questions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"pair_0", "pair_2"}
        instructions = {
            question_id: question["instructions"]
            for question_id, question in payload["questions"].items()
        }
        assert "status.is_active" in instructions["pair_0"]
        assert "status.is_not_active" in instructions["pair_0"]
        assert "response_style.debugging" in instructions["pair_2"]
        assert "communication.debugging_style" in instructions["pair_2"]
        assert "same.key" not in json.dumps(payload)
        return httpx.Response(
            200,
            json=_answer_payload({"pair_0": "no", "pair_2": "yes"}, ["yes", "no"]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        result = await are_claim_key_pairs_equivalent_batch(
            client,
            "typesafe/jev-latest",
            [
                ("status.is_active", "status.is_not_active"),
                ("same.key", "same.key"),
                ("response_style.debugging", "communication.debugging_style"),
            ],
            user_id="usr_1",
        )
    assert result == [False, True, True]
