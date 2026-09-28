"""Native finite-choice consequence decisions preserve production semantics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import httpx
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.memory.consequence_detector import ConsequenceDetector
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMProvider,
    RetryPolicy,
    TransientLLMError,
)
from atagia.services.providers.typesafe import TypeSafeProvider


class GenerativeProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = dict(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata["purpose"])
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs[purpose],
        )


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key="test-openrouter-key",
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openrouter/test-chat",
        llm_component_models={
            "consequence_detector": "openrouter/test/test-generative",
            "consequence_gate": "typesafe/jev-latest",
            "consequence_sentiment": "typesafe/jev-latest",
            "consequence_link": "typesafe/jev-latest",
        },
        llm_finite_decisions_enabled=True,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        embedding_model=None,
    )


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_user_1",
        workspace_id="wrk_1",
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


def _generative_outputs() -> dict[str, str]:
    return {
        "consequence_action_card": "Apply the narrow patch.",
        "consequence_outcome_card": "The tests pass now.",
        "consequence_language_card": "en",
    }


def _choice_response(
    payload: dict[str, Any],
    choice: str,
    *,
    probabilities: dict[str, float] | None = None,
) -> httpx.Response:
    question_id = next(iter(payload["questions"]))
    criteria = payload["questions"][question_id]["criteria"]
    distribution = probabilities or {
        candidate: 1.0 if candidate == choice else 0.0 for candidate in criteria
    }
    return httpx.Response(
        200,
        json={
            "model": "jev-latest",
            "answers": {
                question_id: {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": distribution,
                    "confidence": max(distribution.values()),
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    )


def _detector(
    generative: GenerativeProvider,
    http: httpx.AsyncClient,
) -> ConsequenceDetector:
    return ConsequenceDetector(
        llm_client=LLMClient(
            providers=[generative, TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        ),
        clock=FrozenClock(datetime(2026, 9, 17, tzinfo=timezone.utc)),
        settings=_settings(),
        card_concurrency=5,
    )


@pytest.mark.asyncio
async def test_native_consequence_decisions_use_three_separate_choice_requests() -> None:
    payloads: list[dict[str, Any]] = []
    choices = {
        "is_consequence": "yes",
        "outcome_sentiment": "negative",
        "likely_action_message_id": "msg_assistant_1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert len(payload["questions"]) == 1
        question_id = next(iter(payload["questions"]))
        return _choice_response(payload, choices[question_id])

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        signal = await _detector(generative, http).detect(
            message_text="That patch broke the tests.",
            role="user",
            conversation_context=_context(),
            recent_assistant_messages=[
                {"id": "msg_assistant_1", "text": "Apply the narrow patch."},
                {"id": "msg_assistant_2", "text": "Restart the worker."},
            ],
        )

    assert signal is not None
    assert signal.outcome_sentiment.value == "negative"
    assert signal.likely_action_message_id == "msg_assistant_1"
    assert {next(iter(payload["questions"])) for payload in payloads} == set(choices)
    assert all(len(payload["questions"]) == 1 for payload in payloads)
    link_payload = next(
        payload for payload in payloads if "likely_action_message_id" in payload["questions"]
    )
    assert set(link_payload["questions"]["likely_action_message_id"]["criteria"]) == {
        "none",
        "msg_assistant_1",
        "msg_assistant_2",
    }
    assert {request.metadata["purpose"] for request in generative.requests} == {
        "consequence_action_card",
        "consequence_outcome_card",
        "consequence_language_card",
    }


@pytest.mark.asyncio
async def test_native_negative_gate_stops_before_enrichment() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return _choice_response(payload, "no")

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        signal = await _detector(generative, http).detect(
            message_text="I will try that tomorrow.",
            role="user",
            conversation_context=_context(),
            recent_assistant_messages=[
                {"id": "msg_assistant_1", "text": "Apply the narrow patch."}
            ],
        )

    assert signal is None
    assert [next(iter(payload["questions"])) for payload in payloads] == [
        "is_consequence"
    ]
    assert generative.requests == []


@pytest.mark.asyncio
async def test_empty_link_candidates_skip_the_link_request() -> None:
    payloads: list[dict[str, Any]] = []
    choices = {"is_consequence": "yes", "outcome_sentiment": "positive"}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        question_id = next(iter(payload["questions"]))
        return _choice_response(payload, choices[question_id])

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        signal = await _detector(generative, http).detect(
            message_text="The narrow patch worked.",
            role="user",
            conversation_context=_context(),
            recent_assistant_messages=[],
        )

    assert signal is not None
    assert signal.likely_action_message_id is None
    assert {next(iter(payload["questions"])) for payload in payloads} == set(choices)
    assert "consequence_link_card" not in {
        request.metadata["purpose"] for request in generative.requests
    }


@pytest.mark.asyncio
async def test_native_choice_probability_mismatch_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        question_id = next(iter(payload["questions"]))
        if question_id == "is_consequence":
            return _choice_response(payload, "yes")
        if question_id == "outcome_sentiment":
            return _choice_response(payload, "negative")
        return _choice_response(
            payload,
            "msg_assistant_1",
            probabilities={"none": 0.6, "msg_assistant_1": 0.4},
        )

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMError, match="invalid choice distribution"):
            await _detector(generative, http).detect(
                message_text="That patch broke the tests.",
                role="user",
                conversation_context=_context(),
                recent_assistant_messages=[
                    {"id": "msg_assistant_1", "text": "Apply the narrow patch."}
                ],
            )


@pytest.mark.asyncio
async def test_native_link_rejects_an_ineligible_message_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        question_id = next(iter(payload["questions"]))
        if question_id == "is_consequence":
            return _choice_response(payload, "yes")
        if question_id == "outcome_sentiment":
            return _choice_response(payload, "negative")
        return _choice_response(
            payload,
            "msg_unknown",
            probabilities={"none": 0.0, "msg_assistant_1": 0.0, "msg_unknown": 1.0},
        )

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMError, match="invalid choice distribution"):
            await _detector(generative, http).detect(
                message_text="That patch broke the tests.",
                role="user",
                conversation_context=_context(),
                recent_assistant_messages=[
                    {"id": "msg_assistant_1", "text": "Apply the narrow patch."}
                ],
            )


@pytest.mark.asyncio
async def test_native_transport_failure_is_not_converted_to_no_consequence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic transport failure", request=request)

    generative = GenerativeProvider(_generative_outputs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(TransientLLMError, match="transport failed"):
            await _detector(generative, http).detect(
                message_text="That patch broke the tests.",
                role="user",
                conversation_context=_context(),
                recent_assistant_messages=[
                    {"id": "msg_assistant_1", "text": "Apply the narrow patch."}
                ],
            )


@pytest.mark.asyncio
async def test_native_schema_failure_is_not_converted_to_no_consequence() -> None:
    generative = GenerativeProvider(_generative_outputs())
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"model": "jev-latest", "answers": {}})
    )
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LLMError, match="invalid evaluation response"):
            await _detector(generative, http).detect(
                message_text="That patch broke the tests.",
                role="user",
                conversation_context=_context(),
                recent_assistant_messages=[
                    {"id": "msg_assistant_1", "text": "Apply the narrow patch."}
                ],
            )
