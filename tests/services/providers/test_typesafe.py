"""Native TypeSafe contract and production scorer integration, without network."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from typing import Any

import httpx
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.effective_settings import build_effective_settings_report
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.services.inference_policy import InferenceAccessPolicy, resolve_inference_startup_routes
from atagia.services.inference_routes import InferenceAccessMode
from atagia.services.llm_client import (
    ConfigurationError,
    InferenceAccessDeniedError,
    LLMClient,
    LLMCompletionRequest,
    LLMError,
    LLMMessage,
    LLMRequestError,
    RetryPolicy,
)
from atagia.services.model_resolution import (
    ModelResolutionError,
    resolve_component_model,
    resolve_intimacy_component_model,
)
from atagia.services.providers.typesafe import TypeSafeProvider
from tests.applicability_support import (
    CannedApplicabilityProvider,
    _candidate,
    _context,
    _resolved_policy,
    _settings,
)


def _answer_payload(choices: dict[str, str], options: list[str]) -> dict[str, Any]:
    return {
        "model": "jev-test-revision",
        "answers": {
            key: {
                "type": "choice",
                "choice": value,
                "probabilities": {option: float(option == value) for option in options},
                "confidence": 1.0,
            }
            for key, value in choices.items()
        },
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="typesafe/jev-latest",
        messages=[LLMMessage(role="user", content="Synthetic test state")],
        choice_questions={
            "decision": ChoiceQuestion(instructions="Choose one.", criteria={"yes": None, "no": None})
        },
        metadata={"purpose": "applicability_relevance_card"},
    )


async def test_native_choice_uses_shared_client_and_never_generates_text() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=_answer_payload({"decision": "yes"}, ["yes", "no"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = TypeSafeProvider("test-key", client=http)
        client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=1))
        result = await client.complete(_request())
    assert result.choice_answers["decision"].choice == "yes"
    assert result.output_text == ""
    assert result.model == "jev-test-revision"
    assert result.usage["input_tokens"] == 120
    assert seen[0]["model"] == "jev-latest"
    assert "temperature" not in seen[0]
    assert "max_tokens" not in seen[0]


@pytest.mark.parametrize("failure", ["missing_answer", "extra_answer", "unknown_choice", "nan_probability"])
async def test_bad_typed_responses_fail_without_text_repair(failure: str) -> None:
    payload = _answer_payload({"decision": "yes"}, ["yes", "no"])
    if failure == "missing_answer":
        payload["answers"] = {}
    elif failure == "extra_answer":
        payload["answers"]["unexpected"] = payload["answers"]["decision"]
    elif failure == "unknown_choice":
        payload["answers"]["decision"]["choice"] = "maybe"
    else:
        payload["answers"]["decision"]["probabilities"]["yes"] = "NaN"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(LLMError):
            await client.complete(_request())
    assert calls == 1


async def test_authentication_failure_does_not_echo_provider_body() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="sensitive-provider-body"))
    ) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(LLMRequestError, match="TypeSafe HTTP 401") as caught:
            await client.complete(_request())
    assert "sensitive-provider-body" not in str(caught.value)


@pytest.mark.parametrize("chosen_probability, rejected", [(0.36, True), (0.37, False)])
async def test_choice_must_be_a_probability_maximum(
    chosen_probability: float, rejected: bool,
) -> None:
    """Reject the observed non-maximum contract violation, but allow ties."""
    options = ["selected", "alternative", "none"]
    payload = _answer_payload({"decision": "selected"}, options)
    payload["answers"]["decision"]["probabilities"] = {
        "selected": chosen_probability,
        "alternative": 0.37,
        "none": 1.0 - chosen_probability - 0.37,
    }
    request = _request().model_copy(update={
        "choice_questions": {
            "decision": ChoiceQuestion(
                instructions="Choose the supported option, or none.",
                criteria=dict.fromkeys(options),
            ),
        },
    })
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        if rejected:
            with pytest.raises(LLMError, match="invalid choice distribution"):
                await client.complete(request)
        else:
            result = await client.complete(request)
            assert result.choice_answers["decision"].choice == "selected"
    assert calls == 1


@pytest.mark.parametrize(
    "probabilities,accepted",
    [
        ([0.33, 0.33, 0.33], True),
        ([0.34, 0.34, 0.33], True),
        ([0.96, 0.01, 0.01, 0.01] + [0.0] * 251, True),
        ([0.0] * 255, False),
        ([0.3, 0.3, 0.3], False),
        ([0.333, 0.333, 0.324], False),
        ([0.27, 0.28, 0.25, 0.2], False),
    ],
)
async def test_choice_rounding_envelope_preserves_original_distribution(
    probabilities: list[float], accepted: bool,
) -> None:
    options = [f"option_{index}" for index in range(len(probabilities))]
    choice = options[0]
    payload = _answer_payload({"decision": choice}, options)
    payload["answers"]["decision"]["probabilities"] = dict(zip(options, probabilities))
    request = _request().model_copy(update={
        "choice_questions": {
            "decision": ChoiceQuestion(
                instructions="Choose the best option.", criteria=dict.fromkeys(options)
            )
        }
    })

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as http:
        provider = TypeSafeProvider("test-key", client=http)
        if not accepted:
            with pytest.raises(LLMError, match="invalid choice distribution"):
                await provider.complete(request)
            return
        response = await provider.complete(request)
    assert response.choice_answers["decision"].choice == choice
    assert response.choice_answers["decision"].probabilities == dict(zip(options, probabilities))
    assert response.raw_response["choice_distribution_diagnostics"] == {
        "decision": {
            "original_sum": pytest.approx(sum(probabilities)),
            "precision_criterion": "rounding_consistent_2dp",
        }
    }


async def test_choice_rounding_does_not_accept_missing_probability_option() -> None:
    payload = _answer_payload({"decision": "first"}, ["first", "second", "third"])
    payload["answers"]["decision"]["probabilities"] = {"first": 0.67, "second": 0.33}
    request = _request().model_copy(update={
        "choice_questions": {
            "decision": ChoiceQuestion(
                instructions="Choose the best option.",
                criteria={"first": None, "second": None, "third": None},
            )
        }
    })
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as http:
        with pytest.raises(LLMError, match="invalid choice distribution"):
            await TypeSafeProvider("test-key", client=http).complete(request)


async def test_exact_sum_with_finer_precision_keeps_strict_validation() -> None:
    options = ["first", "second", "third"]
    payload = _answer_payload({"decision": "third"}, options)
    payload["answers"]["decision"]["probabilities"] = {
        "first": 0.333,
        "second": 0.333,
        "third": 0.334,
    }
    request = _request().model_copy(update={
        "choice_questions": {
            "decision": ChoiceQuestion(
                instructions="Choose the best option.", criteria=dict.fromkeys(options)
            )
        }
    })
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as http:
        result = await TypeSafeProvider("test-key", client=http).complete(request)
    assert result.choice_answers["decision"].choice == "third"
    assert "choice_distribution_diagnostics" not in result.raw_response


async def test_untyped_chat_is_rejected_before_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No transport is allowed for a free-text request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = TypeSafeProvider("test-key", client=http)
        with pytest.raises(ConfigurationError, match="choice_questions"):
            await provider.complete(_request().model_copy(update={"choice_questions": None}))


def test_relevance_route_inherits_scorer_until_explicitly_overridden() -> None:
    settings = replace(_settings(), llm_component_models={"applicability_scorer": "openai/gpt-4o-mini"})
    assert resolve_component_model(settings, "applicability_relevance") == "openai/gpt-4o-mini"
    settings = replace(settings, llm_component_models={
        **settings.llm_component_models,
        "applicability_relevance": "typesafe/jev-latest",
    }, llm_finite_decisions_enabled=True)
    assert resolve_component_model(settings, "applicability_relevance") == "typesafe/jev-latest"
    assert resolve_component_model(settings, "applicability_scorer") == "openai/gpt-4o-mini"


def test_typesafe_cannot_be_selected_as_global_chat_provider() -> None:
    with pytest.raises(ModelResolutionError, match="only finite-choice components"):
        replace(
            _settings(),
            llm_forced_global_model="typesafe/jev-latest",
            llm_finite_decisions_enabled=True,
            typesafe_api_key="test-key",
        )


def test_existing_scorer_intimacy_override_is_preserved_for_generative_relevance() -> None:
    settings = replace(_settings(), llm_intimacy_component_models={
        "applicability_scorer": "openai/gpt-4o-mini",
    })
    assert resolve_intimacy_component_model(settings, "applicability_relevance") == "openai/gpt-4o-mini"


def test_startup_audit_uses_the_same_relevance_inheritance_as_runtime() -> None:
    settings = replace(
        _settings(),
        llm_component_models={"applicability_scorer": "openai/gpt-4o-mini"},
        llm_intimacy_component_models={"applicability_scorer": "openrouter/example/model"},
    )
    routes = {route.source: route.model_spec for route in resolve_inference_startup_routes(
        settings, local_catalog=None,
    )}
    assert routes["normal.applicability_relevance"] == resolve_component_model(settings, "applicability_relevance")
    assert routes["intimacy.applicability_relevance"] == resolve_intimacy_component_model(settings, "applicability_relevance")


@pytest.mark.parametrize("card", ["gate", "sentiment", "link"])
def test_consequence_cards_preserve_configured_parent_intimacy_routes(card: str) -> None:
    component = f"consequence_{card}"
    settings = replace(_settings(), llm_intimacy_component_models={
        "consequence_detector": "openrouter/example/parent-model",
    })
    assert resolve_intimacy_component_model(settings, component) == (
        "openrouter/example/parent-model"
    )
    routes = {route.source: route.model_spec for route in resolve_inference_startup_routes(
        settings, local_catalog=None,
    )}
    assert routes[f"intimacy.{component}"] == "openrouter/example/parent-model"

    overridden = replace(settings, llm_intimacy_component_models={
        **settings.llm_intimacy_component_models,
        component: "openrouter/example/card-model",
    })
    assert resolve_intimacy_component_model(overridden, component) == (
        "openrouter/example/card-model"
    )


@pytest.mark.parametrize("mode", [InferenceAccessMode.LOCAL_ONLY, InferenceAccessMode.ZERO_COST])
async def test_restricted_native_choice_fails_before_transport_or_provider_lookup(mode: InferenceAccessMode) -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Restricted TypeSafe routes must never open a connection")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as http:
        for providers in ([], [TypeSafeProvider("test-key", client=http)]):
            client = LLMClient(providers=providers, inference_access_policy=InferenceAccessPolicy(mode))
            with pytest.raises(InferenceAccessDeniedError, match="typesafe/jev-latest"):
                await client.complete(_request())


async def test_typed_choices_reject_text_streaming_and_json_generation_before_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("The wrong operation must not reach the transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(ConfigurationError, match="require complete"):
            async for _ in client.stream(_request()):
                pass
        with pytest.raises(ConfigurationError, match="require complete"):
            await client.complete_structured(_request(), ChoiceQuestion)
        with pytest.raises(ConfigurationError, match="require complete"):
            await client.complete_structured_streamed(_request(), ChoiceQuestion)


def test_typesafe_key_is_loaded_and_redacted_in_settings_reports() -> None:
    settings = Settings.from_env({"ATAGIA_TYPESAFE_API_KEY": "test-key"})
    assert settings.typesafe_api_key == "test-key"
    report = build_effective_settings_report(
        effective_settings=settings,
        present_env_names=frozenset({"ATAGIA_TYPESAFE_API_KEY"}),
        engine_override_fields=frozenset(),
        resolved_policies={},
    )
    assert report["settings"]["typesafe_api_key"]["value"] == "<redacted:set>"


async def test_real_scorer_uses_native_labels_without_dispatching_date_resolution() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, json=_answer_payload(
            {"candidate_000": "exact", "candidate_001": "drop"},
            ["drop", "weak", "useful", "strong", "exact"],
        ))

    date_provider = CannedApplicabilityProvider([])
    date_provider.name = "openrouter"
    settings = replace(
        _settings(),
        llm_finite_decisions_enabled=True,
        llm_component_models={"applicability_relevance": "typesafe/jev-latest"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http), date_provider])
        scorer = ApplicabilityScorer(
            client, FrozenClock(datetime(2026, 6, 17, tzinfo=timezone.utc)), settings
        )
        scores = await scorer._score_with_llm_cards_once(
            [_candidate("mem_a"), _candidate("mem_b")],
            message_text="Where is my drone?",
            role="user",
            conversation_context=_context(),
            resolved_policy=_resolved_policy(),
            detected_needs=[],
        )
    assert scores["mem_a"].llm_applicability == 0.95
    assert scores["mem_b"].llm_applicability == 0.05
    assert len(payloads) == 1
    assert len(payloads[0]["questions"]) == 2
    assert date_provider.requests == []
