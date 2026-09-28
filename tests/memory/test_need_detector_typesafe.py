"""Native finite-choice need cards through the production detector."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from atagia.memory.need_detector import (
    NeedCardCall,
    NeedCardName,
    NeedDetector,
    _authority_context_from_extraction_context,
    _card_task,
    _parse_card_output,
)
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.memory.need_choices import (
    CHOICE_CARD_NAMES,
    LANGUAGE_DECISIONS,
    build_need_choice_questions,
)
from atagia.core.language_codes import ISO_639_1_LANGUAGE_CODES
from atagia.models.schemas_memory import (
    ExplicitLanguagePreference,
    LanguageProfileSourceRef,
    MemoryDependence,
    UserCommunicationProfile,
)
from atagia.services.llm_client import LLMClient, LLMRequestError
from atagia.services.providers.typesafe import TypeSafeProvider
from tests.memory.test_need_detector import (
    CannedCardProvider,
    _clock,
    _context,
    _default_outputs,
    _resolved_policy,
    _settings,
)
from tests.memory.test_need_detector_language_profile import GatedLanguageProvider
from tests.services.providers.test_typesafe import _answer_payload


def _typesafe_settings(component_models: dict[str, str]) -> Any:
    explicit_models = {
        "need_detector_language" if card in {"query_language", "answer_language"}
        else f"need_detector_{card}": "openrouter/test/generative-test-model"
        for card in CHOICE_CARD_NAMES
    }
    explicit_models.update(component_models)
    return replace(
        _settings(),
        llm_finite_decisions_enabled=True,
        llm_component_models=explicit_models,
    )


async def test_native_cards_stay_independent_and_merge_with_generative_cards() -> None:
    payloads = []
    decisions = {"memory": "personal", "exact": "yes", "shape": "slot"}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert len(payload["questions"]) == 1
        card = next(iter(payload["questions"]))
        question = payload["questions"][card]
        return httpx.Response(200, json=_answer_payload(
            {card: decisions[card]}, list(question["criteria"]),
        ))

    generative = CannedCardProvider(_default_outputs())
    generative.name = "openrouter"
    settings = _typesafe_settings({
        f"need_detector_{card}": "typesafe/jev-latest" for card in decisions
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http), generative])
        detector = NeedDetector(client, _clock(), settings)
        trace = []
        result = await detector.detect(
            message_text="What was the locker code you recommended?", role="user",
            conversation_context=_context(), resolved_policy=_resolved_policy(),
            content_language_profile=[], card_call_trace_sink=trace,
        )
    assert len(payloads) == 3
    assert len(generative.requests) == 6
    assert all(request.choice_questions is None for request in generative.requests)
    assert result.memory_dependence == MemoryDependence.PERSONAL
    assert result.exact_recall_needed is True
    assert result.query_type == "slot_fill"
    assert all(call.parse_valid for call in trace)
    assert {call.card_name for call in trace if call.model == "typesafe/jev-latest"} == set(decisions)


@pytest.mark.parametrize("card", ["memory", "exact", "shape", "query_language", "answer_language", "needs", "facets", "callback"])
async def test_native_card_errors_are_not_swallowed_as_missing_decisions(card: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    ) as http:
        detector = NeedDetector(
            LLMClient(providers=[TypeSafeProvider("test-key", client=http)]),
            _clock(), _typesafe_settings({
                "need_detector_language" if card in {"query_language", "answer_language"}
                else f"need_detector_{card}": "typesafe/jev-latest"
            }),
        )
        with pytest.raises(LLMRequestError, match="401"):
            await detector._run_card(
                card_name=card, message_text="Where is my locker?", role="user",
                context=_context(), resolved_policy=_resolved_policy(),
                content_language_profile=[], user_communication_profile=None,
                prompt_authority_context=_authority_context_from_extraction_context(
                    _context(), purpose="need_detection",
                ),
            )


async def test_native_language_error_keeps_its_type_and_trace_through_detect() -> None:
    generative = CannedCardProvider(_default_outputs())
    generative.name = "openrouter"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    ) as http:
        detector = NeedDetector(
            LLMClient(providers=[TypeSafeProvider("test-key", client=http), generative]),
            _clock(), _typesafe_settings({"need_detector_language": "typesafe/jev-latest"}),
        )
        trace = []
        with pytest.raises(LLMRequestError, match="401"):
            await detector.detect(
                message_text="Where is my locker?", role="user",
                conversation_context=_context(), resolved_policy=_resolved_policy(),
                content_language_profile=[], card_call_trace_sink=trace,
            )

    query_attempt = next(call for call in trace if call.card_name == "query_language")
    assert query_attempt.error is not None and "401" in query_attempt.error
    assert query_attempt.prompt is not None


async def test_free_text_card_cannot_be_routed_to_typesafe() -> None:
    with pytest.raises(ValueError, match="finite-choice components"):
        _typesafe_settings(
            {"need_detector_search_words": "typesafe/jev-latest"}
        )


async def _run_choice_card(
    card: NeedCardName, handler: Callable[[httpx.Request], httpx.Response], *,
    policy: ResolvedRetrievalPolicy | None = None,
    query_language: str | None = None,
) -> NeedCardCall:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        detector = NeedDetector(
            LLMClient(providers=[TypeSafeProvider("test-key", client=http)]), _clock(),
            _typesafe_settings({
                "need_detector_language" if card in {"query_language", "answer_language"}
                else f"need_detector_{card}": "typesafe/jev-latest"
            }),
        )
        return await detector._run_card(
            card_name=card, message_text="On és el meu paquet? Respon en italià.", role="user",
            context=_context(), resolved_policy=policy or _resolved_policy(),
            content_language_profile=[], user_communication_profile=None,
            prompt_authority_context=_authority_context_from_extraction_context(_context(), purpose="need_detection"),
            query_language=query_language,
        )


@pytest.mark.parametrize("decisions", [
    {"query_language": "ca", "answer_language": "it"},
    {"query_language": "unknown", "answer_language": "it"},
    {"query_language": "unknown", "answer_language": "unknown"},
])
async def test_language_choices_cover_the_catalog_and_keep_decisions_separate(decisions: dict[str, str]) -> None:
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert len(payload["questions"]) == 1
        key, question = next(iter(payload["questions"].items()))
        assert set(question["criteria"]) == ISO_639_1_LANGUAGE_CODES | {"unknown"}
        assert "us" not in question["criteria"]
        assert "Write two language codes" not in json.dumps(payload)
        if key == "query_language":
            assert "Choose the language the assistant should answer in" not in json.dumps(payload)
        else:
            assert "Known query language: " + (
                "unknown" if decisions["query_language"] == "unknown"
                else decisions["query_language"]
            ) in json.dumps(payload)
        return httpx.Response(200, json=_answer_payload({key: decisions[key]}, list(question["criteria"])))

    query_result = await _run_choice_card("query_language", handler)
    answer_result = await _run_choice_card(
        "answer_language", handler,
        query_language=query_result.parsed["query_language"],
    )
    assert len(payloads) == 2
    assert query_result.parse_valid and answer_result.parse_valid
    assert {**query_result.parsed, **answer_result.parsed} == {
        key: None if value == "unknown" else value for key, value in decisions.items()
    }


@pytest.mark.parametrize("card", ["query_language", "answer_language"])
def test_language_choices_and_plain_outputs_share_the_full_contract(card: str) -> None:
    question = build_need_choice_questions(card, enabled_needs={})[card]
    instruction, _, _ = _card_task(card)

    assert LANGUAGE_DECISIONS[card] in instruction
    assert LANGUAGE_DECISIONS[card] in question.instructions
    assert set(question.criteria) == ISO_639_1_LANGUAGE_CODES | {"unknown"}
    for option in question.criteria:
        parsed, valid = _parse_card_output(card, option)
        assert valid
        assert parsed == {card: None if option == "unknown" else option}


async def test_language_cards_use_independent_plain_model_overrides_and_examples() -> None:
    provider = CannedCardProvider({
        **_default_outputs(),
        "need_detection_query_language_card": "ca",
        "need_detection_answer_language_card": "fr",
    })
    provider.name = "openrouter"
    settings = replace(
        _settings(),
        llm_component_models={
            "need_detector_query_language": "openrouter/test/query-model",
            "need_detector_answer_language": "openrouter/test/answer-model",
        },
        llm_component_examples={
            "need_detector_language": False,
            "need_detector_answer_language": True,
        },
    )
    detector = NeedDetector(LLMClient(providers=[provider]), _clock(), settings)
    result = await detector.detect(
        message_text="On és el meu paquet? Respon en francès.", role="user",
        conversation_context=_context(), resolved_policy=_resolved_policy(),
        content_language_profile=[],
    )

    requests = {
        request.metadata["purpose"]: request for request in provider.requests
    }
    query = requests["need_detection_query_language_card"]
    answer = requests["need_detection_answer_language_card"]
    assert (result.query_language, result.answer_language) == ("ca", "fr")
    assert query.model == "test/query-model"
    assert answer.model == "test/answer-model"
    assert query.choice_questions is None and answer.choice_questions is None
    assert "User message: Hola, que tal?" not in query.messages[-1].content
    assert "User message: Translate \"house\" to French." in answer.messages[-1].content
    assert "Known query language: ca" in answer.messages[-1].content


@pytest.mark.parametrize("native_card", ["query_language", "answer_language"])
async def test_language_card_model_overrides_select_native_and_plain_routes(
    native_card: str,
) -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert list(payload["questions"]) == [native_card]
        question = payload["questions"][native_card]
        answer = "ca" if native_card == "query_language" else "fr"
        return httpx.Response(200, json=_answer_payload(
            {native_card: answer}, list(question["criteria"]),
        ))

    generative = CannedCardProvider({
        **_default_outputs(),
        "need_detection_query_language_card": "ca",
        "need_detection_answer_language_card": "fr",
    })
    generative.name = "openrouter"
    plain_card = "answer_language" if native_card == "query_language" else "query_language"
    settings = _typesafe_settings({
        f"need_detector_{native_card}": "typesafe/jev-latest",
        f"need_detector_{plain_card}": f"openrouter/test/{plain_card}-model",
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        detector = NeedDetector(
            LLMClient(providers=[TypeSafeProvider("test-key", client=http), generative]),
            _clock(), settings,
        )
        result = await detector.detect(
            message_text="On és el meu paquet? Respon en francès.", role="user",
            conversation_context=_context(), resolved_policy=_resolved_policy(),
            content_language_profile=[],
        )

    assert (result.query_language, result.answer_language) == ("ca", "fr")
    assert len(payloads) == 1
    plain_request = next(
        request for request in generative.requests
        if request.metadata["purpose"] == f"need_detection_{plain_card}_card"
    )
    assert plain_request.model == f"test/{plain_card}-model"
    assert plain_request.choice_questions is None
    if native_card == "answer_language":
        assert "Known query language: ca" in json.dumps(payloads[0])


@pytest.mark.parametrize("fail_answer", [False, True])
async def test_native_language_chain_keeps_profile_order_and_card_independence(
    fail_answer: bool,
) -> None:
    generative = GatedLanguageProvider(_default_outputs())
    generative.name = "openrouter"
    answer_started = asyncio.Event()
    payloads: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        await generative.memory_started.wait()
        payload = json.loads(request.content)
        payloads.append(payload)
        assert len(payload["questions"]) == 1
        card, question = next(iter(payload["questions"].items()))
        if card == "answer_language":
            answer_started.set()
            if fail_answer:
                return httpx.Response(401)
        answer = "ca" if card == "query_language" else "fr"
        return httpx.Response(200, json=_answer_payload(
            {card: answer}, list(question["criteria"]),
        ))

    source = LanguageProfileSourceRef(
        source_kind="source_message", source_message_id="msg_profile",
        conversation_id="cnv_1",
    )
    profile = UserCommunicationProfile(explicit_language_preferences=[
        ExplicitLanguagePreference(
            language_code="fr", preference_kind="contextual_answer_language",
            context_label="coding_debug", source_refs=[source], confidence=0.9,
        ),
    ])
    message = "On és el meu paquet? Respon en francès."
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        detector = NeedDetector(
            LLMClient(providers=[TypeSafeProvider("test-key", client=http), generative]),
            _clock(), _typesafe_settings({"need_detector_language": "typesafe/jev-latest"}),
        )
        detection = asyncio.create_task(detector.detect(
            message_text=message, role="user", conversation_context=_context(),
            resolved_policy=_resolved_policy(), content_language_profile=[],
            user_communication_profile=profile,
        ))
        try:
            await asyncio.wait_for(answer_started.wait(), 2)
            assert not generative.release_memory.is_set()
            assert [next(iter(payload["questions"])) for payload in payloads] == [
                "query_language", "answer_language",
            ]
            query_state, answer_state = (
                "\n".join(message["content"] for message in payload["state"])
                for payload in payloads
            )
            for state in (query_state, answer_state):
                assert message in state
                assert "fr/contextual_answer_language/coding_debug" in state
            assert "Known query language: ca" not in query_state
            assert "Known query language: ca" in answer_state
            if fail_answer:
                with pytest.raises(LLMRequestError, match="401"):
                    await asyncio.wait_for(detection, 2)
                assert generative.memory_cancelled.is_set()
            else:
                assert not detection.done()
                generative.release_memory.set()
                result = await asyncio.wait_for(detection, 2)
                assert (result.query_language, result.answer_language) == ("ca", "fr")
        finally:
            generative.release_memory.set()
            if not detection.done():
                await asyncio.wait_for(
                    asyncio.gather(detection, return_exceptions=True), 2,
                )


async def test_facets_can_select_multiple_values_without_collapsing_to_one_label() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        questions = json.loads(request.content)["questions"]
        assert len(questions) == 10
        return httpx.Response(200, json=_answer_payload(
            {key: "yes" if key in {"quantity", "medication"} else "no" for key in questions}, ["yes", "no"],
        ))

    result = await _run_choice_card("facets", handler)
    assert set(result.parsed["exact_facets"]) == {"quantity", "medication"}


async def test_needs_only_evaluate_enabled_types_and_allow_multiple_or_none() -> None:
    policy = _resolved_policy()
    enabled = {need.value for need in policy.need_triggers}
    selected = set(sorted(enabled)[:2])

    def handler(request: httpx.Request) -> httpx.Response:
        questions = json.loads(request.content)["questions"]
        assert set(questions) == enabled
        return httpx.Response(200, json=_answer_payload(
            {key: "yes" if key in selected else "no" for key in questions}, ["yes", "no"],
        ))

    result = await _run_choice_card("needs", handler, policy=policy)
    assert {need.need_type.value for need in result.parsed["needs"]} == selected
    selected.clear()
    result = await _run_choice_card("needs", handler, policy=policy)
    assert result.parse_valid and result.parsed == {"needs": []}


async def test_empty_enabled_need_set_does_not_call_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("An empty enabled set needs no semantic decision")

    result = await _run_choice_card("needs", handler, policy=_resolved_policy().model_copy(update={"need_triggers": []}))
    assert result.parse_valid and result.parsed == {"needs": []}


async def test_all_native_planner_cards_preserve_original_query_and_generative_aliases() -> None:
    payloads: list[dict[str, Any]] = []
    choices = {"memory": "personal", "exact": "yes", "shape": "slot", "callback": "yes",
               "query_language": "es", "answer_language": "it", "code": "yes"}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            answers.update(_answer_payload({key: choices.get(key, "no")}, list(question["criteria"]))["answers"])
        return httpx.Response(200, json={"model": "jev-test", "answers": answers,
                                        "usage": {"input_tokens": 100, "output_tokens": 0}})

    generative = CannedCardProvider({**_default_outputs(), "need_detection_search_words_other_language_card": "none"})
    generative.name = "openrouter"
    settings = _typesafe_settings({
        "need_detector_language" if card in {"query_language", "answer_language"}
        else f"need_detector_{card}": "typesafe/jev-latest"
        for card in CHOICE_CARD_NAMES
    })
    query = "¿Qué código me recomendaste? Responde en italiano."
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        detector = NeedDetector(LLMClient(providers=[TypeSafeProvider("test-key", client=http), generative]), _clock(), settings)
        result = await detector.detect(
            message_text=query, role="user", conversation_context=_context(), resolved_policy=_resolved_policy(),
            content_language_profile=[{"language_code": "en", "memory_count": 1}],
        )
    assert len(payloads) == 8
    assert result.sub_queries == [query]
    assert result.query_language == "es" and result.answer_language == "it"
    assert result.exact_recall_needed and result.callback_bias
    assert set(result.exact_facets) == {"code"}
    assert {request.metadata["purpose"] for request in generative.requests} == {
        "need_detection_search_words_card", "need_detection_search_words_other_language_card",
    }
