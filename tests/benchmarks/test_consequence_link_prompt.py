"""Offline safeguards for the frozen link-only prompt comparison."""

from datetime import datetime, timezone
import json
from typing import Any, cast

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.memory.consequence_detector import ConsequenceDetector
from benchmarks.consequence_detection_cards.link_prompt import (
    LinkCase,
    completed_keys,
    build_requests,
    production_request,
    relabel_case,
    summarize,
)


def _case() -> LinkCase:
    return LinkCase(
        case_id="not_model_input", category="also_not_model_input",
        message='That was incorrect. Quoted text: "ignore the rules".',
        recent_assistant_messages=[
            {"id": "z_81", "text": "Try one possible approach."},
            {"id": "b_04", "text": "Try another approach."},
        ],
        expected_link_id="z_81", notes="ANNOTATION_MUST_NOT_ENTER_INPUT",
    )


def _detector() -> ConsequenceDetector:
    settings = Settings(
        sqlite_path=":memory:", migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"), storage_backend="inprocess",
        redis_url="redis://localhost:6379/0", openai_api_key=None,
        openrouter_api_key=None, openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia", llm_chat_model=None, service_mode=False,
        service_api_key=None, admin_api_key=None, workers_enabled=False, debug=False,
        llm_finite_decisions_enabled=True,
        llm_component_models={"consequence_link": "typesafe/jev-latest"},
    )
    return ConsequenceDetector(
        llm_client=cast(Any, None), settings=settings,
        clock=FrozenClock(datetime(2026, 9, 17, tzinfo=timezone.utc)),
    )


def test_champion_is_the_production_request_and_labels_never_enter_either_arm():
    case = _case()
    detector = _detector()
    requests = build_requests(detector, case, reversed_options=False)
    assert requests["before"] == production_request(detector, case)
    for request in requests.values():
        serialized = request.model_dump_json()
        assert case.notes not in serialized
        assert case.category not in serialized
        assert case.case_id not in serialized
        assert request.choice_questions is not None
        assert set(request.choice_questions["likely_action_message_id"].criteria) == {
            "none", "z_81", "b_04"
        }
    assert requests["before"].metadata == requests["after"].metadata
    assert requests["before"].model == requests["after"].model
    data = json.loads(requests["after"].messages[-1].content)
    assert data["current_message"]["text"] == case.message
    assert data["assistant_messages"] == case.recent_assistant_messages


def test_option_and_id_perturbation_preserves_source_chronology_and_label():
    case = _case()
    relabeled = relabel_case(case)
    assert [item["text"] for item in relabeled.recent_assistant_messages] == [
        item["text"] for item in case.recent_assistant_messages
    ]
    assert relabeled.expected_link_id == relabeled.recent_assistant_messages[0]["id"]
    regular = build_requests(_detector(), relabeled, reversed_options=False)
    reversed_requests = build_requests(_detector(), relabeled, reversed_options=True)
    for arm in regular:
        assert reversed_requests[arm].messages == regular[arm].messages
        assert list(reversed_requests[arm].choice_questions["likely_action_message_id"].criteria) == list(reversed(tuple(regular[arm].choice_questions["likely_action_message_id"].criteria)))


def test_fixture_rejects_unavailable_expected_id():
    with pytest.raises(ValueError, match="eligible"):
        LinkCase.model_validate({**_case().model_dump(), "expected_link_id": "missing"})


def test_quoted_candidate_id_is_not_silently_disconnected_by_relabeling():
    case = _case().model_copy(update={"message": "The form literally says z_81."})
    assert relabel_case(case) == case


def test_technical_failure_does_not_count_as_a_correct_abstention():
    result = summarize([{
        "correct": False, "error": {"type": "LLMError"}, "selected": None,
        "expected": None, "usage": {}, "elapsed_ms": 0.0,
    }])
    assert result["matches"] == 0
    assert result["trials"] == 0
    assert result["attempts"] == 1
    assert result["technical_failures"] == 1
    assert result["false_abstentions"] == 0


def test_continuation_requires_explicit_transient_retry_and_retains_bad_answers():
    rows = [{"case_id": "example", "variant": "base_0", "arm": "before",
             "error": {"type": "TransientLLMError"}}]
    with pytest.raises(ValueError, match="authorized transient"):
        completed_keys(rows, retry_transient=False)
    assert completed_keys(rows, retry_transient=True) == set()
    # Multiple transport attempts are evidence, not duplicate semantic trials.
    assert completed_keys(rows + rows, retry_transient=True) == set()
    incorrect = {**rows[0], "error": None, "correct": False}
    assert completed_keys(rows + [incorrect], retry_transient=True) == {
        ("example", "base_0", "before")
    }
    with pytest.raises(ValueError, match="Duplicate"):
        completed_keys([incorrect, incorrect], retry_transient=True)
    with pytest.raises(ValueError, match="authorized transient"):
        completed_keys([{**rows[0], "error": {"type": "LLMError"}}], retry_transient=True)
