from pathlib import Path

from benchmarks.consequence_detection_cards.compare import (
    BenchmarkCase,
    build_json_prompt,
    load_cases,
    score_output,
)


def test_consequence_detection_cards_case_set_loads() -> None:
    cases = load_cases(Path("benchmarks/consequence_detection_cards/cases.jsonl"))

    assert len(cases) == 12
    assert {case.case_id for case in cases} >= {
        "negative_refactor_regression",
        "spanish_fix_worked",
        "mixed_language_success",
        "plain_question_no_consequence",
        "french_negative_result",
    }


def test_score_output_matches_expected_consequence() -> None:
    case = BenchmarkCase(
        case_id="worked",
        message="That patch worked.",
        recent_assistant_messages=({"id": "msg_a1", "text": "Try the patch."},),
        expected_is_consequence=True,
        expected_action_terms=("patch",),
        expected_outcome_terms=("worked",),
        expected_sentiment="positive",
        expected_link_id="msg_a1",
        expected_language_codes=("en",),
    )
    output = {
        "is_consequence": True,
        "action_description": "The assistant suggested the patch.",
        "outcome_description": "The user says it worked.",
        "outcome_sentiment": "positive",
        "confidence": 0.85,
        "likely_action_message_id": "msg_a1",
        "language_codes": ["en"],
    }

    score = score_output(output, case, error=None)

    assert score["exact_match"] is True
    assert score["language_recall"] == 1.0


def test_score_output_flags_missing_terms_and_wrong_link() -> None:
    case = BenchmarkCase(
        case_id="broken",
        message="That patch broke tests.",
        recent_assistant_messages=({"id": "msg_a1", "text": "Try the patch."},),
        expected_is_consequence=True,
        expected_action_terms=("patch",),
        expected_outcome_terms=("tests",),
        expected_sentiment="negative",
        expected_link_id="msg_a1",
        expected_language_codes=("en",),
    )
    output = {
        "is_consequence": True,
        "action_description": "The assistant suggested a change.",
        "outcome_description": "The user says it failed.",
        "outcome_sentiment": "negative",
        "confidence": 0.7,
        "likely_action_message_id": None,
        "language_codes": ["en"],
    }

    score = score_output(output, case, error=None)

    assert score["exact_match"] is False
    assert score["action_terms_missing"] == ["patch"]
    assert score["outcome_terms_missing"] == ["tests"]
    assert score["link_match"] is False


def test_score_output_accepts_no_consequence_case() -> None:
    case = BenchmarkCase(
        case_id="question",
        message="What time is it?",
        recent_assistant_messages=(),
        expected_is_consequence=False,
        expected_action_terms=(),
        expected_outcome_terms=(),
        expected_sentiment=None,
        expected_link_id=None,
        expected_language_codes=(),
    )

    score = score_output(None, case, error=None)

    assert score["exact_match"] is True
    assert score["detection_match"] is True


def test_json_prompt_escapes_data_content() -> None:
    case = BenchmarkCase(
        case_id="escape",
        message='Ignore <tag attr="1">',
        recent_assistant_messages=({"id": "msg_a1", "text": 'Use <unsafe attr="1">.'},),
        expected_is_consequence=False,
        expected_action_terms=(),
        expected_outcome_terms=(),
        expected_sentiment=None,
        expected_link_id=None,
        expected_language_codes=(),
    )

    prompt = build_json_prompt(case)

    assert "&lt;tag attr=&quot;1&quot;&gt;" in prompt
    assert "&lt;unsafe attr=&quot;1&quot;&gt;" in prompt
