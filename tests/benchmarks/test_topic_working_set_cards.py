from __future__ import annotations

from benchmarks.topic_working_set_cards.compare import (
    _DEFAULT_CASES_PATH,
    load_cases,
    normalize_plan,
    project_topic_state,
    score_plan,
)
from atagia.memory.topic_working_set import (
    TopicUpdateAction,
    TopicUpdateActionType,
    TopicWorkingSetPlan,
)


def test_topic_working_set_card_cases_load() -> None:
    cases = load_cases(_DEFAULT_CASES_PATH)

    assert len(cases) == 5
    assert cases[0].case_id == "create_artifact_manifest"
    assert cases[0].messages[0]["metadata_json"]["attachments"][0]["artifact_id"] == (
        "art_manifest"
    )


def test_score_plan_accepts_expected_create_with_artifact() -> None:
    case = load_cases(_DEFAULT_CASES_PATH, limit=1)[0]
    plan = TopicWorkingSetPlan(
        actions=[
            TopicUpdateAction(
                action=TopicUpdateActionType.CREATE,
                title="Benchmark manifest",
                source_message_ids=["msg_manifest"],
                artifact_ids=["art_manifest"],
                privacy_level=0,
                intimacy_boundary="ordinary",
            )
        ]
    )

    normalized = normalize_plan(plan)
    score = score_plan(normalized, case.expected)

    assert score["exact_match"] is True
    assert score["matched_action_count"] == 1


def test_score_plan_reports_missing_required_artifact() -> None:
    case = load_cases(_DEFAULT_CASES_PATH, limit=1)[0]
    plan = TopicWorkingSetPlan(
        actions=[
            TopicUpdateAction(
                action=TopicUpdateActionType.CREATE,
                title="Benchmark manifest",
                source_message_ids=["msg_manifest"],
                privacy_level=0,
                intimacy_boundary="ordinary",
            )
        ]
    )

    score = score_plan(normalize_plan(plan), case.expected)

    assert score["exact_match"] is False
    assert score["requirement_failures"] == [
        {"field": "artifact_ids", "missing": ["art_manifest"]}
    ]


def test_project_topic_state_applies_park_action() -> None:
    case = [
        loaded
        for loaded in load_cases(_DEFAULT_CASES_PATH)
        if loaded.case_id == "park_paused_model_comparison"
    ][0]
    plan = TopicWorkingSetPlan(
        actions=[
            TopicUpdateAction(
                action=TopicUpdateActionType.PARK,
                topic_id="tpc_qwen",
                source_message_ids=["msg_pause_qwen"],
            )
        ]
    )

    projection = project_topic_state(case.snapshot, normalize_plan(plan))

    assert [topic["id"] for topic in projection["active_topics"]] == []
    assert [topic["id"] for topic in projection["parked_topics"]] == ["tpc_qwen"]
