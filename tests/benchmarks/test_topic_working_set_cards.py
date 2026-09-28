from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from benchmarks.topic_working_set_cards.compare import (
    _DEFAULT_CASES_PATH,
    _run_content_cards,
    BenchmarkCase,
    load_cases,
    normalize_plan,
    project_topic_state,
    score_plan,
)
from atagia.memory.topic_working_set import (
    _CONTENT_CARD_NAMES,
    _CONTENT_FIELDS,
    _TopicRoute,
    TopicUpdateAction,
    TopicUpdateActionType,
    TopicWorkingSetPlan,
    TopicWorkingSetUpdater,
)
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)


@pytest.mark.asyncio
async def test_topic_harness_calls_each_production_content_builder() -> None:
    class ContentProvider(LLMProvider):
        name = "topic-harness"

        def __init__(self) -> None:
            self.outputs = ["Shed planning", "The user plans a shed.", "none", "none", "none"]
            self.requests: list[LLMCompletionRequest] = []

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=self.outputs.pop(0)
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are unused")

    provider = ContentProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])
    updater = TopicWorkingSetUpdater(
        llm_client=client,
        clock=FrozenClock(datetime(2026, 4, 26, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=replace(Settings.from_env(), llm_forced_global_model="openai/test-model"),
    )
    case = BenchmarkCase(
        case_id="synthetic_content",
        conversation_id="cnv_1",
        snapshot={"active_topics": [], "parked_topics": []},
        messages=[{"id": "msg_1", "role": "user", "text": "Plan a backyard shed."}],
        expected={"actions": []},
    )
    route = _TopicRoute(TopicUpdateActionType.CREATE, "tmp1", ("msg_1",))

    results = await _run_content_cards(
        updater=updater, client=client, model="openai/test-model", case=case, routes=(route,)
    )

    assert [result.card_name for result in results] == list(_CONTENT_FIELDS)
    assert all(result.parse_valid for result in results)
    assert [request.messages for request in provider.requests] == [
        updater._card_request(
            card_name=_CONTENT_CARD_NAMES[field_name],
            user_id="benchmark-user",
            conversation_id=case.conversation_id,
            prompt=updater._build_content_prompt(
                field_name=field_name,
                messages=case.messages,
                route=route,
                existing_topic=None,
            ),
            snapshot=case.snapshot,
            target_id=route.target_id,
        ).messages
        for field_name in _CONTENT_FIELDS
    ]


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
