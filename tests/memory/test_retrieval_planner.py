"""Tests for retrieval planning and candidate search."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.memory.retrieval_planner import RetrievalPlanner, build_retrieval_fts_queries
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    NeedTrigger,
    RetrievalPlan,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _resolved_policy(mode_id: str = "coding_debug"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


def _need(need_type: NeedTrigger, confidence: float = 0.8) -> DetectedNeed:
    return DetectedNeed(
        need_type=need_type,
        confidence=confidence,
        reasoning=f"Detected {need_type.value}.",
    )


@pytest.mark.asyncio
async def test_default_plan_matches_policy_retrieval_params() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="Need help debugging a websocket timeout in FastAPI.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.max_candidates == policy.retrieval_params.fts_limit
    assert plan.max_context_items == policy.retrieval_params.final_context_items
    assert plan.status_filter == [MemoryStatus.ACTIVE]
    assert plan.scope_filter == [
        MemoryScope.CONVERSATION,
        MemoryScope.WORKSPACE,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.GLOBAL_USER,
    ]
    assert plan.consequence_search_enabled is False
    assert plan.fts_queries == [
        "need help debugging websocket",
        "need help debugging",
        "need OR help OR debugging OR websocket OR timeout OR fastapi",
    ]


@pytest.mark.asyncio
async def test_ambiguity_need_broadens_scope_and_increases_limit() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="Need help debugging a websocket timeout in FastAPI.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[_need(NeedTrigger.AMBIGUITY)],
        cold_start=False,
    )

    assert plan.max_candidates == 37
    assert plan.scope_filter == [
        MemoryScope.GLOBAL_USER,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
    ]
    assert plan.consequence_search_enabled is True
    assert plan.need_driven_boosts[NeedTrigger.AMBIGUITY] == 1.2


@pytest.mark.asyncio
async def test_high_stakes_need_requires_regrounding() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="I need the safest next step for a production outage.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[_need(NeedTrigger.HIGH_STAKES)],
        cold_start=False,
    )

    assert plan.require_evidence_regrounding is True
    assert plan.max_candidates == 37
    assert plan.consequence_search_enabled is True


@pytest.mark.asyncio
async def test_under_specified_request_broadens_scope_increases_limit_and_enables_consequence_search() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What should we do next?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[_need(NeedTrigger.UNDER_SPECIFIED_REQUEST)],
        cold_start=False,
    )

    assert plan.max_candidates == 37
    assert plan.scope_filter == [
        MemoryScope.GLOBAL_USER,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
    ]
    assert plan.consequence_search_enabled is True


@pytest.mark.asyncio
async def test_pattern_query_targets_episode_then_atomic_levels() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What patterns or preferences do I usually show when debugging?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.retrieval_levels == [1, 0]


@pytest.mark.asyncio
async def test_abstract_multi_session_query_targets_thematic_then_episode_then_atomic_levels() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What themes recur across conversations overall?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.retrieval_levels == [2, 1, 0]


@pytest.mark.asyncio
async def test_temporal_query_detects_named_month_and_removes_date_tokens_from_fts() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What was true in January 2024 about my travel plans?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert plan.temporal_query_range.end == datetime(2024, 1, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    joined_queries = " ".join(plan.fts_queries)
    assert "january" not in joined_queries
    assert "2024" not in joined_queries
    assert plan.retrieval_levels == [0, 1]


@pytest.mark.asyncio
async def test_temporal_query_detects_prefixed_month_without_explicit_year() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What changed in March for the project?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    assert plan.temporal_query_range.end == datetime(2026, 3, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert "march" not in " ".join(plan.fts_queries)


@pytest.mark.asyncio
async def test_temporal_query_detects_iso_date_and_removes_date_tokens_from_fts() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What happened on 2026-01-15 with the deploy?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
    assert plan.temporal_query_range.end == datetime(2026, 1, 15, 23, 59, 59, 999999, tzinfo=timezone.utc)
    joined_queries = " ".join(plan.fts_queries)
    assert "2026" not in joined_queries
    assert "01" not in joined_queries
    assert "15" not in joined_queries


@pytest.mark.asyncio
async def test_temporal_query_detects_relative_day_from_clock() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What happened yesterday with the deploy?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == datetime(2026, 4, 4, 0, 0, tzinfo=timezone.utc)
    assert plan.temporal_query_range.end == datetime(2026, 4, 4, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert "yesterday" not in " ".join(plan.fts_queries)


@pytest.mark.asyncio
async def test_temporal_query_detects_last_month_window() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="What changed last month in the project?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    assert plan.temporal_query_range.end == datetime(2026, 3, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_text", "expected_start", "expected_end"),
    [
        (
            "What do I have next week?",
            datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 12, 23, 59, 59, 999999, tzinfo=timezone.utc),
        ),
        (
            "What changes next month?",
            datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
        ),
        (
            "What is planned next year?",
            datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2027, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
        ),
    ],
)
async def test_temporal_query_detects_next_relative_spans(
    message_text: str,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text=message_text,
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is not None
    assert plan.temporal_query_range.start == expected_start
    assert plan.temporal_query_range.end == expected_end


@pytest.mark.asyncio
async def test_temporal_query_does_not_treat_modal_may_as_month() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="May I get a status update on the deploy?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is None
    assert "deploy" in " ".join(plan.fts_queries)


@pytest.mark.asyncio
async def test_temporal_query_does_not_treat_non_temporal_march_as_month() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="Should we march on the deploy plan now?",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=False,
    )

    assert plan.temporal_query_range is None
    assert "march" in " ".join(plan.fts_queries)


@pytest.mark.asyncio
async def test_cold_start_sets_skip_retrieval() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="Need help debugging a websocket timeout in FastAPI.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[],
        cold_start=True,
    )

    assert plan.skip_retrieval is True


@pytest.mark.asyncio
async def test_frustration_reduces_context_items() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="We keep repeating ourselves and I just need the shortest path out.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[_need(NeedTrigger.FRUSTRATION)],
        cold_start=False,
    )

    assert plan.max_context_items == 6


@pytest.mark.asyncio
async def test_multiple_needs_compose_without_losing_high_stakes_flags() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="We are stuck in the same production outage loop.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[
            _need(NeedTrigger.HIGH_STAKES, confidence=0.9),
            _need(NeedTrigger.LOOP, confidence=0.8),
        ],
        cold_start=False,
    )

    assert plan.require_evidence_regrounding is True
    assert plan.max_candidates == 55
    assert plan.scope_filter == [
        MemoryScope.GLOBAL_USER,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
    ]
    assert plan.need_driven_boosts[NeedTrigger.HIGH_STAKES] == 1.25
    assert plan.need_driven_boosts[NeedTrigger.LOOP] == 1.25
    assert plan.consequence_search_enabled is True


@pytest.mark.asyncio
async def test_mode_shift_resets_first_then_other_needs_reapply_adjustments() -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="Switch context, but this is still a high-stakes production issue.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[
            _need(NeedTrigger.HIGH_STAKES, confidence=0.9),
            _need(NeedTrigger.MODE_SHIFT, confidence=0.7),
        ],
        cold_start=False,
    )

    assert plan.require_evidence_regrounding is True
    assert plan.max_candidates == 37
    assert plan.scope_filter == [
        MemoryScope.CONVERSATION,
        MemoryScope.WORKSPACE,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.GLOBAL_USER,
    ]
    assert plan.need_driven_boosts[NeedTrigger.MODE_SHIFT] == 1.0
    assert plan.need_driven_boosts[NeedTrigger.HIGH_STAKES] == 1.25
    assert plan.consequence_search_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "need_type",
    [
        NeedTrigger.FOLLOW_UP_FAILURE,
        NeedTrigger.LOOP,
        NeedTrigger.HIGH_STAKES,
        NeedTrigger.UNDER_SPECIFIED_REQUEST,
        NeedTrigger.AMBIGUITY,
    ],
)
async def test_consequence_search_needs_set_explicit_plan_flag(need_type: NeedTrigger) -> None:
    planner = RetrievalPlanner(FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)))
    policy = _resolved_policy()

    plan = planner.build_plan(
        message_text="We are stuck and need the next safest step.",
        conversation_context=_context(),
        resolved_policy=policy,
        detected_needs=[_need(need_type)],
        cold_start=False,
    )

    assert plan.consequence_search_enabled is True


async def _build_candidate_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "User Two")
    return connection, memories, search


def _plan(
    *,
    scope_filter: list[MemoryScope],
    status_filter: list[MemoryStatus],
    retrieval_levels: list[int] | None = None,
    skip_retrieval: bool = False,
) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=["websocket retry"],
        scope_filter=scope_filter,
        status_filter=status_filter,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=retrieval_levels or [0],
        require_evidence_regrounding=False,
        need_driven_boosts={},
        skip_retrieval=skip_retrieval,
    )


@pytest.mark.asyncio
async def test_candidate_search_filters_by_user_before_ranking() -> None:
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )
        await memories.create_memory_object(
            user_id="usr_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _plan(scope_filter=[MemoryScope.ASSISTANT_MODE], status_filter=[MemoryStatus.ACTIVE]),
            user_id="usr_1",
        )

        assert [candidate["user_id"] for candidate in candidates] == ["usr_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_respects_scope_filter() -> None:
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _plan(scope_filter=[MemoryScope.CONVERSATION], status_filter=[MemoryStatus.ACTIVE]),
            user_id="usr_1",
        )

        assert [candidate["scope"] for candidate in candidates] == ["conversation"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_filters_summary_mirrors_by_retrieval_levels() -> None:
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="debugging preference memory",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_atomic",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="episode debugging preference summary",
            payload={
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "source_object_ids": ["mem_atomic"],
            },
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.7,
            privacy_level=0,
            memory_id="sum_mem_episode",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="thematic debugging preference summary",
            payload={
                "summary_kind": "thematic_profile",
                "hierarchy_level": 2,
                "source_object_ids": ["mem_atomic"],
            },
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.7,
            privacy_level=0,
            memory_id="sum_mem_thematic",
        )

        plan_episode = _plan(
            scope_filter=[MemoryScope.ASSISTANT_MODE, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[1, 0],
        ).model_copy(update={"fts_queries": ["debugging preference"]})
        plan_atomic = _plan(
            scope_filter=[MemoryScope.ASSISTANT_MODE, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[0],
        ).model_copy(update={"fts_queries": ["debugging preference"]})
        plan_all = _plan(
            scope_filter=[MemoryScope.ASSISTANT_MODE, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[2, 1, 0],
        ).model_copy(update={"fts_queries": ["debugging preference"]})

        episode_results = await search.search(plan_episode, user_id="usr_1")
        atomic_results = await search.search(plan_atomic, user_id="usr_1")
        all_results = await search.search(plan_all, user_id="usr_1")

        assert {candidate["id"] for candidate in episode_results} == {"mem_atomic", "sum_mem_episode"}
        assert [candidate["id"] for candidate in atomic_results] == ["mem_atomic"]
        assert {candidate["id"] for candidate in all_results} == {
            "mem_atomic",
            "sum_mem_episode",
            "sum_mem_thematic",
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_respects_status_filter() -> None:
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.CONVERSATION,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
        )
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.CONVERSATION,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.INFERRED,
            confidence=0.7,
            privacy_level=0,
            status=MemoryStatus.SUPERSEDED,
        )

        active_only = await search.search(
            _plan(scope_filter=[MemoryScope.CONVERSATION], status_filter=[MemoryStatus.ACTIVE]),
            user_id="usr_1",
        )
        with_superseded = await search.search(
            _plan(
                scope_filter=[MemoryScope.CONVERSATION],
                status_filter=[MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED],
            ),
            user_id="usr_1",
        )

        assert len(active_only) == 1
        assert len(with_superseded) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_returns_empty_list_when_skip_retrieval_is_true() -> None:
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _plan(
                scope_filter=[MemoryScope.CONVERSATION],
                status_filter=[MemoryStatus.ACTIVE],
                skip_retrieval=True,
            ),
            user_id="usr_1",
        )

        assert candidates == []
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Tests for build_retrieval_fts_queries
# ---------------------------------------------------------------------------


def test_retrieval_fts_queries_generates_multiple_queries() -> None:
    queries = build_retrieval_fts_queries("When Jon has lost his job as a banker?")
    assert len(queries) == 3
    # First query is AND with top 4 tokens
    assert queries[0] == "jon lost job banker"
    # Second query is AND with top 3 tokens
    assert queries[1] == "jon lost job"
    # Third query is OR with all content tokens
    assert queries[2] == "jon OR lost OR job OR banker"


def test_retrieval_fts_queries_strips_question_words() -> None:
    queries = build_retrieval_fts_queries("What do Jon and Gina have in common?")
    all_terms: set[str] = set()
    for query in queries:
        for part in query.replace(" OR ", " ").split():
            all_terms.add(part)
    forbidden = {"what", "do", "and", "have", "in"}
    assert all_terms.isdisjoint(forbidden), f"Found stopwords: {all_terms & forbidden}"
    assert "jon" in all_terms
    assert "gina" in all_terms
    assert "common" in all_terms


def test_retrieval_fts_queries_short_input() -> None:
    queries = build_retrieval_fts_queries("coffee preference")
    assert len(queries) == 1
    assert queries[0] == "coffee preference"


def test_retrieval_fts_queries_single_word() -> None:
    queries = build_retrieval_fts_queries("hello")
    assert len(queries) == 1
    assert queries[0] == "hello"


def test_retrieval_fts_queries_empty_after_stopwords() -> None:
    queries = build_retrieval_fts_queries("what is the")
    assert queries == []


def test_retrieval_fts_queries_three_content_tokens() -> None:
    queries = build_retrieval_fts_queries("Jon lost banker")
    assert len(queries) == 2
    assert queries[0] == "jon lost banker"
    assert queries[1] == "jon OR lost OR banker"


@pytest.mark.asyncio
async def test_question_shaped_query_finds_candidates_via_or_fallback() -> None:
    """Integration: seed memories and verify a question-shaped query finds them."""
    connection, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.ASSISTANT_MODE,
            canonical_text="Jon lost his job as a banker last year",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )

        # Build plan from a question that would produce 0 hits with a single AND query
        planner = RetrievalPlanner(
            FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
        )
        policy = _resolved_policy()
        plan = planner.build_plan(
            message_text="When has Jon lost his job as a banker?",
            conversation_context=_context(),
            resolved_policy=policy,
            detected_needs=[],
            cold_start=False,
        )

        assert len(plan.fts_queries) >= 2, "Expected multiple FTS queries"

        candidates = await search.search(plan, user_id="usr_1")
        assert len(candidates) >= 1, (
            f"Expected at least 1 candidate but got {len(candidates)}; "
            f"fts_queries={plan.fts_queries}"
        )
        assert "banker" in candidates[0]["canonical_text"]
    finally:
        await connection.close()
