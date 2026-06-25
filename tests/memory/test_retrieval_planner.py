"""Tests for retrieval planning and candidate search."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.memory.retrieval_planner import (
    RetrievalPlanner,
    build_retrieval_fts_queries,
    build_retrieval_fts_query_specs,
)
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExtractionConversationContext,
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    MindTopology,
    NeedTrigger,
    PlannedSubQuery,
    QueryIntelligenceResult,
    RetrievalPlan,
    RuntimeAliasGroupTrace,
    RuntimeAliasSurfaceTrace,
    RuntimeAnchor,
    RuntimeAnchorAlias,
    ScoredCandidate,
    SparseQueryHint,
    SpaceBoundaryMode,
    TemporalQueryRange,
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
async def test_plan_uses_query_intelligence_fields_and_sub_queries() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    temporal_range = TemporalQueryRange(
        start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 4, 30, 23, 59, 59, 999999, tzinfo=timezone.utc),
    )

    plan = planner.build_plan(
        original_query="When did the production retry websocket backoff fail?",
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=temporal_range,
            sub_queries=[
                "retry websocket backoff",
                "production failure outcome",
            ],
            query_type="temporal",
            retrieval_levels=[0, 1],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.original_query == "When did the production retry websocket backoff fail?"
    assert plan.max_candidates == policy.retrieval_params.fts_limit
    assert plan.max_context_items == policy.retrieval_params.final_context_items
    assert plan.status_filter == [MemoryStatus.ACTIVE]
    assert plan.scope_filter == [
        MemoryScope.EPHEMERAL_SESSION,
        MemoryScope.CONVERSATION,
        MemoryScope.WORKSPACE,
        MemoryScope.GLOBAL_USER,
    ]
    assert plan.consequence_search_enabled is False
    assert plan.callback_bias is False
    assert plan.query_type == "temporal"
    assert plan.retrieval_levels == [0, 1]
    assert plan.temporal_query_range == temporal_range
    assert [sub_query.text for sub_query in plan.sub_query_plans] == [
        "retry websocket backoff",
        "production failure outcome",
    ]
    assert plan.sub_query_plans[0].fts_queries == build_retrieval_fts_queries("retry websocket backoff")
    assert plan.fts_queries == [
        *build_retrieval_fts_queries("retry websocket backoff"),
        *build_retrieval_fts_queries("production failure outcome"),
    ]


@pytest.mark.asyncio
async def test_plan_propagates_raw_context_access_mode_from_query_intelligence() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()

    plan = planner.build_plan(
        original_query="Please quote the exact wording from the hidden attachment.",
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["quote hidden attachment"],
            raw_context_access_mode="verbatim",
            query_type="slot_fill",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.raw_context_access_mode == "verbatim"
    assert plan.query_type == "slot_fill"


@pytest.mark.asyncio
async def test_plan_accepts_non_latin_sub_queries() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()

    plan = planner.build_plan(
        original_query="東京駅で待ち合わせたのはいつ？",
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["東京駅で待ち合わせ"],
            query_type="default",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert [sub_query.text for sub_query in plan.sub_query_plans] == ["東京駅で待ち合わせ"]
    assert plan.sub_query_plans[0].fts_queries == ["東京駅で待ち合わせ"]
    assert plan.fts_queries == ["東京駅で待ち合わせ"]


@pytest.mark.asyncio
async def test_plan_uses_sparse_query_hints_for_content_bearing_rewrites() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    original_query = "When did Caroline go to the LGBTQ support group?"

    plan = planner.build_plan(
        original_query=original_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[original_query],
            callback_bias=False,
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=original_query,
                    fts_phrase="Caroline LGBTQ support group",
                    quoted_phrases=["LGBTQ support group"],
                    must_keep_terms=["Caroline"],
                )
            ],
            query_type="temporal",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.original_query == original_query
    assert plan.sub_query_plans[0].text == original_query
    assert plan.sub_query_plans[0].sparse_phrase == "Caroline LGBTQ support group"
    assert plan.sub_query_plans[0].quoted_phrases == ["LGBTQ support group"]
    assert plan.sub_query_plans[0].must_keep_terms == ["Caroline"]
    assert plan.sub_query_plans[0].fts_queries == [
        '"lgbtq support group"',
        "caroline lgbtq support group",
        "caroline OR lgbtq OR support OR group",
    ]
    assert plan.fts_queries == plan.sub_query_plans[0].fts_queries


@pytest.mark.asyncio
async def test_plan_materializes_runtime_anchor_aliases_as_fts_variants() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    sub_query = "current amount of molecula_x for PERSON_A"

    plan = planner.build_plan(
        original_query="¿Cuál es la cantidad actual de molecula_x para PERSON_A?",
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[sub_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=sub_query,
                    fts_phrase="molecula_x current amount PERSON_A",
                    must_keep_terms=["molecula_x", "PERSON_A", "current amount"],
                )
            ],
            anchors=[
                RuntimeAnchor(
                    sub_query_text=sub_query,
                    anchor_type="concept",
                    original_surface="molecula_x",
                    aliases=[
                        RuntimeAnchorAlias(
                            surface="compound_x",
                            alias_language="en",
                            alias_kind="translation",
                            confidence=0.82,
                        )
                    ],
                )
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
            exact_recall_needed=True,
            exact_facets=["quantity"],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    sub_query_plan = plan.sub_query_plans[0]
    assert any(
        query.startswith("compound_x") for query in sub_query_plan.fts_queries
    )
    assert any(
        query == "compound_x OR current OR amount OR person_a"
        for query in sub_query_plan.fts_queries
    )
    assert all(
        not ("molecula_x" in query and "compound_x" in query)
        for query in sub_query_plan.fts_queries
    )
    assert any(
        kind.startswith("anchor_alias_") for kind in sub_query_plan.fts_query_kinds
    )


@pytest.mark.asyncio
async def test_plan_adds_person_anchor_backoff_prefix_for_exact_recall() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    sub_query = "What did Caroline research?"

    plan = planner.build_plan(
        original_query=sub_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[sub_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=sub_query,
                    fts_phrase="Caroline research",
                    must_keep_terms=["Caroline", "research"],
                )
            ],
            anchors=[
                RuntimeAnchor(
                    sub_query_text=sub_query,
                    anchor_type="person",
                    original_surface="Caroline",
                    preserve_verbatim=True,
                )
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
            exact_recall_needed=True,
            exact_facets=["other_verbatim"],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    sub_query_plan = plan.sub_query_plans[0]
    assert sub_query_plan.fts_queries == [
        "caroline research",
        "research*",
        "resear*",
    ]
    assert sub_query_plan.fts_query_kinds == [
        "anchor_first_and",
        "non_evidential_person_anchor_backoff_prefix",
        "non_evidential_person_anchor_backoff_short_prefix_or",
    ]


@pytest.mark.asyncio
async def test_plan_adds_short_prefix_backoff_for_inflected_person_anchor_terms() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    sub_query = "Which cities has Jon visited?"

    plan = planner.build_plan(
        original_query=sub_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[sub_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=sub_query,
                    fts_phrase="Jon visited cities",
                    must_keep_terms=["Jon", "visited", "cities"],
                )
            ],
            anchors=[
                RuntimeAnchor(
                    sub_query_text=sub_query,
                    anchor_type="person",
                    original_surface="Jon",
                    preserve_verbatim=True,
                )
            ],
            query_type="broad_list",
            retrieval_levels=[0],
            exact_recall_needed=True,
            exact_facets=["location", "other_verbatim"],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.max_candidates >= 30
    sub_query_plan = plan.sub_query_plans[0]
    assert sub_query_plan.fts_queries == [
        "jon visited cities",
        "visited cities",
        "visited* cities*",
        "visit* OR citi*",
        "jon OR visited OR cities",
    ]
    assert sub_query_plan.fts_query_kinds == [
        "anchor_first_and",
        "must_keep_tail_and",
        "non_evidential_person_anchor_backoff_prefix",
        "non_evidential_person_anchor_backoff_short_prefix_or",
        "broad_or",
    ]


@pytest.mark.asyncio
async def test_plan_adds_person_anchor_backoff_for_broad_list_without_exact_recall() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    sub_query = "Which cities has Jon visited?"

    plan = planner.build_plan(
        original_query=sub_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[sub_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=sub_query,
                    fts_phrase="Jon visited cities",
                    must_keep_terms=["Jon", "visited", "cities"],
                )
            ],
            anchors=[
                RuntimeAnchor(
                    sub_query_text=sub_query,
                    anchor_type="person",
                    original_surface="Jon",
                    preserve_verbatim=True,
                )
            ],
            query_type="broad_list",
            retrieval_levels=[0],
            exact_recall_needed=False,
            exact_facets=[],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    sub_query_plan = plan.sub_query_plans[0]
    assert sub_query_plan.fts_queries == [
        "jon visited cities",
        "visited* cities*",
        "visit* OR citi*",
        "jon OR visited OR cities",
    ]
    assert sub_query_plan.fts_query_kinds == [
        "anchor_first_and",
        "non_evidential_person_anchor_backoff_prefix",
        "non_evidential_person_anchor_backoff_short_prefix_or",
        "broad_or",
    ]


@pytest.mark.asyncio
async def test_plan_adds_must_keep_tail_backoff_for_exact_anchor_phrase() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    sub_query = "What country is Caroline's grandma from?"

    plan = planner.build_plan(
        original_query=sub_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[sub_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=sub_query,
                    fts_phrase="Caroline grandma country",
                    must_keep_terms=["Caroline", "grandma", "country"],
                )
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
            exact_recall_needed=True,
            exact_facets=["location", "person_name"],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    sub_query_plan = plan.sub_query_plans[0]
    assert sub_query_plan.fts_queries == [
        "caroline grandma country",
        "grandma country",
        "caroline OR grandma OR country",
    ]
    assert sub_query_plan.fts_query_kinds == [
        "anchor_first_and",
        "must_keep_tail_and",
        "broad_or",
    ]


@pytest.mark.asyncio
async def test_plan_derives_sparse_phrase_from_partial_hint_without_question_shaped_fallback() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    original_query = "What was that apple recipe you told me about?"

    plan = planner.build_plan(
        original_query=original_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[original_query],
            callback_bias=True,
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=original_query,
                    quoted_phrases=["apple recipe"],
                    must_keep_terms=["cinnamon"],
                )
            ],
            query_type="default",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.callback_bias is True
    assert plan.sub_query_plans[0].sparse_phrase == "apple recipe cinnamon"
    assert plan.sub_query_plans[0].fts_queries == [
        '"apple recipe"',
        "apple recipe cinnamon",
        "apple OR recipe OR cinnamon",
    ]


@pytest.mark.asyncio
async def test_plan_slot_fill_promotes_fts_phrase_to_precision_anchor() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    original_query = "Where did Caroline move from after the four-year period?"

    plan = planner.build_plan(
        original_query=original_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[original_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=original_query,
                    fts_phrase="Caroline move from home country four years",
                )
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.query_type == "slot_fill"
    assert plan.sub_query_plans[0].quoted_phrases == ["Caroline move from home country four years"]
    assert plan.sub_query_plans[0].fts_queries == [
        '"caroline move from home country four years"',
        "caroline move from home country four",
        "caroline OR move OR from OR home OR country OR four OR years",
    ]


@pytest.mark.asyncio
async def test_plan_prioritizes_anchor_first_rewrite_when_must_keep_terms_extend_sparse_phrase() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    original_query = "What rollback checklist did we use for Phoenix?"

    plan = planner.build_plan(
        original_query=original_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[original_query],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text=original_query,
                    fts_phrase="rollback checklist",
                    must_keep_terms=["Phoenix"],
                )
            ],
            query_type="default",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.sub_query_plans[0].fts_queries == [
        "phoenix rollback checklist",
        "rollback checklist phoenix",
        "rollback OR checklist OR phoenix",
    ]
    assert plan.sub_query_plans[0].fts_query_kinds == [
        "anchor_first_and",
        "sparse_and",
        "broad_or",
    ]


@pytest.mark.asyncio
async def test_plan_exact_recall_materializes_anchor_only_fts_query() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()
    original_query = "¿Cuál es la dosis exacta del medicamento?"

    plan = planner.build_plan(
        original_query=original_query,
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["dose for amlodipine"],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text="dose for amlodipine",
                    fts_phrase="dose amlodipine 10 mg",
                    must_keep_terms=["amlodipine", "10 mg"],
                )
            ],
            query_type="slot_fill",
            retrieval_levels=[0],
            exact_recall_needed=True,
            exact_facets=["medication", "quantity"],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.exact_recall_mode is True
    assert plan.sub_query_plans[0].fts_queries == [
        '"10 mg"',
        "dose amlodipine 10 mg",
        "amlodipine 10 mg",
        "dose OR amlodipine OR 10 OR mg",
    ]
    assert plan.sub_query_plans[0].fts_query_kinds == [
        "quoted_phrase",
        "sparse_and",
        "anchor_only_and",
        "broad_or",
    ]


def test_query_intelligence_broad_list_promotes_fts_phrase_to_explicit_anchor() -> None:
    intelligence = QueryIntelligenceResult(
        needs=[],
        temporal_range=None,
        sub_queries=["team retreat logistics"],
        sparse_query_hints=[
            SparseQueryHint(
                sub_query_text="team retreat logistics",
                fts_phrase="team retreat logistics",
            )
        ],
        query_type="broad_list",
        retrieval_levels=[0],
    )

    assert intelligence.sparse_query_hints[0].quoted_phrases == ["team retreat logistics"]
    assert intelligence.sparse_query_hints[0].must_keep_terms == []


def test_query_intelligence_rejects_duplicate_sparse_hints_for_same_sub_query() -> None:
    with pytest.raises(ValidationError, match="SparseQueryHint.sub_query_text values must be unique"):
        QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["apple recipe question"],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text="apple recipe question",
                    quoted_phrases=["apple recipe"],
                ),
                SparseQueryHint(
                    sub_query_text="apple recipe question",
                    must_keep_terms=["cinnamon"],
                ),
            ],
            query_type="default",
            retrieval_levels=[0],
        )


def test_query_intelligence_rejects_duplicate_facet_hints_for_broad_list() -> None:
    with pytest.raises(
        ValidationError,
        match="broad_list sparse_query_hints must preserve distinct facet anchors",
    ):
        QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=[
                "team retreat logistics",
                "team retreat activities",
            ],
            sparse_query_hints=[
                SparseQueryHint(
                    sub_query_text="team retreat logistics",
                    fts_phrase="team retreat",
                ),
                SparseQueryHint(
                    sub_query_text="team retreat activities",
                    fts_phrase="team retreat",
                ),
            ],
            query_type="broad_list",
            retrieval_levels=[0],
        )


@pytest.mark.asyncio
async def test_query_intelligence_needs_compose_plan_adjustments() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()

    plan = planner.build_plan(
        original_query="What should we do about the production outage?",
        query_intelligence=QueryIntelligenceResult(
            needs=[
                _need(NeedTrigger.AMBIGUITY),
                _need(NeedTrigger.HIGH_STAKES, confidence=0.9),
            ],
            temporal_range=None,
            sub_queries=["production outage next step"],
            query_type="broad_list",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.max_candidates == 55
    assert plan.scope_filter == [
        MemoryScope.GLOBAL_USER,
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
        MemoryScope.EPHEMERAL_SESSION,
    ]
    assert plan.consequence_search_enabled is True
    assert plan.query_type == "broad_list"


@pytest.mark.asyncio
async def test_sensitive_context_preserves_resolved_privacy_ceiling() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy("personal_assistant")
    context = _context().model_copy(update={"assistant_mode_id": "personal_assistant"})

    plan = planner.build_plan(
        original_query="What is my medical dose?",
        query_intelligence=QueryIntelligenceResult(
            needs=[_need(NeedTrigger.SENSITIVE_CONTEXT)],
            temporal_range=None,
            sub_queries=["medical dose"],
            query_type="slot_fill",
            retrieval_levels=[0],
        ),
        conversation_context=context,
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.privacy_ceiling == policy.privacy_ceiling == 3
    assert plan.max_candidates > policy.retrieval_params.fts_limit


@pytest.mark.asyncio
async def test_mode_shift_resets_first_then_other_needs_reapply_adjustments() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()

    plan = planner.build_plan(
        original_query="Switch context and re-ground this answer.",
        query_intelligence=QueryIntelligenceResult(
            needs=[
                _need(NeedTrigger.HIGH_STAKES, confidence=0.9),
                _need(NeedTrigger.MODE_SHIFT, confidence=0.7),
            ],
            temporal_range=None,
            sub_queries=["switch context"],
            query_type="default",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=False,
    )

    assert plan.require_evidence_regrounding is True
    assert plan.max_candidates == 37
    assert plan.scope_filter == [
        MemoryScope.EPHEMERAL_SESSION,
        MemoryScope.CONVERSATION,
        MemoryScope.WORKSPACE,
        MemoryScope.GLOBAL_USER,
    ]
    assert plan.consequence_search_enabled is True


@pytest.mark.asyncio
async def test_cold_start_sets_skip_retrieval() -> None:
    planner = RetrievalPlanner()
    policy = _resolved_policy()

    plan = planner.build_plan(
        original_query="debugging websocket timeout",
        query_intelligence=QueryIntelligenceResult(
            needs=[],
            temporal_range=None,
            sub_queries=["debugging websocket timeout"],
            query_type="default",
            retrieval_levels=[0],
        ),
        conversation_context=_context(),
        resolved_policy=policy,
        cold_start=True,
    )

    assert plan.skip_retrieval is True


async def _build_candidate_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "User One")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "User Two")
    return connection, messages, memories, search


def _plan(
    *,
    scope_filter: list[MemoryScope],
    status_filter: list[MemoryStatus],
    retrieval_levels: list[int] | None = None,
    sub_queries: list[str] | None = None,
    original_query: str | None = None,
    callback_bias: bool = False,
    query_type: str = "default",
    temporal_query_range: TemporalQueryRange | None = None,
    skip_retrieval: bool = False,
) -> RetrievalPlan:
    resolved_sub_queries = sub_queries or ["websocket retry"]
    return RetrievalPlan(
        original_query=original_query or resolved_sub_queries[0],
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=[
            query
            for sub_query in resolved_sub_queries
            for query in build_retrieval_fts_queries(sub_query)
        ],
        sub_query_plans=[
            PlannedSubQuery(
                text=sub_query,
                fts_queries=build_retrieval_fts_queries(sub_query),
            )
            for sub_query in resolved_sub_queries
        ],
        callback_bias=callback_bias,
        query_type=query_type,
        scope_filter=scope_filter,
        status_filter=status_filter,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=retrieval_levels or [0],
        temporal_query_range=temporal_query_range,
        require_evidence_regrounding=False,
        skip_retrieval=skip_retrieval,
    )


def _surface_diagnostic_plan(
    fts_query: str,
    *,
    exact_recall_mode: bool = False,
    query_type: str = "default",
    privacy_ceiling: int = 1,
    allow_intimacy_context: bool = False,
    allow_private_sensitivity: bool = False,
    privacy_enforcement: str = "enforce",
    active_space_id: str | None = None,
    active_space_boundary_mode: SpaceBoundaryMode | None = None,
    active_mind_id: str | None = None,
    mind_topology: MindTopology | None = None,
    active_embodiment_id: str | None = None,
    active_realm_id: str | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=fts_query,
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(
                text=fts_query,
                fts_queries=[fts_query],
                fts_query_kinds=["surface_probe"],
            )
        ],
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        query_type=query_type,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=privacy_ceiling,
        allow_intimacy_context=allow_intimacy_context,
        allow_private_sensitivity=allow_private_sensitivity,
        privacy_enforcement=privacy_enforcement,
        retrieval_levels=[0],
        exact_recall_mode=exact_recall_mode,
        active_space_id=active_space_id,
        active_space_boundary_mode=active_space_boundary_mode,
        active_mind_id=active_mind_id,
        mind_topology=mind_topology or MindTopology.UNIMIND,
        active_embodiment_id=active_embodiment_id,
        active_realm_id=active_realm_id,
    )


def _runtime_alias_policy_plan(
    *,
    privacy_enforcement: str = "enforce",
) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
        platform_id="default",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(
                text="consulta de alias phase8",
                sparse_phrase="consulta alias",
                fts_queries=["consulta alias"],
                fts_query_kinds=["sparse_and"],
            )
        ],
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        privacy_enforcement=privacy_enforcement,
        retrieval_levels=[0],
        exact_recall_mode=True,
        active_space_id="space_active",
        active_space_boundary_mode=SpaceBoundaryMode.FOCUS,
    )


def _phase8_runtime_alias_groups() -> list[RuntimeAliasGroupTrace]:
    return [
        RuntimeAliasGroupTrace(
            sub_query_text="consulta de alias phase8",
            anchor_type="concept",
            original_surface="alias phase8",
            anchor_confidence=0.88,
            aliases=[
                RuntimeAliasSurfaceTrace(
                    surface="phase8aliasbridge",
                    alias_kind="translation",
                    alias_language="en",
                    confidence=0.84,
                )
            ],
        )
    ]


def _persisted_surface_audit_entries(
    fts_query_audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        entry
        for entry in fts_query_audit
        if entry.get("source") == "persisted_surface"
    ]


async def _create_persisted_surface_memory(
    memories: MemoryObjectRepository,
    surfaces: MemoryRetrievalSurfaceRepository,
    *,
    memory_id: str,
    surface_text: str,
    canonical_text: str | None = None,
    surface_type: str = "alias",
    alias_kind: str | None = None,
    language_code: str | None = None,
    preserve_verbatim: bool = False,
    privacy_level: int = 0,
    **memory_kwargs: object,
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.GLOBAL_USER,
        canonical_text=canonical_text or f"Base memory for {memory_id}.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=privacy_level,
        memory_id=memory_id,
        **memory_kwargs,
    )
    await surfaces.upsert_surface(
        user_id="usr_1",
        memory_id=memory_id,
        surface_type=surface_type,
        surface_text=surface_text,
        alias_kind=alias_kind,
        language_code=language_code,
        preserve_verbatim=preserve_verbatim,
    )


@pytest.mark.asyncio
async def test_candidate_search_filters_by_user_before_ranking() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )
        await memories.create_memory_object(
            user_id="usr_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _plan(scope_filter=[MemoryScope.GLOBAL_USER], status_filter=[MemoryStatus.ACTIVE]),
            user_id="usr_1",
        )

        assert [candidate["user_id"] for candidate in candidates] == ["usr_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_person_anchor_backoff_prefix_recovers_first_person_memory_without_name() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text=(
                "Researching adoption agencies has been a long-term family goal."
            ),
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_adoption_research",
        )
        sub_query = "What did Caroline research?"
        plan = RetrievalPlanner().build_plan(
            original_query=sub_query,
            query_intelligence=QueryIntelligenceResult(
                needs=[],
                temporal_range=None,
                sub_queries=[sub_query],
                sparse_query_hints=[
                    SparseQueryHint(
                        sub_query_text=sub_query,
                        fts_phrase="Caroline research",
                        must_keep_terms=["Caroline", "research"],
                    )
                ],
                anchors=[
                    RuntimeAnchor(
                        sub_query_text=sub_query,
                        anchor_type="person",
                        original_surface="Caroline",
                        preserve_verbatim=True,
                    )
                ],
                query_type="slot_fill",
                retrieval_levels=[0],
                exact_recall_needed=True,
                exact_facets=["other_verbatim"],
            ),
            conversation_context=_context(),
            resolved_policy=_resolved_policy(),
            cold_start=False,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_adoption_research"
        ]
        assert any(
            entry["kind"] == "non_evidential_person_anchor_backoff_prefix"
            and entry["query"] == "research*"
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_must_keep_tail_backoff_recovers_first_person_relation_memory() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text=(
                "This necklace is a gift from my grandma in my home country, Sweden."
            ),
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_grandma_country",
        )
        sub_query = "What country is Caroline's grandma from?"
        plan = RetrievalPlanner().build_plan(
            original_query=sub_query,
            query_intelligence=QueryIntelligenceResult(
                needs=[],
                temporal_range=None,
                sub_queries=[sub_query],
                sparse_query_hints=[
                    SparseQueryHint(
                        sub_query_text=sub_query,
                        fts_phrase="Caroline grandma country",
                        must_keep_terms=["Caroline", "grandma", "country"],
                    )
                ],
                query_type="slot_fill",
                retrieval_levels=[0],
                exact_recall_needed=True,
                exact_facets=["location", "person_name"],
            ),
            conversation_context=_context(),
            resolved_policy=_resolved_policy(),
            cold_start=False,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_grandma_country"
        ]
        assert any(
            entry["kind"] == "must_keep_tail_and"
            and entry["query"] == "grandma country"
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_reports_visible_rows_without_candidates() -> None:
    from atagia.services.retrieval_pipeline import RetrievalPipeline

    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The remembered address fact is stored in English.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_surface_visible",
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_surface_visible",
            surface_type="alias",
            surface_text="apartamento",
            alias_kind="translation",
            language_code="es",
        )
        plan = _surface_diagnostic_plan("apartamento")
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert candidates == []
        assert _persisted_surface_audit_entries(fts_query_audit) == [
            {
                "subquery": "apartamento",
                "query": "apartamento",
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "raw_rows": 1,
                "source": "persisted_surface",
                "non_evidential": True,
            }
        ]
        trace = RetrievalPipeline._build_candidate_search_trace(
            candidates,
            plan,
            1.0,
            fts_query_audit,
        )
        assert trace.fts_candidates_count == 0
        assert trace.per_subquery_counts[0].fts == 0
        persisted_execution = [
            execution
            for execution in trace.per_subquery_counts[0].fts_query_executions
            if execution.source == "persisted_surface"
        ]
        assert len(persisted_execution) == 1
        assert persisted_execution[0].raw_rows == 1
        assert persisted_execution[0].candidates == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_respects_user_status_privacy_intimacy_and_platform_gates() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Wrong user base memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_surface_wrong_user",
        )
        await surfaces.upsert_surface(
            user_id="usr_2",
            memory_id="mem_surface_wrong_user",
            surface_type="alias",
            surface_text="wrongusersurface",
        )
        blocked_specs = [
            ("mem_surface_archived", "archivedsurface", {"status": MemoryStatus.ARCHIVED}),
            ("mem_surface_deleted", "deletedsurface", {"status": MemoryStatus.DELETED}),
            ("mem_surface_private", "privatesurface", {"privacy_level": 3}),
            (
                "mem_surface_intimate",
                "intimatesurface",
                {
                    "intimacy_boundary": IntimacyBoundary.ROMANTIC_PRIVATE,
                    "intimacy_boundary_confidence": 0.9,
                },
            ),
            (
                "mem_surface_platform",
                "platformsurface",
                {
                    "platform_locked": True,
                    "platform_id_lock": "mobile",
                },
            ),
        ]
        for memory_id, surface_text, memory_kwargs in blocked_specs:
            normalized_memory_kwargs = dict(memory_kwargs)
            privacy_level = int(normalized_memory_kwargs.pop("privacy_level", 0))
            await memories.create_memory_object(
                user_id="usr_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.GLOBAL_USER,
                canonical_text=f"Base memory for {memory_id}.",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.8,
                privacy_level=privacy_level,
                memory_id=memory_id,
                **normalized_memory_kwargs,
            )
            await surfaces.upsert_surface(
                user_id="usr_1",
                memory_id=memory_id,
                surface_type="alias",
                surface_text=surface_text,
            )

        for query in [
            "wrongusersurface",
            "archivedsurface",
            "deletedsurface",
            "privatesurface",
            "intimatesurface",
            "platformsurface",
        ]:
            fts_query_audit: list[dict[str, object]] = []
            candidates = await search.search(
                _surface_diagnostic_plan(query),
                user_id="usr_1",
                fts_query_audit=fts_query_audit,
            )
            assert candidates == []
            assert _persisted_surface_audit_entries(fts_query_audit) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase8_persisted_surface_diagnostics_distinguish_policy_modes() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_phase8_visible_surface",
            surface_text="phase8privatesurface",
            canonical_text="Visible base memory should pass product gates.",
            privacy_level=0,
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_phase8_private_surface",
            surface_text="phase8privatesurface",
            canonical_text="Private base memory should stay base-memory gated.",
            privacy_level=3,
            sensitivity=MemorySensitivity.SECRET,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
        )

        enforced_audit: list[dict[str, object]] = []
        enforced_candidates = await search.search(
            _surface_diagnostic_plan(
                "phase8privatesurface",
                exact_recall_mode=True,
                privacy_enforcement="enforce",
            ),
            user_id="usr_1",
            fts_query_audit=enforced_audit,
        )
        audit_only_audit: list[dict[str, object]] = []
        audit_only_candidates = await search.search(
            _surface_diagnostic_plan(
                "phase8privatesurface",
                exact_recall_mode=True,
                privacy_enforcement="audit_only",
            ),
            user_id="usr_1",
            fts_query_audit=audit_only_audit,
        )
        off_audit: list[dict[str, object]] = []
        off_candidates = await search.search(
            _surface_diagnostic_plan(
                "phase8privatesurface",
                exact_recall_mode=True,
                privacy_ceiling=3,
                allow_intimacy_context=True,
                allow_private_sensitivity=True,
                privacy_enforcement="off",
            ),
            user_id="usr_1",
            fts_query_audit=off_audit,
            )

        assert {candidate["id"] for candidate in enforced_candidates} == {
            "mem_phase8_visible_surface"
        }
        assert {candidate["id"] for candidate in audit_only_candidates} == {
            "mem_phase8_visible_surface"
        }
        assert _persisted_surface_audit_entries(enforced_audit) == [
            {
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "non_evidential": True,
                "query": "phase8privatesurface",
                "raw_rows": 1,
                "source": "persisted_surface",
                "subquery": "phase8privatesurface",
            }
        ]
        assert _persisted_surface_audit_entries(audit_only_audit) == [
            {
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "non_evidential": True,
                "query": "phase8privatesurface",
                "raw_rows": 1,
                "source": "persisted_surface",
                "subquery": "phase8privatesurface",
            }
        ]
        assert {candidate["id"] for candidate in off_candidates} == {
            "mem_phase8_private_surface",
            "mem_phase8_visible_surface",
        }
        assert _persisted_surface_audit_entries(off_audit) == [
            {
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "non_evidential": True,
                "query": "phase8privatesurface",
                "raw_rows": 2,
                "source": "persisted_surface",
                "subquery": "phase8privatesurface",
            }
        ]
        for candidate in off_candidates:
            assert candidate["fts_query_matches"][0]["source"] == "persisted_surface"
            assert candidate["fts_query_matches"][0]["non_evidential"] is True
            assert "phase8privatesurface" not in candidate["canonical_text"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase8_runtime_alias_policy_matrix_respects_base_memory_gates() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        memory_specs = [
            (
                "mem_phase8_alias_visible",
                "Visible runtime alias base memory.",
                {"privacy_level": 0, "space_id": "space_active"},
            ),
            (
                "mem_phase8_alias_private",
                "Restricted runtime alias base memory.",
                {
                    "privacy_level": 3,
                    "sensitivity": MemorySensitivity.SECRET,
                    "intimacy_boundary": IntimacyBoundary.ROMANTIC_PRIVATE,
                    "intimacy_boundary_confidence": 0.9,
                    "space_id": "space_active",
                },
            ),
            (
                "mem_phase8_alias_platform",
                "Wrong platform runtime alias base memory.",
                {
                    "privacy_level": 0,
                    "platform_locked": True,
                    "platform_id_lock": "ios",
                    "space_id": "space_active",
                },
            ),
            (
                "mem_phase8_alias_space",
                "Wrong space runtime alias base memory.",
                {"privacy_level": 0, "space_id": "space_other"},
            ),
        ]
        for memory_id, canonical_text, memory_kwargs in memory_specs:
            await memories.create_memory_object(
                user_id="usr_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.GLOBAL_USER,
                canonical_text=canonical_text,
                index_text="phase8aliasbridge",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.8,
                memory_id=memory_id,
                **memory_kwargs,
            )
        await memories.create_memory_object(
            user_id="usr_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Wrong user runtime alias base memory.",
            index_text="phase8aliasbridge",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_phase8_alias_wrong_user",
            space_id="space_active",
        )

        mode_results: dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]] = {}
        for mode in ("enforce", "audit_only", "off"):
            audit: list[dict[str, object]] = []
            candidates = await search.search(
                _runtime_alias_policy_plan(privacy_enforcement=mode),
                user_id="usr_1",
                fts_query_audit=audit,
                runtime_alias_groups=_phase8_runtime_alias_groups(),
            )
            mode_results[mode] = (candidates, audit)

        for mode in ("enforce", "audit_only"):
            candidates, audit = mode_results[mode]
            assert {candidate["id"] for candidate in candidates} == {
                "mem_phase8_alias_visible"
            }
            alias_audit = [
                entry
                for entry in audit
                if entry.get("source") == "alias_anchor"
            ]
            assert alias_audit == [
                {
                    "kind": "runtime_alias_or",
                    "match_mode": "implicit_and",
                    "non_evidential": True,
                    "query": "phase8aliasbridge",
                    "raw_rows": 1,
                    "source": "alias_anchor",
                    "subquery": "consulta de alias phase8",
                }
            ]

        off_candidates, off_audit = mode_results["off"]
        assert {candidate["id"] for candidate in off_candidates} == {
            "mem_phase8_alias_private",
            "mem_phase8_alias_visible",
        }
        off_alias_audit = [
            entry
            for entry in off_audit
            if entry.get("source") == "alias_anchor"
        ]
        assert off_alias_audit == [
            {
                "kind": "runtime_alias_or",
                "match_mode": "implicit_and",
                "non_evidential": True,
                "query": "phase8aliasbridge",
                "raw_rows": 2,
                "source": "alias_anchor",
                "subquery": "consulta de alias phase8",
            }
        ]
        for candidate in off_candidates:
            assert candidate["fts_query_matches"][0]["source"] == "alias_anchor"
            assert candidate["fts_query_matches"][0]["non_evidential"] is True
            assert "phase8aliasbridge" not in candidate["canonical_text"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_respects_active_space_gate() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Visible active space base memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_surface_active_space",
            space_id="space_active",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_surface_active_space",
            surface_type="alias",
            surface_text="spacevisible",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Other space base memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_surface_other_space",
            space_id="space_other",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_surface_other_space",
            surface_type="alias",
            surface_text="spaceblocked",
        )

        visible_audit: list[dict[str, object]] = []
        visible_candidates = await search.search(
            _surface_diagnostic_plan(
                "spacevisible",
                active_space_id="space_active",
                active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
            ),
            user_id="usr_1",
            fts_query_audit=visible_audit,
        )
        blocked_audit: list[dict[str, object]] = []
        blocked_candidates = await search.search(
            _surface_diagnostic_plan(
                "spaceblocked",
                active_space_id="space_active",
                active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
            ),
            user_id="usr_1",
            fts_query_audit=blocked_audit,
        )

        assert visible_candidates == []
        assert _persisted_surface_audit_entries(visible_audit)[0]["raw_rows"] == 1
        assert blocked_candidates == []
        assert _persisted_surface_audit_entries(blocked_audit) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_respects_active_mind_gate() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_active_mind",
            surface_text="mindvisible",
            memory_owner_id="mind_alpha",
            source_mind_id="mind_alpha",
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_other_mind",
            surface_text="mindblocked",
            memory_owner_id="mind_beta",
            source_mind_id="mind_beta",
        )

        visible_audit: list[dict[str, object]] = []
        visible_candidates = await search.search(
            _surface_diagnostic_plan(
                "mindvisible",
                active_mind_id="mind_alpha",
                mind_topology=MindTopology.MULTI_MIND,
            ),
            user_id="usr_1",
            fts_query_audit=visible_audit,
        )
        blocked_audit: list[dict[str, object]] = []
        blocked_candidates = await search.search(
            _surface_diagnostic_plan(
                "mindblocked",
                active_mind_id="mind_alpha",
                mind_topology=MindTopology.MULTI_MIND,
            ),
            user_id="usr_1",
            fts_query_audit=blocked_audit,
        )

        assert visible_candidates == []
        assert _persisted_surface_audit_entries(visible_audit)[0]["raw_rows"] == 1
        assert blocked_candidates == []
        assert _persisted_surface_audit_entries(blocked_audit) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_respects_active_embodiment_gate() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_active_embodiment",
            surface_text="embodimentvisible",
            embodiment_id="body_drone",
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_other_embodiment",
            surface_text="embodimentblocked",
            embodiment_id="body_desktop",
        )

        visible_audit: list[dict[str, object]] = []
        visible_candidates = await search.search(
            _surface_diagnostic_plan(
                "embodimentvisible",
                active_embodiment_id="body_drone",
            ),
            user_id="usr_1",
            fts_query_audit=visible_audit,
        )
        blocked_audit: list[dict[str, object]] = []
        blocked_candidates = await search.search(
            _surface_diagnostic_plan(
                "embodimentblocked",
                active_embodiment_id="body_drone",
            ),
            user_id="usr_1",
            fts_query_audit=blocked_audit,
        )

        assert visible_candidates == []
        assert _persisted_surface_audit_entries(visible_audit)[0]["raw_rows"] == 1
        assert blocked_candidates == []
        assert _persisted_surface_audit_entries(blocked_audit) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_diagnostic_respects_active_realm_gate() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_active_realm",
            surface_text="realmvisible",
            realm_id="realm_real",
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_other_realm",
            surface_text="realmblocked",
            realm_id="realm_aincrad",
        )

        visible_audit: list[dict[str, object]] = []
        visible_candidates = await search.search(
            _surface_diagnostic_plan(
                "realmvisible",
                active_realm_id="realm_real",
            ),
            user_id="usr_1",
            fts_query_audit=visible_audit,
        )
        blocked_audit: list[dict[str, object]] = []
        blocked_candidates = await search.search(
            _surface_diagnostic_plan(
                "realmblocked",
                active_realm_id="realm_real",
            ),
            user_id="usr_1",
            fts_query_audit=blocked_audit,
        )

        assert visible_candidates == []
        assert _persisted_surface_audit_entries(visible_audit)[0]["raw_rows"] == 1
        assert blocked_candidates == []
        assert _persisted_surface_audit_entries(blocked_audit) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_recovers_english_memory_from_spanish_query() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_spanish_to_english",
            surface_text="apartamento",
            canonical_text="Ben's new apartment address is 742 Evergreen Terrace.",
            alias_kind="translation",
            language_code="es",
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            _surface_diagnostic_plan(
                "apartamento",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_surface_spanish_to_english"
        ]
        assert candidates[0]["canonical_text"] == (
            "Ben's new apartment address is 742 Evergreen Terrace."
        )
        assert "apartamento" not in candidates[0]["canonical_text"].lower()
        assert candidates[0]["retrieval_sources"] == ["fts"]
        assert candidates[0]["fts_query_matches"] == [
            {
                "subquery": "apartamento",
                "query": "apartamento",
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "position_rank": 1,
                "source": "persisted_surface",
                "non_evidential": True,
            }
        ]
        assert _persisted_surface_audit_entries(fts_query_audit) == [
            {
                "subquery": "apartamento",
                "query": "apartamento",
                "kind": "persisted_surface_surface_probe",
                "match_mode": "implicit_and",
                "raw_rows": 1,
                "source": "persisted_surface",
                "non_evidential": True,
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_recovers_spanish_memory_from_english_query() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_english_to_spanish",
            surface_text="address",
            canonical_text="La direccion nueva de Ben esta en la calle Olmo 742.",
            alias_kind="translation",
            language_code="en",
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            _surface_diagnostic_plan(
                "address",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_surface_english_to_spanish"
        ]
        assert candidates[0]["canonical_text"] == (
            "La direccion nueva de Ben esta en la calle Olmo 742."
        )
        assert "address" not in candidates[0]["canonical_text"].lower()
        assert _persisted_surface_audit_entries(fts_query_audit)[0]["raw_rows"] == 1
        assert candidates[0]["fts_query_matches"][0]["source"] == "persisted_surface"
        assert candidates[0]["fts_query_matches"][0]["non_evidential"] is True
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_runs_for_slot_fill_without_exact_recall() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_slot_fill",
            surface_text="direccion",
            canonical_text="Ben's apartment address is 742 Evergreen Terrace.",
            alias_kind="translation",
            language_code="es",
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            _surface_diagnostic_plan(
                "direccion",
                query_type="slot_fill",
            ),
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_surface_slot_fill"
        ]
        assert candidates[0]["fts_query_matches"][0]["source"] == "persisted_surface"
        assert _persisted_surface_audit_entries(fts_query_audit)[0]["raw_rows"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_does_not_match_similar_proper_name() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_wrong_person",
            surface_text="Carolina",
            canonical_text="The remembered fact belongs to a different person.",
            alias_kind="translation",
            language_code="en",
        )

        candidates = await search.search(
            _surface_diagnostic_plan(
                "Caroline",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert candidates == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_does_not_match_similar_code_literal() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_wrong_code",
            surface_text="SA43",
            canonical_text="The remembered fact belongs to a different ticket.",
            alias_kind="translation",
            language_code="en",
        )

        candidates = await search.search(
            _surface_diagnostic_plan(
                "SA42",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert candidates == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_excludes_preserve_verbatim_surfaces() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_preserve_verbatim",
            surface_text="literalshared",
            canonical_text="Protected literal memory should not be recovered by active surfaces.",
            preserve_verbatim=True,
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_non_verbatim",
            surface_text="ordinaryshared",
            canonical_text="Ordinary surface memory can be recovered.",
            preserve_verbatim=False,
        )

        preserve_verbatim_candidates = await search.search(
            _surface_diagnostic_plan(
                "literalshared",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )
        ordinary_candidates = await search.search(
            _surface_diagnostic_plan(
                "ordinaryshared",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert preserve_verbatim_candidates == []
        assert [candidate["id"] for candidate in ordinary_candidates] == [
            "mem_surface_non_verbatim"
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_excludes_stale_and_deleted_surfaces() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_stale_lifecycle",
            surface_text="stalelifecycle",
            canonical_text="Stale surface memory should not be recovered.",
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_deleted_lifecycle",
            surface_text="deletedlifecycle",
            canonical_text="Deleted surface memory should not be recovered.",
        )
        await surfaces.mark_surfaces_stale_for_memory(
            user_id="usr_1",
            memory_id="mem_surface_stale_lifecycle",
        )
        await surfaces.mark_surfaces_deleted_for_memory(
            user_id="usr_1",
            memory_id="mem_surface_deleted_lifecycle",
        )

        stale_candidates = await search.search(
            _surface_diagnostic_plan(
                "stalelifecycle",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )
        deleted_candidates = await search.search(
            _surface_diagnostic_plan(
                "deletedlifecycle",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert stale_candidates == []
        assert deleted_candidates == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_respects_user_and_privacy_gates() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    surfaces = MemoryRetrievalSurfaceRepository(
        connection,
        FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)),
    )
    try:
        await memories.create_memory_object(
            user_id="usr_2",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Wrong user memory should not leak.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_surface_other_user_active",
        )
        await surfaces.upsert_surface(
            user_id="usr_2",
            memory_id="mem_surface_other_user_active",
            surface_type="alias",
            surface_text="sharedsurface",
        )
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_private_active",
            surface_text="privatesurface",
            canonical_text="Private surface memory should stay gated.",
            privacy_level=3,
        )

        wrong_user_candidates = await search.search(
            _surface_diagnostic_plan(
                "sharedsurface",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )
        private_candidates = await search.search(
            _surface_diagnostic_plan(
                "privatesurface",
                exact_recall_mode=True,
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert wrong_user_candidates == []
        assert private_candidates == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_persisted_surface_candidate_composition_uses_base_memory_text_not_surface_text() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    clock = FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
    surfaces = MemoryRetrievalSurfaceRepository(connection, clock)
    try:
        await _create_persisted_surface_memory(
            memories,
            surfaces,
            memory_id="mem_surface_context_text",
            surface_text="apartamento",
            canonical_text="Ben's new apartment address is 742 Evergreen Terrace.",
            alias_kind="translation",
            language_code="es",
        )
        candidates = await search.search(
            _surface_diagnostic_plan(
                "apartamento",
                exact_recall_mode=True,
                query_type="slot_fill",
            ),
            user_id="usr_1",
            fts_query_audit=[],
        )
        assert len(candidates) == 1

        composed = ContextComposer(clock).compose(
            [
                ScoredCandidate(
                    memory_id=str(candidates[0]["id"]),
                    memory_object=candidates[0],
                    llm_applicability=1.0,
                    retrieval_score=float(candidates[0].get("rrf_score", 0.0)),
                    vitality_boost=0.0,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=1.0,
                )
            ],
            current_contract={},
            user_state=None,
            resolved_policy=_resolved_policy(),
            conversation_messages=[],
            query_text="apartamento",
            query_type="slot_fill",
            exact_recall_mode=True,
        )

        assert composed.selected_memory_ids == ["mem_surface_context_text"]
        assert "Ben's new apartment address is 742 Evergreen Terrace." in composed.memory_block
        assert "apartamento" not in composed.memory_block.lower()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_records_raw_fts_rows_before_candidate_merge() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Falcon deployment account uses token rotation",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="Falcon account",
                    fts_queries=["falcon", "missingtoken"],
                    fts_query_kinds=["default_and", "default_and"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["canonical_text"] for candidate in candidates] == [
            "Falcon deployment account uses token rotation"
        ]
        assert fts_query_audit == [
            {
                "subquery": "Falcon account",
                "query": "falcon",
                "kind": "default_and",
                "match_mode": "implicit_and",
                "raw_rows": 1,
            },
            {
                "subquery": "Falcon account",
                "query": "missingtoken",
                "kind": "default_and",
                "match_mode": "implicit_and",
                "raw_rows": 0,
            },
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_exact_recall_adds_corpus_near_fts_when_planned_queries_have_zero_raw_rows() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_amlodipine",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis actual de amlodipino",
                    sparse_phrase="dosis amlodipino",
                    must_keep_terms=["dosis", "amlodipino"],
                    fts_queries=[
                        "amlodipino dosis",
                        "dosis amlodipino",
                        "dosis OR amlodipino",
                    ],
                    fts_query_kinds=[
                        "anchor_first_and",
                        "sparse_and",
                        "broad_or",
                    ],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_amlodipine"]
        corpus_near_audit = [
            entry
            for entry in fts_query_audit
            if entry["kind"] == "corpus_near_or"
        ]
        assert corpus_near_audit == [
            {
                "subquery": "dosis actual de amlodipino",
                "query": "amlodipine",
                "kind": "corpus_near_or",
                "match_mode": "implicit_and",
                "raw_rows": 1,
            }
        ]
        assert candidates[0]["fts_query_matches"] == [
            {
                "subquery": "dosis actual de amlodipino",
                "query": "amlodipine",
                "kind": "corpus_near_or",
                "match_mode": "implicit_and",
                "position_rank": 1,
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_corpus_near_fts_is_exact_recall_only() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_amlodipine",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis actual de amlodipino",
                    sparse_phrase="dosis amlodipino",
                    must_keep_terms=["dosis", "amlodipino"],
                    fts_queries=["dosis OR amlodipino"],
                    fts_query_kinds=["broad_or"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=False,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert candidates == []
        assert [entry["kind"] for entry in fts_query_audit] == ["broad_or"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase3_multilingual_smoke_recovers_english_query_to_spanish_memory() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="La usuaria toma amlodipino los martes.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_amlodipino_es",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="current amlodipine dose",
                    sparse_phrase="current amlodipine",
                    must_keep_terms=["amlodipine"],
                    fts_queries=["current amlodipine"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_amlodipino_es"]
        assert any(
            entry["kind"] == "corpus_near_or"
            and entry["query"] == "amlodipino"
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase3_multilingual_smoke_keeps_sentence_initial_near_token() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_sentence_initial_amlodipine",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="Amlodipino dosis",
                    sparse_phrase="amlodipino dosis",
                    must_keep_terms=["amlodipino"],
                    fts_queries=["amlodipino dosis"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_sentence_initial_amlodipine"
        ]
        assert any(
            entry["kind"] == "corpus_near_or"
            and entry["query"] == "amlodipine"
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase3_multilingual_smoke_handles_code_switching_without_aliases() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The Tuesday amlodipino plan is active.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_codeswitch",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis amlodipine martes",
                    sparse_phrase="dosis amlodipine martes",
                    must_keep_terms=["amlodipine"],
                    fts_queries=["dosis amlodipine martes"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_codeswitch"]
        assert any(
            entry["kind"] == "corpus_near_or"
            and entry["query"] == "amlodipino"
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase3_multilingual_smoke_does_not_near_match_literals_or_proper_names() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Falcon uses SA42 for deployment approvals.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_sa42",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Carolina owns the deploy calendar.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_carolina",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="SA43",
                    sparse_phrase="SA43",
                    must_keep_terms=["SA43"],
                    fts_queries=["sa43"],
                    fts_query_kinds=["sparse_and"],
                ),
                PlannedSubQuery(
                    text="Caroline",
                    sparse_phrase="caroline",
                    must_keep_terms=["Caroline"],
                    fts_queries=["caroline"],
                    fts_query_kinds=["sparse_and"],
                ),
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert candidates == []
        assert not any(entry["kind"] == "corpus_near_or" for entry in fts_query_audit)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_runtime_alias_fts_recovers_visible_exact_recall_memory() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_runtime_alias_amlodipine",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis actual de amlodipino",
                    sparse_phrase="dosis amlodipino",
                    must_keep_terms=["dosis", "amlodipino"],
                    fts_queries=["amlodipino dosis"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        alias_groups = [
            RuntimeAliasGroupTrace(
                sub_query_text="dosis actual de amlodipino",
                anchor_type="concept",
                original_surface="amlodipino",
                anchor_confidence=0.88,
                aliases=[
                    RuntimeAliasSurfaceTrace(
                        surface="amlodipine",
                        alias_kind="translation",
                        alias_language="en",
                        confidence=0.84,
                    )
                ],
            )
        ]
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
            runtime_alias_groups=alias_groups,
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_runtime_alias_amlodipine"
        ]
        assert any(
            entry["kind"] == "runtime_alias_or"
            and entry["query"] == "amlodipine"
            and entry["source"] == "alias_anchor"
            and entry["non_evidential"] is True
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
        assert not any(entry["kind"] == "corpus_near_or" for entry in fts_query_audit)
        assert candidates[0]["fts_query_matches"] == [
            {
                "subquery": "dosis actual de amlodipino",
                "query": "amlodipine",
                "kind": "runtime_alias_or",
                "match_mode": "implicit_and",
                "position_rank": 1,
                "source": "alias_anchor",
                "non_evidential": True,
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_runtime_alias_fts_is_exact_recall_or_slot_fill_only() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_runtime_alias_default",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis actual de amlodipino",
                    sparse_phrase="dosis amlodipino",
                    fts_queries=["amlodipino dosis"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            query_type="default",
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=False,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
            runtime_alias_groups=[
                RuntimeAliasGroupTrace(
                    sub_query_text="dosis actual de amlodipino",
                    anchor_type="concept",
                    original_surface="amlodipino",
                    anchor_confidence=0.88,
                    aliases=[
                        RuntimeAliasSurfaceTrace(
                            surface="amlodipine",
                            alias_kind="translation",
                            confidence=0.84,
                        )
                    ],
                )
            ],
        )

        assert candidates == []
        assert not any(entry.get("source") == "alias_anchor" for entry in fts_query_audit)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_runtime_alias_fts_recovers_slot_fill_without_exact_recall() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user takes amlodipine tablets on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_runtime_alias_slot_fill",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="dosis actual de amlodipino",
                    sparse_phrase="dosis amlodipino",
                    fts_queries=["amlodipino dosis"],
                    fts_query_kinds=["sparse_and"],
                )
            ],
            query_type="slot_fill",
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=False,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
            runtime_alias_groups=[
                RuntimeAliasGroupTrace(
                    sub_query_text="dosis actual de amlodipino",
                    anchor_type="concept",
                    original_surface="amlodipino",
                    anchor_confidence=0.88,
                    aliases=[
                        RuntimeAliasSurfaceTrace(
                            surface="amlodipine",
                            alias_kind="translation",
                            confidence=0.84,
                        )
                    ],
                )
            ],
        )

        assert [candidate["id"] for candidate in candidates] == [
            "mem_runtime_alias_slot_fill"
        ]
        assert any(
            entry["kind"] == "runtime_alias_or"
            and entry["query"] == "amlodipine"
            and entry["source"] == "alias_anchor"
            and entry["non_evidential"] is True
            and entry["raw_rows"] == 1
            for entry in fts_query_audit
        )
        assert candidates[0]["fts_query_matches"] == [
            {
                "subquery": "dosis actual de amlodipino",
                "query": "amlodipine",
                "kind": "runtime_alias_or",
                "match_mode": "implicit_and",
                "position_rank": 1,
                "source": "alias_anchor",
                "non_evidential": True,
            }
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_runtime_alias_fts_does_not_expand_proper_names_or_codes() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Carolina owns the deploy calendar.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_wrong_carolina",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Falcon uses SA42 for deployment approvals.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_wrong_sa42",
        )
        plan = RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            sub_query_plans=[
                PlannedSubQuery(
                    text="Caroline",
                    sparse_phrase="caroline",
                    must_keep_terms=["Caroline"],
                    fts_queries=["caroline"],
                    fts_query_kinds=["sparse_and"],
                ),
                PlannedSubQuery(
                    text="SA43",
                    sparse_phrase="SA43",
                    must_keep_terms=["SA43"],
                    fts_queries=["sa43"],
                    fts_query_kinds=["sparse_and"],
                ),
            ],
            query_type="slot_fill",
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=10,
            max_context_items=5,
            privacy_ceiling=1,
            retrieval_levels=[0],
            exact_recall_mode=True,
        )
        fts_query_audit: list[dict[str, object]] = []

        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
            runtime_alias_groups=[
                RuntimeAliasGroupTrace(
                    sub_query_text="Caroline",
                    anchor_type="proper_name",
                    original_surface="Caroline",
                    preserve_verbatim=True,
                    anchor_confidence=0.95,
                    aliases=[
                        RuntimeAliasSurfaceTrace(
                            surface="Carolina",
                            alias_kind="translation",
                            confidence=0.7,
                        )
                    ],
                ),
                RuntimeAliasGroupTrace(
                    sub_query_text="SA43",
                    anchor_type="code",
                    original_surface="SA43",
                    preserve_verbatim=True,
                    anchor_confidence=0.95,
                    aliases=[
                        RuntimeAliasSurfaceTrace(
                            surface="SA42",
                            alias_kind="spelling_variant",
                            confidence=0.7,
                        )
                    ],
                ),
            ],
        )

        assert candidates == []
        assert not any(entry.get("source") == "alias_anchor" for entry in fts_query_audit)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_respects_scope_filter() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
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
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="websocket retry loop in FastAPI",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
        )

        candidates = await search.search(
            _plan(scope_filter=[MemoryScope.CONVERSATION], status_filter=[MemoryStatus.ACTIVE]),
            user_id="usr_1",
        )

        assert [candidate["scope"] for candidate in candidates] == ["chat"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_filters_summary_mirrors_by_retrieval_levels() -> None:
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
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
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[1, 0],
            sub_queries=["debugging preference"],
        )
        plan_atomic = _plan(
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[0],
            sub_queries=["debugging preference"],
        )
        plan_all = _plan(
            scope_filter=[MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            retrieval_levels=[2, 1, 0],
            sub_queries=["debugging preference"],
        )

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
    connection, _messages, memories, search = await _build_candidate_runtime()
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
    connection, _messages, memories, search = await _build_candidate_runtime()
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


def test_retrieval_fts_queries_generate_multiple_mechanical_rewrites() -> None:
    queries = build_retrieval_fts_queries("Jon lost banker job last year")
    assert len(queries) == 3
    assert queries[0] == "jon lost banker job"
    assert queries[1] == "jon lost banker"
    assert queries[2] == "jon OR lost OR banker OR job OR last OR year"


def test_retrieval_fts_queries_preserve_distinct_content_terms_after_operator_cleanup() -> None:
    queries = build_retrieval_fts_queries("Jon and Gina share photography project")
    all_terms: set[str] = set()
    for query in queries:
        for part in query.replace(" OR ", " ").split():
            all_terms.add(part)
    expected = {"jon", "gina", "share", "photography", "project"}
    assert expected.issubset(all_terms)
    assert "and" not in all_terms


def test_retrieval_fts_queries_strip_fts5_operators() -> None:
    queries = build_retrieval_fts_queries("NOT related OR nearby topics")
    all_terms: set[str] = set()
    for query in queries:
        for part in query.replace(" OR ", " ").split():
            all_terms.add(part)
    assert "related" in all_terms
    assert "nearby" in all_terms
    assert "topics" in all_terms
    assert "not" not in all_terms


def test_retrieval_fts_queries_unicode_tokens_survive_mechanical_rewrite() -> None:
    queries = build_retrieval_fts_queries("María viajó Bogotá conferencia anual")
    assert queries == [
        "maría viajó bogotá conferencia",
        "maría viajó bogotá",
        "maría OR viajó OR bogotá OR conferencia OR anual",
    ]


def test_retrieval_fts_queries_cjk_input_produces_a_query() -> None:
    queries = build_retrieval_fts_queries("東京駅で待ち合わせ")
    assert queries == ["東京駅で待ち合わせ"]


def test_retrieval_fts_queries_short_input() -> None:
    queries = build_retrieval_fts_queries("coffee preference")
    assert len(queries) == 1
    assert queries[0] == "coffee preference"


def test_retrieval_fts_queries_single_word() -> None:
    queries = build_retrieval_fts_queries("hello")
    assert len(queries) == 1
    assert queries[0] == "hello"


def test_retrieval_fts_queries_three_content_tokens() -> None:
    queries = build_retrieval_fts_queries("Jon lost banker")
    assert len(queries) == 2
    assert queries[0] == "jon lost banker"
    assert queries[1] == "jon OR lost OR banker"


def test_retrieval_fts_query_specs_label_mechanical_rewrites() -> None:
    specs = build_retrieval_fts_query_specs("Jon lost banker")
    assert [(spec.query, spec.kind) for spec in specs] == [
        ("jon lost banker", "default_and"),
        ("jon OR lost OR banker", "broad_or"),
    ]


def test_exact_recall_fts_queries_add_anchor_only_and_without_changing_default_shape() -> None:
    default_queries = build_retrieval_fts_queries(
        "dose amlodipine 10 mg",
        must_keep_terms=["amlodipine", "10 mg"],
    )
    exact_specs = build_retrieval_fts_query_specs(
        "dose amlodipine 10 mg",
        must_keep_terms=["amlodipine", "10 mg"],
        exact_recall=True,
    )

    assert default_queries == [
        "amlodipine 10 mg dose",
        "dose amlodipine 10 mg",
        "dose OR amlodipine OR 10 OR mg",
    ]
    assert [(spec.query, spec.kind) for spec in exact_specs] == [
        ("amlodipine 10 mg dose", "anchor_first_and"),
        ("dose amlodipine 10 mg", "sparse_and"),
        ("amlodipine 10 mg", "anchor_only_and"),
        ("dose OR amlodipine OR 10 OR mg", "broad_or"),
    ]


def test_exact_recall_preserves_broad_or_when_noisy_precise_hints_fill_budget() -> None:
    specs = build_retrieval_fts_query_specs(
        "did jamie initially share information about",
        quoted_phrases=["dr nguyen", "march 12th", "february 2nd"],
        must_keep_terms=[
            "Jamie",
            "allergy",
            "Dr Nguyen",
            "Taylor",
            "March 12th",
            "February 2nd",
        ],
        exact_recall=True,
    )

    assert [(spec.query, spec.kind) for spec in specs] == [
        ('"dr nguyen"', "quoted_phrase"),
        ('"march 12th"', "quoted_phrase"),
        ('"february 2nd"', "quoted_phrase"),
        ("did jamie initially share information about", "sparse_and"),
        (
            "did OR jamie OR initially OR share OR information OR about OR allergy "
            "OR dr OR nguyen OR taylor OR march OR 12th OR february OR 2nd",
            "broad_or",
        ),
    ]


@pytest.mark.asyncio
async def test_content_bearing_query_finds_candidates_via_or_fallback() -> None:
    """Integration: seed memories and verify a content-bearing query finds them."""
    connection, _messages, memories, search = await _build_candidate_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Jon lost his job as a banker last year",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )

        planner = RetrievalPlanner()
        policy = _resolved_policy()
        plan = planner.build_plan(
            original_query="What happened to Jon's banking job?",
            query_intelligence=QueryIntelligenceResult(
                needs=[],
                temporal_range=None,
                sub_queries=["jon lost job as banker"],
                query_type="default",
                retrieval_levels=[0],
            ),
            conversation_context=_context(),
            resolved_policy=policy,
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


@pytest.mark.asyncio
async def test_callback_bias_prefers_assistant_originated_memories() -> None:
    connection, messages, memories, search = await _build_candidate_runtime()
    try:
        await messages.create_message(
            message_id="msg_assistant_1",
            conversation_id="cnv_1",
            role="assistant",
            seq=1,
            text="Try the apple recipe with cinnamon and lemon zest.",
            commit=True,
        )
        await messages.create_message(
            message_id="msg_user_1",
            conversation_id="cnv_1",
            role="user",
            seq=2,
            text="I wrote down an apple recipe with cinnamon too.",
            commit=True,
        )
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="apple recipe cinnamon lemon zest",
            payload={"source_message_ids": ["msg_assistant_1"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_z_assistant",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="apple recipe cinnamon lemon zest",
            payload={"source_message_ids": ["msg_user_1"]},
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            memory_id="mem_a_user",
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-03-30T20:00:00+00:00", "mem_z_assistant"),
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-03-30T20:05:00+00:00", "mem_a_user"),
        )
        await connection.commit()

        callback_candidates = await search.search(
            _plan(
                scope_filter=[MemoryScope.CONVERSATION],
                status_filter=[MemoryStatus.ACTIVE],
                sub_queries=["What was that apple recipe you told me about?"],
                callback_bias=True,
            ),
            user_id="usr_1",
        )
        neutral_candidates = await search.search(
            _plan(
                scope_filter=[MemoryScope.CONVERSATION],
                status_filter=[MemoryStatus.ACTIVE],
                sub_queries=["What was that apple recipe you told me about?"],
                callback_bias=False,
            ),
            user_id="usr_1",
        )

        assert neutral_candidates[0]["id"] == "mem_a_user"
        assert callback_candidates[0]["id"] == "mem_z_assistant"
        assert callback_candidates[0]["assistant_source_match"] is True
    finally:
        await connection.close()
