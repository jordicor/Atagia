"""Tests for intimacy-bound memory policy gates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.intimacy_boundary_policy import INTIMACY_FILTER_REASON
from atagia.memory.intimacy_boundary_policy import (
    coalesced_intimacy_sql_clause,
    memory_object_intimacy_sql_clause,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.services.chat_support import render_topic_working_set_block
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalPlan,
)
from atagia.services.llm_client import LLMClient

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class _UnusedLLMClient(LLMClient[object]):
    async def complete(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM is not used by deterministic policy tests")

    async def complete_structured(self, request, response_model):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM is not used by deterministic policy tests")


async def _build_runtime() -> tuple[object, MemoryObjectRepository, CandidateSearch, FrozenClock]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    search = CandidateSearch(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "general_qa", "Chat")
    return connection, memories, search, clock


def _plan(*, allow_intimacy_context: bool) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id="general_qa",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["pottery"],
        sub_query_plans=[{"text": "pottery", "fts_queries": ["pottery"]}],
        scope_filter=[MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=3,
        allow_intimacy_context=allow_intimacy_context,
        retrieval_levels=[0],
    )


@pytest.mark.asyncio
async def test_candidate_search_filters_intimacy_boundary_until_authorized() -> None:
    connection, memories, search, _clock = await _build_runtime()
    try:
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The user keeps a pottery checklist for the studio.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_ordinary",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The user keeps a private pottery note for intimacy mode.",
            index_text="The user keeps a private pottery-related note for explicit intimacy context.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=2,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.9,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_intimate",
        )

        ordinary_results = await search.search(_plan(allow_intimacy_context=False), "usr_1")
        authorized_results = await search.search(_plan(allow_intimacy_context=True), "usr_1")

        assert [row["id"] for row in ordinary_results] == ["mem_ordinary"]
        assert {row["id"] for row in authorized_results} == {"mem_ordinary", "mem_intimate"}
    finally:
        await connection.close()


def test_applicability_scorer_reports_intimacy_filter_reason() -> None:
    clock = FrozenClock(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc))
    scorer = ApplicabilityScorer(_UnusedLLMClient(), clock)
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["general_qa"]
    policy = PolicyResolver().resolve(manifest, None, None)
    reason = scorer.candidate_filter_reason(
        {
            "id": "mem_intimate",
            "scope": "conversation",
            "status": "active",
            "privacy_level": 2,
            "intimacy_boundary": "romantic_private",
        },
        policy,
        [],
        retrieval_plan=_plan(allow_intimacy_context=False),
    )

    assert reason == INTIMACY_FILTER_REASON


def test_topic_working_set_block_filters_intimacy_boundary_by_default() -> None:
    snapshot = {
        "active_topics": [
            {
                "id": "tpc_public",
                "status": "active",
                "title": "Pottery",
                "summary": "Studio checklist",
                "intimacy_boundary": "ordinary",
            },
            {
                "id": "tpc_private",
                "status": "active",
                "title": "Private mode",
                "summary": "Private continuity",
                "intimacy_boundary": "romantic_private",
            },
        ],
        "parked_topics": [],
        "freshness": {"status": "fresh"},
    }

    ordinary_block = render_topic_working_set_block(snapshot)
    authorized_block = render_topic_working_set_block(snapshot, allow_intimacy_context=True)

    assert "Pottery" in ordinary_block
    assert "Private mode" not in ordinary_block
    assert "Private mode" in authorized_block


def test_intimacy_sql_clause_builders_default_null_boundaries_to_ordinary() -> None:
    assert (
        memory_object_intimacy_sql_clause("mo", allow_intimacy_context=False)
        == "COALESCE(mo.intimacy_boundary, 'ordinary') = 'ordinary'"
    )
    assert (
        coalesced_intimacy_sql_clause("tendency", "outcome", allow_intimacy_context=False)
        == "COALESCE(tendency.intimacy_boundary, outcome.intimacy_boundary, 'ordinary') = 'ordinary'"
    )


def test_intimacy_sql_clause_builders_reject_untrusted_aliases() -> None:
    with pytest.raises(ValueError, match="Invalid SQL alias"):
        memory_object_intimacy_sql_clause("mo; DROP TABLE memory_objects", allow_intimacy_context=False)
