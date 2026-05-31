"""Tests for the flag-gated fact/facet retrieval channel."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_fact_facet_repository import MemoryFactFacetRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExactFacet,
    MemoryEvidenceSupportKind,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings(**overrides: object) -> Settings:
    base = Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )
    return Settings(**{**asdict(base), **overrides})


def _plan() -> RetrievalPlan:
    return RetrievalPlan(
        original_query="What is my location?",
        assistant_mode_id="coding_debug",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        fts_queries=["location"],
        sub_query_plans=[
            PlannedSubQuery(
                text="location",
                sparse_phrase="location",
                must_keep_terms=["location"],
                fts_queries=["location"],
            )
        ],
        query_type="slot_fill",
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.WORKSPACE],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        retrieval_levels=[0],
        exact_recall_mode=True,
        exact_facets=[ExactFacet.LOCATION],
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
    )


async def _seed_fact_facet(
    connection,
    clock: FrozenClock,
    *,
    surface_class: str = "structured",
) -> str:
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    evidence = MemoryEvidenceRepository(connection, clock)
    facets = MemoryFactFacetRepository(connection, clock)

    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    await messages.create_message(
        "msg_1",
        "cnv_1",
        "user",
        1,
        "My current city is Paris.",
        6,
        {},
        "2026-01-02T10:00:00+00:00",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.BELIEF,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Paris",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.91,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_city",
        platform_id="default",
        character_id="wrk_1",
        scope_canonical=MemoryScope.CHAT.value,
        language_codes=["en"],
        commit=False,
    )
    packet = await evidence.create_support_edge_with_spans(
        user_id="usr_1",
        memory_id="mem_city",
        support_kind=MemoryEvidenceSupportKind.DIRECT,
        confidence=0.91,
        spans=[
            {
                "span_role": "source",
                "message_id": "msg_1",
                "conversation_id": "cnv_1",
                "quote_text": "My current city is Paris.",
                "occurred_at": "2026-01-02T10:00:00+00:00",
            }
        ],
        commit=False,
    )
    source_span = packet["spans"][0]
    fact = await facets.upsert_fact_facet(
        user_id="usr_1",
        memory_id="mem_city",
        source_span_id=str(source_span["id"]),
        source_message_id="msg_1",
        conversation_id="cnv_1",
        subject_surface="usr_1",
        surface_class=surface_class,
        facet_label="location.current_city",
        value_text="Paris",
        list_group_key="location.current_city",
        support_kind=MemoryEvidenceSupportKind.DIRECT.value,
        observed_at="2026-01-02T10:00:00+00:00",
        current_state=True,
        language_code="en",
        confidence=0.91,
        commit=False,
    )
    await connection.commit()
    return str(fact["id"])


@pytest.mark.asyncio
async def test_fact_facet_candidate_channel_is_flag_gated() -> None:
    clock = FrozenClock(datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc))
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        fact_id = await _seed_fact_facet(connection, clock)

        disabled_search = CandidateSearch(
            connection,
            clock,
            settings=_settings(fact_facet_retrieval_enabled=False),
        )
        disabled_results = await disabled_search.search(_plan(), "usr_1")
        assert all("fact_facet" not in row.get("retrieval_sources", []) for row in disabled_results)

        enabled_search = CandidateSearch(
            connection,
            clock,
            settings=_settings(fact_facet_retrieval_enabled=True),
        )
        results = await enabled_search.search(_plan(), "usr_1")

        fact_candidates = [
            row for row in results if row.get("is_fact_facet_candidate") is True
        ]
        assert len(fact_candidates) == 1
        candidate = fact_candidates[0]
        assert candidate["id"] == fact_id
        assert candidate["fact_facet_memory_id"] == "mem_city"
        assert candidate["retrieval_sources"] == ["fact_facet"]
        assert candidate["channel_ranks"]["fact_facet"] == 1
        assert candidate["payload_json"]["value_text"] == "Paris"
        assert candidate["payload_json"]["value_norm_key"] == "paris"
        assert candidate["payload_json"]["surface_class"] == "structured"
        assert candidate["payload_json"]["fact_facet"]["surface_class"] == "structured"
        assert candidate["fact_facet_surface_class"] == "structured"
        assert candidate["payload_json"]["fact_facet"]["facet_label"] == "location.current_city"
        evidence_packet = candidate["evidence_packets"][0]
        assert "evidence_polarity" not in evidence_packet
        assert "status" not in evidence_packet
        assert evidence_packet["spans"][0]["quote_text"] == "My current city is Paris."
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_fact_facet_structured_only_excludes_generic_rows() -> None:
    clock = FrozenClock(datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc))
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_fact_facet(connection, clock, surface_class="generic")

        broad_search = CandidateSearch(
            connection,
            clock,
            settings=_settings(
                fact_facet_retrieval_enabled=True,
                fact_facet_structured_only=False,
            ),
        )
        broad_results = await broad_search.search(_plan(), "usr_1")
        assert any(row.get("is_fact_facet_candidate") is True for row in broad_results)

        structured_search = CandidateSearch(
            connection,
            clock,
            settings=_settings(
                fact_facet_retrieval_enabled=True,
                fact_facet_structured_only=True,
            ),
        )
        structured_results = await structured_search.search(_plan(), "usr_1")
        assert all(
            row.get("is_fact_facet_candidate") is not True
            for row in structured_results
        )
    finally:
        await connection.close()


def test_fact_facet_pointer_does_not_dedupe_verbatim_evidence_window() -> None:
    aggregated = {
        "mff_city": {
            "id": "mff_city",
            "is_fact_facet_candidate": True,
            "payload_json": {"source_message_ids": ["msg_1"]},
        },
        "vew_msg_1": {
            "id": "vew_msg_1",
            "is_verbatim_evidence_window": True,
            "verbatim_evidence_window_message_ids": ["msg_1"],
        },
        "mem_direct": {
            "id": "mem_direct",
            "payload_json": {"source_message_ids": ["msg_2"]},
        },
        "vew_msg_2": {
            "id": "vew_msg_2",
            "is_verbatim_evidence_window": True,
            "verbatim_evidence_window_message_ids": ["msg_2"],
        },
    }

    CandidateSearch._dedupe_verbatim_evidence_windows_against_memories(
        aggregated,
        plan=_plan().model_copy(update={"exact_recall_mode": False}),
    )

    assert "vew_msg_1" in aggregated
    assert "vew_msg_2" not in aggregated


def test_fact_facet_query_terms_do_not_apply_english_operator_stoplist() -> None:
    plan = _plan()
    sub_query = PlannedSubQuery(
        text="and near clinic",
        sparse_phrase="and near clinic",
        must_keep_terms=[],
        fts_queries=["and near clinic"],
    )

    terms = CandidateSearch._fact_facet_query_terms(plan, sub_query)

    assert "and" in terms
    assert "near" in terms
    assert "clinic" in terms


def test_fact_facet_like_pattern_escapes_sql_wildcards() -> None:
    assert CandidateSearch._fact_facet_like_pattern(r"50%_done\x") == r"%50\%\_done\\x%"
