"""Tests for the verbatim-evidence-search search channel (Wave 1 batch 2, task 1-C)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExactFacet,
    IntimacyBoundary,
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    PlannedSubQuery,
    MemoryStatus,
    RetrievalPlan,
    SpaceBoundaryMode,
    SummaryViewKind,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = dict(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
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
    )
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


async def _build_runtime(settings: Settings | None = None):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    resolved_settings = settings or _settings()
    search = CandidateSearch(connection, clock, settings=resolved_settings)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    return (
        connection,
        users,
        workspaces,
        conversations,
        messages,
        memories,
        search,
        resolved_settings,
    )


async def _create_conversation(
    conversations: ConversationRepository,
    *,
    conversation_id: str,
    user_id: str,
    workspace_id: str | None,
    assistant_mode_id: str,
    title: str,
) -> None:
    await conversations.create_conversation(
        conversation_id,
        user_id,
        workspace_id,
        assistant_mode_id,
        title,
    )


async def _seed_messages(
    messages: MessageRepository,
    *,
    conversation_id: str,
    texts: list[tuple[str, str]],
    space_id: str | None = None,
) -> list[dict[str, object]]:
    """Insert a sequence of (role, text) messages into a conversation."""
    created: list[dict[str, object]] = []
    for role, text in texts:
        row = await messages.create_message(
            message_id=None,
            conversation_id=conversation_id,
            role=role,
            seq=None,
            text=text,
            space_id=space_id,
        )
        created.append(row)
    return created


def _plan(
    *,
    assistant_mode_id: str,
    conversation_id: str,
    workspace_id: str | None,
    privacy_ceiling: int,
    fts_query: str,
    scope_filter: list[MemoryScope] | None = None,
    query_type: str = "default",
    exact_recall_mode: bool = False,
    raw_context_access_mode: str = "normal",
    privacy_enforcement: str = "enforce",
    active_space_id: str | None = None,
    active_space_boundary_mode: SpaceBoundaryMode | None = None,
    incognito: bool = False,
    remember_across_chats: bool = True,
    allow_intimacy_context: bool = False,
    exact_facets: list[ExactFacet] | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        assistant_mode_id=assistant_mode_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        fts_queries=[fts_query],
        sub_query_plans=[
            {
                "text": fts_query,
                "fts_queries": [fts_query],
            }
        ],
        query_type=query_type,  # type: ignore[arg-type]
        scope_filter=scope_filter
        or [
            MemoryScope.CONVERSATION,
            MemoryScope.WORKSPACE,
            MemoryScope.ASSISTANT_MODE,
            MemoryScope.GLOBAL_USER,
        ],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=privacy_ceiling,
        privacy_enforcement=privacy_enforcement,  # type: ignore[arg-type]
        allow_intimacy_context=allow_intimacy_context,
        incognito=incognito,
        remember_across_chats=remember_across_chats,
        retrieval_levels=[0],
        require_evidence_regrounding=False,
        skip_retrieval=False,
        exact_recall_mode=exact_recall_mode,
        exact_facets=exact_facets or [],
        raw_context_access_mode=raw_context_access_mode,  # type: ignore[arg-type]
        active_space_id=active_space_id,
        active_space_boundary_mode=active_space_boundary_mode,
    )


def test_verbatim_evidence_scope_keeps_active_chat_window_narrow_under_cross_search() -> None:
    plan = _plan(
        assistant_mode_id="general_qa",
        conversation_id="cnv_active",
        workspace_id=None,
        privacy_ceiling=3,
        fts_query="visited places",
        scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
        query_type="broad_list",
        exact_recall_mode=True,
    )

    active_scope = CandidateSearch._resolve_verbatim_evidence_window_scope(
        channel_scope=MemoryScope.GLOBAL_USER,
        plan=plan,
        conversation_id="cnv_active",
    )
    other_scope = CandidateSearch._resolve_verbatim_evidence_window_scope(
        channel_scope=MemoryScope.GLOBAL_USER,
        plan=plan,
        conversation_id="cnv_other",
    )

    assert active_scope is MemoryScope.CONVERSATION
    assert other_scope is MemoryScope.GLOBAL_USER


def test_verbatim_evidence_exact_recall_can_show_artifact_backed_message_text() -> None:
    messages = [
        {
            "id": "msg_artifact",
            "seq": 4,
            "role": "assistant",
            "text": "Jon: I have been to Paris.",
            "include_raw": False,
            "skip_by_default": True,
            "policy_reason": "artifact_backed",
            "content_kind": "artifact",
        }
    ]

    placeholder = CandidateSearch._format_verbatim_evidence_window_text(
        messages,
        include_skipped_raw=False,
    )
    raw = CandidateSearch._format_verbatim_evidence_window_text(
        messages,
        include_skipped_raw=True,
    )

    assert "Skipped message" in placeholder
    assert "Jon: I have been to Paris." in raw


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_returns_conversation_windows() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "I was born on 14 march 1988 in Barcelona"),
                ("assistant", "Thanks, noted your birthday."),
                ("user", "Please remind me about passport renewal"),
                ("assistant", "Sure."),
            ],
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Private biographical anchor.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=2,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_private_birth_anchor",
            payload={"source_message_ids": [str(seeded[0]["id"])]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="born march 1988 barcelona",
            ),
            "usr_1",
        )

        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, "expected at least one verbatim evidence search window candidate"
        top_window = evidence_windows[0]
        assert top_window["object_type"] == MemoryObjectType.EVIDENCE.value
        assert top_window["source_kind"] == MemorySourceKind.VERBATIM.value
        assert "verbatim_evidence_search" in top_window["channel_ranks"]
        assert "verbatim_evidence_search" in top_window["retrieval_sources"]
        assert top_window["verbatim_evidence_window_conversation_id"] == "cnv_1"
        assert top_window["privacy_level"] == 2
        assert "march 1988" in top_window["canonical_text"].lower()
        payload = top_window["payload_json"]
        assert isinstance(payload, dict)
        assert payload["source_message_window_start_occurred_at"]
        assert payload["source_message_window_end_occurred_at"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_keeps_own_severance_space_window() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await SpaceRepository(connection, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))).resolve_space(
            owner_user_id="usr_1",
            space_id="space_severed",
            boundary_mode=SpaceBoundaryMode.SEVERANCE,
            display_name="Severed",
            source_kind="explicit",
            source_id="space_severed",
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            space_id="space_severed",
            texts=[
                ("user", "The exact azimuth marker is copper seventeen."),
                ("assistant", "I will keep the marker local to this room."),
            ],
        )

        outside_candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=1,
                fts_query="azimuth copper seventeen",
                exact_recall_mode=True,
                raw_context_access_mode="verbatim",
            ),
            "usr_1",
        )
        inside_candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=1,
                fts_query="azimuth copper seventeen",
                exact_recall_mode=True,
                raw_context_access_mode="verbatim",
                active_space_id="space_severed",
                active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
            ),
            "usr_1",
        )

        outside_windows = [
            candidate
            for candidate in outside_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        inside_windows = [
            candidate
            for candidate in inside_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        assert outside_windows == []
        assert inside_windows
        assert inside_windows[0]["space_id"] == "space_severed"
        assert inside_windows[0]["space_boundary_mode"] == SpaceBoundaryMode.SEVERANCE.value
        assert "azimuth marker" in inside_windows[0]["canonical_text"].lower()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_window_privacy_level_filters_above_plan_ceiling() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "The private orchid code is violet seven."),
                ("assistant", "I will treat that carefully."),
            ],
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Sensitive code anchor.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=3,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_sensitive_code_anchor",
            payload={"source_message_ids": [str(seeded[0]["id"])]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="orchid violet seven",
            ),
            "usr_1",
        )

        assert not any(candidate.get("is_verbatim_evidence_window") for candidate in candidates)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_marks_secret_source_windows_for_redaction() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My vault PIN is 7391."),
                ("assistant", "I will treat that as protected."),
            ],
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Protected credential anchor.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.95,
            privacy_level=3,
            status=MemoryStatus.ACTIVE,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
            preserve_verbatim=True,
            intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
            intimacy_boundary_confidence=0.72,
            memory_id="mem_vault_pin",
            payload={"source_message_ids": [str(seeded[0]["id"])]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=3,
                fts_query="vault PIN 7391",
                allow_intimacy_context=True,
            ),
            "usr_1",
        )
        evidence_window = next(
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        )

        assert evidence_window["privacy_level"] == 3
        assert evidence_window["memory_category"] == MemoryCategory.PIN_OR_PASSWORD.value
        assert evidence_window["preserve_verbatim"] is True
        assert evidence_window["intimacy_boundary"] == IntimacyBoundary.ROMANTIC_PRIVATE.value
        assert evidence_window["intimacy_boundary_confidence"] == pytest.approx(0.72)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_exact_recall_falls_back_when_precise_query_returns_no_windows() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                (
                    "user",
                    "On Tuesdays I take amlodipine 10 mg and it makes me dizzy for a few hours.",
                ),
                ("assistant", "Thanks, I noted the Tuesday amlodipine dose."),
            ],
        )

        plan = RetrievalPlan(
            original_query="What is Rosa's current amlodipine dose?",
            assistant_mode_id="general_qa",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            fts_queries=[
                "rosa amlodipine dose current",
                "rosa OR amlodipine OR dose OR current",
            ],
            sub_query_plans=[
                PlannedSubQuery(
                    text="What is Rosa's current amlodipine dose?",
                    sparse_phrase="Rosa amlodipine dose",
                    must_keep_terms=["Rosa", "amlodipine", "dose", "current"],
                    fts_queries=[
                        "rosa amlodipine dose current",
                        "rosa OR amlodipine OR dose OR current",
                    ],
                )
            ],
            query_type="slot_fill",
            scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            vector_limit=0,
            max_candidates=10,
            max_context_items=8,
            privacy_ceiling=2,
            retrieval_levels=[0],
            exact_recall_mode=True,
            exact_facets=[ExactFacet.MEDICATION, ExactFacet.QUANTITY],
        )

        candidates = await search.search(plan, "usr_1")

        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, "expected verbatim-evidence-search fallback to recover a broader transcript window"
        top_window = evidence_windows[0]
        assert "amlodipine 10 mg" in str(top_window["canonical_text"]).lower()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_exact_recall_tries_original_query_rewrites() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My public name is PAU, like peace in Catalan."),
                ("assistant", "Noted."),
                ("user", "My biography later discusses the Paulet family."),
            ],
        )

        plan = RetrievalPlan(
            original_query=(
                "Did I explicitly say PAU means peace in Catalan, "
                "or only that it comes from Paulet?"
            ),
            assistant_mode_id="general_qa",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            fts_queries=["pau paulet biography"],
            sub_query_plans=[
                PlannedSubQuery(
                    text="Does PAU come from Paulet in the biography?",
                    sparse_phrase="PAU Paulet biography",
                    must_keep_terms=["PAU", "Paulet", "biography"],
                    fts_queries=["pau paulet biography"],
                )
            ],
            query_type="slot_fill",
            scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            vector_limit=0,
            max_candidates=10,
            max_context_items=8,
            privacy_ceiling=2,
            retrieval_levels=[0],
            exact_recall_mode=True,
            exact_facets=[ExactFacet.OTHER_VERBATIM],
        )

        candidates = await search.search(plan, "usr_1")

        evidence_texts = [
            str(candidate["canonical_text"]).lower()
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        assert any("like peace in catalan" in text for text in evidence_texts)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_exact_recall_keeps_fallback_windows_when_precise_query_is_sparse() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _resolved_settings = (
        await _build_runtime(_settings(verbatim_evidence_search_limit=4))
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "The biography mentions David and Arnau, but this note has no relationship detail."),
                ("assistant", "Noted."),
                ("user", "Filler note about a different topic."),
                ("assistant", "Noted."),
                (
                    "user",
                    "Relationship details for David and Arnau: David is my nephew, and Arnau is also my nephew.",
                ),
            ],
        )

        plan = RetrievalPlan(
            original_query="Who are David and Arnau in my biography?",
            assistant_mode_id="general_qa",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            fts_queries=[
                "david arnau biography",
                "david arnau relationship",
            ],
            sub_query_plans=[
                PlannedSubQuery(
                    text="Who are David and Arnau in my biography?",
                    sparse_phrase="David Arnau biography",
                    must_keep_terms=["David", "Arnau", "biography"],
                    fts_queries=[
                        "david arnau biography",
                        "david arnau relationship",
                    ],
                )
            ],
            query_type="broad_list",
            scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
            status_filter=[MemoryStatus.ACTIVE],
            vector_limit=0,
            max_candidates=10,
            max_context_items=8,
            privacy_ceiling=2,
            retrieval_levels=[0],
            exact_recall_mode=True,
            exact_facets=[ExactFacet.PERSON_NAME],
        )

        candidates = await search.search(plan, "usr_1")

        evidence_texts = [
            str(candidate["canonical_text"]).lower()
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        assert any("relationship details for david and arnau" in text for text in evidence_texts)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_exact_slot_fill_includes_follow_up_window() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("assistant", "Earlier filler about the week."),
                (
                    "user",
                    "Gina and Jon discuss the studio Jon opened yesterday.",
                ),
                ("assistant", "Jon says the lighting and floors are finished."),
                ("assistant", "Jon says the launch plan is almost ready."),
                ("user", "Gina: The studio looks amazing."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="gina jon studio opened",
                query_type="slot_fill",
                exact_recall_mode=True,
                exact_facets=[ExactFacet.OTHER_VERBATIM],
            ),
            "usr_1",
        )

        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows
        top_text = str(evidence_windows[0]["canonical_text"]).lower()
        assert "gina and jon discuss the studio jon opened" in top_text
        assert "studio looks amazing" in top_text
        assert evidence_windows[0]["verbatim_evidence_window_variant"] == "follow_up"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_deduped_against_memory_object() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My allergy is peanut and shellfish"),
                ("assistant", "Noted."),
            ],
        )
        verbatim_message_id = str(seeded[0]["id"])

        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="User allergy is peanut and shellfish",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_allergy",
            payload={"source_message_ids": [verbatim_message_id]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="allergy peanut shellfish",
            ),
            "usr_1",
        )

        ids = [candidate["id"] for candidate in candidates]
        assert "mem_allergy" in ids
        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows == [], "evidence window overlapping a retrieved memory must be deduped"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_not_deduped_against_summary_view() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "I met David and Jean while volunteering at the shelter"),
                ("assistant", "That sounds meaningful."),
            ],
        )
        verbatim_message_id = str(seeded[0]["id"])

        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.CONVERSATION,
            canonical_text=(
                "The user talked about David and Jean while volunteering "
                "at the shelter."
            ),
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="sum_volunteering",
            payload={
                "hierarchy_level": 0,
                "summary_kind": SummaryViewKind.CONVERSATION_CHUNK.value,
                "source_message_ids": [verbatim_message_id],
            },
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="David Jean shelter volunteering",
            ),
            "usr_1",
        )

        ids = [candidate["id"] for candidate in candidates]
        assert "sum_volunteering" in ids
        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, (
            "summary coverage must not suppress the verbatim evidence safety net"
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_exact_recall_preserves_verbatim_evidence_window_overlapping_memory_object() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My allergy is peanut and shellfish"),
                ("assistant", "Noted."),
            ],
        )
        verbatim_message_id = str(seeded[0]["id"])

        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="User allergy is peanut and shellfish",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_allergy",
            payload={"source_message_ids": [verbatim_message_id]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="allergy peanut shellfish",
                exact_recall_mode=True,
            ),
            "usr_1",
        )

        ids = [candidate["id"] for candidate in candidates]
        assert "mem_allergy" in ids
        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, "exact recall should keep overlapping verbatim evidence available"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_privacy_ceiling_blocks_therapy_mode() -> None:
    """Mandatory privacy test: therapy-mode messages never leak to general_qa.

    The SQL filter on ``assistant_modes.privacy_ceiling`` must exclude
    conversations belonging to modes whose privacy ceiling exceeds the
    current retrieval ceiling BEFORE any ranking or composition runs.
    """
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        # ``personal_assistant`` has privacy_ceiling=3 in the manifest
        # so we use it as the high-privacy surrogate for therapy mode.
        await _create_conversation(
            conversations,
            conversation_id="cnv_therapy",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="personal_assistant",
            title="Therapy",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_therapy",
            texts=[
                (
                    "user",
                    "I take sertraline 50mg every morning since september",
                ),
                ("assistant", "Thank you for sharing that."),
            ],
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_general",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="General",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_general",
            texts=[
                ("user", "Let me tell you about my sertraline dosage."),
                ("assistant", "Go ahead."),
            ],
        )

        # Sanity check: the repository-level privacy filter excludes
        # the therapy conversation outright.
        repo_rows = await MessageRepository(
            connection, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))
        ).search_messages_with_privacy(
            user_id="usr_1",
            query="sertraline",
            privacy_ceiling=2,
            limit=10,
        )
        returned_conversations = {row["conversation_id"] for row in repo_rows}
        assert "cnv_therapy" not in returned_conversations
        assert "cnv_general" in returned_conversations

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_general",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="sertraline",
            ),
            "usr_1",
        )

        for candidate in candidates:
            assert candidate.get("verbatim_evidence_window_conversation_id") != "cnv_therapy"
            payload = candidate.get("payload_json") or {}
            assert not isinstance(payload, dict) or "cnv_therapy" not in str(payload)
        # Positive assertion: the general_qa raw window must actually
        # surface for "sertraline" from the full pipeline. Without this
        # the privacy assertion above would pass trivially if the raw
        # channel silently returned zero results for any reason.
        assert len(candidates) >= 1
        assert any(
            c.get("verbatim_evidence_window_conversation_id") == "cnv_general" for c in candidates
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_multi_facet_exact_recall_can_surface_cross_conversation_raw_window() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_prior",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Prior apartment search",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_prior",
            texts=[
                (
                    "user",
                    "The apartment budget we set was exactly 2800 dollars per month.",
                ),
                ("assistant", "Got it."),
            ],
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_current",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Current apartment update",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_current",
            texts=[
                ("user", "The apartment we chose costs $2,650 per month."),
                ("assistant", "Understood."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_current",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="apartment budget 2800",
                query_type="broad_list",
                exact_recall_mode=True,
            ),
            "usr_1",
        )

        evidence_windows = [
            candidate
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        ]
        assert any(
            window.get("verbatim_evidence_window_conversation_id") == "cnv_prior"
            and "2800 dollars" in str(window.get("canonical_text"))
            and window.get("scope_canonical") == MemoryScope.USER.value
            for window in evidence_windows
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_multi_facet_exact_recall_cross_raw_allows_slot_fill_multi_exact_facets() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_prior",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Prior apartment search",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_prior",
            texts=[
                (
                    "user",
                    "The apartment budget we set was exactly 2800 dollars per month.",
                ),
            ],
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_current",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Current apartment update",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_current",
            texts=[
                ("user", "The apartment we chose costs $2,650 per month."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_current",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="apartment budget 2800",
                query_type="slot_fill",
                exact_recall_mode=True,
                exact_facets=[ExactFacet.QUANTITY, ExactFacet.PERSON_NAME],
            ),
            "usr_1",
        )

        assert any(
            candidate.get("verbatim_evidence_window_conversation_id") == "cnv_prior"
            and "2800 dollars" in str(candidate.get("canonical_text"))
            and candidate.get("scope_canonical") == MemoryScope.USER.value
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_single_facet_slot_fill_exact_recall_can_surface_cross_conversation_raw_window() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_prior",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Prior pharmacy update",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_prior",
            texts=[
                (
                    "user",
                    "My current pharmacy phone number is 555-0389.",
                ),
            ],
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_current",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Current general update",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_current",
            texts=[
                ("user", "My emergency contact phone number is 555-0263."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_current",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="pharmacy phone number",
                query_type="slot_fill",
                exact_recall_mode=True,
                exact_facets=[ExactFacet.PHONE],
            ),
            "usr_1",
        )

        assert any(
            candidate.get("verbatim_evidence_window_conversation_id") == "cnv_prior"
            and "555-0389" in str(candidate.get("canonical_text"))
            and candidate.get("scope_canonical") == MemoryScope.USER.value
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_multi_facet_exact_recall_cross_raw_respects_remember_gate() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_prior",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Prior apartment search",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_prior",
            texts=[
                ("user", "The apartment budget was exactly 2800 dollars per month."),
            ],
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_current",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Current apartment update",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_current",
            texts=[
                ("user", "The apartment we chose costs 2650 dollars per month."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_current",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="apartment budget 2800",
                query_type="broad_list",
                exact_recall_mode=True,
                remember_across_chats=False,
            ),
            "usr_1",
        )

        assert not any(
            candidate.get("verbatim_evidence_window_conversation_id") == "cnv_prior"
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_multi_facet_exact_recall_cross_raw_filters_pending_consent_sources() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_prior",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Prior apartment search",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_prior",
            texts=[
                ("user", "The apartment budget was exactly 2800 dollars per month."),
            ],
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_prior",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Pending apartment budget evidence.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            memory_id="mem_pending_budget",
            payload={"source_message_ids": [str(seeded[0]["id"])]},
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_current",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="general_qa",
            title="Current apartment update",
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_current",
                workspace_id=None,
                privacy_ceiling=2,
                fts_query="apartment budget 2800",
                query_type="broad_list",
                exact_recall_mode=True,
            ),
            "usr_1",
        )

        assert not any(
            candidate.get("verbatim_evidence_window_conversation_id") == "cnv_prior"
            for candidate in candidates
            if candidate.get("is_verbatim_evidence_window")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_allows_active_conversation_despite_privacy_ceiling() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_private",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="personal_assistant",
            title="Private",
        )
        await _create_conversation(
            conversations,
            conversation_id="cnv_other_private",
            user_id="usr_1",
            workspace_id=None,
            assistant_mode_id="personal_assistant",
            title="Other Private",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_private",
            texts=[
                (
                    "user",
                    "On Tuesdays I take amlodipine 10 mg and it makes me dizzy for a few hours.",
                ),
                ("assistant", "I noted the Tuesday amlodipine dose."),
            ],
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_other_private",
            texts=[
                ("user", "My therapist is Dr. Reeves."),
                ("assistant", "I will keep that private."),
            ],
        )

        candidates = await search.search(
            RetrievalPlan(
                original_query="What is Rosa's current amlodipine dose?",
                assistant_mode_id="personal_assistant",
                workspace_id=None,
                conversation_id="cnv_private",
                fts_queries=[
                    "rosa amlodipine dose current",
                    "rosa OR amlodipine OR dose OR current",
                ],
                sub_query_plans=[
                    PlannedSubQuery(
                        text="What is Rosa's current amlodipine dose?",
                        sparse_phrase="Rosa amlodipine dose",
                        must_keep_terms=["Rosa", "amlodipine", "dose", "current"],
                        fts_queries=[
                            "rosa amlodipine dose current",
                            "rosa OR amlodipine OR dose OR current",
                        ],
                    )
                ],
                query_type="slot_fill",
                scope_filter=[MemoryScope.CONVERSATION, MemoryScope.GLOBAL_USER],
                status_filter=[MemoryStatus.ACTIVE],
                vector_limit=0,
                max_candidates=10,
                max_context_items=8,
                privacy_ceiling=1,
                retrieval_levels=[0],
                exact_recall_mode=True,
                exact_facets=[ExactFacet.MEDICATION, ExactFacet.QUANTITY],
            ),
            "usr_1",
        )

        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows, "expected current conversation to bypass mode privacy ceiling"
        assert any(
            candidate.get("verbatim_evidence_window_conversation_id") == "cnv_private"
            and "amlodipine 10 mg" in str(candidate.get("canonical_text", "")).lower()
            for candidate in evidence_windows
        )
        assert all(
            candidate.get("verbatim_evidence_window_conversation_id") != "cnv_other_private"
            for candidate in evidence_windows
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_filters_pending_consent_messages() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My phone number is 555-0143 please remember it"),
                ("assistant", "Got it."),
                ("user", "Also my favorite colour is teal"),
            ],
        )
        gated_message_id = str(seeded[0]["id"])

        # A pending memory references the gated message; the raw
        # channel must exclude anything it sources.
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Phone number 555-0143",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            memory_id="mem_pending_phone",
            payload={"source_message_ids": [gated_message_id]},
        )

        repo = MessageRepository(
            connection, FrozenClock(datetime(2026, 4, 4, 11, 0, tzinfo=timezone.utc))
        )
        gated_rows = await repo.search_messages_with_privacy(
            user_id="usr_1",
            query="phone 555",
            privacy_ceiling=2,
            limit=10,
        )
        assert all(row["id"] != gated_message_id for row in gated_rows)

        # Unrelated lookups still work through the verbatim evidence search channel.
        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="favorite colour teal",
            ),
            "usr_1",
        )
        evidence_windows = [c for c in candidates if c.get("is_verbatim_evidence_window")]
        assert evidence_windows, "non-gated messages must still flow through verbatim evidence search"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_gates_pending_consent_neighbor_messages() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My private recovery code is alpha-9999"),
                ("assistant", "The shipping label is ready for pickup"),
                ("user", "Thanks."),
            ],
        )
        gated_message_id = str(seeded[0]["id"])
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Private recovery code alpha-9999",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            memory_id="mem_pending_recovery_code",
            payload={"source_message_ids": [gated_message_id]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="shipping label ready pickup",
                raw_context_access_mode="skipped_raw",
            ),
            "usr_1",
        )
        evidence_windows = [c for c in candidates if c.get("is_verbatim_evidence_window")]

        assert evidence_windows
        canonical_text = str(evidence_windows[0]["canonical_text"])
        assert "shipping label is ready" in canonical_text.lower()
        assert "alpha-9999" not in canonical_text
        assert "pending_user_confirmation" in canonical_text
        assert gated_message_id not in canonical_text
        assert gated_message_id not in evidence_windows[0]["payload_json"]["source_message_ids"]
        assert gated_message_id not in evidence_windows[0]["verbatim_evidence_window_message_ids"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_privacy_off_bypasses_pending_consent_gates() -> None:
    connection, _users, _workspaces, conversations, messages, memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "My private recovery code is alpha-9999"),
                ("assistant", "The shipping label is ready for pickup"),
                ("user", "Thanks."),
            ],
        )
        gated_message_id = str(seeded[0]["id"])
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Private recovery code alpha-9999",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            memory_id="mem_pending_recovery_code",
            payload={"source_message_ids": [gated_message_id]},
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=0,
                fts_query="private recovery alpha",
                privacy_enforcement="off",
            ),
            "usr_1",
        )
        evidence_windows = [c for c in candidates if c.get("is_verbatim_evidence_window")]

        assert evidence_windows
        canonical_text = str(evidence_windows[0]["canonical_text"])
        assert "alpha-9999" in canonical_text
        assert gated_message_id in evidence_windows[0]["payload_json"]["source_message_ids"]
        assert gated_message_id in evidence_windows[0]["verbatim_evidence_window_message_ids"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_uses_placeholder_for_protected_messages() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        protected = await messages.create_message(
            message_id=None,
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text="protected invoice token alpha-7788 should stay hidden",
            metadata={
                "include_raw": False,
                "skip_by_default": True,
                "content_kind": "attachment",
                "policy_reason": "heavy_content",
            },
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="protected invoice alpha",
            ),
            "usr_1",
        )
        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]

        assert evidence_windows
        assert "alpha-7788" not in evidence_windows[0]["canonical_text"]
        assert f"id={protected['id']}" in evidence_windows[0]["canonical_text"]
        assert "policy=heavy_content" in evidence_windows[0]["canonical_text"]

        raw_access_candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="protected invoice alpha",
                raw_context_access_mode="skipped_raw",
            ),
            "usr_1",
        )
        raw_access_windows = [
            candidate
            for candidate in raw_access_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]

        assert raw_access_windows
        assert "alpha-7788" not in raw_access_windows[0]["canonical_text"]
        assert f"id={protected['id']}" in raw_access_windows[0]["canonical_text"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_can_surface_mechanical_heavy_raw_on_explicit_access() -> None:
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        heavy = await messages.create_message(
            message_id=None,
            conversation_id="cnv_1",
            role="user",
            seq=None,
            text=(
                "mechanical heavy perfume alpha-7788 detail is safe to reveal on explicit recall. "
                + "padding "
                * 3000
            ),
        )

        normal_candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="perfume alpha",
            ),
            "usr_1",
        )
        normal_windows = [
            candidate
            for candidate in normal_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]

        assert normal_windows
        assert "alpha-7788" not in normal_windows[0]["canonical_text"]
        assert f"id={heavy['id']}" in normal_windows[0]["canonical_text"]

        raw_candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="perfume alpha",
                raw_context_access_mode="skipped_raw",
            ),
            "usr_1",
        )
        raw_windows = [
            candidate
            for candidate in raw_candidates
            if candidate.get("is_verbatim_evidence_window")
        ]

        assert raw_windows
        assert "alpha-7788" in raw_windows[0]["canonical_text"]
        assert f"id={heavy['id']}" not in raw_windows[0]["canonical_text"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_enabled_disabled_by_settings() -> None:
    settings = _settings(verbatim_evidence_search_enabled=False)
    connection, _users, _workspaces, conversations, messages, _memories, search, _settings_value = (
        await _build_runtime(settings)
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                ("user", "I love kiwi fruit very much"),
                ("assistant", "Nice."),
            ],
        )

        candidates = await search.search(
            _plan(
                assistant_mode_id="general_qa",
                conversation_id="cnv_1",
                workspace_id="wrk_1",
                privacy_ceiling=2,
                fts_query="kiwi fruit",
            ),
            "usr_1",
        )

        assert not any(candidate.get("is_verbatim_evidence_window") for candidate in candidates)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_exact_recall_plan_surfaces_raw_evidence_over_belief() -> None:
    """The exact recall boost must rank raw evidence above abstracted beliefs."""
    from atagia.core.clock import FrozenClock as _FrozenClock
    from atagia.memory.applicability_scorer import ApplicabilityScorer

    clock = _FrozenClock(datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc))
    scorer = ApplicabilityScorer.__new__(ApplicabilityScorer)
    scorer._clock = clock  # type: ignore[attr-defined]
    scorer._settings = _settings()  # type: ignore[attr-defined]

    plan = _plan(
        assistant_mode_id="general_qa",
        conversation_id="cnv_1",
        workspace_id=None,
        privacy_ceiling=2,
        fts_query="1988",
        exact_recall_mode=True,
    )

    evidence_candidate = {
        "id": "mem_evidence",
        "object_type": MemoryObjectType.EVIDENCE.value,
        "is_verbatim_evidence_window": True,
        "maya_score": 0.0,
        "temporal_type": "unknown",
    }
    belief_candidate = {
        "id": "mem_belief",
        "object_type": MemoryObjectType.BELIEF.value,
        "maya_score": 0.0,
        "temporal_type": "unknown",
    }
    summary_candidate = {
        "id": "mem_summary",
        "object_type": MemoryObjectType.SUMMARY_VIEW.value,
        "payload_json": {"hierarchy_level": 1, "summary_kind": "episode"},
        "maya_score": 0.0,
        "temporal_type": "unknown",
    }

    assert scorer._exact_recall_boost(evidence_candidate, plan) == pytest.approx(0.15)
    assert scorer._exact_recall_boost(belief_candidate, plan) == pytest.approx(-0.05)
    assert scorer._exact_recall_boost(summary_candidate, plan) == pytest.approx(-0.05)

    plan_no_recall = plan.model_copy(update={"exact_recall_mode": False})
    assert scorer._exact_recall_boost(evidence_candidate, plan_no_recall) == 0.0
    assert scorer._exact_recall_boost(belief_candidate, plan_no_recall) == 0.0


@pytest.mark.asyncio
async def test_verbatim_evidence_search_deduped_across_sub_queries() -> None:
    """Cross-sub-query dedup: a raw window fetched under sub-query B must
    be deduped against a memory object fetched under sub-query A when
    both cover the same source message.
    """
    connection, _users, _workspaces, conversations, messages, memories, search, _settings_value = (
        await _build_runtime()
    )
    try:
        await _create_conversation(
            conversations,
            conversation_id="cnv_1",
            user_id="usr_1",
            workspace_id="wrk_1",
            assistant_mode_id="general_qa",
            title="Main",
        )
        seeded = await _seed_messages(
            messages,
            conversation_id="cnv_1",
            texts=[
                (
                    "user",
                    "My medicine is ibuprofen 400mg for headaches",
                ),
                ("assistant", "Understood."),
            ],
        )
        verbatim_message_id = str(seeded[0]["id"])

        # The memory object's canonical_text matches sub-query A but NOT
        # sub-query B. The verbatim evidence search text matches sub-query B but NOT
        # sub-query A. Both reference the same source message.
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            assistant_mode_id="general_qa",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="User takes prescription medication",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_prescription",
            payload={"source_message_ids": [verbatim_message_id]},
        )

        plan = RetrievalPlan(
            assistant_mode_id="general_qa",
            workspace_id="wrk_1",
            conversation_id="cnv_1",
            fts_queries=["prescription medication", "ibuprofen 400mg"],
            sub_query_plans=[
                {
                    "text": "prescription medication",
                    "fts_queries": ["prescription medication"],
                },
                {
                    "text": "ibuprofen 400mg",
                    "fts_queries": ["ibuprofen 400mg"],
                },
            ],
            query_type="default",
            scope_filter=[
                MemoryScope.CONVERSATION,
                MemoryScope.WORKSPACE,
                MemoryScope.ASSISTANT_MODE,
                MemoryScope.GLOBAL_USER,
            ],
            status_filter=[MemoryStatus.ACTIVE],
            vector_limit=0,
            max_candidates=10,
            max_context_items=8,
            privacy_ceiling=2,
            retrieval_levels=[0],
            require_evidence_regrounding=False,
            skip_retrieval=False,
        )
        candidates = await search.search(plan, "usr_1")
        ids = [candidate["id"] for candidate in candidates]
        assert "mem_prescription" in ids
        # The verbatim evidence window surfaced under sub-query B references the
        # same source message as the memory object surfaced under
        # sub-query A, so it must be deduped in the final aggregated
        # candidate set.
        evidence_windows = [
            candidate for candidate in candidates if candidate.get("is_verbatim_evidence_window")
        ]
        assert evidence_windows == [], (
            "cross-sub-query dedup must drop evidence windows overlapping a memory "
            "object retrieved under a different sub-query"
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_verbatim_evidence_search_rrf_weight_affects_final_rank() -> None:
    """Lowering ``verbatim_evidence_search_rrf_weight`` must reduce the final rrf_score
    of a raw-only candidate relative to a memory-object candidate.
    """
    async def _run(raw_weight: float) -> dict[str, float]:
        settings = _settings(verbatim_evidence_search_rrf_weight=raw_weight)
        connection, _users, _workspaces, conversations, messages, memories, search, _settings_value = (
            await _build_runtime(settings)
        )
        try:
            await _create_conversation(
                conversations,
                conversation_id="cnv_1",
                user_id="usr_1",
                workspace_id="wrk_1",
                assistant_mode_id="general_qa",
                title="Main",
            )
            await _seed_messages(
                messages,
                conversation_id="cnv_1",
                texts=[
                    (
                        "user",
                        "Verbatim transcript alpha bravo charlie delta echo",
                    ),
                    ("assistant", "Noted."),
                ],
            )

            # A standalone memory object unrelated to the verbatim evidence search
            # text so it only surfaces via its own FTS terms.
            await memories.create_memory_object(
                user_id="usr_1",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                assistant_mode_id="general_qa",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="Curated memory foxtrot golf hotel",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                status=MemoryStatus.ACTIVE,
                memory_id="mem_curated",
                payload={"source_message_ids": []},
            )

            plan = RetrievalPlan(
                assistant_mode_id="general_qa",
                workspace_id="wrk_1",
                conversation_id="cnv_1",
                fts_queries=["alpha bravo", "foxtrot golf"],
                sub_query_plans=[
                    {
                        "text": "alpha bravo",
                        "fts_queries": ["alpha bravo"],
                    },
                    {
                        "text": "foxtrot golf",
                        "fts_queries": ["foxtrot golf"],
                    },
                ],
                query_type="default",
                scope_filter=[
                    MemoryScope.CONVERSATION,
                    MemoryScope.WORKSPACE,
                    MemoryScope.ASSISTANT_MODE,
                    MemoryScope.GLOBAL_USER,
                ],
                status_filter=[MemoryStatus.ACTIVE],
                vector_limit=0,
                max_candidates=10,
                max_context_items=8,
                privacy_ceiling=2,
                retrieval_levels=[0],
                require_evidence_regrounding=False,
                skip_retrieval=False,
            )
            candidates = await search.search(plan, "usr_1")
            scores: dict[str, float] = {}
            for candidate in candidates:
                if candidate.get("is_verbatim_evidence_window"):
                    scores["raw"] = float(candidate.get("rrf_score", 0.0))
                elif candidate.get("id") == "mem_curated":
                    scores["curated"] = float(candidate.get("rrf_score", 0.0))
            return scores
        finally:
            await connection.close()

    scores_full = await _run(1.0)
    scores_half = await _run(0.5)
    assert "raw" in scores_full and "curated" in scores_full
    assert "raw" in scores_half and "curated" in scores_half
    # Curated memory is a full-weight channel and its score must be
    # independent of the raw weight knob.
    assert scores_full["curated"] == pytest.approx(scores_half["curated"])
    # The raw-only candidate must move with the weight: halving the
    # raw weight strictly reduces its final rrf_score.
    assert scores_half["raw"] < scores_full["raw"]


@pytest.mark.asyncio
async def test_query_intelligence_accepts_exact_recall_fields() -> None:
    from atagia.models.schemas_memory import QueryIntelligenceResult

    result = QueryIntelligenceResult(
        needs=[],
        temporal_range=None,
        sub_queries=["birthday date"],
        sparse_query_hints=[
            {"sub_query_text": "birthday date", "fts_phrase": "birthday date"}
        ],
        query_type="slot_fill",
        retrieval_levels=[0],
        exact_recall_needed=True,
        exact_facets=[ExactFacet.DATE, ExactFacet.PERSON_NAME],
    )
    assert result.exact_recall_needed is True
    assert result.exact_facets == [ExactFacet.DATE, ExactFacet.PERSON_NAME]
