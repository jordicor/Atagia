"""Tests for materializing prepared initial-context packages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atagia.core import json_utils
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
from atagia.core.summary_repository import SummaryRepository
from atagia.core.topic_repository import TopicRepository
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    SpaceBoundaryMode,
)
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageProfileItem,
)
from atagia.services.initial_context_package_builder import (
    InitialContextPackageBuildBudget,
    InitialContextPackageBuilder,
)
from atagia.services.initial_context_package_curator import InitialContextPackageCurator
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _seed_runtime() -> tuple[aiosqlite.Connection, FrozenClock, Any]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    manifests = ManifestLoader(MANIFESTS_DIR).load_all()
    await sync_assistant_modes(connection, manifests, clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await UserRepository(connection, clock).create_user("usr_2")
    await WorkspaceRepository(connection, clock).create_workspace(
        "wrk_1",
        "usr_1",
        "Workspace",
    )
    conversations = ConversationRepository(connection, clock)
    await conversations.create_conversation(
        "cnv_1",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Active chat",
        user_persona_id="persona_jordi",
        platform_id="aurvek",
        character_id="assistant_alpha",
    )
    await conversations.create_conversation(
        "cnv_empty",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Empty chat",
        user_persona_id="persona_jordi",
        platform_id="aurvek",
        character_id="assistant_alpha",
    )
    resolved_policy = PolicyResolver().resolve(manifests["coding_debug"], None, None)
    return connection, clock, resolved_policy


async def _create_belief(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    *,
    user_id: str = "usr_1",
    memory_id: str,
    text: str,
    conversation_id: str | None = None,
    scope: MemoryScope = MemoryScope.USER,
    maya_score: float = 0.7,
    confidence: float = 0.9,
    stability: float = 0.8,
    vitality: float = 0.6,
    tension_score: float = 0.0,
    tension_updated_at: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    valid_to: str | None = None,
    payload: dict[str, Any] | None = None,
    user_persona_id: str | None = "persona_jordi",
    platform_id: str | None = "aurvek",
    character_id: str | None = None,
    workspace_id: str | None = "wrk_1",
    space_id: str | None = None,
    space_boundary_mode: str | None = None,
) -> dict[str, Any]:
    repository = MemoryObjectRepository(connection, clock)
    memory = await repository.create_memory_object(
        user_id=user_id,
        memory_id=memory_id,
        object_type=MemoryObjectType.BELIEF,
        scope=scope,
        canonical_text=text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=confidence,
        stability=stability,
        vitality=vitality,
        maya_score=maya_score,
        status=status,
        privacy_level=0,
        payload=payload,
        valid_to=valid_to,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        workspace_id=workspace_id if user_id == "usr_1" else None,
        user_persona_id=user_persona_id if user_id == "usr_1" else None,
        platform_id=platform_id,
        character_id=character_id
        if character_id is not None
        else ("assistant_alpha" if scope is MemoryScope.CHARACTER else None),
        scope_canonical=scope.value,
        space_id=space_id,
        space_boundary_mode=space_boundary_mode,
    )
    if tension_score or tension_updated_at is not None:
        await connection.execute(
            """
            UPDATE memory_objects
            SET tension_score = ?,
                tension_updated_at = ?
            WHERE user_id = ?
              AND id = ?
            """,
            (
                tension_score,
                tension_updated_at or clock.now().isoformat(),
                user_id,
                memory_id,
            ),
        )
        await connection.commit()
        refreshed = await repository.get_memory_object(memory_id, user_id)
        assert refreshed is not None
        return refreshed
    return memory


def _curation_settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_component_models={
            "initial_context_package_curation": "openai/curation-test-model",
        },
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=True,
        debug=False,
        allow_insecure_http=True,
    )


class CurationProvider(LLMProvider):
    name = "initial-context-package-curation-tests"

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        assert request.metadata["purpose"] == "initial_context_package_curation"
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json_utils.dumps(self.output, sort_keys=True),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in this test: {request.model}")


def _curator(output: dict[str, Any]) -> InitialContextPackageCurator:
    provider = CurationProvider(output)
    return InitialContextPackageCurator(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        settings=_curation_settings(),
    )


def _curator_with_provider(
    output: dict[str, Any],
) -> tuple[InitialContextPackageCurator, CurationProvider]:
    provider = CurationProvider(output)
    curator = InitialContextPackageCurator(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        settings=_curation_settings(),
    )
    return curator, provider


@pytest.mark.asyncio
async def test_baseline_package_includes_visible_profile_with_sources_only_for_user() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        memory = await _create_belief(
            connection,
            clock,
            memory_id="mem_visible",
            text="The user prefers direct Spanish replies.",
        )
        await _create_belief(
            connection,
            clock,
            user_id="usr_2",
            memory_id="mem_other_user",
            text="Other user prefers verbose answers.",
        )
        await connection.execute(
            """
            INSERT INTO memory_support_edges(
                id,
                user_id,
                memory_id,
                support_kind,
                evidence_polarity,
                confidence,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, 'direct', 'supports', 0.95, ?, ?)
            """,
            (
                "edge_visible",
                "usr_1",
                memory["id"],
                clock.now().isoformat(),
                clock.now().isoformat(),
            ),
        )
        await connection.commit()

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        block = package.blocks_json.prepared_memory_profile_block
        assert "direct Spanish replies" in block
        assert "Other user" not in block
        assert package.blocks_json.profile_items
        item = package.blocks_json.profile_items[0]
        assert item.scope_json["user_id"] == "usr_1"
        assert item.scope_json["scope_canonical"] == "user"
        assert item.source_refs[0]["memory_id"] == "mem_visible"
        assert any(ref["source_kind"] == "memory_evidence" for ref in item.source_refs)
        assert package.source_refs_json["profile_items"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_package_includes_summary_topic_and_recent_seed() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        messages = MessageRepository(connection, clock)
        await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Estamos disenando el paquete inicial preparado.",
            8,
            {},
        )
        await messages.create_message(
            "msg_2",
            "cnv_1",
            "assistant",
            2,
            "Tiene que ser rapido y no bloquear la respuesta.",
            10,
            {},
        )
        await SummaryRepository(connection, clock).create_summary(
            "usr_1",
            {
                "id": "sum_1",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 1,
                "source_message_end_seq": 2,
                "summary_kind": "conversation_chunk",
                "summary_text": "Se esta planificando un paquete inicial preparado.",
                "source_object_ids_json": [],
                "maya_score": 1.0,
                "model": "test-model",
                "created_at": clock.now().isoformat(),
            },
        )
        topics = TopicRepository(connection, clock)
        await topics.create_topic(
            user_id="usr_1",
            conversation_id="cnv_1",
            topic_id="tpc_1",
            title="Initial package planning",
            summary="Prepared recognition context for fast turn start.",
            active_goal="Define the package without replacing retrieval.",
            source_message_start_seq=1,
            source_message_end_seq=2,
            last_touched_seq=2,
        )
        await topics.link_source(
            user_id="usr_1",
            topic_id="tpc_1",
            source_kind="message",
            source_id="msg_1",
        )
        await topics.create_topic(
            user_id="usr_1",
            conversation_id="cnv_1",
            topic_id="tpc_private",
            title="Private hidden topic",
            summary="This source must not appear in refs.",
            privacy_level=2,
            source_message_start_seq=1,
            source_message_end_seq=2,
            last_touched_seq=2,
        )
        await topics.link_source(
            user_id="usr_1",
            topic_id="tpc_private",
            source_kind="message",
            source_id="msg_2",
        )

        package = await InitialContextPackageBuilder(connection, clock).build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_1",
            resolved_policy=resolved_policy,
        )

        assert "historical context only" in package.blocks_json.conversation_summary_block
        assert "paquete inicial preparado" in package.blocks_json.conversation_summary_block
        assert "Initial package planning" in package.blocks_json.working_topic_block
        assert len(package.blocks_json.recent_verbatim_seed) == 2
        assert package.blocks_json.recent_verbatim_seed[0]["message_id"] == "msg_1"
        assert package.blocks_json.empty_markers["conversation_summary_empty"] is False
        assert package.blocks_json.empty_markers["working_topic_empty"] is False
        assert package.blocks_json.empty_markers["recent_verbatim_seed_empty"] is False
        assert package.source_refs_json["conversation_summary"][0]["summary_id"] == "sum_1"
        assert package.source_refs_json["working_topic"][0]["topic_id"] == "tpc_1"
        assert "Private hidden topic" not in package.blocks_json.working_topic_block
        assert {
            ref["topic_id"] for ref in package.source_refs_json["working_topic"]
        } == {"tpc_1"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_package_uses_space_boundary_mode_from_signature() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await SpaceRepository(connection, clock).resolve_space(
            owner_user_id="usr_1",
            space_id="space_severed",
            boundary_mode=SpaceBoundaryMode.SEVERANCE,
            display_name="Severed space",
            source_kind="explicit",
            source_id="space_severed",
        )
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_severed",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Severed chat",
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            active_space_id="space_severed",
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_unscoped",
            text="Unscoped fact must not cross into severance.",
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_severed",
            text="Severed-space fact is visible.",
            space_id="space_severed",
            space_boundary_mode=SpaceBoundaryMode.SEVERANCE.value,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_severed",
            resolved_policy=resolved_policy,
        )

        assert "Severed-space fact is visible" in package.blocks_json.prepared_memory_profile_block
        assert "Unscoped fact must not cross" not in package.blocks_json.prepared_memory_profile_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_baseline_without_platform_does_not_mix_persona_rows() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_persona_jordi",
            text="Persona Jordi fact is visible.",
            platform_id=None,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_persona_other",
            text="Other persona fact must stay hidden.",
            user_persona_id="persona_other",
            platform_id=None,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_other_character",
            text="Other character in same workspace must stay hidden.",
            scope=MemoryScope.CHARACTER,
            user_persona_id="persona_jordi",
            platform_id=None,
            character_id="assistant_beta",
            workspace_id="wrk_1",
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id=None,
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        assert "Persona Jordi fact is visible" in package.blocks_json.prepared_memory_profile_block
        assert "Other persona fact" not in package.blocks_json.prepared_memory_profile_block
        assert "Other character in same workspace" not in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_empty_conversation_package_marks_same_chat_sections_empty() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        package = await InitialContextPackageBuilder(connection, clock).build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_empty",
            resolved_policy=resolved_policy,
        )

        assert package.blocks_json.empty_markers["same_chat_history_known_empty"] is True
        assert package.blocks_json.empty_markers["conversation_summary_empty"] is True
        assert package.blocks_json.empty_markers["working_topic_empty"] is True
        assert package.blocks_json.empty_markers["recent_verbatim_seed_empty"] is True
        assert package.blocks_json.conversation_summary_block == ""
        assert package.blocks_json.working_topic_block == ""
        assert package.blocks_json.recent_verbatim_seed == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_topic_block_stays_within_configured_budget() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        long_text = " ".join(["oversized-topic-detail"] * 120)
        await TopicRepository(connection, clock).create_topic(
            user_id="usr_1",
            conversation_id="cnv_1",
            topic_id="tpc_huge",
            title="Huge topic",
            summary=long_text,
            active_goal=long_text,
            open_questions=[long_text],
            decisions=[long_text],
            source_message_start_seq=1,
            source_message_end_seq=1,
            last_touched_seq=1,
        )
        builder = InitialContextPackageBuilder(
            connection,
            clock,
            budget=InitialContextPackageBuildBudget(topic_block_budget_tokens=80),
        )

        package = await builder.build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_1",
            resolved_policy=resolved_policy,
        )

        topic_block = package.blocks_json.working_topic_block
        assert ContextComposer.estimate_tokens(topic_block) <= 80
        if not topic_block:
            assert package.source_refs_json["working_topic"] == []
            assert package.blocks_json.empty_markers["working_topic_empty"] is True
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_selection_applies_salience_limit_and_reports_drops() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        for index in range(5):
            await _create_belief(
                connection,
                clock,
                memory_id=f"mem_profile_{index}",
                text=f"Profile fact {index}",
                maya_score=1.0 - (index * 0.1),
                vitality=0.2 + (index * 0.15),
            )

        builder = InitialContextPackageBuilder(
            connection,
            clock,
            budget=InitialContextPackageBuildBudget(
                max_profile_items=2,
                max_profile_candidates=5,
            ),
        )
        package = await builder.build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        assert len(package.blocks_json.profile_items) == 2
        assert package.blocks_json.source_counts["profile_items"] == 2
        assert package.diagnostics_json.selected_profile_items == 2
        assert package.diagnostics_json.dropped_profile_items == 3
        assert "Profile fact 4" in package.blocks_json.prepared_memory_profile_block
        assert "Profile fact 0" not in package.blocks_json.prepared_memory_profile_block
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_package_curation_is_stored_inside_signed_blocks_with_sources() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_pivotal",
            text="The user had a small but pivotal trust repair conversation with the assistant.",
            maya_score=1.0,
            vitality=0.95,
            stability=0.9,
            confidence=0.95,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_work",
            text="The user often discusses work planning.",
            maya_score=0.1,
            vitality=0.2,
            stability=0.7,
            confidence=0.8,
        )
        builder = InitialContextPackageBuilder(
            connection,
            clock,
            curator=_curator(
                {
                    "items": [
                        {
                            "candidate_ids": ["memory:mem_pivotal"],
                            "text": (
                                "A small trust repair conversation is unusually "
                                "important context for this user."
                            ),
                            "status": "current",
                            "salience": 0.96,
                            "reason_category": "relationship_orientation",
                        }
                    ],
                    "nothing_to_add": False,
                }
            ),
        )

        package = await builder.build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        assert package.blocks_json.curated_items
        assert "trust repair" in package.blocks_json.curated_orientation_block
        assert package.blocks_json.source_counts["curated_items"] == 1
        assert package.source_refs_json["curated_orientation"][0]["memory_id"] == "mem_pivotal"
        assert package.diagnostics_json.selected_curated_items == 1
        assert package.diagnostics_json.warnings == []
        assert package.build_status.value == "active"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_package_curation_drops_items_that_do_not_cite_package_sources() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_grounded",
            text="The user prefers direct Spanish replies.",
        )
        builder = InitialContextPackageBuilder(
            connection,
            clock,
            curator=_curator(
                {
                    "items": [
                        {
                            "candidate_ids": ["memory:not_in_package"],
                            "text": "Unsupported claim should not be stored.",
                            "status": "current",
                            "salience": 0.9,
                        }
                    ],
                    "nothing_to_add": False,
                }
            ),
        )

        package = await builder.build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        assert package.blocks_json.curated_items == []
        assert package.blocks_json.curated_orientation_block == ""
        assert package.source_refs_json["curated_orientation"] == []
        assert "curation_dropped_ungrounded_item" in package.diagnostics_json.warnings
        assert "curation_produced_no_valid_items" in package.diagnostics_json.warnings
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_package_curation_skips_recent_seed_marked_skip_by_default() -> None:
    curator, provider = _curator_with_provider(
        {
            "items": [
                {
                    "candidate_ids": ["recent_seed:msg_skip"],
                    "text": "Skipped raw message must not be curated.",
                    "status": "current",
                    "salience": 0.9,
                }
            ],
            "nothing_to_add": False,
        }
    )

    result = await curator.curate(
        user_id="usr_1",
        package_kind="conversation",
        retrieval_profile_id="coding_debug",
        coordinate_complete=True,
        profile_items=[],
        current_state_block="",
        current_state_refs=[],
        conversation_summary_block="",
        summary_refs=[],
        working_topic_block="",
        topic_refs=[],
        recent_verbatim_seed=[
            {
                "message_id": "msg_skip",
                "conversation_id": "cnv_1",
                "seq": 1,
                "text": "Raw content intentionally hidden by default.",
                "skip_by_default": True,
            }
        ],
    )

    assert result.items == []
    assert result.warnings == ["curation_skipped_no_candidates"]
    assert provider.requests == []


@pytest.mark.asyncio
async def test_package_curation_prompt_preserves_ambiguity_and_old_emotional_state() -> None:
    curator, provider = _curator_with_provider(
        {
            "items": [],
            "nothing_to_add": True,
        }
    )

    await curator.curate(
        user_id="usr_1",
        package_kind="baseline",
        retrieval_profile_id="coding_debug",
        coordinate_complete=True,
        profile_items=[
            InitialContextPackageProfileItem(
                item_id="memory:mem_current",
                text="The user is unsure whether they like the tool.",
                reason_category="preference_state",
                source_refs=[{"source_kind": "memory_object", "memory_id": "mem_current"}],
                status="ambiguous",
                salience=0.8,
            )
        ],
        current_state_block="",
        current_state_refs=[],
        conversation_summary_block="",
        summary_refs=[],
        working_topic_block="",
        topic_refs=[],
        recent_verbatim_seed=[],
    )

    assert provider.requests
    prompt = provider.requests[0].messages[1].content
    assert "preserve the ambiguity" in prompt
    assert "flattening the contradiction" in prompt
    assert "old emotional states" in prompt
    assert "explain the present" in prompt


@pytest.mark.asyncio
async def test_package_curation_does_not_promote_noncurrent_sources_to_current() -> None:
    curator = _curator(
        {
            "items": [
                {
                    "candidate_ids": ["memory:mem_superseded"],
                    "text": "The user was previously angry with the assistant.",
                    "status": "current",
                    "salience": 0.9,
                }
            ],
            "nothing_to_add": False,
        }
    )

    result = await curator.curate(
        user_id="usr_1",
        package_kind="baseline",
        retrieval_profile_id="coding_debug",
        coordinate_complete=True,
        profile_items=[
            InitialContextPackageProfileItem(
                item_id="memory:mem_superseded",
                text="The user was angry with the assistant.",
                reason_category="relationship_state",
                source_refs=[
                    {"source_kind": "memory_object", "memory_id": "mem_superseded"}
                ],
                status="superseded",
                salience=0.8,
            )
        ],
        current_state_block="",
        current_state_refs=[],
        conversation_summary_block="",
        summary_refs=[],
        working_topic_block="",
        topic_refs=[],
        recent_verbatim_seed=[],
    )

    assert result.items[0].status == "superseded"
    assert "curation_corrected_noncurrent_status" in result.warnings


@pytest.mark.asyncio
async def test_package_curation_ignores_model_warnings_in_diagnostics() -> None:
    curator = _curator(
        {
            "items": [
                {
                    "candidate_ids": ["memory:mem_grounded"],
                    "text": "The user prefers direct Spanish replies.",
                    "status": "current",
                    "salience": 0.9,
                }
            ],
            "nothing_to_add": False,
            "warnings": ["do not store this freeform model warning"],
        }
    )

    result = await curator.curate(
        user_id="usr_1",
        package_kind="baseline",
        retrieval_profile_id="coding_debug",
        coordinate_complete=True,
        profile_items=[
            InitialContextPackageProfileItem(
                item_id="memory:mem_grounded",
                text="The user prefers direct Spanish replies.",
                reason_category="preference",
                source_refs=[
                    {"source_kind": "memory_object", "memory_id": "mem_grounded"}
                ],
                status="current",
                salience=0.8,
            )
        ],
        current_state_block="",
        current_state_refs=[],
        conversation_summary_block="",
        summary_refs=[],
        working_topic_block="",
        topic_refs=[],
        recent_verbatim_seed=[],
    )

    assert result.items
    assert "do not store this freeform model warning" not in result.warnings
    assert result.warnings == []


@pytest.mark.asyncio
async def test_package_curation_can_exclude_recent_seed_from_curation() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    curator, provider = _curator_with_provider(
        {
            "items": [
                {
                    "candidate_ids": ["recent_seed:msg_1"],
                    "text": "Raw recent message must not be curated.",
                    "status": "current",
                    "salience": 0.9,
                }
            ],
            "nothing_to_add": False,
        }
    )
    try:
        await MessageRepository(connection, clock).create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Recent raw message should remain stored but not curated.",
        )
        builder = InitialContextPackageBuilder(
            connection,
            clock,
            curator=curator,
            curate_recent_verbatim_seed=False,
        )

        package = await builder.build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_1",
            resolved_policy=resolved_policy,
        )

        assert package.blocks_json.recent_verbatim_seed
        assert package.blocks_json.curated_items == []
        assert provider.requests == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_includes_recent_superseded_and_historical_items_with_status() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_current",
            text="The user and assistant are friends again.",
            vitality=0.8,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_superseded",
            text="The user was angry with the assistant.",
            status=MemoryStatus.SUPERSEDED,
            vitality=0.9,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_historical",
            text="The user previously preferred only terse answers.",
            valid_to="2026-06-07T09:00:00+00:00",
            payload={"profile_status": "historical"},
            vitality=0.7,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        statuses = {item.item_id: item.status for item in package.blocks_json.profile_items}
        assert statuses["memory:mem_current"] == "current"
        assert statuses["memory:mem_superseded"] == "superseded"
        assert statuses["memory:mem_historical"] == "historical"
        assert "[superseded] The user was angry" in (
            package.blocks_json.prepared_memory_profile_block
        )
        assert "[historical] The user previously preferred" in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_db_lifecycle_status_overrides_payload_current_hint() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_superseded_payload_current",
            text="The user was angry with the assistant.",
            status=MemoryStatus.SUPERSEDED,
            payload={"profile_status": "current"},
            vitality=0.9,
        )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_historical_payload_current",
            text="The user previously preferred only terse answers.",
            valid_to="2026-06-07T09:00:00+00:00",
            payload={"profile_status": "current"},
            vitality=0.8,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        statuses = {item.item_id: item.status for item in package.blocks_json.profile_items}
        assert statuses["memory:mem_superseded_payload_current"] == "superseded"
        assert statuses["memory:mem_historical_payload_current"] == "historical"
        assert "[superseded] The user was angry" in (
            package.blocks_json.prepared_memory_profile_block
        )
        assert "[historical] The user previously preferred" in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_reserves_review_required_items_as_ambiguous_when_active_rows_saturate() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        for index in range(13):
            await _create_belief(
                connection,
                clock,
                memory_id=f"mem_current_{index:02d}",
                text=f"Current profile fact {index}.",
                vitality=0.9,
                stability=0.9,
                confidence=0.9,
            )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_review_required",
            text="The user's preference for the tool is under review due to conflict.",
            status=MemoryStatus.REVIEW_REQUIRED,
            vitality=0.9,
            stability=0.9,
            confidence=0.9,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        statuses = {item.item_id: item.status for item in package.blocks_json.profile_items}
        assert statuses["memory:mem_review_required"] == "ambiguous"
        assert "[ambiguous] The user's preference for the tool is under review" in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_marks_high_tension_active_items_as_ambiguous() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        await _create_belief(
            connection,
            clock,
            memory_id="mem_tense_active",
            text="The user may or may not want verbose explanations.",
            tension_score=0.9,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        statuses = {item.item_id: item.status for item in package.blocks_json.profile_items}
        assert statuses["memory:mem_tense_active"] == "ambiguous"
        assert "[ambiguous] The user may or may not want verbose explanations" in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_reserves_slots_for_historical_items_when_active_rows_saturate() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        for index in range(13):
            await _create_belief(
                connection,
                clock,
                memory_id=f"mem_current_{index:02d}",
                text=f"Current profile fact {index}.",
                vitality=0.9,
                stability=0.9,
                confidence=0.9,
            )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_superseded",
            text="The user was previously angry with the assistant.",
            status=MemoryStatus.SUPERSEDED,
            vitality=0.9,
            stability=0.9,
            confidence=0.9,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        item_ids = {item.item_id for item in package.blocks_json.profile_items}
        assert len(package.blocks_json.profile_items) == 12
        assert "memory:mem_superseded" in item_ids
        assert "Current profile fact 12" not in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_profile_does_not_reserve_old_low_signal_superseded_items() -> None:
    connection, clock, resolved_policy = await _seed_runtime()
    try:
        for index in range(13):
            await _create_belief(
                connection,
                clock,
                memory_id=f"mem_current_{index:02d}",
                text=f"Current profile fact {index}.",
                vitality=0.9,
                stability=0.9,
                confidence=0.9,
            )
        await _create_belief(
            connection,
            clock,
            memory_id="mem_old_superseded",
            text="The user once disliked all detailed answers.",
            status=MemoryStatus.SUPERSEDED,
            valid_to="2025-01-01T09:00:00+00:00",
            tension_score=0.0,
            vitality=0.9,
            stability=0.9,
            confidence=0.9,
        )

        package = await InitialContextPackageBuilder(connection, clock).build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            workspace_id="wrk_1",
        )

        item_ids = {item.item_id for item in package.blocks_json.profile_items}
        assert "memory:mem_old_superseded" not in item_ids
        assert "once disliked all detailed answers" not in (
            package.blocks_json.prepared_memory_profile_block
        )
    finally:
        await connection.close()
