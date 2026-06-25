"""Integration tests for admin inspection helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.embodiment_repository import EmbodimentRepository
from atagia.core.mind_repository import DEFAULT_OVERSEER_MIND_ID, MindRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.overseer_grant_repository import OverseerGrantRepository
from atagia.core.presence_repository import PresenceRepository
from atagia.core.realm_repository import RealmRepository
from atagia.core.retrieval_event_repository import AdminAuditRepository, RetrievalEventRepository
from atagia.core.space_repository import SpaceRepository
from atagia.memory.inspector import (
    MemoryInspector,
    _boundary_explanation,
    _retrieval_plan_from_event,
)
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    CrossRealmMode,
    EmbodimentBoundaryMode,
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    MindKind,
    MindTopology,
    OverseerGrantKind,
    OverseerGrantTargetKind,
    PresenceKind,
    SpaceBoundaryMode,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc))


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = _clock()
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    events = RetrievalEventRepository(connection, clock)
    audits = AdminAuditRepository(connection, clock)
    inspector = MemoryInspector(connection, clock)

    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Debug")
    await conversations.create_conversation("cnv_2", "usr_2", None, "coding_debug", "Other")
    await messages.create_message("msg_1", "cnv_1", "user", 1, "Need help with retries", 5, {})
    await messages.create_message("msg_2", "cnv_1", "assistant", 2, "Let's inspect the queue", 6, {})
    await messages.create_message("msg_3", "cnv_2", "user", 1, "Other user prompt", 4, {})
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.BELIEF,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User prefers patch-style fixes.",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.9,
        privacy_level=2,
        intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
        intimacy_boundary_confidence=0.86,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_belief",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="User is debugging retry behavior.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_evidence",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.INTERACTION_CONTRACT,
        scope=MemoryScope.ASSISTANT_MODE,
        canonical_text="User prefers concise direct answers.",
        source_kind=MemorySourceKind.INFERRED,
        confidence=0.75,
        privacy_level=1,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_contract",
    )
    await memories.create_memory_object(
        user_id="usr_2",
        conversation_id="cnv_2",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Other user's memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id="mem_other",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Pending sensitive memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=3,
        status=MemoryStatus.PENDING_USER_CONFIRMATION,
        memory_id="mem_pending",
    )
    await memories.create_memory_object(
        user_id="usr_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Declined sensitive memory.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=3,
        status=MemoryStatus.DECLINED,
        memory_id="mem_declined",
    )
    await connection.execute(
        """
        INSERT INTO belief_versions(
            belief_id,
            version,
            claim_key,
            claim_value_json,
            condition_json,
            support_count,
            contradict_count,
            supersedes_version,
            is_current,
            created_at
        )
        VALUES
            (?, 1, 'response_style.debugging', '{"label":"patch_first"}', '{}', 1, 0, NULL, 0, ?),
            (?, 2, 'response_style.debugging', '{"label":"patch_first"}', '{}', 2, 0, 1, 1, ?)
        """,
        (
            "mem_belief",
            "2026-03-31T00:00:00+00:00",
            "mem_belief",
            "2026-03-31T00:05:00+00:00",
        ),
    )
    await connection.commit()
    event = await events.create_event(
        {
            "user_id": "usr_1",
            "conversation_id": "cnv_1",
            "request_message_id": "msg_1",
            "response_message_id": "msg_2",
            "assistant_mode_id": "coding_debug",
            "retrieval_plan_json": {"fts_queries": ["retry queue"], "skip_retrieval": False},
            "selected_memory_ids_json": ["mem_belief", "mem_evidence"],
            "context_view_json": {
                "selected_memory_ids": ["mem_belief", "mem_evidence"],
                "total_tokens_estimate": 72,
                "items_included": 2,
                "items_dropped": 1,
            },
            "outcome_json": {"zero_candidates": False},
        }
    )
    return connection, inspector, audits, event


def test_boundary_explanation_keeps_missing_grant_block_out_of_attribution() -> None:
    explanation = _boundary_explanation(
        {
            "mind": {
                "allowed": False,
                "decision": "blocked",
                "reason": "blocked_missing_overseer_grant",
            }
        }
    )

    assert explanation["decision"] == "blocked"
    assert explanation["blocked_reasons"] == ["blocked_missing_overseer_grant"]
    assert explanation["attribution_reasons"] == []


def test_boundary_explanation_keeps_invalid_realm_bridge_block_out_of_attribution() -> None:
    explanation = _boundary_explanation(
        {
            "realm": {
                "allowed": False,
                "decision": "blocked",
                "reason": "blocked_by_realm_bridge_missing_or_invalid",
            }
        }
    )

    assert explanation["decision"] == "blocked"
    assert explanation["blocked_reasons"] == ["blocked_by_realm_bridge_missing_or_invalid"]
    assert explanation["attribution_reasons"] == []


def test_boundary_explanation_includes_allowed_overseer_grant_attribution() -> None:
    explanation = _boundary_explanation(
        {
            "mind": {
                "allowed": True,
                "decision": "allowed",
                "reason": "allowed_by_overseer_grant",
            }
        }
    )

    assert explanation["decision"] == "allowed"
    assert explanation["blocked_reasons"] == []
    assert explanation["attribution_reasons"] == ["allowed_by_overseer_grant"]


def test_boundary_explanation_includes_allowed_realm_bridge_attribution() -> None:
    explanation = _boundary_explanation(
        {
            "realm": {
                "allowed": True,
                "decision": "allowed",
                "reason": "allowed_by_realm_bridge_attributed",
            }
        }
    )

    assert explanation["decision"] == "allowed"
    assert explanation["blocked_reasons"] == []
    assert explanation["attribution_reasons"] == ["allowed_by_realm_bridge_attributed"]


def test_retrieval_plan_from_event_round_trips_clean_plan_dump() -> None:
    """A production-shaped retrieval-plan dump must round-trip for the inspector.

    ``RetrievalPlan`` is strict (``extra="forbid"``), so the persisted
    ``retrieval_plan_json`` has to stay a clean plan dump. The adaptive-gate
    block lives only on ``retrieval_diagnostics_for_guard``; it must never be
    injected into the plan, or coordinate-trace reconstruction silently breaks
    for every event that falls back to event-plan reconstruction.
    """
    event = {
        "assistant_mode_id": "coding_debug",
        "conversation_id": "cnv_1",
        "retrieval_plan_json": {
            "fts_queries": ["retry queue"],
            "skip_retrieval": False,
        },
    }
    plan, error = _retrieval_plan_from_event(event)
    assert error is None
    assert plan is not None
    assert plan.fts_queries == ["retry queue"]


def test_retrieval_plan_from_event_rejects_adaptive_gate_pollution() -> None:
    """An ``adaptive_gate`` key in the persisted plan breaks strict round-trip.

    This locks in why the gate block must stay out of ``source_retrieval_plan``:
    if it leaks back in, the inspector cannot reconstruct the coordinate trace
    and reports the source as unavailable for every affected decision.
    """
    event = {
        "assistant_mode_id": "coding_debug",
        "conversation_id": "cnv_1",
        "retrieval_plan_json": {
            "fts_queries": ["retry queue"],
            "skip_retrieval": False,
            "adaptive_gate": {
                "status": "skipped",
                "classification": "world",
            },
        },
    }
    plan, error = _retrieval_plan_from_event(event)
    assert plan is None
    assert error is not None
    assert error.startswith("retrieval_plan_unparseable:")


@pytest.mark.asyncio
async def test_inspect_memory_returns_full_object_and_logs_audit() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        memory = await inspector.inspect_memory("mem_belief", "usr_1", admin_user_id="adm_1")

        assert memory is not None
        assert memory["canonical_text"] == "User prefers patch-style fixes."
        assert memory["intimacy_boundary"] == "romantic_private"
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "memory_object"
        assert audit_rows[-1]["target_id"] == "mem_belief"
        assert audit_rows[-1]["metadata_json"]["user_id"] == "usr_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_memory_coordinates_returns_joined_coordinate_truth() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        clock = _clock()
        await PresenceRepository(connection, clock).resolve_presence(
            owner_user_id="usr_1",
            presence_id="presence_detective",
            kind=PresenceKind.OWNED_FACET,
            display_name="Detective",
            source_kind="explicit",
            source_id="presence_detective",
        )
        await PresenceRepository(connection, clock).resolve_presence(
            owner_user_id="usr_1",
            presence_id="presence_accountant",
            kind=PresenceKind.OWNED_FACET,
            display_name="Accountant",
            source_kind="explicit",
            source_id="presence_accountant",
        )
        await SpaceRepository(connection, clock).resolve_space(
            owner_user_id="usr_1",
            space_id="space_vault",
            boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
            display_name="Vault",
            source_kind="explicit",
            source_id="space_vault",
        )
        await MindRepository(connection, clock).resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_detective",
            kind=MindKind.OWNED_FACET,
            display_name="Detective Mind",
            source_kind="explicit",
            source_id="mind_detective",
        )
        await EmbodimentRepository(connection, clock).resolve_embodiment(
            owner_user_id="usr_1",
            embodiment_id="drone_body",
            cross_embodiment_mode=EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY,
            display_name="Drone Body",
            source_kind="explicit",
            source_id="drone_body",
        )
        await RealmRepository(connection, clock).resolve_realm(
            owner_user_id="usr_1",
            realm_id="realm_real",
            cross_realm_mode=CrossRealmMode.NONE,
            display_name="Real World",
            source_kind="explicit",
            source_id="realm_real",
        )
        await MindRepository(connection, clock).resolve_active_mind(
            owner_user_id="usr_1",
            mind_id=None,
            active_presence_id=None,
            topology=MindTopology.OJOCENTAURI,
        )
        await OverseerGrantRepository(connection, clock).upsert_grant(
            owner_user_id="usr_1",
            overseer_mind_id=DEFAULT_OVERSEER_MIND_ID,
            target_kind=OverseerGrantTargetKind.MIND,
            target_id="mind_detective",
            grant_kind=OverseerGrantKind.READ,
        )
        await MemoryObjectRepository(connection, clock).create_memory_object(
            memory_id="mem_coordinate",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="Coordinate inspection memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=1,
            sensitivity=MemorySensitivity.PRIVATE,
            payload={
                "presence_attribution": {"active_presence_id": "presence_detective"},
                "space_boundary": {"space_id": "space_vault"},
                "mind_perspective": {"memory_owner_id": "mind_detective"},
                "embodiment": {"active_embodiment_id": "drone_body"},
                "realm": {"active_realm_id": "realm_real"},
            },
            active_presence_id="presence_detective",
            source_presence_id="presence_accountant",
            presence_cluster_id="cluster_shared",
            space_id="space_vault",
            space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT.value,
            memory_owner_id="mind_detective",
            source_mind_id="mind_detective",
            embodiment_id="drone_body",
            realm_id="realm_real",
        )

        inspection = await inspector.inspect_memory_coordinates(
            "mem_coordinate",
            "usr_1",
            admin_user_id="adm_1",
        )

        assert inspection is not None
        assert inspection["coordinates"]["presence"]["active_presence"]["display_name"] == "Detective"
        assert inspection["coordinates"]["presence"]["source_presence"]["display_name"] == "Accountant"
        assert inspection["coordinates"]["space"]["space"]["boundary_mode"] == "privacy_vault"
        assert inspection["coordinates"]["mind"]["memory_owner"]["kind"] == "owned_facet"
        assert inspection["coordinates"]["embodiment"]["embodiment"]["display_name"] == "Drone Body"
        assert inspection["coordinates"]["realm"]["realm"]["display_name"] == "Real World"
        assert inspection["coordinates"]["overseer_grants"][0]["grant_kind"] == "read"
        assert inspection["provenance"]["payload_coordinates"]["realm"]["active_realm_id"] == "realm_real"

        assert await inspector.inspect_memory_coordinates(
            "mem_coordinate",
            "usr_2",
            admin_user_id="adm_1",
        ) is None
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-2]["action"] == "inspect_memory_coordinates"
        assert audit_rows[-2]["metadata_json"]["found"] is True
        assert audit_rows[-1]["metadata_json"]["found"] is False
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_correct_memory_coordinates_audits_history_and_invalidates_cache() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        clock = _clock()
        await SpaceRepository(connection, clock).resolve_space(
            owner_user_id="usr_1",
            space_id="space_focus",
            boundary_mode=SpaceBoundaryMode.FOCUS,
            display_name="Focus",
            source_kind="explicit",
            source_id="space_focus",
        )
        await SpaceRepository(connection, clock).resolve_space(
            owner_user_id="usr_1",
            space_id="space_private",
            boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
            display_name="Private",
            source_kind="explicit",
            source_id="space_private",
        )
        await MindRepository(connection, clock).resolve_mind(
            owner_user_id="usr_1",
            mind_id="mind_alpha",
            kind=MindKind.OWNED_AI,
            display_name="Alpha",
            source_kind="explicit",
            source_id="mind_alpha",
        )
        await MemoryObjectRepository(connection, clock).create_memory_object(
            memory_id="mem_correct_coordinates",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="Coordinate correction memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            space_id="space_focus",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        invalidated: list[str] = []

        async def invalidate(user_id: str) -> None:
            invalidated.append(user_id)

        corrected = await inspector.correct_memory_coordinates(
            "mem_correct_coordinates",
            "usr_1",
            admin_user_id="adm_1",
            updates={
                "space_id": "space_private",
                "memory_owner_id": "mind_alpha",
            },
            reason="imported into the wrong coordinate",
            invalidate_user_cache=invalidate,
        )

        assert corrected is not None
        assert corrected["coordinates"]["space"]["space_id"] == "space_private"
        assert corrected["coordinates"]["space"]["space_boundary_mode"] == "privacy_vault"
        assert corrected["coordinates"]["mind"]["memory_owner_id"] == "mind_alpha"
        assert invalidated == ["usr_1"]

        audit_rows = await audits.list_entries("adm_1")
        correction = next(row for row in audit_rows if row["action"] == "correct_memory_coordinates")
        assert correction["metadata_json"]["before"]["space_id"] == "space_focus"
        assert correction["metadata_json"]["after"]["space_id"] == "space_private"
        assert correction["metadata_json"]["after"]["memory_owner_id"] == "mind_alpha"
        assert correction["metadata_json"]["reason"] == "imported into the wrong coordinate"

        history = await inspector.inspect_coordinate_correction_history(
            "mem_correct_coordinates",
            "usr_1",
            admin_user_id="adm_1",
        )
        assert [row["id"] for row in history] == [correction["id"]]

        with pytest.raises(ValueError, match="Unknown mind coordinate"):
            await inspector.correct_memory_coordinates(
                "mem_correct_coordinates",
                "usr_1",
                admin_user_id="adm_1",
                updates={"memory_owner_id": "ghost_mind"},
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_retrieval_memory_decision_uses_custody_coordinate_trace() -> None:
    connection, inspector, _audits, event = await _build_runtime()
    try:
        await RetrievalEventRepository(connection, _clock()).update_outcome_fields(
            event["id"],
            "usr_1",
            {
                "retrieval_custody_v2": [
                    {
                        "candidate_id": "mem_evidence",
                        "selected": True,
                        "composer_decision": "selected",
                        "coordinate_trace_v1": {
                            "space": {
                                "allowed": True,
                                "decision": "allowed",
                                "reason": "allowed_same_space",
                            },
                            "mind": {
                                "allowed": True,
                                "decision": "allowed",
                                "reason": "allowed_same_mind",
                            },
                        },
                    }
                ]
            },
        )

        decision = await inspector.inspect_retrieval_memory_decision(
            event["id"],
            "mem_evidence",
            "usr_1",
            admin_user_id="adm_1",
        )

        assert decision is not None
        assert decision["decision"] == "selected"
        assert decision["coordinate_trace_source"] == "retrieval_custody_v2"
        assert decision["coordinate_trace_v1"]["space"]["reason"] == "allowed_same_space"
        assert decision["boundary_explanation"]["decision"] == "allowed"
        assert decision["memory_coordinates"]["memory"]["id"] == "mem_evidence"

        missing = await inspector.inspect_retrieval_memory_decision(
            event["id"],
            "mem_belief",
            "usr_1",
            admin_user_id="adm_1",
        )
        assert missing is not None
        assert missing["decision"] == "not_present_in_candidate_custody"
        assert missing["coordinate_trace_source"] == "reconstructed_from_event_plan"
        assert missing["coordinate_trace_v1"]["space"]["reason"] == "allowed_unscoped_space"
        assert missing["boundary_explanation"]["decision"] == "allowed"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_retrieval_memory_decision_reconstructs_blocked_gate_without_custody() -> None:
    connection, inspector, _audits, event = await _build_runtime()
    try:
        clock = _clock()
        spaces = SpaceRepository(connection, clock)
        await spaces.resolve_space(
            owner_user_id="usr_1",
            space_id="space_a",
            boundary_mode=SpaceBoundaryMode.SEVERANCE,
            display_name="A",
            source_kind="explicit",
            source_id="space_a",
        )
        await spaces.resolve_space(
            owner_user_id="usr_1",
            space_id="space_b",
            boundary_mode=SpaceBoundaryMode.FOCUS,
            display_name="B",
            source_kind="explicit",
            source_id="space_b",
        )
        await MemoryObjectRepository(connection, clock).create_memory_object(
            memory_id="mem_blocked_space_trace",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="Blocked space decision memory.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            space_id="space_b",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        await connection.execute(
            """
            UPDATE retrieval_events
            SET retrieval_plan_json = ?
            WHERE id = ?
              AND user_id = 'usr_1'
            """,
            (
                json_utils.dumps(
                    {
                        "fts_queries": ["space"],
                        "active_space_id": "space_a",
                        "active_space_boundary_mode": "severance",
                    },
                    sort_keys=True,
                ),
                event["id"],
            ),
        )
        await connection.commit()

        decision = await inspector.inspect_retrieval_memory_decision(
            event["id"],
            "mem_blocked_space_trace",
            "usr_1",
            admin_user_id="adm_1",
        )

        assert decision is not None
        assert decision["decision"] == "not_present_in_candidate_custody"
        assert decision["coordinate_trace_source"] == "reconstructed_from_event_plan"
        assert decision["coordinate_trace_v1"]["space"]["allowed"] is False
        assert decision["coordinate_trace_v1"]["space"]["reason"] == "blocked_by_space_severance"
        assert decision["boundary_explanation"]["decision"] == "blocked"
        assert decision["boundary_explanation"]["blocked_reasons"] == ["blocked_by_space_severance"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_retrieval_event_returns_event_with_plan_and_logs_audit() -> None:
    connection, inspector, audits, event = await _build_runtime()
    try:
        inspected = await inspector.inspect_retrieval_event(event["id"], "usr_1", admin_user_id="adm_1")

        assert inspected is not None
        assert inspected["retrieval_plan_json"]["fts_queries"] == ["retry queue"]
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "retrieval_event"
        assert audit_rows[-1]["target_id"] == event["id"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_user_memories_filters_by_type_scope_and_status() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        filtered = await inspector.inspect_user_memories(
            "usr_1",
            admin_user_id="adm_1",
            object_type="belief",
            scope="conversation",
            status="active",
            intimacy_boundary="romantic_private",
            limit=10,
        )

        assert [item["id"] for item in filtered] == ["mem_belief"]
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "user_memory_collection"
        assert audit_rows[-1]["metadata_json"]["result_count"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_user_memories_shows_pending_and_declined_rows() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        memories = await inspector.inspect_user_memories(
            "usr_1",
            admin_user_id="adm_1",
            limit=10,
        )

        statuses = {item["id"]: item["status"] for item in memories}
        assert statuses["mem_pending"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert statuses["mem_declined"] == MemoryStatus.DECLINED.value
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["metadata_json"]["result_count"] == len(memories)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspect_belief_history_returns_all_versions_and_logs_audit() -> None:
    connection, inspector, audits, _event = await _build_runtime()
    try:
        history = await inspector.inspect_belief_history(
            "mem_belief",
            "usr_1",
            admin_user_id="adm_1",
        )

        assert [item["version"] for item in history] == [1, 2]
        assert {item["parent_intimacy_boundary"] for item in history} == {"romantic_private"}
        audit_rows = await audits.list_entries("adm_1")
        assert audit_rows[-1]["target_type"] == "belief_history"
        assert audit_rows[-1]["target_id"] == "mem_belief"
        assert audit_rows[-1]["metadata_json"]["result_count"] == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_inspector_respects_user_isolation() -> None:
    connection, inspector, audits, event = await _build_runtime()
    try:
        assert await inspector.inspect_memory("mem_other", "usr_1", admin_user_id="adm_1") is None
        assert await inspector.inspect_retrieval_event(event["id"], "usr_2", admin_user_id="adm_1") is None
        assert await inspector.inspect_belief_history("mem_belief", "usr_2", admin_user_id="adm_1") == []

        audit_rows = await audits.list_entries("adm_1")
        assert len(audit_rows) == 3
        assert [row["metadata_json"]["user_id"] for row in audit_rows[:2]] == ["usr_1", "usr_2"]
        assert audit_rows[2]["metadata_json"]["user_id"] == "usr_2"
    finally:
        await connection.close()
