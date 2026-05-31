"""Tests for initial context package signatures and freshness helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.embodiment_repository import EmbodimentRepository
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.mind_repository import MindRepository
from atagia.core.overseer_grant_repository import OverseerGrantRepository
from atagia.core.presence_repository import PresenceRepository
from atagia.core.realm_repository import RealmRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.lifecycle_runner import cache_generation_key
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageKey,
    InitialContextPackageKind,
)
from atagia.models.schemas_memory import (
    CrossRealmMode,
    EmbodimentBoundaryMode,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MindKind,
    MindTopology,
    OverseerGrantKind,
    OverseerGrantTargetKind,
    PresenceKind,
    SpaceBoundaryMode,
)
from atagia.services.initial_context_package_signatures import (
    build_initial_context_package_coordinate_signature,
    build_initial_context_package_policy_signature,
    build_initial_context_package_source_fingerprint,
    invalidate_initial_context_package_dependency,
)
from atagia.services.prompt_authority import normalize_request_authority_context

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
DATABASE_ID = "/tmp/atagia-initial-context-package-test.db"


async def _seed_connection():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await UserRepository(connection, clock).create_user("usr_2")
    await WorkspaceRepository(connection, clock).create_workspace(
        "wrk_1",
        "usr_1",
        "Workspace",
    )
    return connection, clock


async def _seed_coordinate_conversation(connection, clock: FrozenClock):
    await PresenceRepository(connection, clock).resolve_presence(
        owner_user_id="usr_1",
        presence_id="assistant_alpha",
        kind=PresenceKind.OWNED_FACET,
        display_name="Assistant Alpha",
        source_kind="explicit",
        source_id="assistant_alpha",
    )
    await SpaceRepository(connection, clock).resolve_space(
        owner_user_id="usr_1",
        space_id="space_focus",
        boundary_mode=SpaceBoundaryMode.FOCUS,
        display_name="Focus Space",
        source_kind="explicit",
        source_id="space_focus",
    )
    await MindRepository(connection, clock).resolve_mind(
        owner_user_id="usr_1",
        mind_id="ojocentauri",
        kind=MindKind.OVERSEER,
        display_name="OjoCentauri",
        source_kind="ojocentauri",
        source_id="ojocentauri",
    )
    await EmbodimentRepository(connection, clock).resolve_embodiment(
        owner_user_id="usr_1",
        embodiment_id="body_mac",
        cross_embodiment_mode=EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY,
        display_name="Mac",
        source_kind="explicit",
        source_id="body_mac",
    )
    realms = RealmRepository(connection, clock)
    await realms.resolve_realm(
        owner_user_id="usr_1",
        realm_id="realm_work",
        cross_realm_mode=CrossRealmMode.ATTRIBUTED,
        display_name="Work",
        source_kind="explicit",
        source_id="realm_work",
    )
    await realms.resolve_realm(
        owner_user_id="usr_1",
        realm_id="realm_research",
        cross_realm_mode=CrossRealmMode.APPLICABLE,
        display_name="Research",
        source_kind="explicit",
        source_id="realm_research",
    )
    await realms.upsert_realm_bridge(
        owner_user_id="usr_1",
        source_realm_id="realm_work",
        target_realm_id="realm_research",
        cross_realm_mode=CrossRealmMode.ATTRIBUTED,
    )
    conversation = await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Coordinate chat",
        character_id="assistant_alpha",
        platform_id="aurvek",
        active_presence_id="assistant_alpha",
        active_space_id="space_focus",
        active_mind_id="ojocentauri",
        mind_topology=MindTopology.OJOCENTAURI,
        active_embodiment_id="body_mac",
        active_realm_id="realm_work",
    )
    await OverseerGrantRepository(connection, clock).upsert_grant(
        owner_user_id="usr_1",
        overseer_mind_id="ojocentauri",
        target_kind=OverseerGrantTargetKind.SPACE,
        target_id="space_focus",
        grant_kind=OverseerGrantKind.READ,
        visibility="attributed",
        expires_at="2026-06-08T10:00:00+00:00",
    )
    return conversation


def test_policy_signature_tracks_authority_and_policy_hash() -> None:
    manifests = ManifestLoader(MANIFESTS_DIR).load_all()
    resolved_policy = PolicyResolver().resolve(
        manifests["coding_debug"],
        None,
        None,
    )

    enforce_signature = build_initial_context_package_policy_signature(
        resolved_policy,
        authority_context=normalize_request_authority_context(
            privacy_enforcement="enforce",
            user_id="usr_1",
            purpose="initial_context_package",
        ),
    )
    off_signature = build_initial_context_package_policy_signature(
        resolved_policy,
        authority_context=normalize_request_authority_context(
            privacy_enforcement="off",
            user_id="usr_1",
            purpose="initial_context_package",
        ),
    )

    assert enforce_signature.effective_policy_hash
    assert enforce_signature.policy_prompt_hash == resolved_policy.prompt_hash
    assert enforce_signature.privacy_enforcement == "enforce"
    assert off_signature.privacy_enforcement == "off"
    assert enforce_signature.markers_json != off_signature.markers_json


@pytest.mark.asyncio
async def test_coordinate_signature_changes_on_coordinate_and_grant_updates() -> None:
    connection, clock = await _seed_connection()
    try:
        conversation = await _seed_coordinate_conversation(connection, clock)

        before = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation=conversation,
            now=clock.now(),
        )
        assert before.complete is True
        assert before.markers_json["space"]["boundary_mode"] == "focus"
        assert before.markers_json["ojocentauri"]["grants"][0]["expired"] is False
        assert before.markers_json["realm"]["bridges"][0]["cross_realm_mode"] == "attributed"

        clock.advance(seconds=7200)
        expired = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation_id="cnv_1",
            now=clock.now(),
        )
        assert expired.coordinate_signature_hash != before.coordinate_signature_hash
        assert expired.markers_json["ojocentauri"]["grants"][0]["expired"] is True

        await SpaceRepository(connection, clock).resolve_space(
            owner_user_id="usr_1",
            space_id="space_focus",
            boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
            display_name="Focus Space",
            source_kind="explicit",
            source_id="space_focus",
        )
        space_changed = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation_id="cnv_1",
            now=clock.now(),
        )
        assert space_changed.coordinate_signature_hash != expired.coordinate_signature_hash
        assert space_changed.markers_json["space"]["boundary_mode"] == "privacy_vault"

        await RealmRepository(connection, clock).upsert_realm_bridge(
            owner_user_id="usr_1",
            source_realm_id="realm_work",
            target_realm_id="realm_research",
            cross_realm_mode=CrossRealmMode.APPLICABLE,
        )
        bridge_changed = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation_id="cnv_1",
            now=clock.now(),
        )
        assert bridge_changed.coordinate_signature_hash != space_changed.coordinate_signature_hash
        assert bridge_changed.markers_json["realm"]["bridges"][0]["cross_realm_mode"] == "applicable"

        await OverseerGrantRepository(connection, clock).revoke_grant(
            owner_user_id="usr_1",
            overseer_mind_id="ojocentauri",
            target_kind=OverseerGrantTargetKind.SPACE,
            target_id="space_focus",
            grant_kind=OverseerGrantKind.READ,
        )
        revoked = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation_id="cnv_1",
            now=clock.now(),
        )
        assert revoked.coordinate_signature_hash != bridge_changed.coordinate_signature_hash
        assert revoked.markers_json["ojocentauri"]["grants"][0]["revoked"] is True
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_coordinate_signature_is_incomplete_for_missing_conversation() -> None:
    connection, clock = await _seed_connection()
    try:
        signature = await build_initial_context_package_coordinate_signature(
            connection,
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            conversation_id="missing_conversation",
            now=clock.now(),
        )

        assert signature.complete is False
        assert "conversation" in signature.markers_json["missing"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_fingerprint_changes_for_user_scoped_source_updates() -> None:
    connection, clock = await _seed_connection()
    try:
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Fingerprint chat",
        )
        memories = MemoryObjectRepository(connection, clock)
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.CONVERSATION,
            canonical_text="The user is testing package freshness.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=1,
            memory_id="mem_usr_1",
        )
        before = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        await memories.create_memory_object(
            user_id="usr_2",
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.USER,
            canonical_text="Other user fact.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=1,
            memory_id="mem_usr_2",
        )
        other_user_change = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert other_user_change.source_fingerprint_hash == before.source_fingerprint_hash

        await connection.execute(
            """
            UPDATE memory_objects
            SET updated_at = ?
            WHERE user_id = ?
              AND id = ?
            """,
            ("2026-06-08T10:30:00+00:00", "usr_1", "mem_usr_1"),
        )
        await connection.commit()
        after = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        assert after.source_fingerprint_hash != before.source_fingerprint_hash
        assert after.source_markers_json["sources"]["memory_objects"]["max_updated_at"] == (
            "2026-06-08T10:30:00+00:00"
        )

        baseline_before = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET tension_updated_at = ?
            WHERE user_id = ?
              AND id = ?
            """,
            ("2026-06-08T10:45:00+00:00", "usr_1", "mem_usr_1"),
        )
        await connection.commit()
        baseline_after = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
        )
        assert baseline_after.source_fingerprint_hash != baseline_before.source_fingerprint_hash
        assert baseline_after.source_markers_json["sources"]["memory_objects"][
            "max_tension_updated_at"
        ] == "2026-06-08T10:45:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_fingerprint_tracks_message_raw_policy_markers() -> None:
    connection, clock = await _seed_connection()
    try:
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Message policy chat",
        )
        await connection.execute(
            """
            INSERT INTO messages(
                id,
                conversation_id,
                role,
                seq,
                text,
                created_at,
                content_kind,
                include_raw,
                skip_by_default,
                heavy_content,
                artifact_backed,
                verbatim_required,
                requires_explicit_request,
                context_placeholder,
                policy_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg_1",
                "cnv_1",
                "user",
                1,
                "Pinned source text",
                "2026-06-08T09:00:00+00:00",
                "text",
                1,
                0,
                0,
                0,
                0,
                0,
                None,
                "a_policy_reason",
            ),
        )
        await connection.execute(
            """
            INSERT INTO messages(
                id,
                conversation_id,
                role,
                seq,
                text,
                created_at,
                content_kind,
                include_raw,
                skip_by_default,
                heavy_content,
                artifact_backed,
                verbatim_required,
                requires_explicit_request,
                context_placeholder,
                policy_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg_2",
                "cnv_1",
                "user",
                2,
                "Another policy marker",
                "2026-06-08T09:01:00+00:00",
                "text",
                1,
                0,
                0,
                0,
                0,
                0,
                None,
                "z_policy_reason",
            ),
        )
        await connection.commit()
        before = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        await connection.execute(
            """
            UPDATE messages
            SET policy_reason = 'b_policy_reason'
            WHERE id = ?
            """,
            ("msg_1",),
        )
        await connection.commit()
        after = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        assert after.source_fingerprint_hash != before.source_fingerprint_hash
        assert after.source_markers_json["sources"]["messages"][
            "requires_explicit_request_count"
        ] == 0
        assert after.source_markers_json["sources"]["messages"][
            "policy_reason_marker_count"
        ] == 2
        assert len(after.source_markers_json["sources"]["messages"]["raw_policy_buckets"]) == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_fingerprint_tracks_raw_policy_bucket_membership() -> None:
    connection, clock = await _seed_connection()
    try:
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Message policy membership chat",
        )
        for seq, reason in (
            (1, "policy_a"),
            (2, "policy_b"),
            (3, "policy_b"),
            (4, "policy_a"),
            (5, "policy_a"),
            (6, "policy_b"),
            (7, "policy_b"),
            (8, "policy_a"),
        ):
            await connection.execute(
                """
                INSERT INTO messages(
                    id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    created_at,
                    content_kind,
                    include_raw,
                    skip_by_default,
                    heavy_content,
                    artifact_backed,
                    verbatim_required,
                    requires_explicit_request,
                    context_placeholder,
                    policy_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"msg_{seq}",
                    "cnv_1",
                    "user",
                    seq,
                    f"Message {seq}",
                    f"2026-06-08T09:{seq:02d}:00+00:00",
                    "text",
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    reason,
                ),
            )
        await connection.commit()
        before = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        await connection.execute(
            "UPDATE messages SET policy_reason = 'policy_b' WHERE id = 'msg_4'"
        )
        await connection.execute(
            "UPDATE messages SET policy_reason = 'policy_a' WHERE id = 'msg_6'"
        )
        await connection.commit()
        after = await build_initial_context_package_source_fingerprint(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        assert before.source_fingerprint_hash != after.source_fingerprint_hash
        before_buckets = before.source_markers_json["sources"]["messages"][
            "raw_policy_buckets"
        ]
        after_buckets = after.source_markers_json["sources"]["messages"][
            "raw_policy_buckets"
        ]
        assert [bucket["row_count"] for bucket in before_buckets] == [4, 4]
        assert [bucket["row_count"] for bucket in after_buckets] == [4, 4]
        assert [bucket["min_seq"] for bucket in before_buckets] == [
            bucket["min_seq"] for bucket in after_buckets
        ]
        assert [bucket["max_seq"] for bucket in before_buckets] == [
            bucket["max_seq"] for bucket in after_buckets
        ]
        assert [bucket["member_hash"] for bucket in before_buckets] != [
            bucket["member_hash"] for bucket in after_buckets
        ]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_package_invalidation_marks_stale_and_clears_dependent_cache() -> None:
    connection, clock = await _seed_connection()
    try:
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Invalidation chat",
        )
        repository = InitialContextPackageRepository(connection, clock)
        key = InitialContextPackageKey(
            version=1,
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        )
        package = await repository.upsert_package(
            package_kind=key.package_kind,
            version=key.version,
            user_id=key.user_id,
            conversation_id=key.conversation_id,
            retrieval_profile_id=key.retrieval_profile_id,
            key_json=key,
            blocks_json=InitialContextPackageBlocks(
                conversation_summary_block="Prepared conversation orientation."
            ),
        )
        backend = InProcessBackend()
        await backend.set_context_view(
            "ctx-usr-1",
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=3600,
        )
        await backend.set_context_view(
            "ctx-usr-2",
            {"user_id": "usr_2", "conversation_id": "cnv_2"},
            ttl_seconds=3600,
        )
        await backend.set_recent_window("usr_1:cnv_1", [{"id": "msg_1"}])

        result = await invalidate_initial_context_package_dependency(
            connection,
            clock=clock,
            storage_backend=backend,
            database_path=DATABASE_ID,
            user_id="usr_1",
            conversation_id="cnv_1",
        )

        stale = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=package.package_key_hash,
        )
        assert stale.status == "stale"
        assert result.stale_package_count == 1
        assert result.deleted_context_views == 1
        assert result.deleted_recent_windows == 1
        assert result.cache_generation == 1
        assert await backend.get_context_view("ctx-usr-1") is None
        assert await backend.get_context_view("ctx-usr-2") is not None
        assert await backend.get_recent_window("usr_1:cnv_1") is None
        assert await backend.get_cache_generation(
            cache_generation_key(DATABASE_ID, "usr_1")
        ) == 1
    finally:
        await connection.close()
