"""Tests for durable prepared initial-context package persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import MigrationManager, initialize_database, open_connection
from atagia.core.ids import generate_prefixed_id
from atagia.core.initial_context_package_repository import (
    InitialContextPackageBuildSupersededError,
    InitialContextPackageRepository,
    InitialContextPackageSourceChangedError,
)
from atagia.core.initial_context_package_revision_repository import (
    InitialContextPackageRevisionRepository,
)
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
    user_erasure_marker_hash,
)
from atagia.services.errors import UserDeletedError
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageCoordinateSignature,
    InitialContextPackageDiagnostics,
    InitialContextPackageKey,
    InitialContextPackageKind,
    InitialContextPackagePolicySignature,
    InitialContextPackageProfileItem,
    InitialContextPackageSourceFingerprint,
    initial_context_package_key_hash,
)
from atagia.services.initial_context_package_sources import (
    assert_initial_context_package_revision_coverage,
)
from atagia.services.initial_context_package_signatures import (
    build_initial_context_package_source_fingerprint,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"


async def _connection_and_clock(
    database_path: str = ":memory:",
) -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    return connection, clock


async def _seed_scope(connection: aiosqlite.Connection, clock: FrozenClock) -> None:
    await UserRepository(connection, clock).create_user("usr_1")
    await UserRepository(connection, clock).create_user("usr_2")
    await connection.execute(
        """
        INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "coding_debug",
            "Coding Debug",
            "hash_1",
            "{}",
            "2026-06-08T09:00:00+00:00",
            "2026-06-08T09:00:00+00:00",
        ),
    )
    await connection.commit()
    conversations = ConversationRepository(connection, clock)
    await conversations.create_conversation(
        "cnv_1",
        "usr_1",
        None,
        "coding_debug",
        "User one chat",
    )
    await conversations.create_conversation(
        "cnv_2",
        "usr_2",
        None,
        "coding_debug",
        "User two chat",
    )


def _key(
    *,
    user_id: str = "usr_1",
    package_kind: InitialContextPackageKind = InitialContextPackageKind.BASELINE,
    conversation_id: str | None = None,
    retrieval_profile_id: str = "default",
    privacy_enforcement: str = "off",
    operational_profile_token: str | None = "normal-token",
) -> InitialContextPackageKey:
    return InitialContextPackageKey(
        version=1,
        package_kind=package_kind,
        user_id=user_id,
        conversation_id=conversation_id,
        retrieval_profile_id=retrieval_profile_id,
        subject_json={
            "platform_id": "aurvek",
            "character_id": "core",
            "workspace_id": "wrk_main",
        },
        policy_json={
            "effective_policy_hash": "policy-main",
            "privacy_enforcement": privacy_enforcement,
        },
        coordinate_json={
            "space_id": "space-main",
            "mind_topology": "unimind",
        },
        operational_json={
            "operational_profile": (
                {"token": operational_profile_token}
                if operational_profile_token is not None
                else None
            )
        },
    )


def _blocks(label: str = "baseline") -> InitialContextPackageBlocks:
    return InitialContextPackageBlocks(
        contract_block=f"{label}: interaction contract.",
        prepared_memory_profile_block=f"{label}: prepared memory profile.",
        current_state_block=f"{label}: current state.",
        coordinate_context_block=f"{label}: coordinate context.",
        conversation_summary_block=f"{label}: summary.",
        working_topic_block=f"{label}: working topic.",
        recent_verbatim_seed=[
            {
                "message_id": f"msg_{label}",
                "role": "user",
                "text": f"{label} recent turn",
            }
        ],
        empty_markers={"same_chat_history_known_empty": label == "baseline"},
        source_counts={"profile_items": 1, "recent_verbatim_seed": 1},
        profile_items=[
            InitialContextPackageProfileItem(
                item_id=f"item_{label}",
                text=f"{label}: the user tends to ask in Spanish.",
                reason_category="communication_profile",
                source_refs=[
                    {
                        "source_kind": "communication_profile",
                        "profile_id": f"ucp_{label}",
                    }
                ],
                freshness_json={"profile_updated_at": "2026-06-08T09:00:00+00:00"},
            )
        ],
    )


def _policy_signature() -> InitialContextPackagePolicySignature:
    return InitialContextPackagePolicySignature(
        effective_policy_hash="policy-main",
        policy_prompt_hash="prompt-main",
        privacy_enforcement="off",
        authority_json={"atagia_master": False},
    )


def _coordinate_signature() -> InitialContextPackageCoordinateSignature:
    return InitialContextPackageCoordinateSignature(
        coordinate_signature_hash="coord-main",
        complete=True,
        markers_json={"space_revision": "space-rev-1"},
    )


def _source_fingerprint(label: str = "baseline") -> InitialContextPackageSourceFingerprint:
    return InitialContextPackageSourceFingerprint(
        source_fingerprint_hash=f"fingerprint-{label}",
        source_markers_json={
            "memory_objects_max_updated_at": "2026-06-08T08:59:00+00:00",
            "communication_profile_updated_at": "2026-06-08T08:58:00+00:00",
        },
    )


async def _upsert(
    repository: InitialContextPackageRepository,
    key: InitialContextPackageKey,
    *,
    label: str = "baseline",
    **extra: object,
):
    return await repository.upsert_package(
        package_kind=key.package_kind,
        version=key.version,
        user_id=key.user_id,
        retrieval_profile_id=key.retrieval_profile_id,
        key_json=key,
        policy_signature_json=_policy_signature(),
        coordinate_signature_json=_coordinate_signature(),
        source_fingerprint_json=_source_fingerprint(label),
        blocks_json=_blocks(label),
        source_refs_json={
            "profile_items": [
                {
                    "source_kind": "communication_profile",
                    "profile_id": f"ucp_{label}",
                }
            ]
        },
        diagnostics_json=InitialContextPackageDiagnostics(
            package_tokens_estimate=256,
            source_counts={"profile_items": 1},
            selected_profile_items=1,
        ),
        **extra,
    )


@pytest.mark.asyncio
async def test_repository_upserts_and_reads_packages_by_user() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        baseline_key = _key()
        baseline = await _upsert(repository, baseline_key)
        baseline_hash = initial_context_package_key_hash(baseline_key)

        assert baseline.package_key_hash == baseline_hash
        assert baseline.source_fingerprint_json.source_fingerprint_hash == "fingerprint-baseline"
        assert baseline.blocks_json.profile_items[0].source_refs[0]["profile_id"] == "ucp_baseline"
        assert await repository.get_by_key_hash(
            user_id="usr_2",
            package_key_hash=baseline_hash,
        ) is None

        read_result = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=baseline_hash,
        )
        assert read_result.status == "hit"
        assert read_result.package is not None
        assert read_result.package.id == baseline.id

        clock.advance(seconds=60)
        updated = await _upsert(repository, baseline_key, label="baseline_updated")
        assert updated.id == baseline.id
        assert updated.created_at == baseline.created_at
        assert updated.updated_at == "2026-06-08T09:01:00+00:00"
        assert updated.source_fingerprint_json.source_fingerprint_hash == (
            "fingerprint-baseline_updated"
        )

        conversation_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
        )
        conversation = await _upsert(
            repository,
            conversation_key,
            label="conversation",
        )

        assert conversation.conversation_id == "cnv_1"
        latest = await repository.get_latest_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert latest is not None
        assert latest.id == conversation.id
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_repository_rejects_mismatched_hash_and_conversation_owner() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        baseline_key = _key()
        with pytest.raises(ValueError, match="package_key_hash must match"):
            await repository.upsert_package(
                package_kind=baseline_key.package_kind,
                version=baseline_key.version,
                user_id=baseline_key.user_id,
                retrieval_profile_id=baseline_key.retrieval_profile_id,
                key_json=baseline_key,
                package_key_hash="icp:v1:not-the-real-hash",
            )

        wrong_owner_key = _key(
            user_id="usr_1",
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_2",
        )
        with pytest.raises(aiosqlite.IntegrityError, match="conversation_id must belong"):
            await _upsert(repository, wrong_owner_key, label="wrong_owner")
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_repository_marks_stale_and_key_family_filters_by_user() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        usr_1_default_key = _key(retrieval_profile_id="default")
        usr_1_alt_key = _key(retrieval_profile_id="alt")
        usr_2_default_key = _key(user_id="usr_2", retrieval_profile_id="default")
        await _upsert(repository, usr_1_default_key, label="usr_1_default")
        await _upsert(repository, usr_1_alt_key, label="usr_1_alt")
        await _upsert(repository, usr_2_default_key, label="usr_2_default")

        default_hash = initial_context_package_key_hash(usr_1_default_key)
        assert await repository.mark_stale_by_key_hash(
            user_id="usr_1",
            package_key_hash=default_hash,
        ) == 1
        assert await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=default_hash,
        ) is None
        stale = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=default_hash,
        )
        assert stale.status == "stale"
        assert stale.fallback_reason == "package_stale"

        assert await repository.mark_stale_for_key_family(
            user_id="usr_1",
            retrieval_profile_id="alt",
        ) == 1
        usr_2_read = await repository.read_by_key_hash(
            user_id="usr_2",
            package_key_hash=initial_context_package_key_hash(usr_2_default_key),
        )
        assert usr_2_read.status == "hit"

        with pytest.raises(ValueError, match="at least one family filter"):
            await repository.mark_stale_for_key_family(user_id="usr_1")

        assert await repository.delete_for_key_family(
            user_id="usr_2",
            retrieval_profile_id="default",
        ) == 1
        deleted_usr_2 = await repository.read_by_key_hash(
            user_id="usr_2",
            package_key_hash=initial_context_package_key_hash(usr_2_default_key),
        )
        assert deleted_usr_2.status == "miss"
        assert deleted_usr_2.package is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_repository_stales_only_matching_package_variant() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        off_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
            privacy_enforcement="off",
            operational_profile_token="profile-a",
        )
        enforce_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
            privacy_enforcement="enforce",
            operational_profile_token="profile-a",
        )
        profile_b_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
            privacy_enforcement="off",
            operational_profile_token="profile-b",
        )
        await _upsert(repository, off_key, label="off")
        await _upsert(repository, enforce_key, label="enforce")
        await _upsert(repository, profile_b_key, label="profile_b")

        assert await repository.mark_stale_for_key_family(
            user_id="usr_1",
            package_kind=InitialContextPackageKind.CONVERSATION,
            retrieval_profile_id="default",
            conversation_id="cnv_1",
            privacy_enforcement="off",
            operational_profile_token="profile-a",
        ) == 1

        off_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(off_key),
        )
        enforce_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(enforce_key),
        )
        profile_b_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(profile_b_key),
        )
        assert off_read.status == "stale"
        assert enforce_read.status == "hit"
        assert profile_b_read.status == "hit"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_repository_stales_family_except_excluded_package_hashes() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        current_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
            operational_profile_token="profile-a",
        )
        older_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
            operational_profile_token="profile-b",
        )
        current_hash = initial_context_package_key_hash(current_key)
        older_hash = initial_context_package_key_hash(older_key)
        await _upsert(repository, current_key, label="current")
        await _upsert(repository, older_key, label="older")

        assert await repository.mark_stale_for_key_family(
            user_id="usr_1",
            package_kind=InitialContextPackageKind.CONVERSATION,
            retrieval_profile_id="default",
            conversation_id="cnv_1",
            exclude_package_key_hashes=[current_hash],
        ) == 1

        current_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=current_hash,
        )
        older_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=older_hash,
        )
        assert current_read.status == "hit"
        assert older_read.status == "stale"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_repository_delete_paths_remove_user_and_conversation_packages() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)

        baseline_key = _key()
        conversation_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
        )
        await _upsert(repository, baseline_key)
        await _upsert(repository, conversation_key, label="conversation")

        assert await repository.delete_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
        ) == 1
        deleted_conversation = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(conversation_key),
        )
        assert deleted_conversation.status == "miss"
        assert deleted_conversation.package is None
        baseline_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(baseline_key),
        )
        assert baseline_read.status == "hit"

        assert await repository.delete_for_user("usr_1") == 1
        deleted_user = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(baseline_key),
        )
        assert deleted_user.status == "miss"
        assert deleted_user.package is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_fingerprint_survives_database_reopen(tmp_path: Path) -> None:
    database_path = str(tmp_path / "initial-context-package.db")
    connection, clock = await _connection_and_clock(database_path)
    baseline_key = _key()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        await _upsert(repository, baseline_key)
    finally:
        await connection.close()

    reopened, reopened_clock = await _connection_and_clock(database_path)
    try:
        repository = InitialContextPackageRepository(reopened, reopened_clock)
        read_result = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(baseline_key),
        )

        assert read_result.status == "hit"
        assert read_result.package is not None
        assert read_result.package.source_fingerprint_json.source_fingerprint_hash == (
            "fingerprint-baseline"
        )
        assert read_result.package.source_fingerprint_json.source_markers_json == {
            "communication_profile_updated_at": "2026-06-08T08:58:00+00:00",
            "memory_objects_max_updated_at": "2026-06-08T08:59:00+00:00",
        }
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_schema_cutover_marks_packages_without_coordinates_stale(
    tmp_path: Path,
) -> None:
    pre_cutover_migrations = tmp_path / "pre-cutover-migrations"
    pre_cutover_migrations.mkdir()
    for migration_path in MIGRATIONS_DIR.glob("*.sql"):
        migration_version = int(migration_path.name.split("_", 1)[0])
        if migration_version <= 56:
            (pre_cutover_migrations / migration_path.name).symlink_to(migration_path)

    database_path = str(tmp_path / "pre-cutover.db")
    connection = await initialize_database(database_path, pre_cutover_migrations)
    clock = FrozenClock(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    try:
        await _seed_scope(connection, clock)
        await connection.execute(
            """
            INSERT INTO initial_context_packages(
                id,
                package_key_hash,
                package_kind,
                version,
                user_id,
                conversation_id,
                retrieval_profile_id,
                key_json,
                policy_signature_json,
                coordinate_signature_json,
                source_fingerprint_json,
                blocks_json,
                source_refs_json,
                diagnostics_json,
                build_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, 'baseline', 1, ?, NULL, ?, '{}', '{}', '{}', '{}', '{}', '{}', '{}', 'active', ?, ?)
            """,
            (
                "icp_pre_cutover",
                "icp:v1:pre-cutover",
                "usr_1",
                "coding_debug",
                "2026-06-08T09:00:00+00:00",
                "2026-06-08T09:00:00+00:00",
            ),
        )
        await connection.commit()

        await MigrationManager(MIGRATIONS_DIR).apply_all(connection)
        await connection.close()
        connection = await open_connection(database_path)
        row = await (
            await connection.execute(
                """
                SELECT build_status,
                       source_user_lifecycle_epoch,
                       source_user_revision,
                       source_conversation_lifecycle_epoch,
                       source_conversation_revision,
                       package_row_version,
                       active_build_attempt_id
                FROM initial_context_packages
                WHERE id = ?
                """,
                ("icp_pre_cutover",),
            )
        ).fetchone()
        assert row is not None
        assert dict(row) == {
            "build_status": "stale",
            "source_user_lifecycle_epoch": None,
            "source_user_revision": None,
            "source_conversation_lifecycle_epoch": None,
            "source_conversation_revision": None,
            "package_row_version": 1,
            "active_build_attempt_id": None,
        }
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_revision_mismatch_never_mutates_newer_active_row() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        key = _key()
        original = await _upsert(repository, key, label="original")
        old_coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id=None,
        )
        assert old_coordinates is not None
        old_generation = await revisions.reserve_refresh_generation(
            user_id="usr_1",
            expected_user_lifecycle_epoch=old_coordinates.user_lifecycle_epoch,
        )
        assert old_generation is not None
        old_attempt = await revisions.begin_build_attempt(
            attempt_id=generate_prefixed_id("ica"),
            user_id="usr_1",
            conversation_id=None,
            package_key_hash=original.package_key_hash,
            refresh_generation=old_generation,
            source_coordinates=old_coordinates,
            refresh_request_job_id="job_old",
        )

        # The diagnostic aggregate does not include this field. The database
        # trigger still advances the exact source revision.
        await connection.execute(
            "UPDATE users SET external_ref = ? WHERE id = ?",
            ("changed-with-equal-aggregate", "usr_1"),
        )
        await connection.commit()
        newer = await _upsert(repository, key, label="newer")
        newer_bytes = newer.model_dump(mode="json")

        with pytest.raises(InitialContextPackageSourceChangedError):
            await _upsert(
                repository,
                key,
                label="old_resumed",
                source_coordinates=old_coordinates,
                build_attempt=old_attempt,
                refresh_request_job_id="job_old",
            )

        after = await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=original.package_key_hash,
            include_inactive=True,
        )
        assert after is not None
        assert after.model_dump(mode="json") == newer_bytes
        attempt_row = await (
            await connection.execute(
                """
                SELECT status
                FROM initial_context_package_build_attempts
                WHERE attempt_id = ?
                """,
                (old_attempt.attempt_id,),
            )
        ).fetchone()
        assert attempt_row["status"] == "source_changed"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_older_generation_cannot_overwrite_or_stale_newer_package() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        key = _key()
        original = await _upsert(repository, key, label="original")
        coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id=None,
        )
        assert coordinates is not None

        old_generation = await revisions.reserve_refresh_generation(
            user_id="usr_1",
            expected_user_lifecycle_epoch=coordinates.user_lifecycle_epoch,
        )
        new_generation = await revisions.reserve_refresh_generation(
            user_id="usr_1",
            expected_user_lifecycle_epoch=coordinates.user_lifecycle_epoch,
        )
        assert old_generation is not None and new_generation is not None
        old_attempt = await revisions.begin_build_attempt(
            attempt_id=generate_prefixed_id("ica"),
            user_id="usr_1",
            conversation_id=None,
            package_key_hash=original.package_key_hash,
            refresh_generation=old_generation,
            source_coordinates=coordinates,
            refresh_request_job_id="job_old_generation",
        )
        new_attempt = await revisions.begin_build_attempt(
            attempt_id=generate_prefixed_id("ica"),
            user_id="usr_1",
            conversation_id=None,
            package_key_hash=original.package_key_hash,
            refresh_generation=new_generation,
            source_coordinates=coordinates,
            refresh_request_job_id="job_new_generation",
        )
        newer = await _upsert(
            repository,
            key,
            label="new_generation",
            source_coordinates=coordinates,
            build_attempt=new_attempt,
            refresh_request_job_id="job_new_generation",
        )
        newer_bytes = newer.model_dump(mode="json")

        with pytest.raises(InitialContextPackageBuildSupersededError):
            await _upsert(
                repository,
                key,
                label="old_generation",
                source_coordinates=coordinates,
                build_attempt=old_attempt,
                refresh_request_job_id="job_old_generation",
            )
        assert await repository.mark_stale_if_row_version(
            user_id="usr_1",
            package_key_hash=original.package_key_hash,
            expected_row_version=original.package_row_version,
        ) is False
        after = await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=original.package_key_hash,
            include_inactive=True,
        )
        assert after is not None
        assert after.model_dump(mode="json") == newer_bytes
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_reader_rejects_active_row_after_exact_revision_change() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        key = _key()
        package = await _upsert(repository, key)
        await connection.execute(
            "UPDATE users SET external_ref = ? WHERE id = ?",
            ("revision-only-change", "usr_1"),
        )
        await connection.commit()

        result = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=package.package_key_hash,
        )
        assert result.status == "stale"
        assert result.fallback_reason == "source_revision_mismatch"
        stored = await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=package.package_key_hash,
            include_inactive=True,
        )
        assert stored is not None
        assert stored.build_status.value == "stale"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_revision_rolls_back_with_canonical_mutation() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        before = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id=None,
        )
        assert before is not None

        await connection.execute("BEGIN IMMEDIATE")
        await connection.execute(
            "UPDATE users SET external_ref = ? WHERE id = ?",
            ("rolled-back", "usr_1"),
        )
        during = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id=None,
        )
        assert during is not None
        assert during.user_revision == before.user_revision + 1
        await connection.rollback()

        after = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id=None,
        )
        assert after == before
        row = await (
            await connection.execute(
                "SELECT external_ref FROM users WHERE id = ?",
                ("usr_1",),
            )
        ).fetchone()
        assert row["external_ref"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_revision_detects_equal_aggregate_raw_sql_changes() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        before_insert = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert before_insert is not None

        await messages.create_message(
            "msg_equal_aggregate",
            "cnv_1",
            "user",
            1,
            "Original text.",
        )
        after_insert = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert after_insert is not None
        assert after_insert.conversation_revision > (
            before_insert.conversation_revision  # type: ignore[operator]
        )
        fingerprint_before_edit = (
            await build_initial_context_package_source_fingerprint(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
            )
        )

        await connection.execute(
            """
            UPDATE messages
            SET text = ?
            WHERE id = ?
              AND conversation_id = ?
            """,
            ("Changed text with identical aggregate markers.", "msg_equal_aggregate", "cnv_1"),
        )
        await connection.commit()
        after_edit = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        fingerprint_after_edit = (
            await build_initial_context_package_source_fingerprint(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
            )
        )
        assert after_edit is not None
        assert after_edit.conversation_revision == (
            after_insert.conversation_revision + 1  # type: ignore[operator]
        )
        assert fingerprint_after_edit == fingerprint_before_edit

        await connection.execute(
            "DELETE FROM messages WHERE id = ? AND conversation_id = ?",
            ("msg_equal_aggregate", "cnv_1"),
        )
        await connection.commit()
        after_delete = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert after_delete is not None
        assert after_delete.conversation_revision == (
            after_edit.conversation_revision + 1  # type: ignore[operator]
        )

        await messages.create_message(
            "msg_equal_aggregate",
            "cnv_1",
            "user",
            1,
            "Changed text with identical aggregate markers.",
        )
        after_reinsert = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        fingerprint_after_reinsert = (
            await build_initial_context_package_source_fingerprint(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
            )
        )
        assert after_reinsert is not None
        assert after_reinsert.conversation_revision > (
            after_delete.conversation_revision  # type: ignore[operator]
        )
        assert fingerprint_after_reinsert == fingerprint_after_edit
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_builder_cannot_activate_after_source_edit() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
        )
        coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert coordinates is not None
        generation = await revisions.reserve_refresh_generation(
            user_id="usr_1",
            expected_user_lifecycle_epoch=coordinates.user_lifecycle_epoch,
        )
        assert generation is not None
        attempt = await revisions.begin_build_attempt(
            attempt_id=generate_prefixed_id("ica"),
            user_id="usr_1",
            conversation_id="cnv_1",
            package_key_hash=initial_context_package_key_hash(key),
            refresh_generation=generation,
            source_coordinates=coordinates,
            refresh_request_job_id="job_conversation_old",
        )

        await MessageRepository(connection, clock).create_message(
            "msg_conversation_edit",
            "cnv_1",
            "user",
            1,
            "Source changed after the conversation builder read it.",
        )
        with pytest.raises(InitialContextPackageSourceChangedError):
            await _upsert(
                repository,
                key,
                label="conversation_old",
                source_coordinates=coordinates,
                build_attempt=attempt,
                refresh_request_job_id="job_conversation_old",
            )

        assert await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(key),
            include_inactive=True,
        ) is None
        attempt_row = await (
            await connection.execute(
                """
                SELECT status
                FROM initial_context_package_build_attempts
                WHERE attempt_id = ?
                """,
                (attempt.attempt_id,),
            )
        ).fetchone()
        assert attempt_row["status"] == "source_changed"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_conversation_owner_move_replaces_lifecycle_epoch() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        old_key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
        )
        old_package = await _upsert(repository, old_key, label="old-owner")
        old_coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert old_coordinates is not None

        await connection.execute(
            "UPDATE conversations SET user_id = ? WHERE id = ?",
            ("usr_2", "cnv_1"),
        )
        await connection.commit()

        assert await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        ) is None
        new_coordinates = await revisions.capture_active_coordinates(
            user_id="usr_2",
            conversation_id="cnv_1",
        )
        assert new_coordinates is not None
        assert (
            new_coordinates.conversation_lifecycle_epoch
            != old_coordinates.conversation_lifecycle_epoch
        )
        old_read = await repository.read_by_key_hash(
            user_id="usr_1",
            package_key_hash=old_package.package_key_hash,
        )
        assert old_read.status == "stale"
        assert old_read.fallback_reason == "source_revision_mismatch"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_retention_eligible_id_reuse_rejects_old_lifecycle_builder() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        revisions = InitialContextPackageRevisionRepository(connection, clock)
        repository = InitialContextPackageRepository(connection, clock)
        key = _key(
            package_kind=InitialContextPackageKind.CONVERSATION,
            conversation_id="cnv_1",
        )
        old_coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert old_coordinates is not None
        old_generation = await revisions.reserve_refresh_generation(
            user_id="usr_1",
            expected_user_lifecycle_epoch=old_coordinates.user_lifecycle_epoch,
        )
        assert old_generation is not None
        old_attempt = await revisions.begin_build_attempt(
            attempt_id=generate_prefixed_id("ica"),
            user_id="usr_1",
            conversation_id="cnv_1",
            package_key_hash=initial_context_package_key_hash(key),
            refresh_generation=old_generation,
            source_coordinates=old_coordinates,
            refresh_request_job_id="job_before_erasure",
        )

        await connection.execute("DELETE FROM conversations WHERE id = ?", ("cnv_1",))
        await connection.execute("DELETE FROM users WHERE id = ?", ("usr_1",))
        await connection.execute(
            """
            INSERT INTO deletion_tombstones(
                id,
                entity_type,
                deleted_at,
                deletion_reason,
                deleted_by,
                scope_summary,
                erasure_protocol_version,
                erasure_cleanup_state,
                cleanup_verified_at,
                cleanup_evidence_manifest_sha256,
                cleanup_evidence_references_json,
                erasure_lifecycle_epoch,
                legacy_reconciled,
                erasure_row_version
            )
            VALUES (?, 'user', ?, 'right_to_erasure', 'system', ?, 1, 'verified', ?, ?, '[]', ?, 0, 1)
            """,
            (
                "del_usr_1",
                "2026-01-01T00:00:00+00:00",
                json.dumps(
                    {"user_id_sha256": user_erasure_marker_hash("usr_1")},
                    sort_keys=True,
                ),
                "2026-01-02T00:00:00+00:00",
                "a" * 64,
                old_coordinates.user_lifecycle_epoch,
            ),
        )
        await connection.commit()

        with pytest.raises(UserDeletedError):
            await UserRepository(connection, clock).create_user("usr_1")

        await connection.execute(
            "DELETE FROM user_lifecycles WHERE user_id = ?",
            ("usr_1",),
        )
        retired = await connection.execute(
            "DELETE FROM deletion_tombstones WHERE id = ?",
            ("del_usr_1",),
        )
        await connection.commit()
        assert int(retired.rowcount or 0) == 1

        await UserRepository(connection, clock).create_user("usr_1")
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "coding_debug",
            "Recreated chat",
        )
        new_coordinates = await revisions.capture_active_coordinates(
            user_id="usr_1",
            conversation_id="cnv_1",
        )
        assert new_coordinates is not None
        assert new_coordinates.user_lifecycle_epoch != (
            old_coordinates.user_lifecycle_epoch
        )
        assert new_coordinates.conversation_lifecycle_epoch != (
            old_coordinates.conversation_lifecycle_epoch
        )

        with pytest.raises(InitialContextPackageSourceChangedError):
            await _upsert(
                repository,
                key,
                label="old_lifecycle",
                source_coordinates=old_coordinates,
                build_attempt=old_attempt,
                refresh_request_job_id="job_before_erasure",
            )
        assert await repository.get_by_key_hash(
            user_id="usr_1",
            package_key_hash=initial_context_package_key_hash(key),
            include_inactive=True,
        ) is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_revision_coverage_guard_fails_for_missing_trigger() -> None:
    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        await assert_initial_context_package_revision_coverage(connection)
        await connection.execute("DROP TRIGGER icp_source_memory_objects_au")
        with pytest.raises(RuntimeError, match="icp_source_memory_objects_au"):
            await assert_initial_context_package_revision_coverage(connection)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_revision_coverage_guard_fails_for_unmapped_builder_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atagia.services import initial_context_package_builder

    connection, clock = await _connection_and_clock()
    try:
        await _seed_scope(connection, clock)
        monkeypatch.setattr(
            initial_context_package_builder,
            "INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES",
            (
                initial_context_package_builder.INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES
                | {"unmapped_builder_input"}
            ),
        )
        with pytest.raises(RuntimeError, match="unmapped_builder_input"):
            await assert_initial_context_package_revision_coverage(connection)
    finally:
        await connection.close()
