"""Tests for durable prepared initial-context package persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.repositories import ConversationRepository, UserRepository
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

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


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
