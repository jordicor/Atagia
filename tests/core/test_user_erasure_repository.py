"""SQLite durability gates for lifecycle-fenced user erasure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import close_connection, initialize_database, open_connection
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import (
    ConversationRepository,
    UserRepository,
    user_erasure_marker_hash,
)
from atagia.core.user_erasure_repository import (
    ErasureCleanupTargetSpec,
    LegacyErasureReconciliationError,
    UserErasureConflictError,
    UserErasureRepository,
)
from atagia.models.schemas_jobs import JobEnvelope, JobType
from atagia.services.errors import UserDeletedError


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
CLOCK = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
EVIDENCE_HASH = "a" * 64
INVENTORY_HASH = "b" * 64


async def _database(path: str = ":memory:") -> aiosqlite.Connection:
    return await initialize_database(path, MIGRATIONS_DIR)


async def _seed_user(
    connection: aiosqlite.Connection,
    *,
    with_conversation: bool = False,
) -> str:
    await UserRepository(connection, CLOCK).create_user("usr_erase")
    lifecycle = await (
        await connection.execute(
            "SELECT lifecycle_epoch FROM user_lifecycles WHERE user_id = 'usr_erase'"
        )
    ).fetchone()
    assert lifecycle is not None
    if with_conversation:
        await connection.execute(
            """
            INSERT INTO assistant_modes(
                id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
            ) VALUES ('general_qa', 'General QA', 'hash', '{}', ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (CLOCK.now().isoformat(), CLOCK.now().isoformat()),
        )
        await connection.commit()
        await ConversationRepository(connection, CLOCK).create_conversation(
            "cnv_erase",
            "usr_erase",
            None,
            "general_qa",
            "Erasure fixture",
        )
    return str(lifecycle["lifecycle_epoch"])


async def _seed_nonterminal_jobs(connection: aiosqlite.Connection) -> None:
    repository = JobRunRepository(connection, CLOCK)
    for status in ("queued", "awaiting_claim", "running", "retrying", "deferred"):
        job_id = f"job_{status}"
        await repository.create_durable_job(
            stream_name="atagia:test",
            target_backend="inprocess",
            envelope=JobEnvelope(
                job_id=job_id,
                job_type=JobType.RUN_EVALUATION,
                user_id="usr_erase",
                conversation_id="cnv_erase",
                payload={"private": f"payload-{status}"},
            ),
            source_token_estimate=10,
            size_bucket="small",
        )
        dispatch_token = (
            f"dsp_{status}" if status in {"awaiting_claim", "running"} else None
        )
        await connection.execute(
            """
            UPDATE worker_job_runs
            SET status = ?,
                dispatch_token = ?,
                dispatch_visibility_deadline = ?,
                execution_owner = ?,
                execution_fence = ?,
                execution_lease_expires_at = ?,
                deferred_until = ?
            WHERE job_id = ?
            """,
            (
                status,
                dispatch_token,
                "2026-07-13T13:00:00+00:00" if dispatch_token else None,
                "worker-paused" if status == "running" else None,
                7 if status == "running" else 0,
                "2026-07-13T13:00:00+00:00" if status == "running" else None,
                "2026-07-13T12:30:00+00:00" if status == "deferred" else None,
                job_id,
            ),
        )
    await connection.commit()


async def _delete_canonical_user(connection: aiosqlite.Connection) -> None:
    await connection.execute("DELETE FROM conversations WHERE user_id = 'usr_erase'")
    await connection.execute("DELETE FROM users WHERE id = 'usr_erase'")


async def _prepare_current(
    connection: aiosqlite.Connection,
    *,
    targets: tuple[ErasureCleanupTargetSpec, ...] = (),
    failpoint=None,
):
    return await UserErasureRepository(connection, CLOCK).prepare_current_erasure(
        user_id="usr_erase",
        scope_counts={"conversation_count": 0},
        target_specs=targets,
        canonical_delete=lambda: _delete_canonical_user(connection),
        cleanup_id="erc_current",
        tombstone_id="tmb_current",
        failpoint=failpoint,
    )


async def _legacy_database(path: str, migration_copy: Path) -> aiosqlite.Connection:
    migration_copy.mkdir()
    for source in MIGRATIONS_DIR.glob("*.sql"):
        if int(source.name.split("_", 1)[0]) <= 58:
            shutil.copy2(source, migration_copy / source.name)
    connection = await initialize_database(path, migration_copy)
    await connection.execute(
        """
        INSERT INTO deletion_tombstones(
            id, entity_type, deleted_at, deletion_reason, deleted_by, scope_summary
        ) VALUES (
            'tmb_legacy', 'user', '2025-01-01T00:00:00+00:00',
            'right_to_erasure', 'system', json_object('user_id_sha256', ?)
        )
        """,
        (user_erasure_marker_hash("usr_legacy"),),
    )
    await connection.commit()
    await close_connection(connection)
    return await initialize_database(path, MIGRATIONS_DIR)


@pytest.mark.asyncio
async def test_current_erasure_revokes_every_nonterminal_state_and_reopens(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "current-erasure.db")
    connection = await _database(database_path)
    old_epoch = await _seed_user(connection, with_conversation=True)
    await _seed_nonterminal_jobs(connection)
    try:
        preparation = await _prepare_current(
            connection,
            targets=(
                ErasureCleanupTargetSpec(
                    target_kind="transient_backend",
                    backend_name="inprocess",
                    target_key="old-namespace",
                ),
            ),
        )
        assert preparation.record_version == 1
        assert preparation.revoked_job_count == 5
        assert preparation.lifecycle_epoch == old_epoch

        jobs = await (
            await connection.execute(
                """
                SELECT * FROM worker_job_runs
                WHERE user_id = 'usr_erase'
                ORDER BY job_id
                """
            )
        ).fetchall()
        assert len(jobs) == 5
        assert {row["status"] for row in jobs} == {"cancelled"}
        assert all(row["conversation_id"] is None for row in jobs)
        assert all(row["recovery_envelope_json"] is None for row in jobs)
        assert all(row["dispatch_token"] is None for row in jobs)
        assert all(row["execution_owner"] is None for row in jobs)
        assert {row["terminal_diagnostics_json"] for row in jobs} == {
            '{"reason":"user_erasure","cleanup_id":"erc_current",'
            '"revoked_execution_fence":1}',
            '{"reason":"user_erasure","cleanup_id":"erc_current",'
            '"revoked_execution_fence":8}',
        }
        revoked = await UserErasureRepository(connection, CLOCK).list_revoked_jobs(
            preparation.cleanup_id
        )
        assert {row["prior_status"] for row in revoked} == {
            "queued",
            "awaiting_claim",
            "running",
            "retrying",
            "deferred",
        }
        by_job = {row["job_id"]: row for row in revoked}
        assert by_job["job_awaiting_claim"]["dispatch_token"] == "dsp_awaiting_claim"
        assert by_job["job_running"]["dispatch_token"] == "dsp_running"
        assert by_job["job_running"]["invalidated_execution_fence"] == 8
        assert by_job["job_queued"]["purge_state"] == "not_required"
        with pytest.raises(UserDeletedError):
            await UserRepository(connection, CLOCK).create_user("usr_erase")

        await close_connection(connection)
        connection = await open_connection(database_path)
        repository = UserErasureRepository(connection, CLOCK)
        resumable = await repository.list_resumable_cleanups()
        assert [row["cleanup_id"] for row in resumable] == ["erc_current"]
        targets = await repository.list_cleanup_targets("erc_current")
        version = await repository.checkpoint_target(
            cleanup_id="erc_current",
            target_id=str(targets[0]["target_id"]),
            expected_cleanup_version=1,
            expected_target_version=0,
            checkpoint_state="verified",
            evidence_sha256=EVIDENCE_HASH,
            evidence_reference="inprocess:namespace-purged",
        )
        assert version == 2
        with pytest.raises(UserErasureConflictError):
            await repository.checkpoint_target(
                cleanup_id="erc_current",
                target_id=str(targets[0]["target_id"]),
                expected_cleanup_version=1,
                expected_target_version=0,
                checkpoint_state="verified",
                evidence_sha256=EVIDENCE_HASH,
                evidence_reference="stale-owner",
            )
        for revoked_job in await repository.list_revoked_jobs("erc_current"):
            if revoked_job["purge_state"] != "pending":
                continue
            version = await repository.checkpoint_revoked_job(
                cleanup_id="erc_current",
                job_id=str(revoked_job["job_id"]),
                expected_cleanup_version=version,
                expected_job_version=int(revoked_job["row_version"]),
                purge_state="verified",
                evidence_sha256=EVIDENCE_HASH,
                evidence_reference=f"inprocess:{revoked_job['job_id']}:purged",
            )
        verified = await repository.finalize_cleanup(
            cleanup_id="erc_current",
            expected_record_version=version,
            evidence_manifest_sha256=EVIDENCE_HASH,
            evidence_references=["manifest:current-erasure"],
        )
        assert verified["erasure_cleanup_state"] == "verified"
        assert verified["erasure_protocol_version"] == 1
        assert verified["erasure_lifecycle_epoch"] == old_epoch
        assert await repository.get_cleanup("erc_current") is None
        assert await repository.list_revoked_jobs("erc_current") == []
        assert await repository.list_cleanup_targets("erc_current") == []
        assert (
            await (
                await connection.execute(
                    "SELECT 1 FROM user_lifecycles WHERE user_id = 'usr_erase'"
                )
            ).fetchone()
            is None
        )
        assert (
            await (
                await connection.execute(
                    "SELECT 1 FROM worker_job_runs WHERE user_id = 'usr_erase'"
                )
            ).fetchone()
            is None
        )
        with pytest.raises(UserDeletedError):
            await UserRepository(connection, CLOCK).create_user("usr_erase")

        eligible = await repository.list_retention_eligible_tombstones(
            deleted_before="2026-08-01T00:00:00+00:00"
        )
        assert [row["id"] for row in eligible] == ["tmb_current"]
        assert await repository.retire_tombstone(
            "tmb_current",
            expected_row_version=1,
            deleted_before="2026-08-01T00:00:00+00:00",
        )
        await UserRepository(connection, CLOCK).create_user("usr_erase")
        new_lifecycle = await (
            await connection.execute(
                "SELECT lifecycle_epoch FROM user_lifecycles WHERE user_id = 'usr_erase'"
            )
        ).fetchone()
        assert new_lifecycle is not None
        assert new_lifecycle["lifecycle_epoch"] != old_epoch
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failpoint_name",
    ["cleanup_prepared", "jobs_revoked", "canonical_delete_sealed"],
)
async def test_prepare_failpoints_roll_back_user_lifecycle_and_resume_evidence(
    tmp_path: Path,
    failpoint_name: str,
) -> None:
    database_path = str(tmp_path / f"prepare-{failpoint_name}.db")
    connection = await _database(database_path)
    await _seed_user(connection)

    def failpoint(name: str) -> None:
        if name == failpoint_name:
            raise RuntimeError(f"injected:{name}")

    try:
        with pytest.raises(RuntimeError, match="injected"):
            await _prepare_current(connection, failpoint=failpoint)
        await close_connection(connection)
        connection = await open_connection(database_path)
        assert (
            await (
                await connection.execute("SELECT 1 FROM users WHERE id = 'usr_erase'")
            ).fetchone()
            is not None
        )
        lifecycle = await (
            await connection.execute(
                "SELECT state, erasure_cleanup_id FROM user_lifecycles WHERE user_id = 'usr_erase'"
            )
        ).fetchone()
        assert lifecycle is not None
        assert dict(lifecycle) == {"state": "active", "erasure_cleanup_id": None}
        assert (
            await (
                await connection.execute(
                    "SELECT 1 FROM deletion_tombstones WHERE id = 'tmb_current'"
                )
            ).fetchone()
            is None
        )
        assert (
            await UserErasureRepository(connection, CLOCK).get_cleanup("erc_current")
            is None
        )
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cleanup_kind,failpoint_name",
    [
        ("current", "checkpoint_validation_complete"),
        ("current", "tombstone_verified"),
        ("current", "cleanup_record_deleted"),
        ("legacy", "checkpoint_validation_complete"),
        ("legacy", "tombstone_verified"),
        ("legacy", "cleanup_record_deleted"),
    ],
)
async def test_finalize_failpoints_reopen_to_one_resumable_state(
    tmp_path: Path,
    cleanup_kind: str,
    failpoint_name: str,
) -> None:
    database_path = str(tmp_path / f"final-{cleanup_kind}-{failpoint_name}.db")
    if cleanup_kind == "current":
        connection = await _database(database_path)
        await _seed_user(connection)
        preparation = await _prepare_current(connection)
        expected_version = preparation.record_version
        pending_state = "pending"
    else:
        connection = await _legacy_database(
            database_path,
            tmp_path / f"migrations-{failpoint_name}",
        )
        repository = UserErasureRepository(connection, CLOCK)
        preparation = await repository.prepare_legacy_reconciliation(
            tombstone_id="tmb_legacy",
            candidate_user_id="usr_legacy",
            inventory_manifest_sha256=INVENTORY_HASH,
            historical_inventory_complete=True,
            target_specs=(
                ErasureCleanupTargetSpec(
                    target_kind="historical_destination",
                    target_key="retired-cluster",
                ),
            ),
            cleanup_id="erc_legacy",
        )
        target = (await repository.list_cleanup_targets(preparation.cleanup_id))[0]
        expected_version = await repository.checkpoint_target(
            cleanup_id=preparation.cleanup_id,
            target_id=str(target["target_id"]),
            expected_cleanup_version=preparation.record_version,
            expected_target_version=0,
            checkpoint_state="decommissioned",
            evidence_sha256=EVIDENCE_HASH,
            evidence_reference="destruction-certificate:retired-cluster",
        )
        pending_state = "legacy_unknown"

    def failpoint(name: str) -> None:
        if name == failpoint_name:
            raise RuntimeError(f"injected:{name}")

    try:
        repository = UserErasureRepository(connection, CLOCK)
        with pytest.raises(RuntimeError, match="injected"):
            await repository.finalize_cleanup(
                cleanup_id=preparation.cleanup_id,
                expected_record_version=expected_version,
                evidence_manifest_sha256=EVIDENCE_HASH,
                evidence_references=[f"manifest:{cleanup_kind}"],
                failpoint=failpoint,
            )
        await close_connection(connection)
        connection = await open_connection(database_path)
        repository = UserErasureRepository(connection, CLOCK)
        cleanup = await repository.get_cleanup(preparation.cleanup_id)
        assert cleanup is not None
        assert cleanup["record_version"] == expected_version
        marker = await (
            await connection.execute(
                "SELECT * FROM deletion_tombstones WHERE id = ?",
                (preparation.tombstone_id,),
            )
        ).fetchone()
        assert marker is not None
        assert marker["erasure_cleanup_state"] == pending_state
        assert marker["cleanup_verified_at"] is None
        verified = await repository.finalize_cleanup(
            cleanup_id=preparation.cleanup_id,
            expected_record_version=expected_version,
            evidence_manifest_sha256=EVIDENCE_HASH,
            evidence_references=[f"manifest:{cleanup_kind}"],
        )
        assert verified["erasure_cleanup_state"] == "verified"
        assert await repository.get_cleanup(preparation.cleanup_id) is None
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
async def test_legacy_cutover_is_non_retirable_until_hash_verified_reconciliation(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "legacy-cutover.db")
    connection = await _legacy_database(database_path, tmp_path / "pre59-migrations")
    repository = UserErasureRepository(connection, CLOCK)
    try:
        marker = await (
            await connection.execute(
                "SELECT * FROM deletion_tombstones WHERE id = 'tmb_legacy'"
            )
        ).fetchone()
        assert marker is not None
        assert marker["erasure_protocol_version"] == 0
        assert marker["erasure_cleanup_state"] == "legacy_unknown"
        assert marker["cleanup_verified_at"] is None
        assert [
            row["id"] for row in await repository.list_legacy_unknown_tombstones()
        ] == ["tmb_legacy"]
        await connection.execute(
            "DELETE FROM deletion_tombstones WHERE id = 'tmb_legacy'"
        )
        await connection.commit()
        assert (
            await (
                await connection.execute(
                    "SELECT 1 FROM deletion_tombstones WHERE id = 'tmb_legacy'"
                )
            ).fetchone()
            is not None
        )
        assert (
            await repository.list_retention_eligible_tombstones(
                deleted_before="2026-01-01T00:00:00+00:00"
            )
            == []
        )

        with pytest.raises(
            LegacyErasureReconciliationError, match="complete historical"
        ):
            await repository.prepare_legacy_reconciliation(
                tombstone_id="tmb_legacy",
                candidate_user_id="usr_legacy",
                inventory_manifest_sha256=INVENTORY_HASH,
                historical_inventory_complete=False,
                target_specs=(
                    ErasureCleanupTargetSpec(
                        target_kind="historical_destination",
                        target_key="old-cache",
                    ),
                ),
            )
        with pytest.raises(LegacyErasureReconciliationError, match="does not match"):
            await repository.prepare_legacy_reconciliation(
                tombstone_id="tmb_legacy",
                candidate_user_id="usr_wrong",
                inventory_manifest_sha256=INVENTORY_HASH,
                historical_inventory_complete=True,
                target_specs=(
                    ErasureCleanupTargetSpec(
                        target_kind="historical_destination",
                        target_key="old-cache",
                    ),
                ),
            )
        assert (
            await repository.get_erasure_state_for_candidate("usr_legacy") is not None
        )
        with pytest.raises(UserDeletedError):
            await UserRepository(connection, CLOCK).create_user("usr_legacy")
    finally:
        await close_connection(connection)
