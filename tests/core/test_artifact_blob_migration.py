"""Offline retirement tests for legacy local-file artifact blobs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from atagia.artifact_blob_migrate_cli import main_async as migration_main_async
from atagia.core.artifact_payload_repository import ArtifactPayloadRepository
from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.repositories import UserRepository
from atagia.services.artifact_blob_migration import (
    LegacyArtifactBlobStateError,
    assert_artifact_blob_runtime_ready,
    drain_artifact_blob_cleanup_intents,
    inventory_legacy_artifact_blobs,
    migrate_legacy_artifact_blobs,
    verify_artifact_blob_migration,
)


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
CLOCK = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))


async def _seed_artifact(
    connection, artifact_id: str, *, payload_blob_id: str | None = None
) -> None:
    await ArtifactRepository(connection, CLOCK).create_artifact(
        artifact_id=artifact_id,
        user_id="usr_blob",
        workspace_id=None,
        conversation_id=None,
        message_id=None,
        artifact_type="file",
        source_kind="host_embedded",
        payload_blob_id=payload_blob_id,
    )


async def _insert_legacy_artifact_blob(
    connection,
    *,
    artifact_id: str,
    storage_uri: str,
    content: bytes,
    sha256: str | None = None,
    byte_size: int | None = None,
) -> None:
    await connection.execute(
        """
        INSERT INTO artifact_blobs(
            artifact_id, storage_kind, blob_bytes, storage_uri,
            byte_size, sha256, created_at, updated_at
        )
        VALUES (?, 'local_file', NULL, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            storage_uri,
            len(content) if byte_size is None else byte_size,
            sha256 or hashlib.sha256(content).hexdigest(),
            CLOCK.now().isoformat(),
            CLOCK.now().isoformat(),
        ),
    )
    await connection.commit()


async def _insert_legacy_payload_blob(
    connection,
    *,
    payload_id: str,
    storage_uri: str,
    content: bytes,
    status: str = "ready",
) -> None:
    await connection.execute(
        """
        INSERT INTO artifact_payload_blobs(
            id, user_id, storage_kind, identity_kind, content_sha256,
            byte_size, blob_bytes, storage_key, external_uri, status,
            created_at, updated_at
        )
        VALUES (?, 'usr_blob', 'local_file', 'content_sha256', ?, ?, NULL, ?, NULL, ?, ?, ?)
        """,
        (
            payload_id,
            hashlib.sha256(content).hexdigest(),
            len(content),
            storage_uri,
            status,
            CLOCK.now().isoformat(),
            CLOCK.now().isoformat(),
        ),
    )
    await connection.commit()


@pytest.fixture
async def blob_connection():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    await UserRepository(connection, CLOCK).create_user("usr_blob")
    try:
        yield connection
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_shared_identity_migrates_across_tables_and_batches_before_cleanup(
    blob_connection,
    tmp_path: Path,
) -> None:
    content = b"one physical file shared by both artifact tables"
    storage_root = tmp_path / "artifact_blobs"
    storage_path = storage_root / "shared" / "payload.bin"
    storage_path.parent.mkdir(parents=True)
    storage_path.write_bytes(content)
    storage_uri = "shared/payload.bin"

    await _seed_artifact(blob_connection, "art_legacy")
    await _insert_legacy_artifact_blob(
        blob_connection,
        artifact_id="art_legacy",
        storage_uri=storage_uri,
        content=content,
    )
    await _insert_legacy_payload_blob(
        blob_connection,
        payload_id="apb_legacy",
        storage_uri=str(storage_path),
        content=content,
    )
    await _seed_artifact(blob_connection, "art_payload", payload_blob_id="apb_legacy")

    inventory = await inventory_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
    )
    assert inventory.local_reference_count == 2
    assert len(inventory.identities) == 1
    assert inventory.issue_count == 0

    first = await migrate_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
        batch_size=1,
    )
    assert first.migrated_reference_count == 1
    assert first.remaining_local_reference_count == 1
    assert first.cleanup_intent_count == 0
    assert storage_path.is_file()

    second = await migrate_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
        batch_size=1,
    )
    assert second.migrated_reference_count == 1
    assert second.remaining_local_reference_count == 0
    assert second.cleanup_intent_count == 1
    assert storage_path.is_file()

    cleanup = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=storage_root,
    )
    assert cleanup.deleted_file_count == 1
    assert cleanup.remaining_intent_count == 0
    assert not storage_path.exists()
    verification = await verify_artifact_blob_migration(blob_connection)
    assert verification.is_complete

    cursor = await blob_connection.execute(
        """
        SELECT blob_bytes, byte_size, sha256
        FROM artifact_blobs
        WHERE artifact_id = 'art_legacy'
        """
    )
    row = await cursor.fetchone()
    assert bytes(row["blob_bytes"]) == content
    assert row["byte_size"] == len(content)
    assert row["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_inventory_reports_missing_corrupt_and_symlink_escape_without_mutating_rows(
    blob_connection,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (storage_root / "escape.bin").symlink_to(outside)

    cases = (
        ("art_missing", "missing.bin", b"expected", None, None),
        (
            "art_hash",
            "hash.bin",
            b"actual",
            hashlib.sha256(b"different").hexdigest(),
            None,
        ),
        ("art_size", "size.bin", b"actual", None, 99),
        ("art_escape", "escape.bin", b"outside", None, None),
    )
    for artifact_id, uri, content, sha256, byte_size in cases:
        await _seed_artifact(blob_connection, artifact_id)
        if artifact_id not in {"art_missing", "art_escape"}:
            (storage_root / uri).write_bytes(content)
        await _insert_legacy_artifact_blob(
            blob_connection,
            artifact_id=artifact_id,
            storage_uri=uri,
            content=content,
            sha256=sha256,
            byte_size=byte_size,
        )

    inventory = await inventory_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
    )
    assert inventory.local_reference_count == 4
    assert inventory.issue_count >= 4
    assert any("symlink" in issue for issue in inventory.unresolved_issues)
    result = await migrate_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
        batch_size=20,
    )
    assert result.migrated_reference_count == 0
    cursor = await blob_connection.execute(
        "SELECT COUNT(*) AS count FROM artifact_blobs WHERE storage_kind = 'local_file'"
    )
    assert (await cursor.fetchone())["count"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "expected_storage_kind"),
    (
        ("before_blob_write", "local_file"),
        ("before_row_commit", "local_file"),
        ("after_row_commit", "sqlite_blob"),
    ),
)
async def test_migration_is_resumable_at_each_row_failpoint(
    blob_connection,
    tmp_path: Path,
    stage: str,
    expected_storage_kind: str,
) -> None:
    content = b"failpoint payload"
    storage_root = tmp_path / stage
    storage_root.mkdir()
    path = storage_root / "payload.bin"
    path.write_bytes(content)
    artifact_id = f"art_{stage}"
    await _seed_artifact(blob_connection, artifact_id)
    await _insert_legacy_artifact_blob(
        blob_connection,
        artifact_id=artifact_id,
        storage_uri="payload.bin",
        content=content,
    )

    def failpoint(current_stage: str, _identity: str) -> None:
        if current_stage == stage:
            raise RuntimeError(f"injected {stage}")

    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        await migrate_legacy_artifact_blobs(
            blob_connection,
            storage_root=storage_root,
            failpoint=failpoint,
        )
    cursor = await blob_connection.execute(
        "SELECT storage_kind, blob_bytes FROM artifact_blobs WHERE artifact_id = ?",
        (artifact_id,),
    )
    row = await cursor.fetchone()
    assert row["storage_kind"] == expected_storage_kind
    if expected_storage_kind == "local_file":
        assert row["blob_bytes"] is None
    else:
        assert bytes(row["blob_bytes"]) == content
    assert path.is_file()

    resumed = await migrate_legacy_artifact_blobs(
        blob_connection,
        storage_root=storage_root,
    )
    assert resumed.remaining_local_reference_count == 0
    await drain_artifact_blob_cleanup_intents(
        blob_connection, storage_root=storage_root
    )
    assert not path.exists()


@pytest.mark.asyncio
async def test_cleanup_rechecks_all_declared_tables_and_retries_unlink_failpoint(
    blob_connection,
    tmp_path: Path,
) -> None:
    content = b"final global reference check"
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    path = storage_root / "payload.bin"
    path.write_bytes(content)
    await _seed_artifact(blob_connection, "art_first")
    await _insert_legacy_artifact_blob(
        blob_connection,
        artifact_id="art_first",
        storage_uri="payload.bin",
        content=content,
    )
    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)

    await _insert_legacy_payload_blob(
        blob_connection,
        payload_id="apb_late",
        storage_uri="payload.bin",
        content=content,
        status="quarantined",
    )
    deferred = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=storage_root,
    )
    assert deferred.deferred_intent_count == 1
    assert path.is_file()

    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)

    def fail_before_unlink(stage: str, _identity: str) -> None:
        if stage == "before_unlink":
            raise RuntimeError("injected before unlink")

    with pytest.raises(RuntimeError, match="injected before unlink"):
        await drain_artifact_blob_cleanup_intents(
            blob_connection,
            storage_root=storage_root,
            failpoint=fail_before_unlink,
        )
    assert path.is_file()
    assert (
        await verify_artifact_blob_migration(blob_connection)
    ).cleanup_intent_count == 1
    cleanup = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=storage_root,
    )
    assert cleanup.deleted_file_count == 1
    assert not path.exists()


@pytest.mark.asyncio
async def test_unlink_failure_retains_cleanup_intent_for_retry(
    blob_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"retry unlink"
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    path = storage_root / "payload.bin"
    path.write_bytes(content)
    await _seed_artifact(blob_connection, "art_unlink_retry")
    await _insert_legacy_artifact_blob(
        blob_connection,
        artifact_id="art_unlink_retry",
        storage_uri="payload.bin",
        content=content,
    )
    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)

    original_unlink = Path.unlink

    def fail_unlink(_path: Path, *args, **kwargs) -> None:
        del args, kwargs
        raise OSError("injected unlink failure")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    failed = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=storage_root,
    )
    assert failed.deferred_intent_count == 1
    assert failed.remaining_intent_count == 1
    assert path.is_file()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    retried = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=storage_root,
    )
    assert retried.remaining_intent_count == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_existing_sqlite_payload_is_reused_during_local_payload_migration(
    blob_connection,
    tmp_path: Path,
) -> None:
    content = b"deduplicated payload"
    sha256 = hashlib.sha256(content).hexdigest()
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    (storage_root / "payload.bin").write_bytes(content)
    sqlite_payload = await ArtifactPayloadRepository(
        blob_connection, CLOCK
    ).create_payload_blob(
        payload_blob_id="apb_sqlite",
        user_id="usr_blob",
        storage_kind="sqlite_blob",
        identity_kind="content_sha256",
        content_sha256=sha256,
        byte_size=len(content),
        blob_bytes=content,
        storage_key=None,
        external_uri=None,
    )
    await _insert_legacy_payload_blob(
        blob_connection,
        payload_id="apb_local",
        storage_uri="payload.bin",
        content=content,
    )
    await _seed_artifact(
        blob_connection, "art_local_payload", payload_blob_id="apb_local"
    )

    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)
    artifact = await ArtifactRepository(blob_connection, CLOCK).get_artifact(
        "art_local_payload",
        "usr_blob",
    )
    assert artifact["payload_blob_id"] == sqlite_payload["id"]
    cursor = await blob_connection.execute(
        "SELECT COUNT(*) AS count FROM artifact_payload_blobs WHERE id = 'apb_local'"
    )
    assert (await cursor.fetchone())["count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_status",
    ("pending", "gc_pending", "quarantined", "deleted"),
)
async def test_ready_local_payload_migration_preserves_a_readable_ready_survivor(
    blob_connection,
    tmp_path: Path,
    existing_status: str,
) -> None:
    content = f"status-aware payload {existing_status}".encode()
    sha256 = hashlib.sha256(content).hexdigest()
    storage_root = tmp_path / existing_status
    storage_root.mkdir()
    (storage_root / "payload.bin").write_bytes(content)
    repository = ArtifactPayloadRepository(blob_connection, CLOCK)
    await repository.create_payload_blob(
        payload_blob_id=f"apb_existing_{existing_status}",
        user_id="usr_blob",
        storage_kind="sqlite_blob",
        identity_kind="content_sha256",
        content_sha256=sha256,
        byte_size=len(content),
        blob_bytes=content,
        storage_key=None,
        external_uri=None,
        status=existing_status,
    )
    await _insert_legacy_payload_blob(
        blob_connection,
        payload_id=f"apb_local_{existing_status}",
        storage_uri="payload.bin",
        content=content,
        status="ready",
    )
    artifact_id = f"art_{existing_status}"
    await _seed_artifact(
        blob_connection,
        artifact_id,
        payload_blob_id=f"apb_local_{existing_status}",
    )

    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)

    payload = await repository.get_payload_for_artifact(artifact_id, "usr_blob")
    assert payload is not None
    assert payload["status"] == "ready"
    assert bytes(payload["blob_bytes"]) == content
    assert payload["content_sha256"] == sha256
    artifact = await ArtifactRepository(blob_connection, CLOCK).get_artifact(
        artifact_id,
        "usr_blob",
    )
    if existing_status in {"pending", "gc_pending"}:
        assert artifact["payload_blob_id"] == f"apb_existing_{existing_status}"
    else:
        assert artifact["payload_blob_id"] == f"apb_local_{existing_status}"


@pytest.mark.asyncio
async def test_payload_deduplication_rejects_mismatched_existing_sqlite_bytes(
    blob_connection,
    tmp_path: Path,
) -> None:
    content = b"verified local payload"
    sha256 = hashlib.sha256(content).hexdigest()
    storage_root = tmp_path / "mismatch"
    storage_root.mkdir()
    (storage_root / "payload.bin").write_bytes(content)
    await ArtifactPayloadRepository(blob_connection, CLOCK).create_payload_blob(
        payload_blob_id="apb_bad_existing",
        user_id="usr_blob",
        storage_kind="sqlite_blob",
        identity_kind="content_sha256",
        content_sha256=sha256,
        byte_size=len(content),
        blob_bytes=b"x" * len(content),
        storage_key=None,
        external_uri=None,
        status="ready",
    )
    await _insert_legacy_payload_blob(
        blob_connection,
        payload_id="apb_verified_local",
        storage_uri="payload.bin",
        content=content,
    )
    await _seed_artifact(
        blob_connection,
        "art_verified_local",
        payload_blob_id="apb_verified_local",
    )

    with pytest.raises(ValueError, match="do not match"):
        await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)

    cursor = await blob_connection.execute(
        "SELECT storage_kind FROM artifact_payload_blobs WHERE id = 'apb_verified_local'"
    )
    assert (await cursor.fetchone())["storage_kind"] == "local_file"
    artifact = await ArtifactRepository(blob_connection, CLOCK).get_artifact(
        "art_verified_local",
        "usr_blob",
    )
    assert artifact["payload_blob_id"] == "apb_verified_local"


@pytest.mark.asyncio
async def test_pending_deletion_only_becomes_cleanup_intent_and_missing_file_drains(
    blob_connection,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "legacy_root"
    storage_root.mkdir()
    await blob_connection.execute(
        """
        INSERT INTO pending_file_deletions(
            id, storage_uri, storage_root, sha256, reason, tombstone_id, created_at
        )
        VALUES ('pfd_orphan', 'already-gone.bin', ?, ?, 'user_erasure', 'tmb_old', ?)
        """,
        (
            str(storage_root),
            hashlib.sha256(b"gone").hexdigest(),
            CLOCK.now().isoformat(),
        ),
    )
    await blob_connection.commit()
    result = await migrate_legacy_artifact_blobs(
        blob_connection,
        storage_root=tmp_path / "configured_root",
    )
    assert result.cleanup_intent_count == 1
    cleanup = await drain_artifact_blob_cleanup_intents(
        blob_connection,
        storage_root=tmp_path / "configured_root",
    )
    assert cleanup.remaining_intent_count == 0
    cursor = await blob_connection.execute(
        "SELECT deleted_at FROM pending_file_deletions WHERE id = 'pfd_orphan'"
    )
    assert (await cursor.fetchone())["deleted_at"] is not None


@pytest.mark.asyncio
async def test_startup_refusal_precedes_retired_configuration_rejection(
    blob_connection,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    content = b"legacy"
    (storage_root / "payload.bin").write_bytes(content)
    await _seed_artifact(blob_connection, "art_legacy")
    await _insert_legacy_artifact_blob(
        blob_connection,
        artifact_id="art_legacy",
        storage_uri="payload.bin",
        content=content,
    )
    with pytest.raises(
        LegacyArtifactBlobStateError, match="offline migration.*references=1"
    ):
        await assert_artifact_blob_runtime_ready(
            blob_connection,
            configured_storage_kind="local_file",
        )

    await migrate_legacy_artifact_blobs(blob_connection, storage_root=storage_root)
    await drain_artifact_blob_cleanup_intents(
        blob_connection, storage_root=storage_root
    )
    with pytest.raises(LegacyArtifactBlobStateError, match="local_file.*retired"):
        await assert_artifact_blob_runtime_ready(
            blob_connection,
            configured_storage_kind="local_file",
        )
    await assert_artifact_blob_runtime_ready(
        blob_connection,
        configured_storage_kind="sqlite_blob",
    )


@pytest.mark.asyncio
async def test_supported_repositories_reject_new_local_file_rows(
    blob_connection,
) -> None:
    with pytest.raises(ValueError, match="local_file.*retired"):
        await ArtifactPayloadRepository(blob_connection, CLOCK).create_payload_blob(
            user_id="usr_blob",
            storage_kind="local_file",
            identity_kind="content_sha256",
            content_sha256=hashlib.sha256(b"payload").hexdigest(),
            byte_size=7,
            blob_bytes=None,
            storage_key="payload.bin",
            external_uri=None,
        )
    with pytest.raises(ValueError, match="local_file.*retired"):
        await ArtifactRepository(blob_connection, CLOCK).create_artifact(
            artifact_id="art_rejected",
            user_id="usr_blob",
            workspace_id=None,
            conversation_id=None,
            message_id=None,
            artifact_type="file",
            source_kind="host_embedded",
            storage_kind="local_file",
            storage_uri="payload.bin",
        )


@pytest.mark.asyncio
async def test_migration_and_cleanup_intent_survive_database_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "atagia.db"
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    content = b"durable migration state"
    path = storage_root / "payload.bin"
    path.write_bytes(content)
    connection = await initialize_database(str(database_path), MIGRATIONS_DIR)
    await UserRepository(connection, CLOCK).create_user("usr_blob")
    await _seed_artifact(connection, "art_reopen")
    await _insert_legacy_artifact_blob(
        connection,
        artifact_id="art_reopen",
        storage_uri="payload.bin",
        content=content,
    )
    await migrate_legacy_artifact_blobs(connection, storage_root=storage_root)
    await connection.close()

    reopened = await open_connection(str(database_path))
    try:
        cursor = await reopened.execute(
            "SELECT storage_kind, blob_bytes FROM artifact_blobs WHERE artifact_id = 'art_reopen'"
        )
        row = await cursor.fetchone()
        assert row["storage_kind"] == "sqlite_blob"
        assert bytes(row["blob_bytes"]) == content
        assert (
            await verify_artifact_blob_migration(reopened)
        ).cleanup_intent_count == 1
        await drain_artifact_blob_cleanup_intents(reopened, storage_root=storage_root)
        assert (await verify_artifact_blob_migration(reopened)).is_complete
        assert not path.exists()
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_offline_cli_runs_migration_to_completion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "atagia.db"
    storage_root = tmp_path / "artifact_blobs"
    storage_root.mkdir()
    content = b"CLI migration"
    (storage_root / "payload.bin").write_bytes(content)
    second_content = b"second CLI migration"
    (storage_root / "second.bin").write_bytes(second_content)
    connection = await initialize_database(str(database_path), MIGRATIONS_DIR)
    await UserRepository(connection, CLOCK).create_user("usr_blob")
    await _seed_artifact(connection, "art_cli")
    await _insert_legacy_artifact_blob(
        connection,
        artifact_id="art_cli",
        storage_uri="payload.bin",
        content=content,
    )
    await _seed_artifact(connection, "art_cli_second")
    await _insert_legacy_artifact_blob(
        connection,
        artifact_id="art_cli_second",
        storage_uri="second.bin",
        content=second_content,
    )
    await connection.close()

    exit_code = await migration_main_async(
        [
            "--sqlite-path",
            str(database_path),
            "--migrations-path",
            str(MIGRATIONS_DIR),
            "--artifact-blob-storage-path",
            str(storage_root),
            "--batch-size",
            "1",
            "run",
        ]
    )
    assert exit_code == 0
    assert '"is_complete": true' in capsys.readouterr().out
    assert not (storage_root / "payload.bin").exists()
    assert not (storage_root / "second.bin").exists()
