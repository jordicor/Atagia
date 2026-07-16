"""Offline migration from the retired local-file artifact blob backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass(frozen=True, slots=True)
class BlobReferenceSpec:
    """One table participating in the global artifact-blob reference map."""

    table: str
    id_column: str
    uri_column: str
    sha256_column: str
    byte_size_column: str
    blob_column: str
    updated_at_column: str


# Adding another table that can reference local artifact bytes requires adding it
# here. Inventory, migration, startup refusal, and final cleanup checks all use
# this single map.
ARTIFACT_BLOB_REFERENCE_SCHEMA: tuple[BlobReferenceSpec, ...] = (
    BlobReferenceSpec(
        table="artifact_blobs",
        id_column="artifact_id",
        uri_column="storage_uri",
        sha256_column="sha256",
        byte_size_column="byte_size",
        blob_column="blob_bytes",
        updated_at_column="updated_at",
    ),
    BlobReferenceSpec(
        table="artifact_payload_blobs",
        id_column="id",
        uri_column="storage_key",
        sha256_column="content_sha256",
        byte_size_column="byte_size",
        blob_column="blob_bytes",
        updated_at_column="updated_at",
    ),
)


MigrationFailpoint = Callable[[str, str], None]


@dataclass(frozen=True, slots=True)
class LocalBlobReference:
    table: str
    row_id: str
    storage_uri: str
    storage_identity: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class PendingFileDeletion:
    row_id: str
    storage_uri: str
    storage_root: str
    storage_identity: str
    sha256: str | None


@dataclass(slots=True)
class StorageIdentityInventory:
    storage_identity: str
    storage_root: str
    storage_uri: str
    references: list[LocalBlobReference] = field(default_factory=list)
    pending_deletions: list[PendingFileDeletion] = field(default_factory=list)
    expected_sha256: str | None = None
    expected_byte_size: int | None = None
    content_bytes: bytes | None = field(default=None, repr=False)
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ArtifactBlobInventory:
    identities: tuple[StorageIdentityInventory, ...]
    unresolved_issues: tuple[str, ...]
    unresolved_local_reference_count: int
    unresolved_pending_deletion_count: int
    cleanup_intent_count: int

    @property
    def local_reference_count(self) -> int:
        return self.unresolved_local_reference_count + sum(
            len(identity.references) for identity in self.identities
        )

    @property
    def pending_deletion_count(self) -> int:
        return self.unresolved_pending_deletion_count + sum(
            len(identity.pending_deletions) for identity in self.identities
        )

    @property
    def issue_count(self) -> int:
        return len(self.unresolved_issues) + sum(
            len(identity.issues) for identity in self.identities
        )


@dataclass(frozen=True, slots=True)
class ArtifactBlobMigrationResult:
    migrated_reference_count: int
    cleanup_intent_count: int
    skipped_identity_count: int
    remaining_local_reference_count: int
    issue_count: int


@dataclass(frozen=True, slots=True)
class ArtifactBlobCleanupResult:
    processed_intent_count: int
    deleted_file_count: int
    deferred_intent_count: int
    remaining_intent_count: int


@dataclass(frozen=True, slots=True)
class ArtifactBlobVerification:
    local_reference_count: int
    pending_deletion_count: int
    cleanup_intent_count: int
    invalid_sqlite_blob_count: int

    @property
    def is_complete(self) -> bool:
        return (
            self.local_reference_count == 0
            and self.pending_deletion_count == 0
            and self.cleanup_intent_count == 0
            and self.invalid_sqlite_blob_count == 0
        )


class LegacyArtifactBlobStateError(RuntimeError):
    """Normal startup encountered state owned by the offline blob migrator."""


def resolve_legacy_storage_path(storage_uri: str, *, storage_root: str | Path) -> Path:
    """Resolve a legacy key beneath its root and reject symlink/path escapes."""
    root = Path(storage_root).expanduser().resolve(strict=False)
    raw_path = Path(storage_uri).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        lexical_candidate.relative_to(root)
    except ValueError:
        raise ValueError(
            "Artifact blob path escapes configured storage directory"
        ) from None
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            "Artifact blob path escapes configured storage directory through a symlink"
        ) from None
    return resolved


async def inventory_legacy_artifact_blobs(
    connection: aiosqlite.Connection,
    *,
    storage_root: str | Path,
) -> ArtifactBlobInventory:
    """Inventory and verify every declared local-file reference and delete row."""
    configured_root = str(Path(storage_root).expanduser().resolve(strict=False))
    groups: dict[str, StorageIdentityInventory] = {}
    unresolved: list[str] = []
    unresolved_local_references = 0
    unresolved_pending_deletions = 0

    for spec in ARTIFACT_BLOB_REFERENCE_SCHEMA:
        if not await _table_exists(connection, spec.table):
            continue
        cursor = await connection.execute(
            f"""
            SELECT
                {spec.id_column} AS row_id,
                {spec.uri_column} AS storage_uri,
                {spec.sha256_column} AS sha256,
                {spec.byte_size_column} AS byte_size
            FROM {spec.table}
            WHERE storage_kind = 'local_file'
            ORDER BY {spec.id_column} ASC
            """
        )
        for row in await cursor.fetchall():
            row_id = str(row["row_id"])
            storage_uri_value = row["storage_uri"]
            if storage_uri_value is None:
                unresolved.append(
                    f"{spec.table}:{row_id}: local_file row has no storage URI"
                )
                unresolved_local_references += 1
                continue
            storage_uri = str(storage_uri_value)
            try:
                path = resolve_legacy_storage_path(
                    storage_uri, storage_root=configured_root
                )
            except ValueError as exc:
                unresolved.append(f"{spec.table}:{row_id}: {exc}")
                unresolved_local_references += 1
                continue
            identity = str(path)
            group = groups.setdefault(
                identity,
                StorageIdentityInventory(
                    storage_identity=identity,
                    storage_root=configured_root,
                    storage_uri=storage_uri,
                ),
            )
            sha256_value = row["sha256"]
            if sha256_value is None:
                group.issues.append(f"{spec.table}:{row_id}: missing recorded sha256")
                sha256_value = ""
            byte_size_value = row["byte_size"]
            if byte_size_value is None or int(byte_size_value) < 0:
                group.issues.append(
                    f"{spec.table}:{row_id}: invalid recorded byte size"
                )
                byte_size_value = -1
            group.references.append(
                LocalBlobReference(
                    table=spec.table,
                    row_id=row_id,
                    storage_uri=storage_uri,
                    storage_identity=identity,
                    sha256=str(sha256_value),
                    byte_size=int(byte_size_value),
                )
            )

    if await _table_exists(connection, "pending_file_deletions"):
        cursor = await connection.execute(
            """
            SELECT id, storage_uri, storage_root, sha256
            FROM pending_file_deletions
            WHERE deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            """
        )
        for row in await cursor.fetchall():
            row_id = str(row["id"])
            row_root = str(row["storage_root"] or "").strip()
            storage_uri = str(row["storage_uri"] or "").strip()
            if not row_root or not storage_uri:
                unresolved.append(
                    f"pending_file_deletions:{row_id}: missing storage root or URI"
                )
                unresolved_pending_deletions += 1
                continue
            try:
                path = resolve_legacy_storage_path(storage_uri, storage_root=row_root)
            except ValueError as exc:
                unresolved.append(f"pending_file_deletions:{row_id}: {exc}")
                unresolved_pending_deletions += 1
                continue
            identity = str(path)
            group = groups.setdefault(
                identity,
                StorageIdentityInventory(
                    storage_identity=identity,
                    storage_root=str(Path(row_root).expanduser().resolve(strict=False)),
                    storage_uri=storage_uri,
                ),
            )
            group.pending_deletions.append(
                PendingFileDeletion(
                    row_id=row_id,
                    storage_uri=storage_uri,
                    storage_root=row_root,
                    storage_identity=identity,
                    sha256=str(row["sha256"]) if row["sha256"] is not None else None,
                )
            )

    for group in groups.values():
        _verify_inventory_group(group)

    cleanup_intent_count = await _count_rows(
        connection, "artifact_blob_cleanup_intents"
    )
    return ArtifactBlobInventory(
        identities=tuple(
            sorted(groups.values(), key=lambda item: item.storage_identity)
        ),
        unresolved_issues=tuple(unresolved),
        unresolved_local_reference_count=unresolved_local_references,
        unresolved_pending_deletion_count=unresolved_pending_deletions,
        cleanup_intent_count=cleanup_intent_count,
    )


def _verify_inventory_group(group: StorageIdentityInventory) -> None:
    reference_hashes = {
        reference.sha256 for reference in group.references if reference.sha256
    }
    pending_hashes = {
        deletion.sha256
        for deletion in group.pending_deletions
        if deletion.sha256 is not None
    }
    hashes = reference_hashes | pending_hashes
    sizes = {
        reference.byte_size
        for reference in group.references
        if reference.byte_size >= 0
    }
    if len(hashes) > 1:
        group.issues.append("references for this storage identity disagree on sha256")
    if len(sizes) > 1:
        group.issues.append(
            "references for this storage identity disagree on byte size"
        )
    group.expected_sha256 = next(iter(hashes), None)
    group.expected_byte_size = next(iter(sizes), None)
    path = Path(group.storage_identity)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        group.issues.append("referenced local artifact file is missing")
        return
    except OSError as exc:
        group.issues.append(f"failed to read referenced local artifact file: {exc}")
        return
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if group.expected_sha256 is not None and actual_sha256 != group.expected_sha256:
        group.issues.append(
            "referenced local artifact file sha256 does not match recorded metadata"
        )
    if (
        group.expected_byte_size is not None
        and len(content) != group.expected_byte_size
    ):
        group.issues.append(
            "referenced local artifact file size does not match recorded metadata"
        )
    if not group.issues:
        group.content_bytes = content
        group.expected_sha256 = actual_sha256
        group.expected_byte_size = len(content)


async def migrate_legacy_artifact_blobs(
    connection: aiosqlite.Connection,
    *,
    storage_root: str | Path,
    batch_size: int = 500,
    failpoint: MigrationFailpoint | None = None,
) -> ArtifactBlobMigrationResult:
    """Migrate at most ``batch_size`` references, committing each row safely."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inventory = await inventory_legacy_artifact_blobs(
        connection, storage_root=storage_root
    )
    migrated = 0
    skipped = len(inventory.unresolved_issues)
    initial_intents = inventory.cleanup_intent_count
    for group in inventory.identities:
        if group.references:
            if group.issues or group.content_bytes is None:
                skipped += 1
                continue
            for reference in group.references:
                if migrated >= batch_size:
                    break
                changed = await _migrate_one_reference(
                    connection,
                    reference=reference,
                    group=group,
                    configured_storage_root=storage_root,
                    failpoint=failpoint,
                )
                migrated += int(changed)
            if migrated >= batch_size:
                break
        elif group.pending_deletions:
            non_missing_issues = [
                issue
                for issue in group.issues
                if issue != "referenced local artifact file is missing"
            ]
            if non_missing_issues:
                skipped += 1
                continue
            await _create_cleanup_intent_if_unreferenced(
                connection,
                group=group,
                configured_storage_root=storage_root,
            )

    final_intent_count = await _count_rows(connection, "artifact_blob_cleanup_intents")
    return ArtifactBlobMigrationResult(
        migrated_reference_count=migrated,
        cleanup_intent_count=max(0, final_intent_count - initial_intents),
        skipped_identity_count=skipped,
        remaining_local_reference_count=await _count_local_file_references(connection),
        issue_count=inventory.issue_count,
    )


async def _migrate_one_reference(
    connection: aiosqlite.Connection,
    *,
    reference: LocalBlobReference,
    group: StorageIdentityInventory,
    configured_storage_root: str | Path,
    failpoint: MigrationFailpoint | None,
) -> bool:
    spec = _reference_spec(reference.table)
    await connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = await connection.execute(
            f"""
            SELECT *
            FROM {spec.table}
            WHERE {spec.id_column} = ?
              AND storage_kind = 'local_file'
            """,
            (reference.row_id,),
        )
        current = await cursor.fetchone()
        if current is None:
            await connection.rollback()
            return False
        current_uri = current[spec.uri_column]
        if current_uri is None:
            raise ValueError(
                f"{spec.table}:{reference.row_id}: local_file row has no storage URI"
            )
        current_path = resolve_legacy_storage_path(
            str(current_uri),
            storage_root=configured_storage_root,
        )
        if str(current_path) != group.storage_identity:
            raise ValueError(
                f"{spec.table}:{reference.row_id}: storage identity changed during migration"
            )
        if str(current[spec.sha256_column]) != group.expected_sha256:
            raise ValueError(
                f"{spec.table}:{reference.row_id}: sha256 changed during migration"
            )
        if int(current[spec.byte_size_column]) != group.expected_byte_size:
            raise ValueError(
                f"{spec.table}:{reference.row_id}: byte size changed during migration"
            )
        _call_failpoint(failpoint, "before_blob_write", group.storage_identity)
        if spec.table == "artifact_payload_blobs":
            await _migrate_payload_blob_row(
                connection,
                row=current,
                content_bytes=group.content_bytes or b"",
            )
        else:
            await connection.execute(
                f"""
                UPDATE {spec.table}
                SET storage_kind = 'sqlite_blob',
                    {spec.blob_column} = ?,
                    {spec.uri_column} = NULL,
                    {spec.updated_at_column} = ?
                WHERE {spec.id_column} = ?
                  AND storage_kind = 'local_file'
                """,
                (group.content_bytes, _timestamp(), reference.row_id),
            )
        _call_failpoint(failpoint, "before_row_commit", group.storage_identity)
        remaining = await _references_for_identity(
            connection,
            storage_identity=group.storage_identity,
            configured_storage_root=configured_storage_root,
        )
        if not remaining:
            await _insert_cleanup_intent(connection, group)
        await connection.commit()
    except BaseException:
        await connection.rollback()
        raise
    _call_failpoint(failpoint, "after_row_commit", group.storage_identity)
    return True


async def _migrate_payload_blob_row(
    connection: aiosqlite.Connection,
    *,
    row: aiosqlite.Row,
    content_bytes: bytes,
) -> None:
    current_id = str(row["id"])
    current_status = str(row["status"])
    existing: aiosqlite.Row | None = None
    if current_status in {"pending", "ready", "gc_pending"}:
        cursor = await connection.execute(
            """
            SELECT id, status, blob_bytes
            FROM artifact_payload_blobs
            WHERE id <> ?
              AND user_id = ?
              AND storage_kind = 'sqlite_blob'
              AND identity_kind = 'content_sha256'
              AND content_sha256 = ?
              AND byte_size = ?
              AND status IN ('pending', 'ready', 'gc_pending')
            ORDER BY
                CASE status
                    WHEN 'ready' THEN 0
                    WHEN 'pending' THEN 1
                    ELSE 2
                END,
                created_at ASC,
                id ASC
            LIMIT 1
            """,
            (current_id, row["user_id"], row["content_sha256"], row["byte_size"]),
        )
        existing = await cursor.fetchone()
    if existing is not None:
        existing_bytes = bytes(existing["blob_bytes"])
        expected_sha256 = str(row["content_sha256"])
        if (
            len(existing_bytes) != int(row["byte_size"])
            or hashlib.sha256(existing_bytes).hexdigest() != expected_sha256
            or existing_bytes != content_bytes
        ):
            raise ValueError(
                "Existing SQLite artifact payload bytes do not match the "
                "local payload selected for deduplication"
            )
        existing_id = str(existing["id"])
        status_rank = {"gc_pending": 0, "pending": 1, "ready": 2}
        existing_status = str(existing["status"])
        if status_rank[current_status] > status_rank[existing_status]:
            await connection.execute(
                """
                UPDATE artifact_payload_blobs
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (current_status, _timestamp(), existing_id, existing_status),
            )
        await connection.execute(
            """
            UPDATE artifacts
            SET payload_blob_id = ?,
                updated_at = ?
            WHERE user_id = ?
              AND payload_blob_id = ?
            """,
            (existing_id, _timestamp(), row["user_id"], current_id),
        )
        await connection.execute(
            "DELETE FROM artifact_payload_blobs WHERE id = ? AND storage_kind = 'local_file'",
            (current_id,),
        )
        return
    await connection.execute(
        """
        UPDATE artifact_payload_blobs
        SET storage_kind = 'sqlite_blob',
            blob_bytes = ?,
            storage_key = NULL,
            external_uri = NULL,
            updated_at = ?
        WHERE id = ?
          AND storage_kind = 'local_file'
        """,
        (content_bytes, _timestamp(), current_id),
    )


async def _create_cleanup_intent_if_unreferenced(
    connection: aiosqlite.Connection,
    *,
    group: StorageIdentityInventory,
    configured_storage_root: str | Path,
) -> bool:
    await connection.execute("BEGIN IMMEDIATE")
    try:
        remaining = await _references_for_identity(
            connection,
            storage_identity=group.storage_identity,
            configured_storage_root=configured_storage_root,
        )
        if remaining:
            await connection.rollback()
            return False
        await _insert_cleanup_intent(connection, group)
        await connection.commit()
        return True
    except BaseException:
        await connection.rollback()
        raise


async def _insert_cleanup_intent(
    connection: aiosqlite.Connection,
    group: StorageIdentityInventory,
) -> None:
    cursor = await connection.execute(
        """
        SELECT expected_sha256, expected_byte_size
        FROM artifact_blob_cleanup_intents
        WHERE storage_identity = ?
        """,
        (group.storage_identity,),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        if (
            existing["expected_sha256"] != group.expected_sha256
            or existing["expected_byte_size"] != group.expected_byte_size
        ):
            raise ValueError(
                "Existing cleanup intent metadata disagrees for storage identity"
            )
        return
    timestamp = _timestamp()
    await connection.execute(
        """
        INSERT INTO artifact_blob_cleanup_intents(
            storage_identity,
            storage_root,
            storage_uri,
            expected_sha256,
            expected_byte_size,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            group.storage_identity,
            group.storage_root,
            group.storage_uri,
            group.expected_sha256,
            group.expected_byte_size,
            timestamp,
            timestamp,
        ),
    )


async def drain_artifact_blob_cleanup_intents(
    connection: aiosqlite.Connection,
    *,
    storage_root: str | Path,
    limit: int = 100,
    failpoint: MigrationFailpoint | None = None,
) -> ArtifactBlobCleanupResult:
    """Globally recheck and then unlink obsolete local files."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    cursor = await connection.execute(
        """
        SELECT *
        FROM artifact_blob_cleanup_intents
        ORDER BY
            CASE WHEN last_attempt_at IS NULL THEN 0 ELSE 1 END ASC,
            last_attempt_at ASC,
            created_at ASC,
            storage_identity ASC
        LIMIT ?
        """,
        (limit,),
    )
    intents = await cursor.fetchall()
    deleted = 0
    deferred = 0
    for intent in intents:
        storage_identity = str(intent["storage_identity"])
        pending_ids: list[str] = []
        await connection.execute("BEGIN IMMEDIATE")
        try:
            remaining = await _references_for_identity(
                connection,
                storage_identity=storage_identity,
                configured_storage_root=storage_root,
            )
            if remaining:
                await _defer_cleanup_intent(
                    connection,
                    storage_identity,
                    "local_file references still exist in the declared blob schema",
                )
                await connection.commit()
                deferred += 1
                continue
            pending, pending_error = await _pending_deletions_for_identity(
                connection,
                storage_identity=storage_identity,
                expected_sha256=intent["expected_sha256"],
            )
            if pending_error is not None:
                await _defer_cleanup_intent(connection, storage_identity, pending_error)
                await connection.commit()
                deferred += 1
                continue
            pending_ids = [row.row_id for row in pending]
            path = resolve_legacy_storage_path(
                str(intent["storage_uri"]),
                storage_root=str(intent["storage_root"]),
            )
            if str(path) != storage_identity:
                await _defer_cleanup_intent(
                    connection,
                    storage_identity,
                    "cleanup intent no longer resolves to its recorded storage identity",
                )
                await connection.commit()
                deferred += 1
                continue
            verification_error = _cleanup_file_verification_error(
                path,
                expected_sha256=intent["expected_sha256"],
                expected_byte_size=intent["expected_byte_size"],
            )
            if verification_error is not None:
                await _defer_cleanup_intent(
                    connection, storage_identity, verification_error
                )
                await connection.commit()
                deferred += 1
                continue
            timestamp = _timestamp()
            await connection.execute(
                """
                UPDATE artifact_blob_cleanup_intents
                SET attempt_count = attempt_count + 1,
                    verified_at = ?,
                    last_attempt_at = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE storage_identity = ?
                """,
                (timestamp, timestamp, timestamp, storage_identity),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

        try:
            _call_failpoint(failpoint, "before_unlink", storage_identity)
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            await _record_cleanup_failure(connection, storage_identity, str(exc))
            deferred += 1
            continue

        await connection.execute("BEGIN IMMEDIATE")
        try:
            if pending_ids:
                placeholders = ", ".join("?" for _ in pending_ids)
                timestamp = _timestamp()
                await connection.execute(
                    f"""
                    UPDATE pending_file_deletions
                    SET attempted_at = ?,
                        deleted_at = ?,
                        last_error = NULL
                    WHERE id IN ({placeholders})
                      AND deleted_at IS NULL
                    """,
                    (timestamp, timestamp, *pending_ids),
                )
            await connection.execute(
                "DELETE FROM artifact_blob_cleanup_intents WHERE storage_identity = ?",
                (storage_identity,),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    remaining_intents = await _count_rows(connection, "artifact_blob_cleanup_intents")
    return ArtifactBlobCleanupResult(
        processed_intent_count=len(intents),
        deleted_file_count=deleted,
        deferred_intent_count=deferred,
        remaining_intent_count=remaining_intents,
    )


async def _pending_deletions_for_identity(
    connection: aiosqlite.Connection,
    *,
    storage_identity: str,
    expected_sha256: Any,
) -> tuple[list[PendingFileDeletion], str | None]:
    if not await _table_exists(connection, "pending_file_deletions"):
        return [], None
    cursor = await connection.execute(
        """
        SELECT id, storage_uri, storage_root, sha256
        FROM pending_file_deletions
        WHERE deleted_at IS NULL
        ORDER BY created_at ASC, id ASC
        """
    )
    matches: list[PendingFileDeletion] = []
    for row in await cursor.fetchall():
        try:
            resolved = resolve_legacy_storage_path(
                str(row["storage_uri"]),
                storage_root=str(row["storage_root"]),
            )
        except ValueError:
            continue
        if str(resolved) != storage_identity:
            continue
        row_sha256 = str(row["sha256"]) if row["sha256"] is not None else None
        if expected_sha256 is not None and row_sha256 not in {
            None,
            str(expected_sha256),
        }:
            return [], "pending deletion metadata disagrees with cleanup intent sha256"
        matches.append(
            PendingFileDeletion(
                row_id=str(row["id"]),
                storage_uri=str(row["storage_uri"]),
                storage_root=str(row["storage_root"]),
                storage_identity=storage_identity,
                sha256=row_sha256,
            )
        )
    return matches, None


def _cleanup_file_verification_error(
    path: Path,
    *,
    expected_sha256: Any,
    expected_byte_size: Any,
) -> str | None:
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"failed to read file before cleanup: {exc}"
    if expected_sha256 is None:
        return "cleanup intent has no sha256 for an existing file; refusing cleanup"
    if expected_byte_size is not None and len(content) != int(expected_byte_size):
        return "file size changed after migration; refusing cleanup"
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != str(
        expected_sha256
    ):
        return "file sha256 changed after migration; refusing cleanup"
    return None


async def _defer_cleanup_intent(
    connection: aiosqlite.Connection,
    storage_identity: str,
    error: str,
) -> None:
    timestamp = _timestamp()
    await connection.execute(
        """
        UPDATE artifact_blob_cleanup_intents
        SET attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_error = ?,
            updated_at = ?
        WHERE storage_identity = ?
        """,
        (timestamp, error, timestamp, storage_identity),
    )


async def _record_cleanup_failure(
    connection: aiosqlite.Connection,
    storage_identity: str,
    error: str,
) -> None:
    await connection.execute("BEGIN IMMEDIATE")
    try:
        timestamp = _timestamp()
        await connection.execute(
            """
            UPDATE artifact_blob_cleanup_intents
            SET last_attempt_at = ?,
                last_error = ?,
                updated_at = ?
            WHERE storage_identity = ?
            """,
            (timestamp, error, timestamp, storage_identity),
        )
        await connection.commit()
    except BaseException:
        await connection.rollback()
        raise


async def _references_for_identity(
    connection: aiosqlite.Connection,
    *,
    storage_identity: str,
    configured_storage_root: str | Path,
) -> list[LocalBlobReference]:
    matches: list[LocalBlobReference] = []
    for spec in ARTIFACT_BLOB_REFERENCE_SCHEMA:
        if not await _table_exists(connection, spec.table):
            continue
        cursor = await connection.execute(
            f"""
            SELECT
                {spec.id_column} AS row_id,
                {spec.uri_column} AS storage_uri,
                {spec.sha256_column} AS sha256,
                {spec.byte_size_column} AS byte_size
            FROM {spec.table}
            WHERE storage_kind = 'local_file'
            ORDER BY {spec.id_column} ASC
            """
        )
        for row in await cursor.fetchall():
            if row["storage_uri"] is None:
                continue
            try:
                path = resolve_legacy_storage_path(
                    str(row["storage_uri"]),
                    storage_root=configured_storage_root,
                )
            except ValueError:
                continue
            if str(path) != storage_identity:
                continue
            matches.append(
                LocalBlobReference(
                    table=spec.table,
                    row_id=str(row["row_id"]),
                    storage_uri=str(row["storage_uri"]),
                    storage_identity=storage_identity,
                    sha256=str(row["sha256"] or ""),
                    byte_size=int(row["byte_size"] or 0),
                )
            )
    return matches


async def verify_artifact_blob_migration(
    connection: aiosqlite.Connection,
) -> ArtifactBlobVerification:
    """Verify that no local state remains and every SQLite blob is coherent."""
    local_references = 0
    invalid_sqlite = 0
    for spec in ARTIFACT_BLOB_REFERENCE_SCHEMA:
        if not await _table_exists(connection, spec.table):
            continue
        local_references += await _count_where(
            connection,
            spec.table,
            "storage_kind = 'local_file'",
        )
        cursor = await connection.execute(
            f"""
            SELECT
                {spec.id_column} AS row_id,
                {spec.blob_column} AS blob_bytes,
                {spec.sha256_column} AS sha256,
                {spec.byte_size_column} AS byte_size
            FROM {spec.table}
            WHERE storage_kind = 'sqlite_blob'
            """
        )
        for row in await cursor.fetchall():
            blob_bytes = row["blob_bytes"]
            if blob_bytes is None:
                invalid_sqlite += 1
                continue
            content = bytes(blob_bytes)
            if len(content) != int(row["byte_size"]):
                invalid_sqlite += 1
                continue
            if hashlib.sha256(content).hexdigest() != str(row["sha256"]):
                invalid_sqlite += 1
    pending = await _count_where(
        connection,
        "pending_file_deletions",
        "deleted_at IS NULL",
    )
    intents = await _count_rows(connection, "artifact_blob_cleanup_intents")
    return ArtifactBlobVerification(
        local_reference_count=local_references,
        pending_deletion_count=pending,
        cleanup_intent_count=intents,
        invalid_sqlite_blob_count=invalid_sqlite,
    )


async def _count_local_file_references(connection: aiosqlite.Connection) -> int:
    count = 0
    for spec in ARTIFACT_BLOB_REFERENCE_SCHEMA:
        count += await _count_where(
            connection,
            spec.table,
            "storage_kind = 'local_file'",
        )
    return count


async def assert_artifact_blob_runtime_ready(
    connection: aiosqlite.Connection,
    *,
    configured_storage_kind: str,
) -> None:
    """Refuse runtime startup until the retired backend is fully drained."""
    local_reference_count = await _count_local_file_references(connection)
    pending_deletion_count = await _count_where(
        connection,
        "pending_file_deletions",
        "deleted_at IS NULL",
    )
    cleanup_intent_count = await _count_rows(
        connection, "artifact_blob_cleanup_intents"
    )
    if local_reference_count or pending_deletion_count or cleanup_intent_count:
        raise LegacyArtifactBlobStateError(
            "Legacy local_file artifact state requires offline migration "
            f"(references={local_reference_count}, "
            f"pending_deletions={pending_deletion_count}, "
            f"cleanup_intents={cleanup_intent_count}). Stop Atagia writers and GC, then run "
            "`atagia-artifact-blob-migrate run` with the production SQLite path and legacy blob root."
        )
    if configured_storage_kind != "sqlite_blob":
        raise LegacyArtifactBlobStateError(
            "artifact_blob_storage_kind='local_file' is retired. Set "
            "ATAGIA_ARTIFACT_BLOB_STORAGE_KIND=sqlite_blob."
        )


async def _table_exists(connection: aiosqlite.Connection, table: str) -> bool:
    cursor = await connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return await cursor.fetchone() is not None


async def _count_rows(connection: aiosqlite.Connection, table: str) -> int:
    if not await _table_exists(connection, table):
        return 0
    cursor = await connection.execute(f"SELECT COUNT(*) AS count FROM {table}")
    row = await cursor.fetchone()
    return int(row["count"])


async def _count_where(
    connection: aiosqlite.Connection,
    table: str,
    where_clause: str,
) -> int:
    if not await _table_exists(connection, table):
        return 0
    cursor = await connection.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {where_clause}"
    )
    row = await cursor.fetchone()
    return int(row["count"])


def _reference_spec(table: str) -> BlobReferenceSpec:
    for spec in ARTIFACT_BLOB_REFERENCE_SCHEMA:
        if spec.table == table:
            return spec
    raise ValueError(f"Unknown artifact blob reference table: {table}")


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _call_failpoint(
    failpoint: MigrationFailpoint | None,
    stage: str,
    storage_identity: str,
) -> None:
    if failpoint is not None:
        failpoint(stage, storage_identity)
