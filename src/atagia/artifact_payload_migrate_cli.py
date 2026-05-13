"""CLI for migrating legacy artifact_blobs rows to shared payload blobs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, initialize_database
from atagia.core.clock import SystemClock
from atagia.services.artifact_blob_store import ArtifactBlobStore
from atagia.services.artifact_payload_service import ArtifactPayloadService


@dataclass(frozen=True, slots=True)
class MigrationCliResult:
    """JSON-serializable migration command result."""

    mode: str
    legacy_active_artifacts_without_payload: int
    payload_blob_count: int
    migrated_artifacts: int = 0
    skipped_artifacts: int = 0
    error_count: int = 0
    missing_local_files: int = 0
    hash_mismatches: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atagia-artifact-payload-migrate",
        description="Audit, migrate, and verify artifact payload blob deduplication.",
    )
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--migrations-path", default=None)
    parser.add_argument("--artifact-blob-storage-path", default=None)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("audit")
    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--batch-size", type=int, default=500)
    migrate.add_argument(
        "--target-storage-kind",
        choices=("sqlite_blob", "local_file"),
        default=None,
    )
    subparsers.add_parser("verify")
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = Settings.from_env()
    sqlite_path = str(args.sqlite_path or settings.sqlite_path)
    migrations_path = Path(args.migrations_path or settings.migrations_dir())
    storage_path = Path(args.artifact_blob_storage_path or settings.artifact_blobs_dir())
    connection = await initialize_database(sqlite_path, migrations_path)
    try:
        if args.mode == "audit":
            result = await _audit(connection)
        elif args.mode == "migrate":
            result = await _migrate(
                connection,
                storage_path=storage_path,
                target_storage_kind=args.target_storage_kind or settings.artifact_blob_storage_kind,
                batch_size=args.batch_size,
            )
        else:
            result = await _verify(connection, storage_path=storage_path)
    finally:
        await close_connection(connection)
    print(json.dumps(asdict(result), sort_keys=True))
    if result.error_count or result.missing_local_files or result.hash_mismatches:
        return 1
    if args.mode == "verify" and result.legacy_active_artifacts_without_payload:
        return 1
    return 0


async def _audit(connection: aiosqlite.Connection) -> MigrationCliResult:
    return MigrationCliResult(
        mode="audit",
        legacy_active_artifacts_without_payload=await _count_legacy_active_artifacts_without_payload(connection),
        payload_blob_count=await _payload_blob_count(connection),
    )


async def _migrate(
    connection: aiosqlite.Connection,
    *,
    storage_path: Path,
    target_storage_kind: str,
    batch_size: int,
) -> MigrationCliResult:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    blob_store = ArtifactBlobStore(storage_path)
    payload_service = ArtifactPayloadService(connection, SystemClock(), blob_store=blob_store)
    cursor = await connection.execute(
        """
        SELECT
            a.id AS artifact_id,
            a.user_id,
            ab.storage_kind,
            ab.blob_bytes,
            ab.storage_uri,
            ab.byte_size,
            ab.sha256
        FROM artifacts AS a
        JOIN artifact_blobs AS ab ON ab.artifact_id = a.id
        WHERE a.payload_blob_id IS NULL
          AND a.status NOT IN ('deleted', 'purged')
        ORDER BY a.created_at ASC, a.id ASC
        LIMIT ?
        """,
        (batch_size,),
    )
    rows = await cursor.fetchall()
    migrated = 0
    skipped = 0
    errors = 0
    await connection.execute("BEGIN IMMEDIATE")
    try:
        for row in rows:
            try:
                blob = _migration_blob_for_row(
                    row,
                    blob_store=blob_store,
                    target_storage_kind=target_storage_kind,
                )
                payload = await payload_service.get_or_create_payload_blob(blob, user_id=str(row["user_id"]))
                if payload is None:
                    skipped += 1
                    continue
                await connection.execute(
                    """
                    UPDATE artifacts
                    SET payload_blob_id = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND user_id = ?
                      AND payload_blob_id IS NULL
                    """,
                    (
                        payload["id"],
                        SystemClock().now().isoformat(),
                        row["artifact_id"],
                        row["user_id"],
                    ),
                )
                migrated += 1
            except Exception:
                errors += 1
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return MigrationCliResult(
        mode="migrate",
        legacy_active_artifacts_without_payload=await _count_legacy_active_artifacts_without_payload(connection),
        payload_blob_count=await _payload_blob_count(connection),
        migrated_artifacts=migrated,
        skipped_artifacts=skipped,
        error_count=errors,
    )


async def _verify(connection: aiosqlite.Connection, *, storage_path: Path) -> MigrationCliResult:
    blob_store = ArtifactBlobStore(storage_path)
    missing = 0
    mismatches = 0
    cursor = await connection.execute(
        """
        SELECT storage_key, content_sha256
        FROM artifact_payload_blobs
        WHERE storage_kind = 'local_file'
          AND storage_key IS NOT NULL
          AND status IN ('pending', 'ready', 'gc_pending')
        ORDER BY id ASC
        """
    )
    for row in await cursor.fetchall():
        try:
            content = blob_store.read_bytes(str(row["storage_key"]))
        except FileNotFoundError:
            missing += 1
            continue
        if _sha256(content) != str(row["content_sha256"]):
            mismatches += 1
    return MigrationCliResult(
        mode="verify",
        legacy_active_artifacts_without_payload=await _count_legacy_active_artifacts_without_payload(connection),
        payload_blob_count=await _payload_blob_count(connection),
        missing_local_files=missing,
        hash_mismatches=mismatches,
    )


def _migration_blob_for_row(
    row: aiosqlite.Row,
    *,
    blob_store: ArtifactBlobStore,
    target_storage_kind: str,
) -> dict[str, Any]:
    storage_kind = str(row["storage_kind"])
    if storage_kind == "external_ref":
        return {
            "storage_kind": "external_ref",
            "storage_uri": row["storage_uri"],
            "byte_size": int(row["byte_size"] or 0),
            "sha256": row["sha256"],
        }
    if storage_kind == "sqlite_blob":
        blob_bytes = row["blob_bytes"]
        if blob_bytes is None:
            raise ValueError("Legacy SQLite artifact blob is missing bytes")
        if target_storage_kind == "local_file":
            return {
                "storage_kind": "local_file",
                "blob_bytes": bytes(blob_bytes),
                "storage_uri": None,
                "byte_size": int(row["byte_size"]),
                "sha256": row["sha256"],
            }
        return {
            "storage_kind": "sqlite_blob",
            "blob_bytes": bytes(blob_bytes),
            "storage_uri": None,
            "byte_size": int(row["byte_size"]),
            "sha256": row["sha256"],
        }
    if storage_kind != "local_file":
        raise ValueError(f"Unsupported legacy artifact blob storage kind: {storage_kind}")
    content = blob_store.read_bytes(str(row["storage_uri"]))
    if target_storage_kind == "sqlite_blob":
        return {
            "storage_kind": "sqlite_blob",
            "blob_bytes": content,
            "storage_uri": None,
            "byte_size": len(content),
            "sha256": row["sha256"],
        }
    return {
        "storage_kind": "local_file",
        "blob_bytes": content,
        "storage_uri": None,
        "byte_size": len(content),
        "sha256": row["sha256"],
    }


async def _count_legacy_active_artifacts_without_payload(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM artifacts AS a
        JOIN artifact_blobs AS ab ON ab.artifact_id = a.id
        WHERE a.payload_blob_id IS NULL
          AND a.status NOT IN ('deleted', 'purged')
        """
    )
    row = await cursor.fetchone()
    return int(row["count"])


async def _payload_blob_count(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute("SELECT COUNT(*) AS count FROM artifact_payload_blobs")
    row = await cursor.fetchone()
    return int(row["count"])


def _sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))
