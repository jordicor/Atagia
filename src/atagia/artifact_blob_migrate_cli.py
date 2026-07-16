"""Offline CLI for retiring legacy local-file artifact blobs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, initialize_database
from atagia.services.artifact_blob_migration import (
    drain_artifact_blob_cleanup_intents,
    inventory_legacy_artifact_blobs,
    migrate_legacy_artifact_blobs,
    verify_artifact_blob_migration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atagia-artifact-blob-migrate",
        description=(
            "Offline inventory, migration, cleanup, and verification for the retired "
            "local_file artifact backend. Stop Atagia writers and GC before running it."
        ),
    )
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--migrations-path", default=None)
    parser.add_argument("--artifact-blob-storage-path", default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("migrate")
    subparsers.add_parser("cleanup")
    subparsers.add_parser("verify")
    subparsers.add_parser("run")
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    settings = Settings.from_env()
    sqlite_path = str(args.sqlite_path or settings.sqlite_path)
    migrations_path = Path(args.migrations_path or settings.migrations_dir())
    storage_path = Path(
        args.artifact_blob_storage_path or settings.artifact_blobs_dir()
    )
    connection = await initialize_database(sqlite_path, migrations_path)
    try:
        payload, success = await _execute_mode(
            connection,
            mode=str(args.mode),
            storage_path=storage_path,
            batch_size=int(args.batch_size),
        )
    finally:
        await close_connection(connection)
    print(json.dumps(payload, sort_keys=True))
    return 0 if success else 1


async def _execute_mode(
    connection: Any,
    *,
    mode: str,
    storage_path: Path,
    batch_size: int,
) -> tuple[dict[str, Any], bool]:
    if mode == "inventory":
        inventory = await inventory_legacy_artifact_blobs(
            connection, storage_root=storage_path
        )
        payload = _inventory_payload(inventory)
        return payload, inventory.issue_count == 0
    if mode == "migrate":
        result = await migrate_legacy_artifact_blobs(
            connection,
            storage_root=storage_path,
            batch_size=batch_size,
        )
        payload = {"mode": mode, **asdict(result)}
        return payload, result.issue_count == 0
    if mode == "cleanup":
        result = await drain_artifact_blob_cleanup_intents(
            connection,
            storage_root=storage_path,
            limit=batch_size,
        )
        payload = {"mode": mode, **asdict(result)}
        return (
            payload,
            result.deferred_intent_count == 0 and result.remaining_intent_count == 0,
        )
    if mode == "verify":
        verification = await verify_artifact_blob_migration(connection)
        payload = {
            "mode": mode,
            **asdict(verification),
            "is_complete": verification.is_complete,
        }
        return payload, verification.is_complete

    total_migrated = 0
    while True:
        result = await migrate_legacy_artifact_blobs(
            connection,
            storage_root=storage_path,
            batch_size=batch_size,
        )
        total_migrated += result.migrated_reference_count
        if result.migrated_reference_count == 0:
            break
    cleanup_totals = {
        "processed_intent_count": 0,
        "deleted_file_count": 0,
        "deferred_intent_count": 0,
        "remaining_intent_count": 0,
    }
    initial_intent_count: int | None = None
    while True:
        cleanup = await drain_artifact_blob_cleanup_intents(
            connection,
            storage_root=storage_path,
            limit=max(batch_size, 1),
        )
        cleanup_totals["processed_intent_count"] += cleanup.processed_intent_count
        cleanup_totals["deleted_file_count"] += cleanup.deleted_file_count
        cleanup_totals["deferred_intent_count"] += cleanup.deferred_intent_count
        cleanup_totals["remaining_intent_count"] = cleanup.remaining_intent_count
        if initial_intent_count is None:
            initial_intent_count = (
                cleanup.remaining_intent_count
                + cleanup.processed_intent_count
                - cleanup.deferred_intent_count
            )
        if cleanup.remaining_intent_count == 0:
            break
        if cleanup_totals["processed_intent_count"] >= initial_intent_count:
            break
    verification = await verify_artifact_blob_migration(connection)
    payload = {
        "mode": mode,
        "migrated_reference_count": total_migrated,
        "cleanup": cleanup_totals,
        "verification": asdict(verification),
        "is_complete": verification.is_complete,
    }
    return payload, verification.is_complete


def _inventory_payload(inventory: Any) -> dict[str, Any]:
    identities = []
    for identity in inventory.identities:
        identities.append(
            {
                "storage_identity": identity.storage_identity,
                "reference_count": len(identity.references),
                "pending_deletion_count": len(identity.pending_deletions),
                "expected_sha256": identity.expected_sha256,
                "expected_byte_size": identity.expected_byte_size,
                "issues": list(identity.issues),
            }
        )
    return {
        "mode": "inventory",
        "local_reference_count": inventory.local_reference_count,
        "pending_deletion_count": inventory.pending_deletion_count,
        "cleanup_intent_count": inventory.cleanup_intent_count,
        "issue_count": inventory.issue_count,
        "unresolved_issues": list(inventory.unresolved_issues),
        "identities": identities,
    }


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))
