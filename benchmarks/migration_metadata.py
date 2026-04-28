"""Migration reproducibility metadata for benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

from atagia.core.db_sqlite import MigrationManager
from benchmarks.artifact_hash import sha256_directory

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MIGRATIONS_DIR = _PROJECT_ROOT / "migrations"


def benchmark_migration_metadata(
    migrations_dir: str | Path = _DEFAULT_MIGRATIONS_DIR,
) -> dict[str, object]:
    """Return stable migration metadata for benchmark run artifacts."""
    resolved_dir = Path(migrations_dir).expanduser()
    migrations = MigrationManager(resolved_dir).discover()
    versions = [migration.version for migration in migrations]
    return {
        "path": str(resolved_dir),
        "sha256": sha256_directory(resolved_dir),
        "versions": versions,
        "latest_version": max(versions) if versions else None,
    }
