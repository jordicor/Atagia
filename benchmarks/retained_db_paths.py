"""Helpers for choosing durable benchmark DB snapshot locations."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def default_benchmark_db_dir(output_dir: str | Path) -> Path:
    """Return the default durable DB snapshot directory for a benchmark output."""
    return Path(output_dir).expanduser() / "dbs"


def validate_retained_benchmark_db_dir(
    benchmark_db_dir: str | Path,
    *,
    allow_temp_benchmark_db_dir: bool = False,
) -> Path:
    """Reject temporary directories for retained benchmark DB snapshots."""
    db_dir = Path(benchmark_db_dir).expanduser()
    if allow_temp_benchmark_db_dir or not is_temporary_path(db_dir):
        return db_dir

    roots = ", ".join(str(root) for root in temporary_roots())
    raise ValueError(
        "Refusing to retain benchmark DB snapshots under a temporary directory: "
        f"{db_dir}. Use a persistent --benchmark-db-dir, or pass "
        "--allow-temp-benchmark-db-dir for an explicitly disposable run. "
        f"Temporary roots detected: {roots}"
    )


def is_temporary_path(path: str | Path) -> bool:
    """Return whether path is inside a known system temporary directory."""
    candidate = Path(path).expanduser().resolve(strict=False)
    return any(_is_relative_to(candidate, root) for root in temporary_roots())


def temporary_roots() -> tuple[Path, ...]:
    """Return normalized temporary roots worth rejecting for retained artifacts."""
    candidates = [
        tempfile.gettempdir(),
        os.environ.get("TMPDIR"),
        "/tmp",
        "/private/tmp",
        "/var/tmp",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for raw_candidate in candidates:
        if not raw_candidate:
            continue
        root = Path(raw_candidate).expanduser().resolve(strict=False)
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
