"""Hash helpers for benchmark artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 for a file."""
    file_path = Path(path).expanduser()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_if_exists(path: str | Path) -> str | None:
    """Return the SHA-256 for an existing file, otherwise None."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return None
    return sha256_file(file_path)


def sha256_directory(path: str | Path) -> str:
    """Return a stable SHA-256 over file names and file hashes in a directory."""
    directory = Path(path).expanduser()
    if not directory.exists():
        return ""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = file_path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
