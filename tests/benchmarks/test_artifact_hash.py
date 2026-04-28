"""Tests for benchmark artifact hash helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.artifact_hash import sha256_directory, sha256_file, sha256_file_if_exists


def test_sha256_file_helpers_hash_existing_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    payload = b'{"ok": true}'
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_file(artifact) == expected
    assert sha256_file_if_exists(artifact) == expected
    assert sha256_file_if_exists(tmp_path / "missing.json") is None


def test_sha256_directory_hashes_names_and_contents(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("two", encoding="utf-8")

    first_hash = sha256_directory(tmp_path)
    (nested / "b.txt").write_text("three", encoding="utf-8")

    assert first_hash
    assert sha256_directory(tmp_path) != first_hash
    assert sha256_directory(tmp_path / "missing") == ""
