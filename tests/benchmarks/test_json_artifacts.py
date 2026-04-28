"""Tests for benchmark JSON artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.json_artifacts import write_json_atomic


def test_write_json_atomic_creates_parent_and_removes_temp_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "artifact.json"
    unicode_value = "\u00f1"

    result = write_json_atomic(output_path, {"b": 2, "a": unicode_value})

    assert result == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "a": unicode_value,
        "b": 2,
    }
    assert unicode_value in output_path.read_text(encoding="utf-8")
    assert not list(output_path.parent.glob(".artifact.json.*.tmp"))
