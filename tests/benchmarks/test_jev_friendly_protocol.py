"""Offline corpus slot and worktree-fidelity checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from benchmarks.jev_friendly_cards.protocol import (
    imported_root,
    plan_slots,
    remaining_slots,
    terminal_rows,
)


ROOT = Path(__file__).resolve().parents[2]


def test_imports_are_confined_to_selected_worktree() -> None:
    paths = imported_root(ROOT, Path(sys.executable))
    assert Path(paths["atagia"]).is_relative_to(ROOT / "src" / "atagia")
    assert Path(paths["benchmarks"]).is_relative_to(ROOT / "benchmarks")


def test_balanced_slots_and_successful_resume(tmp_path: Path) -> None:
    cases = [
        {"case_id": "one", "primary_family": "classification"},
        {"case_id": "two", "primary_family": "evidence"},
    ]
    slots = plan_slots(cases, 3)
    assert len(slots) == 18
    assert {slot["arm"] for slot in slots} == {
        "A_baseline_llm", "B_shared_llm", "C_shared_jev"
    }
    result_path = tmp_path / "results.jsonl"
    completed = {**slots[0], "status": "success", "grade": "incorrect"}
    result_path.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    rows = terminal_rows(result_path)
    assert len(remaining_slots(slots, rows, {slots[0]["slot"]}, set())) == 17
    with pytest.raises(ValueError, match="reconciled"):
        remaining_slots(slots, rows, {slots[0]["slot"], slots[1]["slot"]}, set())
    with pytest.raises(ValueError, match="reconciled"):
        remaining_slots(slots, rows, {slots[0]["slot"]}, {slots[1]["slot"]})


def test_repetition_count_frozen_per_group() -> None:
    with pytest.raises(ValueError, match="10, 5, or 3"):
        plan_slots([{"case_id": "one", "primary_family": "topics"}], 4)
