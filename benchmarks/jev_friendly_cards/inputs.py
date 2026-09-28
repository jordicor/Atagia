"""Load frozen private evaluation inputs without exposing answer labels to runners."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """The private evaluation inputs differ from their recorded version."""


def verify_input_hashes(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    expected = json.loads((root / "B_input_hashes.json").read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or not expected:
        raise InputError("The input hash manifest is empty")
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise InputError("Input hash manifest contains an invalid path")
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise InputError(f"Private input changed: {name}")
    return expected


def load_model_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read only source cases and runtime fixtures, never the grading file."""
    verify_input_hashes(root)
    cases = [
        json.loads(line)
        for line in (root / "B_corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fixtures = json.loads((root / "B_card_fixtures.json").read_text(encoding="utf-8"))
    ids = [case.get("case_id") for case in cases]
    if len(cases) != 40 or len(set(ids)) != 40 or set(ids) != set(fixtures):
        raise InputError("Evaluation cases and fixtures do not match")
    for case in cases:
        case_id = case["case_id"]
        fixture = fixtures[case_id]
        if case.get("split") != "evaluation" or case.get("role") != "user":
            raise InputError(f"Unsupported source case: {case_id}")
        if not isinstance(case.get("source_text"), str) or not case["source_text"]:
            raise InputError(f"Missing source text: {case_id}")
        if fixture.get("source_message_id") != case_id:
            raise InputError(f"Source message identity changed: {case_id}")
        if fixture.get("mode") != fixture.get("assistant_mode_id"):
            raise InputError(f"Mode and policy disagree: {case_id}")
        if fixture.get("assistant_mode_id") != (
            "personal_assistant" if case["origin"] == "aurvek_local_snapshot" else "general_qa"
        ):
            raise InputError(f"Unexpected policy profile: {case_id}")
        when = datetime.fromisoformat(fixture["occurred_at"])
        if when.tzinfo is None or when.astimezone(timezone.utc).utcoffset() is None:
            raise InputError(f"Timestamp lacks an offset: {case_id}")
        if case["origin"] == "synthetic" and not isinstance(
            fixture.get("card_replay_candidate"), str
        ):
            raise InputError(f"Card replay candidate is missing: {case_id}")
    return cases, fixtures
