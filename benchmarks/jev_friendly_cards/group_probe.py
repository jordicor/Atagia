"""Compare an observed TypeSafe batch with isolated and permuted requests.

Selection is fixed by case, arm, first successful repetition, and first
successful homogeneous batch in that repetition. No model answer influences it.
The selection file is written before dispatch and contains the source captures.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import atagia

from atagia.models.schemas_decisions import ChoiceAnswer, ScoreAnswer
from atagia.services.llm_client import LLMClient, LLMCompletionRequest

from benchmarks.jev_friendly_cards import budget as budget_module
from benchmarks.jev_friendly_cards import protocol as protocol_module
from benchmarks.jev_friendly_cards.budget import (
    BudgetError,
    LaneBudget,
    _BudgetedProvider,
    install_budget,
    slot_scope,
)
from benchmarks.jev_friendly_cards.protocol import (
    ARMS,
    remaining_slots,
    root_fingerprint,
    sha256,
    terminal_rows,
)


CAPTURE_FORMAT = "ordered_typesafe_v1"
SELECTION_VERSION = 1
FROZEN_CASE_IDS = ("S01", "S06", "S07", "S08", "S19", "S23", "S24", "S29")


def _code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(atagia.__file__).resolve().parents[2] != root:
        raise ValueError("Group probe imported production from a different checkout")
    return {
        "group_probe_sha256": sha256(Path(__file__)),
        "budget_sha256": sha256(Path(budget_module.__file__)),
        "protocol_sha256": sha256(Path(protocol_module.__file__)),
        "production": root_fingerprint(root, Path(sys.executable), root),
    }


def _source_attempts(journal: Path, slot: str) -> list[tuple[str, str, dict, str | None]]:
    if not journal.is_file():
        raise ValueError(f"Missing evaluation journal: {journal}")
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT a.id, a.provider, p.request_json, p.response_json "
            "FROM attempts AS a JOIN attempt_payloads AS p ON p.attempt_id=a.id "
            "WHERE a.slot=? AND a.status='success' "
            "ORDER BY a.started_utc, a.id",
            (slot,),
        ).fetchall()
    return [
        (attempt_id, provider, json.loads(request), response)
        for attempt_id, provider, request, response in rows
    ]


def _question_kind(request: dict[str, Any]) -> str | None:
    choices = request.get("choice_questions") or {}
    scores = request.get("score_questions") or {}
    if len(choices) > 1 and not scores:
        return "choice"
    if len(scores) > 1 and not choices:
        return "score"
    return None


def _answers(kind: str, response: dict[str, Any], question_ids: set[str]) -> dict:
    key = f"{kind}_answers"
    answers = response.get(key)
    if not isinstance(answers, dict) or set(answers) != question_ids:
        raise ValueError("Captured answer IDs differ from the grouped questions")
    schema = ChoiceAnswer if kind == "choice" else ScoreAnswer
    return {
        question_id: schema.model_validate(answer).model_dump(mode="json")
        for question_id, answer in answers.items()
    }


def select_grouped_captures(
    *,
    protocol_path: Path,
    evaluation_output: Path,
    journal_paths: dict[str, Path],
) -> dict[str, Any]:
    """Choose source batches without reading answer values for selection."""
    code_fingerprint = _code_fingerprint()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["grouped_individual_subset"]) != FROZEN_CASE_IDS:
        raise ValueError("The frozen grouped subset changed")
    manifest_path = evaluation_output / "manifest.json"
    freeze = json.loads((evaluation_output / "freeze.json").read_text(encoding="utf-8"))
    if freeze["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Evaluation manifest does not match its freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["suite"] != "card_replay" or set(journal_paths) != set(ARMS):
        raise ValueError("The grouped probe needs the complete card replay and three journals")
    planned = manifest["slots"]
    terminal = {
        arm: terminal_rows(evaluation_output / f"results_{arm}.jsonl")
        for arm in ARMS
    }
    selections: list[dict[str, Any]] = []
    for case_id in FROZEN_CASE_IDS:
        for arm in ARMS:
            matching = sorted(
                (slot for slot in planned if slot["case_id"] == case_id and slot["arm"] == arm),
                key=lambda slot: slot["repetition"],
            )
            if len(matching) != manifest["repetitions"] or any(
                slot["slot"] not in terminal[arm] for slot in matching
            ):
                raise ValueError(f"Evaluation is incomplete for {case_id}/{arm}")
            first_valid = next(
                (slot for slot in matching if terminal[arm][slot["slot"]]["status"] == "success"),
                None,
            )
            row: dict[str, Any] = {"case_id": case_id, "arm": arm}
            if first_valid is None:
                row.update(status="ineligible", reason="no_successful_repetition")
                selections.append(row)
                continue
            row.update(repetition=first_valid["repetition"], source_slot=first_valid["slot"])
            candidates = _source_attempts(journal_paths[arm], first_valid["slot"])
            selected = next(
                (
                    (attempt_id, provider, request, response)
                    for attempt_id, provider, request, response in candidates
                    if provider == "typesafe" and _question_kind(request) is not None
                ),
                None,
            )
            if selected is None:
                row.update(status="ineligible", reason="no_grouped_typesafe_request")
                selections.append(row)
                continue
            attempt_id, _, request, response_json = selected
            row["source_attempt_id"] = attempt_id
            if request.get("capture_format") != CAPTURE_FORMAT:
                row.update(status="ineligible", reason="unordered_capture")
                selections.append(row)
                continue
            if response_json is None:
                raise ValueError("Successful grouped attempt has no response capture")
            response = json.loads(response_json)
            kind = _question_kind(request)
            assert kind is not None
            question_ids = set(request[f"{kind}_questions"])
            _answers(kind, response, question_ids)
            if response.get("provider") != "typesafe" or response.get("model") != request.get("model"):
                raise ValueError("Captured TypeSafe route changed")
            row.update(
                status="eligible",
                question_kind=kind,
                request_capture=request,
                response_capture=response,
            )
            selections.append(row)
    return {
        "version": SELECTION_VERSION,
        "code_fingerprint": code_fingerprint,
        "protocol_sha256": sha256(protocol_path),
        "evaluation_manifest_sha256": sha256(manifest_path),
        "selections": selections,
    }


def freeze_selection(path: Path, selection: dict[str, Any]) -> str:
    """Create an immutable dispatch input; a resume must reuse these bytes."""
    if not path.is_absolute() or selection.get("version") != SELECTION_VERSION:
        raise ValueError("Selection needs an absolute path and known version")
    if selection.get("code_fingerprint") != _code_fingerprint():
        raise ValueError("Selected code changed before the probe freeze")
    payload = (json.dumps(selection, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _probe_slots(selection: dict[str, Any]) -> list[dict[str, Any]]:
    planned = []
    for row in selection["selections"]:
        if row["status"] != "eligible":
            continue
        questions = row["request_capture"][f"{row['question_kind']}_questions"]
        stem = f"group_probe:{row['case_id']}:{row['arm']}:{row['source_attempt_id']}"
        for index, question_id in enumerate(questions, 1):
            planned.append({
                "slot": f"{stem}:individual:{index:02}",
                "question_id": question_id,
                "selection": row,
            })
        planned.append({"slot": f"{stem}:permuted", "selection": row})
    return planned


def _request_for_slot(slot: dict[str, Any]) -> tuple[LLMCompletionRequest, dict[str, str]]:
    row = slot["selection"]
    capture = row["request_capture"]
    kind = row["question_kind"]
    questions = capture[f"{kind}_questions"]
    if "question_id" in slot:
        question_id = slot["question_id"]
        selected = {question_id: questions[question_id]}
        id_map = {question_id: question_id}
    else:
        original_ids = list(questions)
        permuted_ids = [f"probe_{index:03}" for index in range(1, len(questions) + 1)]
        if set(permuted_ids) & set(original_ids):
            raise ValueError("Permutation IDs overlap original question IDs")
        id_map = dict(zip(permuted_ids, reversed(original_ids), strict=True))
        selected = {new_id: questions[old_id] for new_id, old_id in id_map.items()}
    request_data = {
        key: capture[key]
        for key in (
            "model", "messages", "temperature", "max_output_tokens",
            "response_schema", "external_answer", "metadata",
        )
    }
    request_data["metadata"] = {**request_data["metadata"], "benchmark_slot": slot["slot"]}
    request_data[f"{kind}_questions"] = selected
    return LLMCompletionRequest.model_validate(request_data), id_map


def _journal_slots(journal: Path) -> tuple[set[str], set[str]]:
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT slot, status FROM attempts").fetchall()
    return {slot for slot, _ in rows}, {slot for slot, status in rows if status == "reserved"}


def _append_terminal(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def run_group_probe(
    *,
    selection_path: Path,
    terminal_path: Path,
    grant_path: Path,
    assignment_id: str,
    client: LLMClient[Any],
) -> list[dict[str, Any]]:
    """Dispatch each new probe slot once through an exact granted LaneBudget."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("version") != SELECTION_VERSION or not selection_path.is_absolute():
        raise ValueError("A frozen, absolute selection file is required")
    if selection.get("code_fingerprint") != _code_fingerprint():
        raise ValueError("Probe code differs from the frozen selection")
    if not terminal_path.is_absolute() or terminal_path.parent.resolve() != selection_path.parent.resolve():
        raise ValueError("Probe terminals must stay beside the private selection")
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    digest = sha256(selection_path)
    if grant.get("freeze_sha256") != digest or grant.get("paid_dispatch_enabled") is not True:
        raise BudgetError("The grant does not authorize this frozen selection")
    journal_path = Path(grant["journal_path"])
    planned = _probe_slots(selection)
    terminal = terminal_rows(terminal_path)
    if any(row.get("selection_sha256") != digest for row in terminal.values()):
        raise ValueError("Probe terminals belong to a different selection")
    if any(row["status"] != "success" for row in terminal.values()):
        raise ValueError("A failed probe terminal requires review before resume")
    attempted, pending = _journal_slots(journal_path) if journal_path.exists() else (set(), set())
    remaining = remaining_slots(planned, terminal, attempted, pending)
    budget = LaneBudget.from_grant(
        grant_path, journal_path, expected_assignment_id=assignment_id
    )
    try:
        install_budget(client, budget)
        provider = client._providers.get("typesafe")
        if not isinstance(provider, _BudgetedProvider) or provider.budget is not budget:
            raise BudgetError("The TypeSafe provider is not bound to the exact grant")
        for slot in remaining:
            request, id_map = _request_for_slot(slot)
            source = slot["selection"]
            kind = source["question_kind"]
            supported = provider.supports_choices if kind == "choice" else provider.supports_scores
            if not supported:
                raise ValueError("The granted provider does not support this question kind")
            with slot_scope(slot["slot"]):
                try:
                    response = await provider.complete(request)
                    observed = _answers(
                        kind,
                        response.model_dump(mode="json"),
                        set(id_map),
                    )
                except BudgetError:
                    raise
                except Exception as exc:
                    statuses = budget.slot_attempt_statuses(slot["slot"])
                    if "error" in statuses or "reserved" in statuses:
                        status = "provider_error"
                    elif "success" in statuses:
                        status = "invalid_response"
                    else:
                        status = "harness_error"
                    result = {
                        "slot": slot["slot"], "status": status,
                        "selection_sha256": digest,
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                    _append_terminal(terminal_path, result)
                    raise
            original = _answers(
                kind,
                source["response_capture"],
                set(source["request_capture"][f"{kind}_questions"]),
            )
            result = {
                "slot": slot["slot"], "status": "success",
                "selection_sha256": digest,
                "source_attempt_id": source["source_attempt_id"],
                "question_kind": kind,
                "id_map": id_map,
                "comparison": {
                    original_id: {"original": original[original_id], "probe": observed[new_id]}
                    for new_id, original_id in id_map.items()
                },
            }
            _append_terminal(terminal_path, result)
            terminal[result["slot"]] = result
        return list(terminal.values())
    finally:
        try:
            await client.aclose()
        finally:
            budget.close()
