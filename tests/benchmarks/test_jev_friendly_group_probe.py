"""Offline selection and exact-grant checks for the grouped decision probe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

import pytest

from atagia.models.schemas_decisions import ChoiceAnswer, ScoreAnswer
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
)
from benchmarks.jev_friendly_cards.budget import BudgetError, GlobalGrantRegistry
from benchmarks.jev_friendly_cards import group_probe
from benchmarks.jev_friendly_cards.group_probe import (
    CAPTURE_FORMAT,
    FROZEN_CASE_IDS,
    _request_for_slot,
    freeze_selection,
    run_group_probe,
    select_grouped_captures,
)
from benchmarks.jev_friendly_cards.protocol import ARMS, sha256


def _choice_request(*, ordered: bool = True) -> dict:
    return {
        "capture_format": CAPTURE_FORMAT if ordered else "old_capture",
        "model": "jev-test",
        "messages": [{"role": "user", "content": "An unchanged state."}],
        "choice_questions": {
            "q_b": {"type": "choice", "instructions": "Select B.", "criteria": {"z": None, "a": "A"}},
            "q_a": {"type": "choice", "instructions": "Select A.", "criteria": {"no": None, "yes": None}},
        },
        "score_questions": {},
        "response_schema": None,
        "max_output_tokens": None,
        "temperature": None,
        "external_answer": False,
        "metadata": {"purpose": "offline_probe", "benchmark_slot": "source"},
    }


def _choice_answer(choice: str, options: list[str]) -> dict:
    return ChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={key: float(key == choice) for key in options},
        confidence=1.0,
    ).model_dump(mode="json")


def _choice_response() -> dict:
    return {
        "provider": "typesafe", "model": "jev-test",
        "choice_answers": {
            "q_b": _choice_answer("z", ["z", "a"]),
            "q_a": _choice_answer("yes", ["no", "yes"]),
        },
        "score_answers": {},
    }


def _score_request() -> dict:
    request = _choice_request()
    request["choice_questions"] = {}
    request["score_questions"] = {
        "first": {"type": "score", "instructions": "Rate first.", "criteria": ["0", "1", "2", "3", "4"]},
        "second": {"type": "score", "instructions": "Rate second.", "criteria": ["0", "1", "2", "3", "4"]},
    }
    return request


def _score_response() -> dict:
    answer = ScoreAnswer(
        type="score", score=2.0,
        legend={str(i): str(i) for i in range(5)},
        probabilities={str(i): float(i == 2) for i in range(5)},
        confidence=1.0,
    ).model_dump(mode="json")
    return {
        "provider": "typesafe", "model": "jev-test", "choice_answers": {},
        "score_answers": {"first": answer, "second": answer},
    }


def _source_journal(path: Path, captures: list[tuple[str, dict, dict]]) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE attempts (id TEXT, slot TEXT, provider TEXT, status TEXT, started_utc TEXT)"
        )
        db.execute(
            "CREATE TABLE attempt_payloads "
            "(attempt_id TEXT, request_json TEXT, response_json TEXT)"
        )
        for index, (slot, request, response) in enumerate(captures):
            attempt_id = f"attempt_{index}"
            db.execute(
                "INSERT INTO attempts VALUES (?, ?, 'typesafe', 'success', ?)",
                (attempt_id, slot, f"2026-09-26T00:00:{index:02}Z"),
            )
            db.execute(
                "INSERT INTO attempt_payloads VALUES (?, ?, ?)",
                (attempt_id, json.dumps(request), json.dumps(response)),
            )


def _evaluation(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    output = tmp_path / "evaluation"
    output.mkdir()
    protocol_path = tmp_path / "B_protocol.json"
    protocol_path.write_text(
        json.dumps({"grouped_individual_subset": list(FROZEN_CASE_IDS)}), encoding="utf-8"
    )
    slots = [
        {
            "slot": f"evaluation:{case_id}:{arm}:{repetition:02}",
            "case_id": case_id, "arm": arm, "repetition": repetition,
        }
        for case_id in FROZEN_CASE_IDS
        for arm in ARMS
        for repetition in (1, 2)
    ]
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps({"suite": "card_replay", "repetitions": 2, "slots": slots}),
        encoding="utf-8",
    )
    (output / "freeze.json").write_text(
        json.dumps({"manifest_sha256": sha256(manifest_path)}), encoding="utf-8"
    )
    journals = {arm: tmp_path / f"{arm}.sqlite" for arm in ARMS}
    for arm in ARMS:
        results = [
            {**slot, "status": "invalid_response" if slot["case_id"] == "S01"
             and slot["arm"] == "C_shared_jev" and slot["repetition"] == 1 else "success"}
            for slot in slots if slot["arm"] == arm
        ]
        (output / f"results_{arm}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8"
        )
        captures: list[tuple[str, dict, dict]] = []
        if arm == "C_shared_jev":
            single_request = _choice_request()
            single_request["choice_questions"] = {
                "q_b": single_request["choice_questions"]["q_b"]
            }
            single_response = _choice_response()
            single_response["choice_answers"] = {
                "q_b": single_response["choice_answers"]["q_b"]
            }
            captures = [
                ("evaluation:S01:C_shared_jev:02", _choice_request(), _choice_response()),
                ("evaluation:S01:C_shared_jev:02", _choice_request(), {
                    **_choice_response(), "choice_answers": {
                        "q_b": _choice_answer("a", ["z", "a"]),
                        "q_a": _choice_answer("no", ["no", "yes"]),
                    },
                }),
                ("evaluation:S06:C_shared_jev:01", _score_request(), _score_response()),
                ("evaluation:S07:C_shared_jev:01", single_request, single_response),
                ("evaluation:S07:C_shared_jev:02", _choice_request(), _choice_response()),
                ("evaluation:S08:C_shared_jev:01", _choice_request(ordered=False), _choice_response()),
            ]
        _source_journal(journals[arm], captures)
    return protocol_path, output, journals


def _selection(tmp_path: Path) -> dict:
    protocol, output, journals = _evaluation(tmp_path)
    return select_grouped_captures(
        protocol_path=protocol, evaluation_output=output, journal_paths=journals
    )


def test_selection_uses_first_valid_repetition_and_preserves_order(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    fingerprint = selection["code_fingerprint"]
    assert len(fingerprint["group_probe_sha256"]) == 64
    assert len(fingerprint["budget_sha256"]) == 64
    assert {
        "src/atagia/services/llm_client.py",
        "src/atagia/services/providers/typesafe.py",
        "src/atagia/models/schemas_decisions.py",
    } <= set(fingerprint["production"]["source_hashes"])
    rows = {(row["case_id"], row["arm"]): row for row in selection["selections"]}
    choice = rows["S01", "C_shared_jev"]
    assert choice["status"] == "eligible"
    assert choice["repetition"] == 2
    assert choice["source_attempt_id"] == "attempt_0"
    assert choice["response_capture"]["choice_answers"]["q_a"]["choice"] == "yes"
    assert list(choice["request_capture"]["choice_questions"]) == ["q_b", "q_a"]
    assert list(choice["request_capture"]["choice_questions"]["q_b"]["criteria"]) == ["z", "a"]
    assert rows["S06", "C_shared_jev"]["question_kind"] == "score"
    assert rows["S07", "C_shared_jev"]["reason"] == "no_grouped_typesafe_request"
    assert rows["S08", "C_shared_jev"]["reason"] == "unordered_capture"
    assert rows["S01", "A_baseline_llm"]["status"] == "ineligible"
    assert len(selection["selections"]) == len(FROZEN_CASE_IDS) * len(ARMS)

    slot = {"slot": "probe", "selection": choice}
    request, id_map = _request_for_slot(slot)
    assert id_map == {"probe_001": "q_a", "probe_002": "q_b"}
    assert list(request.choice_questions or {}) == ["probe_001", "probe_002"]
    assert request.messages[0].content == "An unchanged state."
    assert request.model == "jev-test"
    assert list(request.choice_questions["probe_002"].criteria) == ["z", "a"]
    assert request.choice_questions["probe_001"].instructions == "Select A."


class FakeTypeSafe(LLMProvider):
    name = "typesafe"
    supports_choices = True
    supports_scores = True

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        choices = request.choice_questions or {}
        scores = request.score_questions or {}
        return LLMCompletionResponse(
            provider=self.name, model=request.model,
            choice_answers={
                question_id: ChoiceAnswer.model_validate(
                    _choice_answer(next(iter(question.criteria)), list(question.criteria))
                )
                for question_id, question in choices.items()
            },
            score_answers={
                question_id: ScoreAnswer.model_validate(_score_response()["score_answers"]["first"])
                for question_id in scores
            },
            usage={"input_tokens": 10, "output_tokens": 0},
        )


def _grant(tmp_path: Path, digest: str) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry.sqlite"
    grant_path = tmp_path / "probe_grant.json"
    journal_path = tmp_path / "probe_journal.sqlite"
    registry = GlobalGrantRegistry.create(registry_path, program_id="group-probe-test")
    try:
        registry.issue(grant_path, {
            "registry_path": str(registry_path.resolve()),
            "assignment_id": "probe-assignment",
            "journal_path": str(journal_path.resolve()),
            "freeze_sha256": digest,
            "cap_usd": "1",
            "deadline_utc": "2099-01-01T00:00:00Z",
            "paid_dispatch_enabled": True,
            "price_verified_utc": "2026-09-26T00:00:00Z",
            "concurrency": {"typesafe": 1},
            "prices": [{
                "provider": "typesafe", "model": "jev-test",
                "input_per_million": "1", "output_per_million": "0",
                "context_tokens": 1000,
                "source_url": "https://example.test/prices",
                "verified_utc": "2026-09-26T00:00:00Z",
            }],
        })
    finally:
        registry.close()
    return grant_path, journal_path


def test_granted_probe_records_raw_calls_and_resumes_without_repeats(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    # Restrict execution to one selected batch; selection behavior is checked above.
    selection["selections"] = [
        row for row in selection["selections"] if row["case_id"] == "S01"
        and row["arm"] == "C_shared_jev"
    ]
    selection_path = tmp_path / "selection.json"
    digest = freeze_selection(selection_path, selection)
    with pytest.raises(FileExistsError):
        freeze_selection(selection_path, selection)
    grant_path, journal_path = _grant(tmp_path, digest)
    terminal_path = tmp_path / "probe_results.jsonl"

    fake = FakeTypeSafe()
    rows = asyncio.run(run_group_probe(
        selection_path=selection_path, terminal_path=terminal_path,
        grant_path=grant_path, assignment_id="probe-assignment",
        client=LLMClient(providers=[fake]),
    ))
    assert len(fake.requests) == 3
    assert len(rows) == 3
    assert all(row["status"] == "success" for row in rows)
    assert rows[-1]["id_map"] == {"probe_001": "q_a", "probe_002": "q_b"}
    assert rows[-1]["comparison"]["q_a"]["original"]["choice"] == "yes"
    assert rows[-1]["comparison"]["q_a"]["probe"]["choice"] == "no"
    assert list(fake.requests[-1].choice_questions or {}) == ["probe_001", "probe_002"]
    with sqlite3.connect(journal_path) as db:
        attempts = db.execute("SELECT slot, status FROM attempts").fetchall()
        captures = db.execute(
            "SELECT request_json, response_json FROM attempt_payloads"
        ).fetchall()
    assert len(attempts) == 3
    assert {status for _, status in attempts} == {"success"}
    assert all(request and response for request, response in captures)

    resumed = FakeTypeSafe()
    rows_again = asyncio.run(run_group_probe(
        selection_path=selection_path, terminal_path=terminal_path,
        grant_path=grant_path, assignment_id="probe-assignment",
        client=LLMClient(providers=[resumed]),
    ))
    assert len(rows_again) == 3
    assert resumed.requests == []
    with sqlite3.connect(journal_path) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 3

    original_lines = terminal_path.read_text(encoding="utf-8").splitlines()
    failed = json.loads(original_lines[0])
    failed["status"] = "invalid_response"
    terminal_path.write_text(
        "\n".join([json.dumps(failed), *original_lines[1:]]) + "\n", encoding="utf-8"
    )
    stopped = FakeTypeSafe()
    with pytest.raises(ValueError, match="requires review"):
        asyncio.run(run_group_probe(
            selection_path=selection_path, terminal_path=terminal_path,
            grant_path=grant_path, assignment_id="probe-assignment",
            client=LLMClient(providers=[stopped]),
        ))
    assert stopped.requests == []

    terminal_path.write_text(
        "\n".join(original_lines[:-1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reconciled"):
        asyncio.run(run_group_probe(
            selection_path=selection_path, terminal_path=terminal_path,
            grant_path=grant_path, assignment_id="probe-assignment",
            client=LLMClient(providers=[FakeTypeSafe()]),
        ))


def test_score_probe_maps_permuted_ids_back_to_originals(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    selection["selections"] = [
        row for row in selection["selections"] if row["case_id"] == "S06"
        and row["arm"] == "C_shared_jev"
    ]
    selection_path = tmp_path / "selection.json"
    digest = freeze_selection(selection_path, selection)
    grant_path, _ = _grant(tmp_path, digest)
    provider = FakeTypeSafe()
    rows = asyncio.run(run_group_probe(
        selection_path=selection_path, terminal_path=tmp_path / "probe_results.jsonl",
        grant_path=grant_path, assignment_id="probe-assignment",
        client=LLMClient(providers=[provider]),
    ))
    assert len(provider.requests) == 3
    assert rows[-1]["id_map"] == {"probe_001": "second", "probe_002": "first"}
    assert set(rows[-1]["comparison"]) == {"first", "second"}
    assert rows[-1]["comparison"]["first"]["probe"]["score"] == 2.0
    assert provider.requests[-1].messages == provider.requests[0].messages
    assert provider.requests[-1].model == provider.requests[0].model
    assert provider.requests[-1].score_questions["probe_001"].instructions == "Rate second."


def test_wrong_freeze_rejects_dispatch_before_provider_call(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    selection_path = tmp_path / "selection.json"
    digest = freeze_selection(selection_path, selection)
    grant_path, journal_path = _grant(tmp_path, digest)
    selection_path.write_text(selection_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    provider = FakeTypeSafe()
    with pytest.raises(BudgetError, match="frozen selection"):
        asyncio.run(run_group_probe(
            selection_path=selection_path, terminal_path=tmp_path / "terminals.jsonl",
            grant_path=grant_path, assignment_id="probe-assignment",
            client=LLMClient(providers=[provider]),
        ))
    assert provider.requests == []
    assert not journal_path.exists()


def test_changed_code_fingerprint_rejects_freeze_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    changed = {**selection, "code_fingerprint": {
        **selection["code_fingerprint"], "group_probe_sha256": "0" * 64
    }}
    with pytest.raises(ValueError, match="code changed"):
        freeze_selection(tmp_path / "changed.json", changed)
    assert not (tmp_path / "changed.json").exists()

    selection_path = tmp_path / "selection.json"
    digest = freeze_selection(selection_path, selection)
    grant_path, journal_path = _grant(tmp_path, digest)
    monkeypatch.setattr(group_probe, "_code_fingerprint", lambda: changed["code_fingerprint"])
    provider = FakeTypeSafe()
    with pytest.raises(ValueError, match="code differs"):
        asyncio.run(run_group_probe(
            selection_path=selection_path, terminal_path=tmp_path / "terminals.jsonl",
            grant_path=grant_path, assignment_id="probe-assignment",
            client=LLMClient(providers=[provider]),
        ))
    assert provider.requests == []
    assert not journal_path.exists()


def test_wrong_import_root_rejected_before_fingerprinting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(group_probe.atagia, "__file__", str(tmp_path / "src/atagia/__init__.py"))
    with pytest.raises(ValueError, match="different checkout"):
        group_probe._code_fingerprint()
