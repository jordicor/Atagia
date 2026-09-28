"""Offline checks for the bounded quote comparison and its fixed labels."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from benchmarks.source_quote_selection.cases import load_cases
from benchmarks.source_quote_selection.run import (
    completed_rows,
    grade,
    prepare,
    verify_freeze,
    evaluate,
    validate_resume,
)
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.source_quote_selection.readout import punctuation_boundary_diagnostic


def test_grade_rejects_missing_qualification_and_excess_context():
    source = "Earlier notes. Use BLUE only on Friday. Extra unrelated notes."
    catalog = SourceReferenceCatalog(source)
    case = {
        "source_text": source,
        "candidates": [
            {
                "candidate_id": "cand_001",
                "expected_ranges": [[15, 39]],
                "required_ranges": [[19, 23], [24, 38]],
                "allowed_range": [15, 39],
            }
        ],
    }
    complete = catalog.resolve("r4", "r9")
    assert grade(case, {"cand_001": complete})[0]["adequate"]
    assert not grade(case, {"cand_001": catalog.resolve("r4", "r5")})[0]["adequate"]
    assert not grade(case, {"cand_001": catalog.resolve("r1", "r9")})[0]["adequate"]


def test_no_support_counts_positive_selection_as_false_support():
    source = "Use BLUE."
    case = {
        "source_text": source,
        "candidates": [
            {
                "candidate_id": "cand_001",
                "expected_ranges": [],
                "required_ranges": [],
                "allowed_range": None,
            }
        ],
    }
    assert grade(case, {"cand_001": None})[0]["exact"]
    result = grade(
        case, {"cand_001": SourceReferenceCatalog(source).resolve("r1", "r3")}
    )[0]
    assert result["false_support"] and not result["adequate"]


def test_all_predeclared_exact_ranges_satisfy_required_evidence():
    for case in load_cases():
        catalog = SourceReferenceCatalog(case["source_text"])
        starts = {anchor.char_start: anchor.reference_id for anchor in catalog.anchors}
        ends = {anchor.char_end: anchor.reference_id for anchor in catalog.anchors}
        for candidate in case["candidates"]:
            single = {**case, "candidates": [candidate]}
            for start, end in candidate["expected_ranges"]:
                reference = catalog.resolve(starts[start], ends[end])
                result = grade(single, {candidate["candidate_id"]: reference})[0]
                assert result["adequate"] and result["exact"], case["case_id"]


def test_wrong_completed_slots_remain_terminal_and_duplicate_slots_fail(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {"slot": "case:jev:01", "status": "invalid_selection"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert set(completed_rows(path)) == {row["slot"]}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        completed_rows(path)


def test_freeze_detects_case_mutation_and_cannot_be_overwritten(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("# fixture", encoding="utf-8")
    output = tmp_path / "output"
    prepare(output, helper)
    manifest = verify_freeze(output, helper)
    assert len(manifest["slots"]) == 29 * 3 * 10
    with pytest.raises(ValueError, match="cannot be overwritten"):
        prepare(output, helper)
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen cases"):
        verify_freeze(output, helper)


@pytest.mark.asyncio
async def test_production_evidence_arm_builds_and_parses_without_network():
    class Client:
        def __init__(self):
            self.requests = []

        async def complete_choice_questions(
            self, *, model, messages, questions, metadata, **kwargs
        ):
            self.requests.append(
                LLMCompletionRequest(
                    model=model,
                    messages=messages,
                    choice_questions=questions,
                    metadata=metadata,
                    max_output_tokens=kwargs.get("max_output_tokens", 32),
                )
            )
            return {
                key: "direct" if metadata["purpose"] == "memory_extraction_evidence_support_card" else "no"
                for key in questions
            }

        async def complete(self, request):
            self.requests.append(request)
            return LLMCompletionResponse(
                provider="openrouter",
                model=request.model,
                output_text={
                    "memory_extraction_candidate_language_card": "en",
                    "memory_extraction_source_reference_card": "r1 r3",
                }[request.metadata["purpose"]],
            )

    client = Client()
    references = await evaluate(
        client,
        {
            "case_id": "smoke",
            "source_text": "Use BLUE.",
            "candidates": [{"candidate_id": "cand_001", "canonical_text": "Use BLUE."}],
        },
        "current_evidence",
    )
    assert references["cand_001"].quote("Use BLUE.") == "Use BLUE."
    assert len(client.requests) == 4
    assert all(
        "<candidate>" in request.messages[-1].content
        or all("<candidate>" in question.instructions for question in (request.choice_questions or {}).values())
        for request in client.requests
    )
    assert all("cand_001 |" not in request.messages[-1].content for request in client.requests)


def test_overload_resume_requires_exact_authorization_and_cooldown(tmp_path):
    journal = tmp_path / "budget.sqlite"
    with sqlite3.connect(journal) as db:
        db.execute(
            "CREATE TABLE attempts (id TEXT, slot TEXT, status TEXT, provider TEXT, "
            "error_type TEXT, finished_utc TEXT)"
        )
        db.execute(
            "CREATE TABLE attempt_payloads (attempt_id TEXT, response_json TEXT, error_json TEXT)"
        )
        db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?)",
            (
                "a1",
                "slot",
                "error",
                "typesafe",
                "TransientLLMError",
                "2026-09-24T04:00:00+00:00",
            ),
        )
        db.execute(
            "INSERT INTO attempt_payloads VALUES (?,?,?)",
            ("a1", None, json.dumps({"status_code": 529})),
        )
    now = datetime(2026, 9, 24, 4, 11, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="reconciliation"):
        validate_resume(journal, set(), set(), now)
    with pytest.raises(ValueError, match="cooldown"):
        validate_resume(journal, set(), {"a1"}, now.replace(minute=9))
    validate_resume(journal, set(), {"a1"}, now)
    with sqlite3.connect(journal) as db:
        db.execute(
            "UPDATE attempt_payloads SET error_json=?",
            (json.dumps({"status_code": None}),),
        )
    with pytest.raises(ValueError, match="no-answer"):
        validate_resume(journal, set(), {"a1"}, now)
    journal.with_name("typesafe_http.jsonl").write_text(
        json.dumps({"slot": "slot", "status_code": 529}) + "\n", encoding="utf-8"
    )
    validate_resume(journal, set(), {"a1"}, now)
    validate_resume(journal, {"slot"}, set(), now)
    with pytest.raises(ValueError, match="reconciliation"):
        validate_resume(journal, {"slot"}, {"a1"}, now)
    with sqlite3.connect(journal) as db:
        db.execute("UPDATE attempt_payloads SET response_json='{}'")
    with pytest.raises(ValueError, match="no-answer"):
        validate_resume(journal, set(), {"a1"}, now)


def test_boundary_sensitivity_does_not_excuse_missing_evidence_or_extra_words():
    case = {"source_text": "Use BLUE, extra."}
    candidate = {"allowed_range": [0, 8], "required_ranges": [[0, 3], [4, 8]]}
    assert punctuation_boundary_diagnostic(
        case, candidate, {"adequate": False, "range": [0, 9]}
    )
    assert not punctuation_boundary_diagnostic(
        case, candidate, {"adequate": False, "range": [4, 9]}
    )
    assert not punctuation_boundary_diagnostic(
        case, candidate, {"adequate": False, "range": [0, 16]}
    )
