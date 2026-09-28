"""Offline accounting and per-case readout; never sends model requests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3
import statistics
import unicodedata

from atagia.core.source_references import SourceReferenceCatalog
from benchmarks.source_quote_selection.run import grade, percentile, write_json


def punctuation_boundary_diagnostic(case: dict, candidate: dict, result: dict) -> bool:
    """Post-hoc sensitivity only; never changes the frozen primary grade."""
    interval = result["range"]
    allowed = candidate["allowed_range"]
    if result["adequate"] or interval is None or allowed is None:
        return False
    start, end = interval
    if not all(
        start <= low < high <= end for low, high in candidate["required_ranges"]
    ):
        return False
    source = case["source_text"]
    extra = source[start : allowed[0]] if start < allowed[0] else ""
    extra += source[allowed[1] : end] if end > allowed[1] else ""
    return bool(extra) and all(
        char.isspace() or unicodedata.category(char).startswith("P") for char in extra
    )


def report(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    groups = defaultdict(list)
    for row in rows:
        case = cases[row["case_id"]]
        candidate_by_id = {item["candidate_id"]: item for item in case["candidates"]}
        for result in row.get("grades", []):
            result["punctuation_only_boundary_failure"] = (
                punctuation_boundary_diagnostic(
                    case, candidate_by_id[result["candidate_id"]], result
                )
            )
        groups[row["arm"]].append(row)
    with sqlite3.connect(
        (output / "budget.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        attempts = [
            dict(row)
            for row in db.execute("SELECT * FROM attempts ORDER BY started_utc")
        ]
        requests = {
            row["slot"]: json.loads(row["request_json"])
            for row in db.execute(
                "SELECT a.slot,p.request_json FROM attempts a JOIN attempt_payloads p ON p.attempt_id=a.id "
                "ORDER BY a.started_utc"
            )
        }
        pending = sum(row["status"] == "reserved" for row in attempts)
        violations = db.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    captures = []
    capture_path = output / "typesafe_http.jsonl"
    if capture_path.exists():
        captures = [
            json.loads(line)
            for line in capture_path.read_text(encoding="utf-8").splitlines()
        ]
    http_by_slot = defaultdict(list)
    for capture in captures:
        capture["payload"] = json.loads(capture.pop("body"))
        http_by_slot[capture["slot"]].append(capture)
    diagnostics = []
    for row in groups["jev_reference"]:
        if row["status"] != "provider_error":
            continue
        records = http_by_slot[row["slot"]]
        record = records[-1] if records else None
        detail = {"slot": row["slot"], "raw_body_available": record is not None}
        if record:
            answers = record["payload"].get("answers", {})
            questions = requests[row["slot"]]["choice_questions"]
            failures = []
            for key, answer in answers.items():
                probabilities = answer["probabilities"]
                selected = probabilities.get(answer["choice"])
                total = sum(probabilities.values())
                failure = {"question": key, "sum": total}
                if set(probabilities) != set(questions[key]["criteria"]):
                    failure["options_mismatch"] = True
                if selected is None or selected < max(probabilities.values()) - 1e-6:
                    failure["winner_mismatch"] = True
                if not math.isclose(total, 1, abs_tol=0.002):
                    failure["sum_mismatch"] = True
                if len(failure) > 2:
                    failures.append(failure)
            detail["failures"] = failures
            case = cases[row["case_id"]]
            catalog = SourceReferenceCatalog(case["source_text"])
            references = {}
            try:
                for item in case["candidates"]:
                    key = item["candidate_id"]
                    start, end = (
                        answers[f"{key}.{side}"]["choice"] for side in ("start", "end")
                    )
                    references[key] = (
                        None if start == end == "none" else catalog.resolve(start, end)
                    )
                raw_grades = grade(case, references)
                candidates = {item["candidate_id"]: item for item in case["candidates"]}
                for result in raw_grades:
                    result["punctuation_only_boundary_failure"] = (
                        punctuation_boundary_diagnostic(
                            case, candidates[result["candidate_id"]], result
                        )
                    )
                detail["raw_choice_diagnostic_grades"] = raw_grades
            except (ValueError, KeyError):
                detail["raw_choice_diagnostic_grades"] = None
        diagnostics.append(detail)
    arms = {}
    for arm, matching in groups.items():
        graded = [g for row in matching for g in row.get("grades", [])]
        slots = {row["slot"] for row in matching}
        paid = [attempt for attempt in attempts if attempt["slot"] in slots]
        latency = [row["latency_ms"] for row in matching if row["status"] == "success"]
        observed_tokens = sum(
            capture["payload"].get("usage", {}).get("input_tokens", 0)
            for slot in slots
            for capture in http_by_slot[slot]
        )
        arms[arm] = {
            "statuses": dict(Counter(row["status"] for row in matching)),
            "candidates": sum(
                len(cases[row["case_id"]]["candidates"]) for row in matching
            ),
            "adequate": sum(g["adequate"] for g in graded),
            "exact": sum(g["exact"] for g in graded),
            "punctuation_only_boundary_failures": sum(
                g["punctuation_only_boundary_failure"] for g in graded
            ),
            "false_support": sum(g["false_support"] for g in graded),
            "false_abstention": sum(g["false_abstention"] for g in graded),
            "p50_ms_valid_slots": statistics.median(latency) if latency else None,
            "p95_ms_valid_slots": percentile(latency, 0.95),
            "provider_attempts": len(paid),
            "conservative_usd": sum(
                a["charged"] if a["charged"] is not None else a["reserved"]
                for a in paid
            )
            / 1e9,
            "reported_usd": sum(a["reported_cost"] or 0 for a in paid) / 1e9,
            "typesafe_observed_input_tokens": observed_tokens,
            "typesafe_usage_estimate_usd": observed_tokens * 0.042 / 1e6
            if arm == "jev_reference"
            else None,
            "per_case": {},
            "per_origin": {},
            "by_length": {},
        }
        for name, key_fn in (
            ("per_case", lambda row: row["case_id"]),
            ("per_origin", lambda row: row["origin"]),
            (
                "by_length",
                lambda row: (
                    "long"
                    if len(
                        SourceReferenceCatalog(
                            cases[row["case_id"]]["source_text"]
                        ).anchors
                    )
                    > 254
                    else "short"
                ),
            ),
        ):
            grouped = defaultdict(list)
            for row in matching:
                grouped[key_fn(row)].append(row)
            for key, items in grouped.items():
                latencies = [
                    row["latency_ms"] for row in items if row["status"] == "success"
                ]
                arms[arm][name][key] = {
                    "candidates": sum(
                        len(cases[row["case_id"]]["candidates"]) for row in items
                    ),
                    "adequate": sum(
                        g["adequate"] for row in items for g in row.get("grades", [])
                    ),
                    "statuses": dict(Counter(row["status"] for row in items)),
                    "p50_ms": statistics.median(latencies) if latencies else None,
                }
    result = {
        "cases": len(cases),
        "slots": len(rows),
        "planned_slots": len(manifest["slots"]),
        "arms": arms,
        "provider_diagnostics": diagnostics,
        "pending_reservations": pending,
        "budget_violations": violations,
        "all_attempts_conservative_usd": sum(
            a["charged"] if a["charged"] is not None else a["reserved"]
            for a in attempts
        )
        / 1e9,
        "caveats": [
            "Repeated trials are not independent cases.",
            "Latency describes valid slots; failed trials are reported separately.",
            "Raw choice diagnostics are not validated route successes.",
            "Punctuation-only boundary sensitivity was added after inspecting outputs; the frozen primary grade is unchanged.",
            "The first failed TypeSafe HTTP body was not captured; its full reservation remains charged.",
            "Jev reference selection does not return the full evidence card's ancillary fields.",
        ],
    }
    write_json(output / "readout.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = report(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cases",
                    "slots",
                    "planned_slots",
                    "all_attempts_conservative_usd",
                )
            }
        )
    )
