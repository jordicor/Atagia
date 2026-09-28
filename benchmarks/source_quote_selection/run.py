"""Frozen, budget-reserved comparison of full evidence and reference selectors."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import statistics
from time import perf_counter
from typing import Any

import httpx

from atagia.core.config import Settings
from atagia.core.source_references import SourceReferenceCatalog
from atagia.memory.extraction_cards import (
    CandidateDraft,
    run_evidence_card,
)
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import LLMClient, LLMError, RetryPolicy
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.model_resolution import (
    examples_enabled_for_component,
    resolve_component_model,
)
from atagia.services.providers.openrouter import OpenRouterProvider
from atagia.services.providers.typesafe import TypeSafeProvider
from benchmarks.source_quote_selection import selector

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "current_evidence": "openrouter/openai/gpt-5.6-luna",
    "luna_reference": "openrouter/openai/gpt-5.6-luna",
    "jev_reference": "typesafe/jev-1.13.0",
}
REPETITIONS = 10
SOURCE_FILES = (
    "benchmarks/source_quote_selection/run.py",
    "benchmarks/source_quote_selection/selector.py",
    "benchmarks/source_quote_selection/cases.py",
    "benchmarks/memory_extraction_cards/cases.jsonl",
    "src/atagia/core/source_references.py",
    "src/atagia/memory/extraction_cards.py",
    "src/atagia/memory/evidence_cards.py",
    "src/atagia/models/schemas_memory.py",
    "src/atagia/models/schemas_decisions.py",
    "src/atagia/services/llm_client.py",
    "src/atagia/services/llm_reliability.py",
    "src/atagia/services/model_profiles.py",
    "src/atagia/services/providers/openai.py",
    "src/atagia/services/providers/openrouter.py",
    "src/atagia/services/providers/typesafe.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("quote_budget", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the reservation helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def prepare(output: Path, budget_helper: Path) -> None:
    from benchmarks.source_quote_selection.cases import load_cases

    if (output / "freeze.json").exists():
        raise ValueError("A frozen evaluation cannot be overwritten")
    if not examples_enabled_for_component(Settings.from_env(), "extractor"):
        raise ValueError("The configured evidence-card example setting changed")
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    slots = []
    for repetition in range(1, REPETITIONS + 1):
        for index, case in enumerate(cases):
            arms = list(MODELS)
            rotation = (index + repetition) % len(arms)
            arms = arms[rotation:] + arms[:rotation]
            for arm in arms:
                slots.append(
                    {
                        "slot": f"{case['case_id']}:{arm}:{repetition:02}",
                        "case_id": case["case_id"],
                        "arm": arm,
                        "repetition": repetition,
                    }
                )
    write_json(
        output / "manifest.json",
        {
            "models": MODELS,
            "repetitions": REPETITIONS,
            "cases": cases,
            "slots": slots,
            "source_kinds": dict(Counter(case["origin"]["kind"] for case in cases)),
            "scope": "Fixed-candidate reference selection, not full ingestion or full-card replacement",
        },
    )
    write_json(
        output / "freeze.json",
        {
            "source_hashes": {name: digest(ROOT / name) for name in SOURCE_FILES},
            "budget_helper_sha256": digest(budget_helper),
            "manifest_sha256": digest(output / "manifest.json"),
        },
    )


def verify_freeze(output: Path, budget_helper: Path) -> dict:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if freeze["source_hashes"] != {name: digest(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError("Evaluation source freeze changed")
    if freeze["budget_helper_sha256"] != digest(budget_helper):
        raise ValueError("Reservation helper changed")
    if freeze["manifest_sha256"] != digest(output / "manifest.json"):
        raise ValueError("Frozen cases or slot order changed")
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def grade(case: dict, references: dict) -> list[dict]:
    grades = []
    for candidate in case["candidates"]:
        reference = references[candidate["candidate_id"]]
        expected = candidate["expected_ranges"]
        absent = not expected
        interval = [reference.char_start, reference.char_end] if reference else None
        allowed = candidate["allowed_range"]
        adequate = (
            absent
            if reference is None
            else (
                not absent
                and allowed[0] <= interval[0] < interval[1] <= allowed[1]
                and all(
                    interval[0] <= start < end <= interval[1]
                    for start, end in candidate["required_ranges"]
                )
            )
        )
        minimum = min((end - start for start, end in expected), default=0)
        grades.append(
            {
                "candidate_id": candidate["candidate_id"],
                "adequate": bool(adequate),
                "exact": absent if reference is None else interval in expected,
                "expected_absent": absent,
                "false_support": absent and reference is not None,
                "false_abstention": not absent and reference is None,
                "range": interval,
                "quote": reference.quote(case["source_text"]) if reference else None,
                "extra_characters": max(0, interval[1] - interval[0] - minimum)
                if interval
                else 0,
            }
        )
    return grades


async def evaluate(client: LLMClient, case: dict, arm: str) -> dict:
    candidates = tuple(
        CandidateDraft(
            candidate_id=item["candidate_id"], canonical_text=item["canonical_text"]
        )
        for item in case["candidates"]
    )
    source = case["source_text"]
    if arm == "jev_reference":
        return await selector.select_references(
            client, model=MODELS[arm], source_text=source, candidates=candidates
        )
    if arm == "luna_reference":
        request = selector.build_reference_only_request(
            model=MODELS[arm],
            source_text=source,
            candidates=candidates,
        ).model_copy(update={"max_output_tokens": 1024})
        response = await client.complete(request)
        return selector.parse_reference_only_output(
            response.output_text,
            source_text=source,
            candidates=candidates,
        )
    context = ExtractionConversationContext(
        user_id="quote_bench_user",
        conversation_id="quote_bench_conversation",
        source_message_id=case["case_id"],
        assistant_mode_id="coding_debug",
        mode="coding_debug",
        privacy_enforcement="off",
    )
    catalog = SourceReferenceCatalog(source)
    result = await run_evidence_card(
        client,
        model=MODELS[arm],
        evidence_model=MODELS[arm],
        message_text=source,
        role="user",
        context=context,
        resolved_policy=None,
        allowed_write_scopes=("chat", "user"),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=candidates,
        source_catalog=catalog,
    )
    if set(result.parsed) != {candidate.candidate_id for candidate in candidates}:
        raise ValueError("Evidence result has missing or unexpected candidates")
    references = {}
    for candidate_id, row in result.parsed.items():
        if row["start_ref"] is None and row["end_ref"] is None:
            references[candidate_id] = None
        else:
            references[candidate_id] = catalog.resolve(row["start_ref"], row["end_ref"])
    return references


def completed_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["slot"] in rows:
                raise ValueError("Duplicate terminal slot in results")
            rows[row["slot"]] = row
    return rows


def attempt_stats(journal: Path) -> dict[str, dict]:
    if not journal.exists():
        return {}
    with sqlite3.connect(
        journal.resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        return {
            row["slot"]: dict(row)
            for row in connection.execute(
                "SELECT slot, COUNT(*) AS attempts, SUM(COALESCE(charged,reserved)) AS charged, "
                "SUM(reported_cost) AS reported_cost, SUM(input_tokens) AS input_tokens, "
                "SUM(output_tokens) AS output_tokens, SUM(status='reserved') AS pending, "
                "SUM(status='error') AS errors FROM attempts GROUP BY slot"
            )
        }


def validate_resume(
    journal: Path,
    completed: set[str],
    authorized_attempts: set[str],
    now: datetime,
) -> None:
    """Resume only explicitly reconciled overloads that returned no answer."""
    if not journal.exists():
        if authorized_attempts:
            raise ValueError("Authorized attempts are missing from the journal")
        return
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        unfinished = [
            dict(row)
            for row in db.execute(
                "SELECT a.*, p.response_json, p.error_json FROM attempts a "
                "LEFT JOIN attempt_payloads p ON p.attempt_id=a.id"
            )
            if row["slot"] not in completed
        ]
    if {row["id"] for row in unfinished} != authorized_attempts:
        raise ValueError("An unfinished paid slot requires explicit reconciliation")
    capture_path = journal.with_name("typesafe_http.jsonl")
    captures = {}
    if unfinished and capture_path.exists():
        for line in capture_path.read_text(encoding="utf-8").splitlines():
            capture = json.loads(line)
            captures[capture["slot"]] = capture["status_code"]
    for row in unfinished:
        error = json.loads(row["error_json"] or "{}")
        status_code = error.get("status_code") or captures.get(row["slot"])
        if (
            row["status"] != "error"
            or row["provider"] != "typesafe"
            or row["error_type"] != "TransientLLMError"
            or status_code != 529
            or row["response_json"] is not None
        ):
            raise ValueError("Only a no-answer TypeSafe overload may be resumed")
        available = datetime.fromisoformat(row["finished_utc"]) + timedelta(minutes=10)
        if now < available:
            raise ValueError(f"TypeSafe cooldown lasts until {available.isoformat()}")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarize(output: Path, manifest: dict, rows: dict, budget_snapshot: dict) -> dict:
    attempts = attempt_stats(output / "budget.sqlite")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    arms = {}
    for arm in MODELS:
        matching = [row for row in rows.values() if row["arm"] == arm]
        grades = [item for row in matching for item in row.get("grades", [])]
        charged = sum(attempts[row["slot"]]["charged"] or 0 for row in matching)
        latencies = [
            row["latency_ms"] for row in matching if row["status"] == "success"
        ]
        arms[arm] = {
            "slots": len(matching),
            "valid_slots": sum(row["status"] == "success" for row in matching),
            "invalid_slots": sum(
                row["status"] == "invalid_selection" for row in matching
            ),
            "provider_error_slots": sum(
                row["status"] == "provider_error" for row in matching
            ),
            "graded_candidates": len(grades),
            "adequate": sum(item["adequate"] for item in grades),
            "completed_candidates": sum(
                len(cases[row["case_id"]]["candidates"]) for row in matching
            ),
            "exact": sum(item["exact"] for item in grades),
            "false_support": sum(item["false_support"] for item in grades),
            "false_abstention": sum(item["false_abstention"] for item in grades),
            "p50_ms": statistics.median(latencies) if latencies else None,
            "p95_ms": percentile(latencies, 0.95),
            "conservative_usd": charged / 1e9,
            "reported_usd": sum(
                attempts[row["slot"]]["reported_cost"] or 0 for row in matching
            )
            / 1e9,
            "provider_attempts": sum(
                attempts[row["slot"]]["attempts"] for row in matching
            ),
        }
    result = {
        "cases": len(manifest["cases"]),
        "planned_slots": len(manifest["slots"]),
        "completed_slots": len(rows),
        "arms": arms,
        "budget": budget_snapshot,
    }
    write_json(output / "summary.json", result)
    lines = [
        "# Source quote selection",
        "",
        "Fixed-candidate development comparison; not end-to-end quality.",
        "",
        "| Arm | Valid slots | Adequate / all candidates | Invalid slots | Provider errors | p50 ms | Conservative USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in arms.items():
        lines.append(
            f"| {arm} | {values['valid_slots']} | {values['adequate']}/{values['completed_candidates']} | "
            f"{values['invalid_slots']} | {values['provider_error_slots']} | "
            f"{values['p50_ms']} | {values['conservative_usd']:.6f} |"
        )
    lines += [
        "",
        "Invalid slots must be counted as failures in the complete candidate denominator.",
        "The full evidence arm returns ancillary fields not produced by the selectors.",
        "Per-case references, exact matches, absent support and excess text are in results.jsonl.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


async def run(args: argparse.Namespace) -> None:
    output = args.output_dir
    manifest = verify_freeze(output, args.budget_helper)
    grant = json.loads(args.grant.read_text(encoding="utf-8"))
    if grant.get("freeze_sha256") != digest(output / "freeze.json"):
        raise ValueError("Grant does not authorize this freeze")
    rows = completed_rows(output / "results.jsonl")
    validate_resume(
        output / "budget.sqlite",
        set(rows),
        set(args.resume_transient_attempt),
        datetime.now(timezone.utc),
    )
    settings = Settings.from_env()
    if resolve_component_model(settings, "extractor") != MODELS["current_evidence"]:
        raise ValueError("Configured baseline changed")
    if not examples_enabled_for_component(settings, "extractor"):
        raise ValueError("Configured evidence-card examples changed")
    if not settings.openrouter_api_key or not settings.typesafe_api_key:
        raise ValueError("Both configured provider credentials are required")
    helper = load_helper(args.budget_helper)
    budget = helper.LaneBudget.from_grant(
        args.grant, output / "budget.sqlite", expected_assignment_id=args.assignment_id
    )
    capture_context = {"slot": None}

    async def capture_typesafe(response: httpx.Response) -> None:
        await response.aread()
        with (output / "typesafe_http.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "slot": capture_context["slot"],
                        "status_code": response.status_code,
                        "body": response.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    typesafe_http = httpx.AsyncClient(
        timeout=60,
        follow_redirects=False,
        trust_env=False,
        event_hooks={"response": [capture_typesafe]},
    )
    client = LLMClient(
        providers=[
            OpenRouterProvider(
                settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                request_timeout_seconds=60,
            ),
            TypeSafeProvider(settings.typesafe_api_key, client=typesafe_http),
        ],
        retry_policy=RetryPolicy(attempts=1),
        interactive_retry_policy=RetryPolicy(attempts=1),
        extraction_retry_policy=RetryPolicy(attempts=1),
        structured_output_retry_attempts=0,
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
    )
    helper.install_budget(client, budget)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    try:
        for slot in manifest["slots"]:
            if slot["slot"] in rows:
                continue
            case = cases[slot["case_id"]]
            capture_context["slot"] = slot["slot"]
            started = perf_counter()
            row = {
                **slot,
                "category": case["category"],
                "origin": case["origin"]["kind"],
            }
            try:
                with helper.slot_scope(slot["slot"]):
                    references = await evaluate(client, case, slot["arm"])
                row.update(status="success", grades=grade(case, references))
            except ValueError as exc:
                row.update(
                    status="invalid_selection",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except LLMError as exc:
                if str(exc) != "TypeSafe returned an invalid choice distribution":
                    write_json(
                        output / "stopped.json",
                        {
                            **row,
                            "status": "technical_failure",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise
                # Diagnosed API-contract failures are terminal trials, never repaired
                # or retried. Raw HTTP bodies remain available for separate analysis.
                row.update(
                    status="provider_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except Exception as exc:
                write_json(
                    output / "stopped.json",
                    {
                        **row,
                        "status": "technical_failure",
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            row["latency_ms"] = (perf_counter() - started) * 1000
            with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows[row["slot"]] = row
            write_json(
                output / "progress.json",
                {
                    "completed": len(rows),
                    "planned": len(manifest["slots"]),
                    "last_slot": row["slot"],
                    "budget": budget.snapshot(),
                },
            )
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "planned": len(manifest["slots"]),
                        "slot": row["slot"],
                        "status": row["status"],
                    }
                ),
                flush=True,
            )
    finally:
        await client.aclose()
        await typesafe_http.aclose()
        summarize(output, manifest, rows, budget.snapshot())
        budget.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-helper", type=Path, required=True)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--assignment-id", default="20260924-source-quotes-v1")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume-transient-attempt", action="append", default=[])
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output_dir, args.budget_helper)
    else:
        if args.grant is None:
            parser.error("Paid execution requires a grant")
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
