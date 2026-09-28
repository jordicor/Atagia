"""Frozen, budgeted comparison of the production extraction workflow."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter

import httpx
from openai import AsyncOpenAI

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.source_references import source_sha256
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    sync_assistant_modes,
)
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    ExtractionContextMessage,
)
from atagia.services.llm_client import LLMClient, RetryPolicy
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.providers.openrouter import OpenRouterProvider
from atagia.services.providers.typesafe import TypeSafeProvider
from benchmarks.extractor_workflow.cases import load_cases, load_smoke_cases
from benchmarks.source_quote_selection.run import (
    attempt_stats,
    completed_rows,
    digest,
    load_helper,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "src/atagia/resources/migrations"
MANIFESTS = ROOT / "src/atagia/resources/manifests"
ARMS = {
    "luna56": "openrouter/openai/gpt-5.6-luna",
    "luna6": "openrouter/openai/gpt-6-luna",
    "luna6_jev_classification": "openrouter/openai/gpt-6-luna",
}
EVIDENCE_MODEL = "openrouter/openai/gpt-6-luna"
AUXILIARY_MODEL = "openrouter/openai/gpt-5.6-luna"
JEV_MODEL = "typesafe/jev-1.13.0"
ASSIGNMENT = "20260924-extractor-workflow-v1"
CLOCK_TIME = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
SOURCE_FILES = (
    "benchmarks/extractor_workflow/run.py",
    "benchmarks/extractor_workflow/cases.py",
    "benchmarks/source_quote_selection/run.py",
    "src/atagia/core/config.py",
    "src/atagia/core/db_sqlite.py",
    "src/atagia/core/repositories.py",
    "src/atagia/core/source_references.py",
    "src/atagia/memory/extractor.py",
    "src/atagia/memory/extraction_mapping.py",
    "src/atagia/memory/extraction_cards.py",
    "src/atagia/memory/extraction_temporal.py",
    "src/atagia/memory/evidence_cards.py",
    "src/atagia/memory/coverage_members_card.py",
    "src/atagia/memory/extraction_watchdog.py",
    "src/atagia/memory/chunking_config.py",
    "src/atagia/memory/card_prompt.py",
    "src/atagia/memory/claim_keys.py",
    "src/atagia/memory/intent_classifier.py",
    "src/atagia/memory/source_quote_selector.py",
    "src/atagia/memory/text_chunker.py",
    "src/atagia/memory/policy_manifest.py",
    "src/atagia/models/schemas_memory.py",
    "src/atagia/models/schemas_decisions.py",
    "src/atagia/services/llm_client.py",
    "src/atagia/services/llm_reliability.py",
    "src/atagia/services/model_resolution.py",
    "src/atagia/services/model_profiles.py",
    "src/atagia/services/providers/openai.py",
    "src/atagia/services/providers/openrouter.py",
    "src/atagia/services/providers/typesafe.py",
)


def isolated_settings(arm: str) -> Settings:
    """Use explicit benchmark settings, never ambient extraction overrides."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS),
        manifests_path=str(MANIFESTS),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia extractor workflow benchmark",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_component_models={
            "extractor": ARMS[arm],
            **({
                "extraction_kind": JEV_MODEL,
                "extraction_scope": JEV_MODEL,
                "extraction_confidence": JEV_MODEL,
            } if arm == "luna6_jev_classification" else {}),
            "extraction_evidence": EVIDENCE_MODEL,
            "text_chunker": AUXILIARY_MODEL,
            "intent_classifier": AUXILIARY_MODEL,
            "belief_reviser": AUXILIARY_MODEL,
            "extraction_watchdog": AUXILIARY_MODEL,
        },
        llm_max_concurrent_requests_per_provider=2,
        llm_finite_decisions_enabled=arm == "luna6_jev_classification",
        llm_structured_output_retry_attempts=0,
        llm_structured_output_rescue_enabled=False,
        opf_privacy_filter_enabled=False,
        disable_chunking_extraction=False,
        retrieval_packets_dry_run_enabled=False,
        retrieval_packets_write_enabled=False,
        fact_facet_surfaces_enabled=False,
        graph_projection_enabled=False,
    )


def _cases() -> tuple[list[dict], list[dict]]:
    smoke, evaluation = load_smoke_cases(), load_cases()
    ids = [case["case_id"] for case in (*smoke, *evaluation)]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique across phases")
    for case in (*smoke, *evaluation):
        if case.get("prior_chunk_context") is not None:
            raise ValueError(
                f"Case {case['case_id']} has synthetic prior chunk context"
            )
        if not case["source_text"] or case["role"] not in {"user", "assistant"}:
            raise ValueError(f"Invalid source case: {case['case_id']}")
    return smoke, evaluation


def _slots(cases: list[dict], repetitions: int, phase: str) -> list[dict]:
    slots = []
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            arms = tuple(ARMS)
            offset = (index + repetition - 1) % len(arms)
            arms = arms[offset:] + arms[:offset]
            for arm in arms:
                slots.append(
                    {
                        "slot": f"{phase}:{case['case_id']}:{arm}:{repetition:02}",
                        "phase": phase,
                        "case_id": case["case_id"],
                        "arm": arm,
                        "repetition": repetition,
                    }
                )
    return slots


def prepare(output: Path, budget_helper: Path) -> dict:
    if not budget_helper.is_file():
        raise ValueError("Budget helper is missing")
    forbidden = ("manifest.json", "freeze.json", "budget.sqlite", "results.jsonl")
    if any((output / name).exists() for name in forbidden):
        raise ValueError("Frozen or paid experiment cannot be overwritten")
    if output.exists() and any(
        path.name
        not in {
            "price_verification.json",
            "coordinator_control.json",
            "grant.pending.json",
        }
        and not path.name.endswith("_endpoints.json")
        for path in output.iterdir()
    ):
        raise ValueError("Existing experiment directory has unrecognized files")
    smoke, evaluation = _cases()
    manifest = {
        "assignment_id": ASSIGNMENT,
        "arms": ARMS,
        "evidence_model": EVIDENCE_MODEL,
        "auxiliary_model": AUXILIARY_MODEL,
        "jev_model": JEV_MODEL,
        "clock_utc": CLOCK_TIME.isoformat(),
        "smoke_cases": smoke,
        "evaluation_cases": evaluation,
        "slots": _slots(smoke, 1, "smoke") + _slots(evaluation, 10, "evaluation"),
        "semantic_labels_are_manual": True,
        "workflow": "MemoryExtractor.extract_with_persistence_and_chunk_plan",
    }
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in SOURCE_FILES:
        source = ROOT / name
        if not source.is_file():
            raise ValueError(f"Frozen source missing: {name}")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[name] = digest(target)
    for source in sorted((*MIGRATIONS.glob("*.sql"), *MANIFESTS.glob("*.json"))):
        name = source.relative_to(ROOT).as_posix()
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[name] = digest(target)
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "freeze.json",
        {
            "manifest_sha256": digest(output / "manifest.json"),
            "budget_helper_sha256": digest(budget_helper),
            "source_hashes": source_hashes,
        },
    )
    return manifest


def verify_freeze(output: Path, budget_helper: Path) -> dict:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if digest(output / "manifest.json") != freeze["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    if digest(budget_helper) != freeze["budget_helper_sha256"]:
        raise ValueError("Budget helper changed")
    if set(freeze["source_hashes"]) != set(SOURCE_FILES) | {
        path.relative_to(ROOT).as_posix()
        for path in (*MIGRATIONS.glob("*.sql"), *MANIFESTS.glob("*.json"))
    }:
        raise ValueError("Frozen source set changed")
    for name, expected in freeze["source_hashes"].items():
        if (
            digest(ROOT / name) != expected
            or digest(output / "source_snapshot" / name) != expected
        ):
            raise ValueError(f"Frozen source changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["assignment_id"] != ASSIGNMENT:
        raise ValueError("Wrong experiment assignment")
    return manifest


def _approval(output: Path, phase: str) -> None:
    path = output / f"{phase}_approval.json"
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval != {"freeze_sha256": digest(output / "freeze.json"), "approved": True}:
        raise ValueError(f"{phase} approval does not match the frozen experiment")


async def _run_slot(client: LLMClient, case: dict, arm: str) -> dict:
    """Build one fresh production runtime and extract through SQLite."""
    started_setup = perf_counter()
    connection = await initialize_database(":memory:", MIGRATIONS)
    try:
        clock = FrozenClock(CLOCK_TIME)
        manifests = ManifestLoader(MANIFESTS).load_all()
        await sync_assistant_modes(connection, manifests, clock)
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        messages = MessageRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        await users.create_user("benchmark_user")
        await conversations.create_conversation(
            "benchmark_conversation",
            "benchmark_user",
            None,
            "coding_debug",
            "Benchmark",
        )
        recent = []
        seq = 1
        for item in case["recent_messages"]:
            message_id = f"recent_{seq}"
            await messages.create_message(
                message_id,
                "benchmark_conversation",
                item["role"],
                seq,
                item["content"],
                occurred_at=CLOCK_TIME.isoformat(),
            )
            recent.append(
                ExtractionContextMessage(
                    id=message_id,
                    role=item["role"],
                    content=item["content"],
                    seq=seq,
                    occurred_at=CLOCK_TIME.isoformat(),
                )
            )
            seq += 1
        await messages.create_message(
            "source_message",
            "benchmark_conversation",
            case["role"],
            seq,
            case["source_text"],
            occurred_at=CLOCK_TIME.isoformat(),
        )
        settings = isolated_settings(arm)
        extractor = MemoryExtractor(
            llm_client=client,
            clock=clock,
            message_repository=messages,
            memory_repository=memories,
            storage_backend=InProcessBackend(),
            settings=settings,
        )
        policy = PolicyResolver().resolve(manifests["coding_debug"], None, None)
        context = ExtractionConversationContext(
            user_id="benchmark_user",
            conversation_id="benchmark_conversation",
            source_message_id="source_message",
            assistant_mode_id="coding_debug",
            mode="coding_debug",
            recent_messages=recent,
            privacy_enforcement="off",
        )
        setup_ms = (perf_counter() - started_setup) * 1000
        started_extract = perf_counter()
        before_cards = asyncio.all_tasks()
        try:
            details = await extractor.extract_with_persistence_and_chunk_plan(
                message_text=case["source_text"],
                role=case["role"],
                conversation_context=context,
                resolved_policy=policy,
                occurred_at=CLOCK_TIME.isoformat(),
            )
        finally:
            await _drain_new_tasks(before_cards)
        extraction_ms = (perf_counter() - started_extract) * 1000
        source = case["source_text"]
        reference_checks = []
        for item in (
            *details.result.evidences,
            *details.result.beliefs,
            *details.result.contract_signals,
            *details.result.state_updates,
        ):
            ref = item.source_reference
            if ref is None:
                raise ValueError("Extracted item lacks a source reference")
            literal = ref.quote(source)
            reference_checks.append(
                {
                    "memory_text": item.canonical_text,
                    "reference": ref.model_dump(mode="json"),
                    "literal_quote": literal,
                    "quote_matches": item.source_quote == literal,
                }
            )
            if item.source_quote != literal:
                raise ValueError("Extracted source quote does not match exact source")
        connection.row_factory = sqlite3.Row
        rows = {}
        for table in (
            "memory_objects",
            "memory_support_edges",
            "memory_evidence_spans",
        ):
            cursor = await connection.execute(f"SELECT * FROM {table} ORDER BY id")
            rows[table] = [dict(row) for row in await cursor.fetchall()]
        for span in rows["memory_evidence_spans"]:
            if span["message_id"] == "source_message":
                start, end = span["char_start"], span["char_end"]
                if (
                    start is None
                    or end is None
                    or source[start:end] != span["quote_text"]
                ):
                    raise ValueError("Persisted source span has wrong coordinates")
        return {
            "setup_ms": setup_ms,
            "extraction_persistence_ms": extraction_ms,
            "raw_extraction": details.result.model_dump(mode="json"),
            "persisted": details.persisted,
            "database_rows": rows,
            "source_reference_checks": reference_checks,
            "source_sha256": source_sha256(source),
            "chunk_plan": asdict(details.chunk_plan),
            "chunk_count": len(details.chunk_plan.chunks),
            "grounding_dropped_count": details.grounding_dropped_count,
        }
    finally:
        await close_connection(connection)


async def _drain_new_tasks(before: set[asyncio.Task]) -> None:
    """Settle card fan-out and provider charges before closing a failed slot."""
    current = asyncio.current_task()
    for _ in range(3):
        pending = {
            task
            for task in asyncio.all_tasks()
            if task not in before and task is not current and not task.done()
        }
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise RuntimeError("Extraction left in-flight tasks after drain")


def _journal_calls(path: Path, slot: str) -> list[dict]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute(
                "SELECT a.*, p.request_json, p.response_json, p.error_json "
                "FROM attempts a LEFT JOIN attempt_payloads p ON p.attempt_id=a.id "
                "WHERE a.slot=? ORDER BY a.started_utc, a.id",
                (slot,),
            )
        ]


def _append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _validate_prior_failures(output: Path, rows: dict, freeze_sha256: str) -> None:
    failed = {slot for slot, row in rows.items() if row["status"] != "success"}
    stopped = output / "stopped.json"
    if stopped.exists():
        stopped_slot = json.loads(stopped.read_text(encoding="utf-8"))["slot"]
        if stopped_slot not in failed:
            raise ValueError(
                "Stopped slot lacks a terminal result; reconcile paid attempts"
            )
    if not failed:
        return
    acknowledgment = output / "failure_ack.json"
    if not acknowledgment.exists():
        raise ValueError("Prior technical failure requires coordinator acknowledgment")
    value = json.loads(acknowledgment.read_text(encoding="utf-8"))
    if (
        value.get("freeze_sha256") != freeze_sha256
        or set(value.get("acknowledged_slots", [])) != failed
        or not isinstance(value.get("reason"), str)
        or not value["reason"].strip()
    ):
        raise ValueError("Failure acknowledgment does not match terminal failures")


def _readout(output: Path, manifest: dict, budget_snapshot: dict, grant: dict) -> None:
    rows = completed_rows(output / "results.jsonl")
    stats = attempt_stats(output / "budget.sqlite")
    jev_price = next(
        (
            Decimal(str(entry["input_per_million"]))
            for entry in grant["prices"]
            if entry["provider"] == "typesafe"
            and entry["model"] == JEV_MODEL.removeprefix("typesafe/")
        ),
        None,
    )
    summary = {}
    for phase in ("smoke", "evaluation"):
        for arm in ARMS:
            selected = [
                row
                for row in rows.values()
                if row["phase"] == phase and row["arm"] == arm
            ]
            jev_calls = [
                call
                for row in selected
                for call in row["provider_calls"]
                if call["provider"] == "typesafe"
            ]
            summary[f"{phase}:{arm}"] = {
                "slots": len(selected),
                "statuses": dict(Counter(row["status"] for row in selected)),
                "provider_reported_usd_known": sum(
                    (stats.get(row["slot"]) or {}).get("reported_cost") or 0
                    for row in selected
                )
                / 1e9,
                "jev_input_estimate_usd": (
                    str(
                        sum(
                            Decimal(call["input_tokens"] or 0)
                            * jev_price
                            / Decimal(1_000_000)
                            for call in jev_calls
                        )
                    )
                    if jev_price is not None
                    else None
                ),
                "missing_provider_reported_cost_calls": sum(
                    call["reported_cost_nano"] is None
                    for row in selected
                    for call in row["provider_calls"]
                ),
                "conservative_usd": sum(
                    (stats.get(row["slot"]) or {}).get("charged") or 0
                    for row in selected
                )
                / 1e9,
                "provider_attempts": sum(
                    (stats.get(row["slot"]) or {}).get("attempts") or 0
                    for row in selected
                ),
            }
    write_json(
        output / "readout.json",
        {
            "planned_slots": len(manifest["slots"]),
            "terminal_slots": len(rows),
            "summary": summary,
            "budget": budget_snapshot,
            "interpretation": "Mechanical reference checks only; semantic labels require manual review.",
        },
    )


async def run(output: Path, helper_path: Path, grant_path: Path, phase: str) -> None:
    manifest = verify_freeze(output, helper_path)
    if phase not in {"smoke", "evaluation"}:
        raise ValueError("Unknown phase")
    _approval(output, phase)
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    if grant.get("freeze_sha256") != digest(output / "freeze.json"):
        raise ValueError("Grant does not authorize this freeze")
    rows = completed_rows(output / "results.jsonl")
    known_slots = {slot["slot"] for slot in manifest["slots"]}
    if set(rows) - known_slots:
        raise ValueError("Unknown terminal slot in results")
    attempts = attempt_stats(output / "budget.sqlite")
    if set(attempts) - set(rows):
        raise ValueError("Unreconciled paid attempt; terminal result is missing")
    if any(stats["pending"] for stats in attempts.values()):
        raise ValueError("Unsettled provider reservation")
    _validate_prior_failures(output, rows, digest(output / "freeze.json"))
    if phase == "evaluation" and any(
        slot["slot"] not in rows or rows[slot["slot"]]["status"] != "success"
        for slot in manifest["slots"]
        if slot["phase"] == "smoke"
    ):
        raise ValueError("Smoke phase must complete before evaluation")
    credentials = Settings.from_env()
    if not credentials.openrouter_api_key or not credentials.typesafe_api_key:
        raise ValueError("OpenRouter and TypeSafe credentials are required")
    helper = load_helper(helper_path)
    budget = helper.LaneBudget.from_grant(
        grant_path,
        output / "budget.sqlite",
        expected_assignment_id=ASSIGNMENT,
    )
    capture_slot: dict[str, str | None] = {"slot": None}

    async def capture_request(request: httpx.Request) -> None:
        body = await request.aread()
        _append_row(
            output / "openrouter_http.jsonl",
            {
                "slot": capture_slot["slot"],
                "event": "request",
                "method": request.method,
                "url": str(request.url),
                "body": body.decode("utf-8", errors="replace"),
            },
        )

    async def capture_response(response: httpx.Response) -> None:
        body = await response.aread()
        _append_row(
            output / "openrouter_http.jsonl",
            {
                "slot": capture_slot["slot"],
                "event": "response",
                "status_code": response.status_code,
                "body": body.decode("utf-8", errors="replace"),
            },
        )

    async def capture_typesafe_request(request: httpx.Request) -> None:
        body = await request.aread()
        _append_row(
            output / "typesafe_http.jsonl",
            {
                "slot": capture_slot["slot"],
                "event": "request",
                "method": request.method,
                "url": str(request.url),
                "body": body.decode("utf-8", errors="replace"),
            },
        )

    async def capture_typesafe_response(response: httpx.Response) -> None:
        body = await response.aread()
        _append_row(
            output / "typesafe_http.jsonl",
            {
                "slot": capture_slot["slot"],
                "event": "response",
                "status_code": response.status_code,
                "body": body.decode("utf-8", errors="replace"),
            },
        )

    http_client = httpx.AsyncClient(
        timeout=120,
        follow_redirects=True,
        trust_env=False,
        event_hooks={"request": [capture_request], "response": [capture_response]},
    )
    typesafe_http = httpx.AsyncClient(
        timeout=120,
        follow_redirects=False,
        trust_env=False,
        event_hooks={
            "request": [capture_typesafe_request],
            "response": [capture_typesafe_response],
        },
    )
    sdk_client = AsyncOpenAI(
        api_key=credentials.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": credentials.openrouter_site_url,
            "X-Title": credentials.openrouter_app_name,
        },
        timeout=120,
        max_retries=0,
        http_client=http_client,
    )
    client = LLMClient(
        providers=[
            OpenRouterProvider(
                credentials.openrouter_api_key,
                site_url=credentials.openrouter_site_url,
                app_name=credentials.openrouter_app_name,
                client=sdk_client,
                request_timeout_seconds=120,
            ),
            TypeSafeProvider(
                credentials.typesafe_api_key,
                request_timeout_seconds=120,
                client=typesafe_http,
            ),
        ],
        retry_policy=RetryPolicy(attempts=1),
        interactive_retry_policy=RetryPolicy(attempts=1),
        extraction_retry_policy=RetryPolicy(attempts=1),
        structured_output_retry_attempts=0,
        structured_output_rescue_enabled=False,
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
        max_concurrent_requests_per_provider=2,
    )
    helper.install_budget(client, budget)
    cases = {
        case["case_id"]: case
        for case in (*manifest["smoke_cases"], *manifest["evaluation_cases"])
    }
    try:
        for slot in manifest["slots"]:
            if slot["phase"] != phase or slot["slot"] in rows:
                continue
            case = cases[slot["case_id"]]
            capture_slot["slot"] = slot["slot"]
            before = asyncio.all_tasks()
            started = perf_counter()
            error = None
            details = None
            try:
                with helper.slot_scope(slot["slot"]):
                    details = await _run_slot(client, case, slot["arm"])
            except BaseException as exc:
                error = exc
            finally:
                await _drain_new_tasks(before)
            calls = _journal_calls(output / "budget.sqlite", slot["slot"])
            row = {
                **slot,
                "model": ARMS[slot["arm"]],
                "evidence_model": EVIDENCE_MODEL,
                "wall_ms": (perf_counter() - started) * 1000,
                "status": "success" if error is None else "error",
                "error_type": type(error).__name__ if error else None,
                "error": str(error) if error else None,
                "details": details,
                "provider_calls": [
                    {
                        "attempt_id": call["id"],
                        "purpose": (
                            json.loads(call["request_json"] or "{}").get("metadata")
                            or {}
                        ).get("purpose"),
                        "provider": call["provider"],
                        "model": call["model"],
                        "status": call["status"],
                        "reported_cost_nano": call["reported_cost"],
                        "conservative_debit_nano": call["charged"],
                        "input_tokens": call["input_tokens"],
                        "output_tokens": call["output_tokens"],
                    }
                    for call in calls
                ],
            }
            _append_row(output / "results.jsonl", row)
            rows[slot["slot"]] = row
            _readout(output, manifest, budget.snapshot(), grant)
            if error is not None:
                write_json(
                    output / "stopped.json",
                    {
                        "slot": slot["slot"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "attempt_ids": [call["id"] for call in calls],
                    },
                )
                raise RuntimeError(f"Technical failure in {slot['slot']}") from error
    finally:
        await client.aclose()
        await sdk_client.close()
        await http_client.aclose()
        await typesafe_http.aclose()
        budget.close()
        capture_slot["slot"] = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-helper", type=Path, required=True)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--phase", choices=("smoke", "evaluation"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.budget_helper)
    elif args.command == "verify":
        verify_freeze(args.output, args.budget_helper)
    else:
        if args.grant is None or args.phase is None:
            parser.error("run requires --grant and --phase")
        asyncio.run(run(args.output, args.budget_helper, args.grant, args.phase))


if __name__ == "__main__":
    main()
