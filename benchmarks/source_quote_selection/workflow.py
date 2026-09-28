"""Bounded smoke and acceptance checks through the production evidence workflow."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter

import httpx
from openai import AsyncOpenAI

from atagia.core.config import Settings
from atagia.core.source_references import SourceReferenceCatalog
from atagia.memory.extraction_cards import CandidateDraft, run_evidence_card
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import LLMClient, RetryPolicy
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.providers.openrouter import OpenRouterProvider
from atagia.services.providers.typesafe import TypeSafeProvider
from benchmarks.source_quote_selection.run import (
    ROOT,
    attempt_stats,
    completed_rows,
    digest,
    grade,
    load_helper,
    write_json,
)

ASSIGNMENT = "20260924-source-quote-workflow-v1"
ARMS = {
    "full_luna56": ("openrouter/openai/gpt-5.6-luna", "openrouter/openai/gpt-5.6-luna"),
    "full_luna6": ("openrouter/openai/gpt-6-luna", "openrouter/openai/gpt-6-luna"),
    "jev_with_luna6": ("openrouter/openai/gpt-6-luna", "typesafe/jev-1.13.0"),
}
SOURCES = (
    "src/atagia/memory/extraction_cards.py",
    "src/atagia/memory/evidence_cards.py",
    "src/atagia/memory/source_quote_selector.py",
    "src/atagia/memory/extractor.py",
    "src/atagia/core/source_references.py",
    "src/atagia/services/model_resolution.py",
    "src/atagia/services/model_profiles.py",
    "src/atagia/services/providers/typesafe.py",
    "src/atagia/services/providers/openrouter.py",
    "src/atagia/services/providers/openai.py",
    "src/atagia/services/llm_client.py",
    "benchmarks/source_quote_selection/workflow.py",
    "benchmarks/source_quote_selection/run.py",
    "benchmarks/source_quote_selection/acceptance_cases.py",
)


def smoke_cases() -> list[dict]:
    """Small development probes for contract and wiring, separate from acceptance."""
    cases = [
        {
            "case_id": "smoke_literal",
            "source_text": "Keep the reference ZX-19/B.",
            "recent_messages": [],
            "candidates": [
                {
                    "candidate_id": "cand_001",
                    "canonical_text": "The reference is ZX-19/B.",
                }
            ],
        },
        {
            "case_id": "smoke_reply",
            "source_text": "The second one, please.",
            "recent_messages": [
                {
                    "role": "assistant",
                    "content": "For the badge color, would you prefer orange or violet?",
                }
            ],
            "candidates": [
                {
                    "candidate_id": "cand_001",
                    "canonical_text": "The user chose violet for the badge color.",
                }
            ],
        },
        {
            "case_id": "smoke_mixed",
            "source_text": "The audit is finished; the import has not started.",
            "recent_messages": [],
            "candidates": [
                {
                    "candidate_id": "cand_001",
                    "canonical_text": "The audit is finished.",
                },
                {
                    "candidate_id": "cand_002",
                    "canonical_text": "The import is finished.",
                },
            ],
        },
    ]
    source = "\n".join(f"Item {n}: received {n + 3} samples." for n in range(35))
    cases.append(
        {
            "case_id": "smoke_blocks",
            "source_text": source + "\nThe archive key is DR-28/A.",
            "recent_messages": [],
            "candidates": [
                {
                    "candidate_id": "cand_001",
                    "canonical_text": "The archive key is DR-28/A.",
                }
            ],
        }
    )
    return cases


async def evaluate(client: LLMClient, case: dict, arm: str) -> dict:
    """Invoke the same complete evidence operation used by MemoryExtractor."""
    model, evidence_model = ARMS[arm]
    candidates = tuple(
        CandidateDraft(
            candidate_id=c["candidate_id"], canonical_text=c["canonical_text"]
        )
        for c in case["candidates"]
    )
    context = ExtractionConversationContext(
        user_id="workflow_check",
        conversation_id="workflow_conversation",
        source_message_id=case["case_id"],
        assistant_mode_id="coding_debug",
        mode="coding_debug",
        privacy_enforcement="off",
        recent_messages=case.get("recent_messages", []),
    )
    catalog = SourceReferenceCatalog(case["source_text"])
    result = await run_evidence_card(
        client,
        model=model,
        evidence_model=evidence_model,
        message_text=case["source_text"],
        role=case.get("role", "user"),
        context=context,
        resolved_policy=None,
        allowed_write_scopes=("chat", "user"),
        occurred_at=None,
        prior_chunk_context=case.get("prior_chunk_context"),
        candidates=candidates,
        source_catalog=catalog,
        include_examples=True,
        metadata={},
        semaphore=asyncio.Semaphore(2),
    )
    if set(result.parsed) != {c.candidate_id for c in candidates}:
        raise ValueError("Evidence workflow lost or invented a candidate")
    references = {}
    for candidate_id, row in result.parsed.items():
        pair = row["start_ref"], row["end_ref"]
        if pair == (None, None):
            references[candidate_id] = None
        else:
            references[candidate_id] = catalog.resolve(*pair)
    return {
        "evidence": result.parsed,
        "references": {
            key: value.model_dump() if value else None
            for key, value in references.items()
        },
        "quotes": {
            key: value.quote(case["source_text"]) if value else None
            for key, value in references.items()
        },
        "grades": grade(case, references)
        if "expected_ranges" in case["candidates"][0]
        else [],
    }


def prepare(output: Path, helper: Path) -> None:
    from benchmarks.source_quote_selection.acceptance_cases import load_cases

    if output.exists():
        raise ValueError("Never overwrite a workflow run")
    output.mkdir(parents=True)
    phases = {"smoke": smoke_cases(), "acceptance": load_cases()}
    slots = []
    for phase, cases in phases.items():
        for repetition in range(1, (1 if phase == "smoke" else 2) + 1):
            for index, case in enumerate(cases):
                arms = list(ARMS)
                offset = (index + repetition) % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    slots.append(
                        {
                            "slot": f"{phase}:{case['case_id']}:{arm}:{repetition}",
                            "phase": phase,
                            "case_id": case["case_id"],
                            "arm": arm,
                            "repetition": repetition,
                        }
                    )
    write_json(
        output / "manifest.json",
        {
            "arms": ARMS,
            "cases": phases,
            "slots": slots,
            "scope": "Complete evidence card, fixed candidates; not full ingestion",
        },
    )
    write_json(
        output / "freeze.json",
        {
            "sources": {path: digest(ROOT / path) for path in SOURCES},
            "helper": digest(helper),
            "manifest": digest(output / "manifest.json"),
        },
    )


def verify(output: Path, helper: Path) -> dict:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if freeze["sources"] != {path: digest(ROOT / path) for path in SOURCES}:
        raise ValueError("Frozen workflow code changed")
    if freeze["helper"] != digest(helper) or freeze["manifest"] != digest(
        output / "manifest.json"
    ):
        raise ValueError("Frozen inputs or budget helper changed")
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def validate_prior_failures(output: Path, rows: dict[str, dict]) -> None:
    """Continue only after explicit diagnosis, retaining failed terminal slots."""
    failures = {key: row for key, row in rows.items() if row["status"] != "success"}
    if not failures:
        return
    acknowledgement = output / "acknowledged_failures.json"
    acknowledged = (
        json.loads(acknowledgement.read_text(encoding="utf-8")).get("rows", {})
        if acknowledgement.exists()
        else {}
    )
    if any(acknowledged.get(key) != row for key, row in failures.items()):
        raise ValueError("A recorded workflow failure requires coordinator diagnosis")


async def run(output: Path, helper_path: Path, phase: str) -> None:
    manifest = verify(output, helper_path)
    if phase == "acceptance":
        approval = json.loads(
            (output / "preflight_approved.json").read_text(encoding="utf-8")
        )
        if approval.get("freeze_sha256") != digest(
            output / "freeze.json"
        ) or not approval.get("approved"):
            raise ValueError(
                "Acceptance requires explicit review of this freeze's smoke"
            )
    rows = completed_rows(output / "results.jsonl")
    # Acknowledged failures remain terminal: their slots are never replayed.
    validate_prior_failures(output, rows)
    if (output / "budget.sqlite").exists():
        stats = attempt_stats(output / "budget.sqlite")
        if set(stats) - set(rows) or any(row["pending"] for row in stats.values()):
            raise ValueError("Unreconciled provider attempts prohibit resume")
    helper = load_helper(helper_path)
    grant_path = output / "grant.json"
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    if grant.get("freeze_sha256") != digest(output / "freeze.json"):
        raise ValueError("Grant does not authorize this freeze")
    budget = helper.LaneBudget.from_grant(
        grant_path, output / "budget.sqlite", expected_assignment_id=ASSIGNMENT
    )
    settings = Settings.from_env()
    capture_context = {"slot": None}

    async def capture(response: httpx.Response) -> None:
        await response.aread()
        with (output / "http.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "slot": capture_context["slot"],
                        "host": response.request.url.host,
                        "status": response.status_code,
                        "body": response.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    http = httpx.AsyncClient(
        timeout=60, trust_env=False, event_hooks={"response": [capture]}
    )
    sdk = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        http_client=http,
        timeout=60,
        max_retries=0,
        default_headers={
            "HTTP-Referer": settings.openrouter_site_url,
            "X-Title": settings.openrouter_app_name,
        },
    )
    client = LLMClient(
        providers=[
            OpenRouterProvider(
                settings.openrouter_api_key,
                client=sdk,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
            ),
            TypeSafeProvider(settings.typesafe_api_key, client=http),
        ],
        retry_policy=RetryPolicy(attempts=1),
        interactive_retry_policy=RetryPolicy(attempts=1),
        extraction_retry_policy=RetryPolicy(attempts=1),
        structured_output_retry_attempts=0,
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
    )
    helper.install_budget(client, budget)
    cases = {case["case_id"]: case for case in manifest["cases"][phase]}
    try:
        for slot in manifest["slots"]:
            if slot["phase"] != phase or slot["slot"] in rows:
                continue
            capture_context["slot"] = slot["slot"]
            started = perf_counter()
            row = {**slot, "started_utc": datetime.now(timezone.utc).isoformat()}
            try:
                with helper.slot_scope(slot["slot"]):
                    result = await evaluate(client, cases[slot["case_id"]], slot["arm"])
                row.update(status="success", **result)
            except BaseException as exc:
                row.update(
                    status="technical_failure",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            finally:
                row["latency_ms"] = (perf_counter() - started) * 1000
                with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows[slot["slot"]] = row
                write_json(
                    output / "progress.json",
                    {
                        "phase": phase,
                        "completed": len(rows),
                        "budget": budget.snapshot(),
                    },
                )
                print(
                    json.dumps(
                        {
                            "slot": slot["slot"],
                            "status": row["status"],
                            "completed": len(rows),
                        }
                    ),
                    flush=True,
                )
    finally:
        stats = attempt_stats(output / "budget.sqlite")
        summary = {}
        for arm in ARMS:
            matching = [
                row
                for row in rows.values()
                if row["phase"] == phase and row["arm"] == arm
            ]
            grades = [g for row in matching for g in row.get("grades", [])]
            times = [row["latency_ms"] for row in matching]
            summary[arm] = {
                "slots": len(matching),
                "statuses": dict(Counter(row["status"] for row in matching)),
                "adequate": sum(g["adequate"] for g in grades),
                "graded_candidates": len(grades),
                "false_support": sum(g["false_support"] for g in grades),
                "false_abstention": sum(g["false_abstention"] for g in grades),
                "median_ms": statistics.median(times) if times else None,
                "max_ms": max(times) if times else None,
                "conservative_usd": sum(
                    stats.get(row["slot"], {}).get("charged", 0) for row in matching
                )
                / 1e9,
                "attempts": sum(
                    stats.get(row["slot"], {}).get("attempts", 0) for row in matching
                ),
            }
        write_json(
            output / f"{phase}_readout.json",
            {"arms": summary, "budget": budget.snapshot()},
        )
        await client.aclose()
        await sdk.close()
        await http.aclose()
        budget.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-helper", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "smoke_native", "acceptance"))
    args = parser.parse_args()
    if args.phase:
        asyncio.run(run(args.output, args.budget_helper, args.phase))
    else:
        prepare(args.output, args.budget_helper)


if __name__ == "__main__":
    main()
