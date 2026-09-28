"""Append a model-only GPT-6-Luna comparison to the preserved quote experiment."""

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
from atagia.services.llm_client import LLMClient, RetryPolicy, TransientLLMError
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.model_profiles import MODEL_PROFILES, ModelProfile
from atagia.services.providers.openrouter import OpenRouterProvider
from benchmarks.source_quote_selection import run as original
from benchmarks.source_quote_selection.readout import punctuation_boundary_diagnostic

MODEL = "openrouter/openai/gpt-6-luna"
ARMS = ("current_evidence", "luna_reference")
ASSIGNMENT = "20260924-source-quotes-luna6-v1"


class ModelOnlyClient:
    """Keep the imported builders and parsers; change only their target model."""

    def __init__(self, client):
        self.client = client

    async def complete(self, request):
        if request.model != original.MODELS["current_evidence"]:
            raise ValueError("Unexpected baseline model in the model-only comparison")
        return await self.client.complete(request.model_copy(update={"model": MODEL}))

    async def complete_choice_questions(self, *, model, **kwargs):
        if model != original.MODELS["current_evidence"]:
            raise ValueError("Unexpected baseline model in the model-only comparison")
        return await self.client.complete_choice_questions(model=MODEL, **kwargs)


def prepare(output: Path, baseline: Path, helper: Path) -> None:
    if output.exists():
        raise ValueError("The append-only experiment directory already exists")
    old = original.verify_freeze(baseline, helper)
    if len(original.completed_rows(baseline / "results.jsonl")) != len(old["slots"]):
        raise ValueError("The baseline experiment is not complete")
    slots = [slot for slot in old["slots"] if slot["arm"] in ARMS]
    manifest = {
        **old,
        "models": {arm: MODEL for arm in ARMS},
        "slots": slots,
        "baseline_directory": str(baseline.resolve()),
        "baseline_models": old["models"],
        "reasoning_effort": "none",
        "temperature": "omitted, matching the original OpenRouter GPT-5.6 wire request",
        "comparison": "Append-only model comparison; old results are not rerun or replaced",
    }
    output.mkdir(parents=True)
    original.write_json(output / "manifest.json", manifest)
    old_freeze = json.loads((baseline / "freeze.json").read_text(encoding="utf-8"))
    source_hashes = {
        **old_freeze["source_hashes"],
        "benchmarks/source_quote_selection/run_luna6.py": original.digest(
            Path(__file__)
        ),
        "benchmarks/source_quote_selection/readout.py": original.digest(
            original.ROOT / "benchmarks/source_quote_selection/readout.py"
        ),
    }
    original.write_json(
        output / "freeze.json",
        {
            "source_hashes": source_hashes,
            "manifest_sha256": original.digest(output / "manifest.json"),
            "budget_helper_sha256": original.digest(helper),
            "baseline_hashes": {
                name: original.digest(baseline / name)
                for name in (
                    "manifest.json",
                    "freeze.json",
                    "results.jsonl",
                    "readout.json",
                )
            },
        },
    )


def verify(output: Path, helper: Path) -> dict:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if original.digest(output / "manifest.json") != freeze["manifest_sha256"]:
        raise ValueError("The frozen matrix changed")
    if original.digest(helper) != freeze["budget_helper_sha256"]:
        raise ValueError("The reservation helper changed")
    for name, expected in freeze["source_hashes"].items():
        if original.digest(original.ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    baseline = Path(manifest["baseline_directory"])
    for name, expected in freeze["baseline_hashes"].items():
        if original.digest(baseline / name) != expected:
            raise ValueError(f"Historical result changed: {name}")
    return manifest


def summarize(output: Path, manifest: dict, rows: dict, budget: dict) -> dict:
    attempts = original.attempt_stats(output / "budget.sqlite")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    arms = {}
    for arm in ARMS:
        matching = [row for row in rows.values() if row["arm"] == arm]
        grades = [grade for row in matching for grade in row.get("grades", [])]
        latency = [row["latency_ms"] for row in matching if row["status"] == "success"]
        sensitivity = 0
        per_case = {}
        for case_id, case in cases.items():
            local = [row for row in matching if row["case_id"] == case_id]
            candidate_map = {item["candidate_id"]: item for item in case["candidates"]}
            per_case[case_id] = {
                "candidates": len(local) * len(case["candidates"]),
                "adequate": sum(
                    g["adequate"] for row in local for g in row.get("grades", [])
                ),
                "statuses": dict(Counter(row["status"] for row in local)),
            }
            sensitivity += sum(
                punctuation_boundary_diagnostic(
                    case, candidate_map[g["candidate_id"]], g
                )
                for row in local
                for g in row.get("grades", [])
            )
        arms[arm] = {
            "model": MODEL,
            "slots": len(matching),
            "statuses": dict(Counter(row["status"] for row in matching)),
            "candidates": sum(
                len(cases[row["case_id"]]["candidates"]) for row in matching
            ),
            "adequate": sum(g["adequate"] for g in grades),
            "exact": sum(g["exact"] for g in grades),
            "false_support": sum(g["false_support"] for g in grades),
            "false_abstention": sum(g["false_abstention"] for g in grades),
            "punctuation_only_boundary_failures": sensitivity,
            "p50_ms_valid_slots": statistics.median(latency) if latency else None,
            "p95_ms_valid_slots": original.percentile(latency, 0.95),
            "conservative_usd": sum(
                attempts[row["slot"]]["charged"] for row in matching
            )
            / 1e9,
            "reported_usd": sum(
                attempts[row["slot"]]["reported_cost"] or 0 for row in matching
            )
            / 1e9,
            "provider_attempts": sum(
                attempts[row["slot"]]["attempts"] for row in matching
            ),
            "per_case": per_case,
        }
    result = {
        "cases": len(cases),
        "slots": len(rows),
        "planned_slots": len(manifest["slots"]),
        "models": manifest["models"],
        "arms": arms,
        "budget": budget,
        "caveats": [
            "Same frozen development fixtures; repeated trials are not independent cases.",
            "Added later, without concurrent reruns of the old models; latency is observational.",
            "Primary grades include unavailable results; latency describes valid slots.",
            "Reference-only selection is not a complete evidence-card replacement.",
            "Punctuation sensitivity uses the old diagnostic unchanged; it is not the primary grade.",
        ],
    }
    original.write_json(output / "readout.json", result)
    return result


async def run(output: Path, grant_path: Path, helper_path: Path) -> None:
    manifest = verify(output, helper_path)
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    if grant["freeze_sha256"] != original.digest(output / "freeze.json"):
        raise ValueError("The grant does not authorize this freeze")
    rows = original.completed_rows(output / "results.jsonl")
    if set(original.attempt_stats(output / "budget.sqlite")) - set(rows):
        raise ValueError("An unfinished paid slot requires coordinator reconciliation")
    # This process-local profile preserves the baseline's effective API controls.
    # It does not register or activate GPT-6-Luna in production configuration.
    MODEL_PROFILES[MODEL] = ModelProfile(
        omit_temperature=True, extra_body={"reasoning": {"effort": "none"}}
    )
    settings = Settings.from_env()
    if not settings.openrouter_api_key:
        raise ValueError("OpenRouter credentials are required")
    helper = original.load_helper(helper_path)
    budget = helper.LaneBudget.from_grant(
        grant_path, output / "budget.sqlite", expected_assignment_id=ASSIGNMENT
    )
    capture_slot = {"slot": None}

    async def capture_response(response: httpx.Response) -> None:
        await response.aread()
        with (output / "openrouter_http.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "slot": capture_slot["slot"],
                        "status_code": response.status_code,
                        "body": response.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    http_client = httpx.AsyncClient(
        timeout=60, follow_redirects=True, event_hooks={"response": [capture_response]}
    )
    sdk_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": settings.openrouter_site_url,
            "X-Title": settings.openrouter_app_name,
        },
        timeout=60,
        max_retries=0,
        http_client=http_client,
    )
    client = LLMClient(
        providers=[
            OpenRouterProvider(
                settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                app_name=settings.openrouter_app_name,
                request_timeout_seconds=60,
                client=sdk_client,
            )
        ],
        retry_policy=RetryPolicy(attempts=1),
        interactive_retry_policy=RetryPolicy(attempts=1),
        extraction_retry_policy=RetryPolicy(attempts=1),
        structured_output_retry_attempts=0,
        technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
    )
    helper.install_budget(client, budget)
    selected = ModelOnlyClient(client)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    try:
        for slot in manifest["slots"]:
            if slot["slot"] in rows:
                continue
            case = cases[slot["case_id"]]
            capture_slot["slot"] = slot["slot"]
            row = {
                **slot,
                "model": MODEL,
                "category": case["category"],
                "origin": case["origin"]["kind"],
            }
            started = perf_counter()
            try:
                with helper.slot_scope(slot["slot"]):
                    references = await original.evaluate(selected, case, slot["arm"])
                row.update(status="success", grades=original.grade(case, references))
            except ValueError as exc:
                row.update(
                    status="invalid_selection",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except TransientLLMError as exc:
                if (
                    str(exc)
                    != "openrouter returned no output content (finish_reason=unknown)"
                ):
                    original.write_json(
                        output / "stopped.json",
                        {**row, "error_type": type(exc).__name__},
                    )
                    raise
                row.update(
                    status="provider_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except Exception as exc:
                original.write_json(
                    output / "stopped.json", {**row, "error_type": type(exc).__name__}
                )
                raise
            row["latency_ms"] = (perf_counter() - started) * 1000
            with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows[row["slot"]] = row
            original.write_json(
                output / "progress.json",
                {
                    "completed": len(rows),
                    "planned": len(manifest["slots"]),
                    "last_slot": row["slot"],
                    "budget": budget.snapshot(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
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
        await sdk_client.close()
        summarize(output, manifest, rows, budget.snapshot())
        budget.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget-helper", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.baseline is None:
            parser.error("Preparation requires the preserved baseline")
        prepare(args.output_dir, args.baseline, args.budget_helper)
    else:
        if args.grant is None:
            parser.error("Paid execution requires a grant")
        asyncio.run(run(args.output_dir, args.grant, args.budget_helper))


if __name__ == "__main__":
    main()
