"""Re-score stored Atagia-bench predictions under a chosen judge protocol.

The Atagia-bench ``llm_judge`` rubric follows the ``memory_quality`` doctrine:
all requested information must be present, correct, and correctly
attributed; extras that are TRUE in the conversation pass; fail only on missing,
false, or misattributed information. This tool re-judges the ``llm_judge``
predictions stored in retained Atagia-bench reports under a chosen protocol,
WITHOUT re-running the engine, so a rubric change is isolated from answer
generation (stored predictions are held constant). The CURRENT authored dataset
is the source of truth for question text and ground truth — gold-question
alignment corrections take effect on rejudge, and each verdict records whether
the stored gold differed.

It reuses the production grading path exactly:
``AtagiaBenchRunner._grader_config_for_question`` builds the source-evidence and
full persona transcript, and ``resolve_grader('llm_judge', scorer)`` grades the
prediction. Only the ``llm_judge`` grader is protocol-sensitive; deterministic,
supersession, abstention, and gated graders are reported as unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.atagia_bench.adapter import (
    AtagiaBenchAdapter,
    AtagiaBenchDataset,
    AtagiaBenchPersonaData,
)
from benchmarks.atagia_bench.graders import measurement_layer_for_grader, resolve_grader
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder
from benchmarks.output_root import assert_outside_repo, utc_run_id
from benchmarks.scorer import JudgeProtocol, LLMJudgeScorer
from atagia.core.config import Settings
from atagia.models.schemas_replay import AblationConfig
from atagia.services.providers import build_llm_client

# Load .env before any Settings.from_env() call resolves provider keys.
load_dotenv()

_DEFAULT_JUDGE_MODEL = "openrouter/openai/gpt-5.6-luna,medium"
# GPT-5.6 Luna direct-OpenAI list rates (USD per 1M tokens); deliberately
# conservative vs the current OpenRouter promo — see benchmarks.locomo.rejudge.
_LUNA_INPUT_PRICE = 0.20
_LUNA_CACHED_INPUT_PRICE = 0.02
_LUNA_OUTPUT_PRICE = 1.20
_CHARS_PER_TOKEN = 4.0


@dataclass(slots=True)
class RejudgeConfig:
    """Configuration for one Atagia-bench rejudge invocation."""

    report_paths: list[Path]
    protocol: JudgeProtocol = JudgeProtocol.MEMORY_QUALITY
    judge_model: str = _DEFAULT_JUDGE_MODEL
    data_dir: Path | None = None
    concurrency: int = 4
    max_cost_usd: float = 5.0
    dry_run: bool = False


@dataclass(slots=True)
class _QuestionRecord:
    question_id: str
    persona_id: str
    prediction: str
    # Gold as stored in the report; kept for flip auditing. Grading always uses
    # the CURRENT authored dataset's ground truth, so authored gold corrections
    # take effect on rejudge.
    stored_ground_truth: str
    stored_passed: bool
    source_report: str


@dataclass(slots=True)
class _Verdict:
    question_id: str
    persona_id: str
    ground_truth: str
    stored_ground_truth: str
    stored_passed: bool
    rejudge_passed: bool
    reasoning: str
    source_report: str


@dataclass(slots=True)
class _Skipped:
    question_id: str
    persona_id: str
    grader: str
    measurement_layer: str
    stored_passed: bool


@dataclass(slots=True)
class _LoadedReports:
    llm_judge_records: list[_QuestionRecord] = field(default_factory=list)
    non_llm_judge: list[_Skipped] = field(default_factory=list)
    report_paths: list[Path] = field(default_factory=list)


def _resolve_report_paths(report_specs: list[str]) -> list[Path]:
    """Expand report specs (files or directories) into report json paths."""
    paths: list[Path] = []
    for spec in report_specs:
        path = Path(spec).expanduser()
        if path.is_dir():
            matches = sorted(path.rglob("atagia-bench-report-*.json"))
            if not matches:
                raise FileNotFoundError(
                    f"No atagia-bench-report-*.json found under {path}"
                )
            paths.extend(matches)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Report path not found: {path}")
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_reports(config: RejudgeConfig, dataset: AtagiaBenchDataset) -> _LoadedReports:
    """Load stored predictions and split llm_judge vs other graders.

    The authored dataset is the single source of truth for each question's
    grader; a stored question id that is no longer in the dataset fails fast
    (dataset drift makes a rejudge meaningless).
    """
    report_paths = _resolve_report_paths([str(p) for p in config.report_paths])
    dataset_graders = {
        question.question_id: question.grader
        for persona in dataset.personas
        for question in persona.questions
    }
    loaded = _LoadedReports(report_paths=report_paths)
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for per_question in payload.get("per_question", []):
            question_id = str(per_question.get("question_id") or "")
            persona_id = str(per_question.get("persona_id") or "")
            grade = per_question.get("grade") or {}
            stored_passed = bool(grade.get("passed"))
            grader = dataset_graders.get(question_id)
            if grader is None:
                raise ValueError(
                    f"Stored question {question_id} (report {path}) is not in "
                    "the current Atagia-bench dataset; the dataset has drifted "
                    "since the report was produced."
                )
            if grader != "llm_judge":
                loaded.non_llm_judge.append(
                    _Skipped(
                        question_id=question_id,
                        persona_id=persona_id,
                        grader=grader,
                        measurement_layer=measurement_layer_for_grader(grader),
                        stored_passed=stored_passed,
                    )
                )
                continue
            loaded.llm_judge_records.append(
                _QuestionRecord(
                    question_id=question_id,
                    persona_id=persona_id,
                    prediction=str(per_question.get("prediction") or ""),
                    stored_ground_truth=str(per_question.get("ground_truth") or ""),
                    stored_passed=stored_passed,
                    source_report=str(path),
                )
            )
    return loaded


def _project_cost(records: list[_QuestionRecord], transcripts_chars: int) -> dict[str, Any]:
    input_chars = sum(len(r.prediction) + len(r.stored_ground_truth) for r in records)
    input_chars += transcripts_chars
    input_chars += 1200 * len(records)  # instruction + evidence overhead per call.
    input_tokens = input_chars / _CHARS_PER_TOKEN
    output_tokens = 200 * len(records)
    cost = (
        input_tokens / 1_000_000 * _LUNA_INPUT_PRICE
        + output_tokens / 1_000_000 * _LUNA_OUTPUT_PRICE
    )
    return {
        "llm_calls_planned": len(records),
        "projected_cost_usd": round(cost, 4),
    }


def _actual_cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute honest cost from provider-reported token counts.

    Provider ``input_tokens`` INCLUDES cached tokens, so cache reads are priced
    separately (same accounting as ``benchmarks.locomo.rejudge``).
    """
    input_tokens = 0.0
    cached_input_tokens = 0.0
    output_tokens = 0.0
    for record in records:
        counts = record.get("token_counts") or {}
        input_tokens += float(counts.get("input_tokens") or 0.0)
        cached_input_tokens += float(counts.get("cached_input_tokens") or 0.0)
        output_tokens += float(counts.get("output_tokens") or 0.0)
    non_cached_input = max(0.0, input_tokens - cached_input_tokens)
    cost = (
        non_cached_input / 1_000_000 * _LUNA_INPUT_PRICE
        + cached_input_tokens / 1_000_000 * _LUNA_CACHED_INPUT_PRICE
        + output_tokens / 1_000_000 * _LUNA_OUTPUT_PRICE
    )
    return {
        "llm_calls": len(records),
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": round(cost, 4),
    }


async def run_rejudge(config: RejudgeConfig) -> dict[str, Any]:
    """Re-judge stored llm_judge predictions under the configured protocol."""
    dataset = AtagiaBenchAdapter(config.data_dir).load()
    loaded = load_reports(config, dataset)
    records = loaded.llm_judge_records

    persona_by_id: dict[str, AtagiaBenchPersonaData] = {
        persona.persona.persona_id: persona for persona in dataset.personas
    }
    question_by_id = {
        question.question_id: question
        for persona in dataset.personas
        for question in persona.questions
    }
    ablation_off = AblationConfig(privacy_enforcement="off")

    # Build grader configs once (they carry the persona transcript + evidence).
    grader_configs: dict[str, dict[str, Any]] = {}
    transcripts_chars = 0
    for record in records:
        question = question_by_id.get(record.question_id)
        persona = persona_by_id.get(record.persona_id)
        if question is None or persona is None:
            raise ValueError(
                f"Stored question {record.question_id} (persona "
                f"{record.persona_id}) not found in the current dataset."
            )
        grader_config = AtagiaBenchRunner._grader_config_for_question(
            question,
            ablation_off,
            persona_data=persona,
            answer_stance="reactive",
        )
        grader_configs[record.question_id] = grader_config
        transcripts_chars += len(grader_config.get("conversation_transcript") or "")

    projection = _project_cost(records, transcripts_chars)

    if config.dry_run:
        return {
            "dry_run": True,
            "report_paths": [str(p) for p in loaded.report_paths],
            "llm_judge_questions": len(records),
            "non_llm_judge_questions": len(loaded.non_llm_judge),
            "projection": projection,
        }

    if projection["projected_cost_usd"] > config.max_cost_usd:
        raise RuntimeError(
            f"Projected judge cost ${projection['projected_cost_usd']} exceeds "
            f"--max-cost-usd ${config.max_cost_usd}. Increase the cap or reduce "
            "the report set."
        )

    settings = Settings.from_env()
    client = build_llm_client(settings)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    scorer = LLMJudgeScorer(client, config.judge_model, config.protocol)
    grader = resolve_grader("llm_judge", llm_judge=scorer)

    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def judge_one(record: _QuestionRecord) -> _Verdict:
        # The CURRENT authored dataset is the source of truth for the gold, so
        # gold-question alignment corrections take effect on rejudge; the stored
        # gold is carried in the verdict for flip auditing.
        ground_truth = question_by_id[record.question_id].ground_truth
        async with semaphore:
            grade = await grader.grade(
                prediction=record.prediction,
                ground_truth=ground_truth,
                config=grader_configs[record.question_id],
            )
        return _Verdict(
            question_id=record.question_id,
            persona_id=record.persona_id,
            ground_truth=ground_truth,
            stored_ground_truth=record.stored_ground_truth,
            stored_passed=record.stored_passed,
            rejudge_passed=grade.passed,
            reasoning=grade.reason,
            source_report=record.source_report,
        )

    # Group by persona: each persona has its own transcript prefix, so warming
    # once per persona (then fanning out that persona's remainder) is what lets
    # provider prefix-caching reuse the transcript across its questions.
    by_persona: dict[str, list[_QuestionRecord]] = defaultdict(list)
    for record in records:
        by_persona[record.persona_id].append(record)

    verdicts: list[_Verdict] = []
    for persona_id in sorted(by_persona):
        persona_records = by_persona[persona_id]
        verdicts.append(await judge_one(persona_records[0]))
        if len(persona_records) > 1:
            verdicts.extend(
                await asyncio.gather(*(judge_one(r) for r in persona_records[1:]))
            )

    return _build_result(config, loaded, verdicts, projection, recorder)


def _build_result(
    config: RejudgeConfig,
    loaded: _LoadedReports,
    verdicts: list[_Verdict],
    projection: dict[str, Any],
    recorder: LLMCallRecorder,
) -> dict[str, Any]:
    rejudge_passed = sum(1 for v in verdicts if v.rejudge_passed)
    stored_passed_llm = sum(1 for v in verdicts if v.stored_passed)
    fail_to_pass = [
        v.question_id for v in verdicts if not v.stored_passed and v.rejudge_passed
    ]
    pass_to_fail = [
        v.question_id for v in verdicts if v.stored_passed and not v.rejudge_passed
    ]
    non_llm_passed = sum(1 for s in loaded.non_llm_judge if s.stored_passed)
    non_llm_layers = Counter(s.measurement_layer for s in loaded.non_llm_judge)

    total_questions = len(verdicts) + len(loaded.non_llm_judge)
    # The aligned total combines rejudged llm_judge verdicts with unchanged
    # non-llm_judge graders (their stored grade is not protocol-sensitive).
    aligned_total_passed = rejudge_passed + non_llm_passed

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "benchmarks.atagia_bench.rejudge",
        "judge_protocol": config.protocol.value,
        "judge_model": config.judge_model,
        "source_reports": [
            {"path": str(path), "sha256": sha256_file_if_exists(path)}
            for path in loaded.report_paths
        ],
        "projection": projection,
        "actual_cost": _actual_cost(recorder.records()),
        "coverage": {
            "total_questions": total_questions,
            "llm_judge_questions": len(verdicts),
            "non_llm_judge_questions": len(loaded.non_llm_judge),
            "non_llm_judge_layers": dict(non_llm_layers),
        },
        "llm_judge_rubric": {
            "stored_passed": stored_passed_llm,
            "rejudge_passed": rejudge_passed,
            "fail_to_pass": fail_to_pass,
            "pass_to_fail": pass_to_fail,
        },
        "aligned_total": {
            "passed": aligned_total_passed,
            "total": total_questions,
            "non_llm_judge_passed": non_llm_passed,
        },
        "verdicts": [
            {
                "question_id": v.question_id,
                "persona_id": v.persona_id,
                "ground_truth": v.ground_truth,
                "stored_ground_truth_differs": (
                    v.stored_ground_truth != v.ground_truth
                ),
                "stored_passed": v.stored_passed,
                "rejudge_passed": v.rejudge_passed,
                "flip": (
                    "fail_to_pass"
                    if (not v.stored_passed and v.rejudge_passed)
                    else "pass_to_fail"
                    if (v.stored_passed and not v.rejudge_passed)
                    else "none"
                ),
                "reasoning": v.reasoning,
                "source_report": v.source_report,
            }
            for v in verdicts
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-judge stored Atagia-bench llm_judge predictions under a chosen "
            "judge protocol (isolates a rubric change from answer generation)."
        )
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Atagia-bench report json, or a directory to search for "
            "atagia-bench-report-*.json. Repeatable."
        ),
    )
    parser.add_argument(
        "--judge-protocol",
        choices=tuple(protocol.value for protocol in JudgeProtocol),
        default=JudgeProtocol.MEMORY_QUALITY.value,
    )
    parser.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to write the rejudge json (must be outside the repo).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = RejudgeConfig(
        report_paths=[Path(spec) for spec in args.report],
        protocol=JudgeProtocol(args.judge_protocol),
        judge_model=args.judge_model,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        concurrency=args.concurrency,
        max_cost_usd=args.max_cost_usd,
        dry_run=args.dry_run,
    )
    result = asyncio.run(run_rejudge(config))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output and not args.dry_run:
        out_dir = assert_outside_repo(Path(args.output).expanduser())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"atagia-bench-rejudge-{config.protocol.value}-{utc_run_id()}.json"
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
