"""CLI entry point for running the Atagia-bench v0 benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.atagia_bench.custody_report import (
    build_failed_question_custody_report,
    save_failed_question_custody_report,
)
from benchmarks.atagia_bench.report_diff import (
    build_diff,
    format_diff_summary,
    load_atagia_bench_report,
    save_diff,
)
from benchmarks.atagia_bench.runner import (
    AtagiaBenchReport,
    AtagiaBenchRunner,
    load_holdout_question_ids,
)
from benchmarks.custody_summary import format_retrieval_custody_summary
from benchmarks.failure_taxonomy import (
    build_failure_taxonomy_report,
    failure_taxonomy_manifest_summary,
    format_failure_taxonomy_summary,
    save_failure_taxonomy_report,
)
from benchmarks.retained_db_paths import default_benchmark_db_dir
from atagia.core.config import ANSWER_STANCE_PROMPT_VARIANTS
from atagia.models.schemas_replay import AblationConfig
from atagia.services.model_resolution import COMPONENTS_BY_ID

load_dotenv()

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
_DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_DEFAULT_HOLDOUT_FILE = Path(__file__).resolve().parent / "data" / "holdout_v0.json"
_DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4-7"
_DEFAULT_PRIVACY_ENFORCEMENT = "off"


def _parse_csv_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [item.strip() for item in raw_value.split(",")]
    return [value for value in values if value] or None


def _parse_ablation(raw_value: str | None) -> AblationConfig | None:
    if raw_value is None:
        return None
    presets = {
        "no_contract": AblationConfig(skip_contract_memory=True),
        "no_scope": AblationConfig(force_all_scopes=True),
        "no_need_detection": AblationConfig(skip_need_detection=True),
        "no_revision": AblationConfig(skip_belief_revision=True),
        "no_compaction": AblationConfig(skip_compaction=True),
        "composer_budgeted_marginal": AblationConfig(
            composer_strategy="budgeted_marginal"
        ),
        "composer_evidence_obligation": AblationConfig(
            enable_evidence_obligation_coverage=True
        ),
        "applicability_gate_shadow": AblationConfig(
            applicability_gate_mode="shadow"
        ),
        "applicability_gate_enforced": AblationConfig(
            applicability_gate_mode="enforced"
        ),
        "context_envelope_4k": AblationConfig(
            context_envelope_budget_tokens=4_096,
        ),
        "context_envelope_8k": AblationConfig(
            context_envelope_budget_tokens=8_192,
        ),
        "context_envelope_16k": AblationConfig(
            context_envelope_budget_tokens=16_384,
            override_retrieval_params={
                "rerank_top_k": 20,
                "final_context_items": 12,
            },
        ),
        "context_envelope_24k": AblationConfig(
            context_envelope_budget_tokens=24_576,
            override_retrieval_params={
                "rerank_top_k": 25,
                "final_context_items": 16,
            },
        ),
        "context_envelope_32k": AblationConfig(
            context_envelope_budget_tokens=32_768,
            override_retrieval_params={
                "rerank_top_k": 25,
                "final_context_items": 20,
            },
        ),
    }
    preset = presets.get(raw_value)
    if preset is not None:
        return preset
    return AblationConfig.model_validate(json.loads(raw_value))


def _benchmark_ablation(
    raw_ablation: str | None,
    privacy_enforcement: str | None,
) -> AblationConfig | None:
    ablation = _parse_ablation(raw_ablation)
    effective_privacy_enforcement = privacy_enforcement or _DEFAULT_PRIVACY_ENFORCEMENT
    return (ablation or AblationConfig()).model_copy(
        update={"privacy_enforcement": effective_privacy_enforcement}
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Atagia-bench v0 pilot benchmark."
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="LLM provider name (anthropic, openai, openrouter)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Legacy one-model baseline. Without role-specific model flags this "
            "forces every Atagia LLM component to the same model; with role "
            "flags it is the fallback for unspecified roles."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (or set via environment variable)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "LLM model for scoring; defaults to direct Anthropic "
            "claude-opus-4-7 for benchmark stability"
        ),
    )
    parser.add_argument(
        "--ingest-model",
        default=None,
        help="Model for Atagia ingest components unless overridden by --component-model.",
    )
    parser.add_argument(
        "--retrieval-model",
        default=None,
        help="Model for Atagia retrieval components unless overridden by --component-model.",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Model for Atagia answer-generation components.",
    )
    parser.add_argument(
        "--answer-postcondition-guard",
        action="store_true",
        help=(
            "Enable the generic answer postcondition verifier with one retry "
            "and controlled abstention on failure."
        ),
    )
    parser.add_argument(
        "--answer-stance",
        choices=("reactive", "proactive"),
        default="reactive",
        help=(
            "Answer behavior stance. reactive answers only the exact request; "
            "proactive can add clearly caveated related evidence when useful."
        ),
    )
    parser.add_argument(
        "--answer-stance-prompt-variant",
        choices=ANSWER_STANCE_PROMPT_VARIANTS,
        default="baseline",
        help=(
            "Experimental answer-stance prompt variant. Use baseline unless "
            "running prompt ablations."
        ),
    )
    parser.add_argument(
        "--component-model",
        action="append",
        default=[],
        metavar="COMPONENT=MODEL",
        help=(
            "Override one Atagia LLM component model. Repeatable. "
            "Example: need_detector=openrouter/google/gemini-3.1-flash-lite"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory for the benchmark report JSON",
    )
    parser.add_argument(
        "--personas",
        default=None,
        help="Comma-separated persona ids to run (default: all)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated category tags to filter questions",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Comma-separated question ids to run",
    )
    parser.add_argument(
        "--benchmark-split",
        choices=("all", "development", "holdout"),
        default="all",
        help=(
            "Question split to run. development excludes the frozen holdout; "
            "holdout runs only frozen holdout ids."
        ),
    )
    parser.add_argument(
        "--holdout-file",
        default=str(_DEFAULT_HOLDOUT_FILE),
        help="Frozen Atagia-bench holdout manifest JSON",
    )
    parser.add_argument(
        "--manifests-dir",
        default=str(_DEFAULT_MANIFESTS_DIR),
        help="Path to the Atagia manifests directory",
    )
    parser.add_argument(
        "--embedding-backend",
        default="none",
        help='Embedding backend: "none" or "sqlite_vec"',
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name (required when backend is sqlite_vec)",
    )
    parser.add_argument(
        "--ablation",
        default=None,
        help=(
            "Ablation preset (no_contract, no_scope, no_need_detection, "
            "no_revision, no_compaction, composer_evidence_obligation, "
            "context_envelope_4k, context_envelope_8k, context_envelope_16k, "
            "context_envelope_24k, context_envelope_32k) or JSON "
            "AblationConfig"
        ),
    )
    parser.add_argument(
        "--privacy-enforcement",
        choices=("enforce", "audit_only", "off"),
        default=_DEFAULT_PRIVACY_ENFORCEMENT,
        help=(
            "Request privacy_enforcement for Atagia-bench client calls. Defaults to off. "
            "Core Atagia-bench/default-readiness runs must stay off until "
            "full Atagia-bench, full LoCoMo, and competitor baselines are "
            "matched or beaten without privacy enforcement. "
            "audit_only keeps SQL privacy gates but audits/skips the late "
            "candidate/composer policy filters; off also disables retrieval SQL "
            "privacy/intimacy/sensitivity gates and related context gates, and "
            "grades privacy_check questions as private-fact retrieval checks "
            "while preserving user isolation."
        ),
    )
    parser.add_argument(
        "--diff-against",
        default=None,
        help="Baseline Atagia-bench report JSON to diff against",
    )
    parser.add_argument(
        "--diff-output",
        default=None,
        help="Optional diff artifact output path",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data directory (default: built-in data/)",
    )
    parser.add_argument(
        "--trusted-evaluation",
        action="store_true",
        help=(
            "Use a trusted local evaluation context: raise retrieval privacy "
            "ceiling and activate pending memories for coverage auditing while "
            "preserving ordinary high-risk secret disclosure refusals"
        ),
    )
    parser.add_argument(
        "--parallel-personas",
        type=int,
        default=1,
        help=(
            "Number of Atagia-bench personas to run concurrently. Each persona "
            "uses its own temporary SQLite DB."
        ),
    )
    parser.add_argument(
        "--benchmark-db-dir",
        default=None,
        help="Directory where --keep-db stores reusable benchmark databases.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the ingested benchmark SQLite DB and write reuse metadata.",
    )
    parser.add_argument(
        "--allow-temp-benchmark-db-dir",
        action="store_true",
        help=(
            "Allow --keep-db to retain benchmark DB snapshots under a system "
            "temporary directory. Intended only for explicitly disposable runs."
        ),
    )
    parser.add_argument(
        "--reuse-db",
        default=None,
        help=(
            "Existing benchmark DB file or directory containing benchmark.db. "
            "Requires exactly one selected persona and skips ingestion."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Run retrieval/answer/judge only against --reuse-db.",
    )
    return parser


def _resolve_judge_model(args: argparse.Namespace) -> str | None:
    if args.judge_model is not None:
        return args.judge_model
    return _DEFAULT_JUDGE_MODEL


def _parse_component_model_overrides(raw_values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_value in raw_values or []:
        if "=" not in raw_value:
            raise ValueError(
                f"--component-model must use COMPONENT=MODEL, got {raw_value!r}"
            )
        component_id, model = (part.strip() for part in raw_value.split("=", 1))
        if not component_id or not model:
            raise ValueError(
                "--component-model must include both COMPONENT and MODEL, "
                f"got {raw_value!r}"
            )
        if component_id not in COMPONENTS_BY_ID:
            valid = ", ".join(sorted(COMPONENTS_BY_ID))
            raise ValueError(
                f"Unknown component id for --component-model: {component_id}. "
                f"Valid component ids: {valid}"
            )
        overrides[component_id] = model
    return overrides


def _default_diff_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("atagia-bench-report-")
    return report_path.with_name(f"atagia-bench-diff-{timestamp}.json")


def _default_custody_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("atagia-bench-report-")
    return report_path.with_name(f"atagia-bench-failed-custody-{timestamp}.json")


def _default_taxonomy_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("atagia-bench-report-")
    return report_path.with_name(f"atagia-bench-failure-taxonomy-{timestamp}.json")


def _effective_benchmark_db_dir(args: argparse.Namespace) -> str | None:
    if not args.keep_db:
        return args.benchmark_db_dir
    if args.benchmark_db_dir is not None:
        return args.benchmark_db_dir
    return str(default_benchmark_db_dir(args.output))


def _question_filters_for_split(
    *,
    explicit_question_ids: list[str] | None,
    benchmark_split: str,
    holdout_ids: list[str],
) -> tuple[list[str] | None, list[str] | None]:
    if benchmark_split == "all":
        return explicit_question_ids, None
    holdout_set = set(holdout_ids)
    if benchmark_split == "holdout":
        if explicit_question_ids is None:
            return sorted(holdout_set), None
        return sorted(set(explicit_question_ids).intersection(holdout_set)), None
    return explicit_question_ids, sorted(holdout_set)


async def _run_async(
    args: argparse.Namespace,
) -> tuple[
    AtagiaBenchReport, Path, Path | None, Any, Path, Path, Path, dict[str, object]
]:
    holdout_ids = load_holdout_question_ids(args.holdout_file)
    question_ids, exclude_question_ids = _question_filters_for_split(
        explicit_question_ids=_parse_csv_list(args.questions),
        benchmark_split=args.benchmark_split,
        holdout_ids=holdout_ids,
    )
    component_models = getattr(args, "component_models", None)
    if component_models is None:
        component_models = _parse_component_model_overrides(
            getattr(args, "component_model", []),
        )
    runner = AtagiaBenchRunner(
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_model=args.model,
        judge_model=_resolve_judge_model(args),
        ingest_model=args.ingest_model,
        retrieval_model=args.retrieval_model,
        answer_model=args.answer_model,
        component_models=component_models,
        manifests_dir=args.manifests_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        data_dir=args.data_dir,
        answer_postcondition_guard_enabled=args.answer_postcondition_guard,
        answer_stance=args.answer_stance,
        answer_stance_prompt_variant=args.answer_stance_prompt_variant,
    )
    report = await runner.run(
        persona_ids=_parse_csv_list(args.personas),
        category_tags=_parse_csv_list(args.categories),
        question_ids=question_ids,
        exclude_question_ids=exclude_question_ids,
        benchmark_split=args.benchmark_split,
        holdout_question_ids=holdout_ids,
        ablation=_benchmark_ablation(
            args.ablation,
            getattr(args, "privacy_enforcement", None),
        ),
        trusted_evaluation=args.trusted_evaluation,
        parallel_personas=args.parallel_personas,
        benchmark_db_dir=_effective_benchmark_db_dir(args),
        requested_benchmark_db_dir=args.benchmark_db_dir,
        allow_temp_benchmark_db_dir=args.allow_temp_benchmark_db_dir,
        keep_db=args.keep_db,
        reuse_db=args.reuse_db,
        evaluate_only=args.evaluate_only,
        invocation_args=sys.argv[1:],
    )
    report_path = runner.save_report(report, args.output)

    diff_report = None
    diff_path = None
    if args.diff_against is not None:
        baseline = load_atagia_bench_report(args.diff_against)
        diff_report = build_diff(
            baseline,
            report,
            before_label=Path(args.diff_against).name,
            after_label=report_path.name,
            before_report_sha256=sha256_file_if_exists(args.diff_against),
            after_report_sha256=sha256_file_if_exists(report_path),
        )
        requested_diff_output = (
            Path(args.diff_output).expanduser()
            if args.diff_output is not None
            else _default_diff_output_path(report_path)
        )
        diff_path = save_diff(diff_report, requested_diff_output)

    custody_path = _default_custody_output_path(report_path)
    save_failed_question_custody_report(
        build_failed_question_custody_report(report, source_report=str(report_path)),
        custody_path,
    )
    taxonomy_report = build_failure_taxonomy_report(
        report, source_report=str(report_path)
    )
    taxonomy_summary = failure_taxonomy_manifest_summary(taxonomy_report)
    taxonomy_path = _default_taxonomy_output_path(report_path)
    save_failure_taxonomy_report(taxonomy_report, taxonomy_path)
    manifest_path = runner.save_run_manifest(
        report,
        report_path=report_path,
        diff_path=diff_path,
        holdout_path=args.holdout_file,
        custody_path=custody_path,
        taxonomy_path=taxonomy_path,
        failure_taxonomy_summary=taxonomy_summary,
    )

    return (
        report,
        report_path,
        diff_path,
        diff_report,
        manifest_path,
        custody_path,
        taxonomy_path,
        taxonomy_summary,
    )


def _format_duration(duration_seconds: float) -> str:
    rounded = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_report_summary(
    report: AtagiaBenchReport,
    report_path: Path,
    diff_path: Path | None = None,
    manifest_path: Path | None = None,
    custody_path: Path | None = None,
    taxonomy_path: Path | None = None,
    failure_taxonomy_summary: dict[str, object] | None = None,
) -> str:
    lines = [
        "=" * 50,
        "Atagia-bench v0 Results",
        "=" * 50,
        f"Pass rate: {report.pass_rate:.1%} ({report.total_passed}/{report.total_questions})",
        f"Avg score: {report.avg_score:.3f}",
        f"Priority failures: {report.priority_failure_count}",
        f"Duration: {_format_duration(report.run_duration_seconds)}",
        f"Personas: {', '.join(report.personas_used)}",
        f"Model mode: {report.config.get('model_mode', 'forced_global')}",
        f"Answer model: {report.config.get('provider', '')} / {report.config.get('answer_model', '')}",
        f"Judge model: {report.config.get('judge_model', '')}",
        f"Trusted evaluation: {bool(report.config.get('trusted_evaluation', False))}",
        f"Evaluate only: {bool(report.config.get('evaluate_only', False))}",
        f"Benchmark split: {report.config.get('benchmark_split', 'all')}",
        format_retrieval_custody_summary(
            report.config.get("retrieval_custody_summary")
        ),
        format_failure_taxonomy_summary(
            failure_taxonomy_summary or report.config.get("failure_taxonomy_summary")
        ),
        "",
        "Category breakdown:",
    ]

    for stats in sorted(report.per_category, key=lambda s: s.category):
        lines.append(
            f"  {stats.category:30s}  "
            f"{stats.pass_rate:6.1%} ({stats.pass_count}/{stats.count})  "
            f"avg={stats.avg_score:.3f}"
        )

    # Show failed questions
    failed = [r for r in report.per_question if not r.grade.passed]
    if failed:
        lines.append("")
        lines.append(f"Failed questions ({len(failed)}):")
        for r in failed[:20]:  # Limit output
            lines.append(f"  {r.question_id}: {r.grade.reason[:80]}")
        if len(failed) > 20:
            lines.append(f"  ... and {len(failed) - 20} more")

    lines.extend(
        [
            "",
            f"Report saved to: {report_path}",
            *[
                f"Benchmark DB: {entry.get('db_path')}"
                for entry in (report.config.get("benchmark_databases") or [])
                if isinstance(entry, dict) and entry.get("db_path")
            ],
            *(
                [f"Run manifest saved to: {manifest_path}"]
                if manifest_path is not None
                else []
            ),
            *(
                [f"Failed-question custody saved to: {custody_path}"]
                if custody_path is not None
                else []
            ),
            *(
                [f"Failure taxonomy saved to: {taxonomy_path}"]
                if taxonomy_path is not None
                else []
            ),
            *([f"Diff saved to: {diff_path}"] if diff_path is not None else []),
            "=" * 50,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the benchmark, and print results."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.component_models = _parse_component_model_overrides(args.component_model)
    except ValueError as exc:
        parser.error(str(exc))

    if args.embedding_backend == "sqlite_vec" and not args.embedding_model:
        parser.error(
            "--embedding-model is required when --embedding-backend is sqlite_vec"
        )
    if args.parallel_personas < 1:
        parser.error("--parallel-personas must be at least 1")
    if args.evaluate_only and args.reuse_db is None:
        parser.error("--evaluate-only requires --reuse-db")
    selected_personas = _parse_csv_list(args.personas)
    if args.reuse_db is not None and (
        selected_personas is None or len(selected_personas) != 1
    ):
        parser.error("--reuse-db requires --personas with exactly one persona id")

    (
        report,
        report_path,
        diff_path,
        diff_report,
        manifest_path,
        custody_path,
        taxonomy_path,
        failure_taxonomy_summary,
    ) = asyncio.run(_run_async(args))

    print(
        _format_report_summary(
            report,
            report_path,
            diff_path=diff_path,
            manifest_path=manifest_path,
            custody_path=custody_path,
            taxonomy_path=taxonomy_path,
            failure_taxonomy_summary=failure_taxonomy_summary,
        )
    )

    if diff_report is not None:
        print()
        print(format_diff_summary(diff_report))


if __name__ == "__main__":
    main()
