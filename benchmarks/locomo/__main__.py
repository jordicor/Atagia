"""CLI entry point for running the LoCoMo benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from benchmarks.base import (
    DEFAULT_SCORED_CATEGORIES,
    BenchmarkReport,
    ConversationReport,
    QuestionResult,
)
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from benchmarks.report_diff import build_benchmark_diff, load_benchmark_report, save_benchmark_diff
from atagia.models.schemas_replay import AblationConfig

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
_DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-opus-4-6"
_CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
}


def _parse_csv_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [item.strip() for item in raw_value.split(",")]
    return [value for value in values if value]


def _parse_categories(raw_value: str | None) -> list[int] | None:
    values = _parse_csv_list(raw_value)
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_ablation(raw_value: str | None) -> AblationConfig | None:
    if raw_value is None:
        return None
    presets = {
        "similarity_only": AblationConfig(skip_applicability_scoring=True),
        "no_contract": AblationConfig(skip_contract_memory=True),
        "no_scope": AblationConfig(force_all_scopes=True),
        "no_need_detection": AblationConfig(skip_need_detection=True),
        "no_revision": AblationConfig(skip_belief_revision=True),
        "no_compaction": AblationConfig(skip_compaction=True),
    }
    preset = presets.get(raw_value)
    if preset is not None:
        return preset
    return AblationConfig.model_validate(json.loads(raw_value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LoCoMo benchmark against Atagia.")
    parser.add_argument("--data-path", required=True, help="Path to locomo10.json")
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory where the benchmark report JSON should be written",
    )
    parser.add_argument("--provider", required=True, help="LLM provider name")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--model", required=True, help="LLM model for answer generation")
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Optional LLM model for scoring; defaults to claude-opus-4-6 for "
            "Anthropic, otherwise the answer model"
        ),
    )
    parser.add_argument(
        "--conversations",
        default=None,
        help="Comma-separated LoCoMo conversation ids to run",
    )
    parser.add_argument(
        "--categories",
        default=",".join(str(category) for category in DEFAULT_SCORED_CATEGORIES),
        help="Comma-separated scored categories to evaluate",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Comma-separated question ids to run after conversation/category filtering",
    )
    parser.add_argument(
        "--ablation",
        default=None,
        help=(
            "Ablation preset (similarity_only, no_contract, no_scope, no_need_detection, "
            "no_revision, no_compaction) or a JSON object matching AblationConfig"
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions to score per conversation (for quick validation)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns to ingest per conversation",
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
        "--corrections",
        default=None,
        help="Path to corrections overlay JSON (substitutes ground truths before scoring)",
    )
    parser.add_argument(
        "--community-corrections",
        default=None,
        help="Path to community audit errors.json (dial481/locomo-audit format)",
    )
    parser.add_argument(
        "--diff-against",
        default=None,
        help="Optional baseline benchmark report JSON to diff against after the run",
    )
    parser.add_argument(
        "--diff-output",
        default=None,
        help="Optional diff artifact output path; defaults next to the saved report",
    )
    parser.add_argument(
        "--checkpoint-output",
        default=None,
        help=(
            "Path to a JSON checkpoint overwritten after each scored question; "
            "defaults to locomo-checkpoint.json in the output directory"
        ),
    )
    parser.add_argument(
        "--trusted-evaluation",
        action="store_true",
        help=(
            "Run in controlled local benchmark mode: raise retrieval privacy "
            "ceiling and allow sensitive facts from retrieved benchmark context"
        ),
    )
    return parser


def _resolve_judge_model(args: argparse.Namespace) -> str | None:
    if args.judge_model is not None:
        return args.judge_model
    if args.provider == "anthropic":
        return _DEFAULT_ANTHROPIC_JUDGE_MODEL
    return None


def _default_diff_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("locomo-report-")
    return report_path.with_name(f"locomo-diff-{timestamp}.json")


def _default_checkpoint_output_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser() / "locomo-checkpoint.json"


async def _run_async(
    args: argparse.Namespace,
) -> tuple[BenchmarkReport, Path, Path | None, Path]:
    benchmark = LoCoMoBenchmark(
        data_path=args.data_path,
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_model=args.model,
        judge_model=_resolve_judge_model(args),
        manifests_dir=args.manifests_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        corrections_path=args.corrections,
        community_corrections_path=args.community_corrections,
    )
    checkpoint_path = (
        Path(args.checkpoint_output).expanduser()
        if args.checkpoint_output is not None
        else _default_checkpoint_output_path(args.output)
    )
    print(f"Checkpoint will be updated at: {checkpoint_path}", flush=True)
    report = await benchmark.run(
        ablation=_parse_ablation(args.ablation),
        conversation_ids=_parse_csv_list(args.conversations),
        categories=_parse_categories(args.categories),
        question_ids=_parse_csv_list(args.questions),
        max_questions=args.max_questions,
        max_turns=args.max_turns,
        checkpoint_path=checkpoint_path,
        trusted_evaluation=args.trusted_evaluation,
    )
    report_path = benchmark.save_report(report, args.output)

    diff_path = None
    if args.diff_against is not None:
        baseline_report = load_benchmark_report(args.diff_against)
        diff_report = build_benchmark_diff(
            baseline_report,
            report,
            before_label=Path(args.diff_against).name,
            after_label=report_path.name,
        )
        requested_diff_output = (
            Path(args.diff_output).expanduser()
            if args.diff_output is not None
            else _default_diff_output_path(report_path)
        )
        diff_path = save_benchmark_diff(diff_report, requested_diff_output)
    return report, report_path, diff_path, checkpoint_path


def _format_duration(duration_seconds: float) -> str:
    rounded = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _score_counts(results: list[QuestionResult]) -> tuple[int, int]:
    total = len(results)
    correct = sum(result.score_result.score for result in results)
    return correct, total


def _category_counts(report: BenchmarkReport) -> dict[int, tuple[int, int]]:
    counts: dict[int, tuple[int, int]] = {}
    for conversation in report.conversations:
        for result in conversation.results:
            category = result.question.category
            correct, total = counts.get(category, (0, 0))
            counts[category] = (correct + result.score_result.score, total + 1)
    return counts


def _conversation_counts(conversation: ConversationReport) -> tuple[int, int]:
    return _score_counts(conversation.results)


def _format_report_summary(
    report: BenchmarkReport,
    report_path: Path,
    diff_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> str:
    category_stats = _category_counts(report)
    lines = [
        "=" * 40,
        "LoCoMo Benchmark Results",
        "=" * 40,
        f"Overall accuracy: {report.overall_accuracy * 100:.1f}% ({report.total_correct}/{report.total_questions})",
        f"Duration: {_format_duration(report.duration_seconds)}",
        (
            "Model: "
            f"{report.model_info.get('provider', '')} / "
            f"{report.model_info.get('answer_model', '')}"
        ),
        f"Trusted evaluation: {bool(report.model_info.get('trusted_evaluation', False))}",
        "",
        "Category breakdown:",
    ]
    for category, name in _CATEGORY_NAMES.items():
        correct, total = category_stats.get(category, (0, 0))
        if total == 0:
            continue
        accuracy = (correct / total) * 100.0
        lines.append(
            f"  Cat {category} ({name}): {accuracy:6.1f}% ({correct}/{total})"
        )

    lines.append("")
    lines.append("Per-conversation:")
    for conversation in report.conversations:
        correct, total = _conversation_counts(conversation)
        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        lines.append(
            f"  {conversation.conversation_id}: {accuracy:6.1f}% ({correct}/{total})"
        )

    lines.extend(
        [
            "",
            f"Report saved to: {report_path}",
            *(
                [f"Checkpoint saved to: {checkpoint_path}"]
                if checkpoint_path is not None
                else []
            ),
            *([f"Diff saved to: {diff_path}"] if diff_path is not None else []),
            "=" * 40,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the benchmark, and print the report path."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.embedding_backend == "sqlite_vec" and not args.embedding_model:
        parser.error("--embedding-model is required when --embedding-backend is sqlite_vec")
    report, output_path, diff_path, checkpoint_path = asyncio.run(_run_async(args))
    print(
        _format_report_summary(
            report,
            output_path,
            diff_path=diff_path,
            checkpoint_path=checkpoint_path,
        )
    )


if __name__ == "__main__":
    main()
