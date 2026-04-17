"""CLI entry point for running the Atagia-bench v0 benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from benchmarks.atagia_bench.runner import AtagiaBenchReport, AtagiaBenchRunner
from benchmarks.atagia_bench.report_diff import (
    build_diff,
    format_diff_summary,
    load_atagia_bench_report,
    save_diff,
)
from atagia.models.schemas_replay import AblationConfig

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
_DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-opus-4-6"


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
    }
    preset = presets.get(raw_value)
    if preset is not None:
        return preset
    return AblationConfig.model_validate(json.loads(raw_value))


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
        required=True,
        help="LLM model for answer generation",
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
            "LLM model for scoring; defaults to claude-opus-4-6 for "
            "Anthropic, otherwise the answer model"
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
            "no_revision, no_compaction) or JSON AblationConfig"
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
    return parser


def _resolve_judge_model(args: argparse.Namespace) -> str | None:
    if args.judge_model is not None:
        return args.judge_model
    if args.provider == "anthropic":
        return _DEFAULT_ANTHROPIC_JUDGE_MODEL
    return None


def _default_diff_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("atagia-bench-report-")
    return report_path.with_name(f"atagia-bench-diff-{timestamp}.json")


async def _run_async(
    args: argparse.Namespace,
) -> tuple[AtagiaBenchReport, Path, Path | None]:
    runner = AtagiaBenchRunner(
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_model=args.model,
        judge_model=_resolve_judge_model(args),
        manifests_dir=args.manifests_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        data_dir=args.data_dir,
    )
    report = await runner.run(
        persona_ids=_parse_csv_list(args.personas),
        category_tags=_parse_csv_list(args.categories),
        ablation=_parse_ablation(args.ablation),
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
        )
        requested_diff_output = (
            Path(args.diff_output).expanduser()
            if args.diff_output is not None
            else _default_diff_output_path(report_path)
        )
        diff_path = save_diff(diff_report, requested_diff_output)

    return report, report_path, diff_path, diff_report


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
) -> str:
    lines = [
        "=" * 50,
        "Atagia-bench v0 Results",
        "=" * 50,
        f"Pass rate: {report.pass_rate:.1%} ({report.total_passed}/{report.total_questions})",
        f"Avg score: {report.avg_score:.3f}",
        f"Critical errors: {report.critical_error_count}",
        f"Duration: {_format_duration(report.run_duration_seconds)}",
        f"Personas: {', '.join(report.personas_used)}",
        f"Model: {report.config.get('provider', '')} / {report.config.get('answer_model', '')}",
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
            lines.append(
                f"  {r.question_id}: {r.grade.reason[:80]}"
            )
        if len(failed) > 20:
            lines.append(f"  ... and {len(failed) - 20} more")

    lines.extend([
        "",
        f"Report saved to: {report_path}",
        *(
            [f"Diff saved to: {diff_path}"]
            if diff_path is not None
            else []
        ),
        "=" * 50,
    ])
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the benchmark, and print results."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.embedding_backend == "sqlite_vec" and not args.embedding_model:
        parser.error(
            "--embedding-model is required when --embedding-backend is sqlite_vec"
        )

    report, report_path, diff_path, diff_report = asyncio.run(_run_async(args))

    print(_format_report_summary(report, report_path, diff_path=diff_path))

    if diff_report is not None:
        print()
        print(format_diff_summary(diff_report))


if __name__ == "__main__":
    main()
