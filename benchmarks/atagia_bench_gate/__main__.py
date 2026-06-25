"""CLI entry point for the adaptive retrieval gate diagnostic suite.

Runs the gate suite end to end with the adaptive retrieval flag ON and prints a
classification-accuracy and gate-action summary. Use ``--dry-run`` to validate
the dataset and print its shape without making any LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile

from dotenv import load_dotenv

from benchmarks.atagia_bench_gate.dataset import load_gate_suite
from benchmarks.atagia_bench_gate.runner import GateRunReport, GateSuiteRunner


load_dotenv()

_DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.atagia_bench_gate",
        description="Run the adaptive retrieval gate diagnostic suite.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a gate suite JSON dataset (defaults to the bundled v0).",
    )
    parser.add_argument(
        "--manifests-dir",
        default=str(_DEFAULT_MANIFESTS_DIR),
        help="Directory of assistant mode manifests.",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Directory for per-persona benchmark databases (defaults to a temp dir).",
    )
    parser.add_argument("--provider", default=None, help="LLM provider name.")
    parser.add_argument("--api-key", default=None, help="Provider API key.")
    parser.add_argument(
        "--forced-global-model",
        default=None,
        help="Force a single model for every LLM component.",
    )
    parser.add_argument("--chat-model", default=None, help="Answer model id.")
    parser.add_argument("--ingest-model", default=None, help="Ingestion model id.")
    parser.add_argument("--retrieval-model", default=None, help="Retrieval model id.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print its shape without any LLM calls.",
    )
    return parser


def _print_dataset_summary(dataset_file: str | None) -> None:
    dataset = load_gate_suite(dataset_file)
    pair_ids = sorted(
        {q.pair_id for q in dataset.questions if q.pair_id is not None}
    )
    print(f"Suite: {dataset.name}")
    print(f"Personas: {len(dataset.personas)}")
    print(f"Setup conversations: {dataset.total_conversations}")
    print(f"Probe questions: {dataset.total_questions}")
    print(f"Paired topics: {len(pair_ids)}")
    print(f"Languages: {', '.join(sorted(dataset.languages))}")
    by_kind: dict[str, int] = {}
    for question in dataset.questions:
        by_kind[question.probe_kind] = by_kind.get(question.probe_kind, 0) + 1
    for kind in sorted(by_kind):
        print(f"  {kind}: {by_kind[kind]}")


def _print_run_summary(report: GateRunReport) -> None:
    score = report.score
    print(f"Suite: {report.suite_name}")
    print(f"Adaptive retrieval enabled: {report.adaptive_retrieval_enabled}")
    print(f"Probe questions answered: {len(report.results)}")

    classification_accuracy = score.classification_accuracy
    if classification_accuracy is None:
        print("Classification accuracy: unscored (no classification reported)")
    else:
        print(
            f"Classification accuracy: {classification_accuracy:.1%} "
            f"({score.classification_correct}/{score.classification_scored})"
        )
    for language in sorted(score.classification_accuracy_by_language):
        accuracy = score.classification_accuracy_by_language[language]
        print(f"  {language}: {accuracy:.1%}")

    action_accuracy = score.action_accuracy
    if action_accuracy is None:
        print(
            "Gate action accuracy: unscored "
            "(engine did not report a skip/retrieve status)"
        )
    else:
        print(
            f"Gate action accuracy: {action_accuracy:.1%} "
            f"({score.action_correct}/{score.action_scored})"
        )
    print(f"False skips (dangerous): {score.false_skips}")
    print(f"Correct skips: {score.correct_skips}")
    print(f"Missed skips (benign, latency only): {score.missed_skips}")


async def _run(args: argparse.Namespace) -> GateRunReport:
    runner = GateSuiteRunner(
        dataset=load_gate_suite(args.dataset),
        manifests_dir=args.manifests_dir,
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        forced_global_model=args.forced_global_model,
        chat_model=args.chat_model,
        ingest_model=args.ingest_model,
        retrieval_model=args.retrieval_model,
    )
    if args.db_dir is not None:
        db_dir = Path(args.db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        return await runner.run(db_dir / "gate_suite.db")
    with tempfile.TemporaryDirectory(prefix="atagia_gate_suite_") as tmp:
        return await runner.run(Path(tmp) / "gate_suite.db")


def main() -> None:
    """Parse arguments and run the gate suite or its dry-run summary."""
    args = _build_arg_parser().parse_args()
    if args.dry_run:
        _print_dataset_summary(args.dataset)
        return

    report = asyncio.run(_run(args))
    _print_run_summary(report)
    if args.output is not None:
        Path(args.output).write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
