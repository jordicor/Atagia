"""Run retained LoCoMo conversation shards in parallel and aggregate reports.

This runner is intentionally small: it shells out to the existing
``python -m benchmarks.locomo`` CLI once per retained conversation, so each shard
can use its own copied SQLite DB and output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.custody_report import (
    build_failed_question_custody_report,
    save_failed_question_custody_report,
)
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.report_aggregate import (
    build_combined_report,
    save_combined_report,
    save_combined_run_manifest,
)
from benchmarks.report_diff import load_benchmark_report


@dataclass(frozen=True)
class RetainedLoCoMoJob:
    """One retained LoCoMo conversation shard."""

    name: str
    output: Path
    conversations: list[str]
    questions: list[str]
    reuse_db: Path | None = None
    extra_args: list[str] | None = None


@dataclass(frozen=True)
class RetainedLoCoMoRunConfig:
    """Config for a retained LoCoMo parallel run."""

    base_args: list[str]
    jobs: list[RetainedLoCoMoJob]
    aggregate_output: Path | None = None
    duplicate_strategy: str = "last"


def load_config(path: str | Path) -> RetainedLoCoMoRunConfig:
    """Load a retained LoCoMo runner config from JSON."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Runner config must be a JSON object")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Runner config must contain a non-empty jobs list")
    base_args = payload.get("base_args") or []
    if not isinstance(base_args, list) or not all(isinstance(arg, str) for arg in base_args):
        raise ValueError("base_args must be a list of strings")
    jobs = [_load_job(entry) for entry in raw_jobs]
    aggregate_output = payload.get("aggregate_output")
    duplicate_strategy = str(payload.get("duplicate_strategy") or "last")
    if duplicate_strategy not in {"first", "last", "error"}:
        raise ValueError("duplicate_strategy must be first, last, or error")
    return RetainedLoCoMoRunConfig(
        base_args=list(base_args),
        jobs=jobs,
        aggregate_output=(
            Path(aggregate_output).expanduser()
            if isinstance(aggregate_output, str) and aggregate_output
            else None
        ),
        duplicate_strategy=duplicate_strategy,
    )


def _load_job(entry: object) -> RetainedLoCoMoJob:
    if not isinstance(entry, dict):
        raise ValueError("Each job must be a JSON object")
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("Each job requires a name")
    output = entry.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError(f"Job {name} requires an output path")
    conversations = _string_list(entry.get("conversations") or entry.get("conversation"))
    questions = _string_list(entry.get("questions"))
    if not conversations:
        raise ValueError(f"Job {name} requires at least one conversation")
    if not questions:
        raise ValueError(f"Job {name} requires at least one question")
    reuse_db = entry.get("reuse_db")
    extra_args = entry.get("extra_args") or []
    if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError(f"Job {name} extra_args must be a list of strings")
    return RetainedLoCoMoJob(
        name=name,
        output=Path(output).expanduser(),
        conversations=conversations,
        questions=questions,
        reuse_db=(
            Path(reuse_db).expanduser()
            if isinstance(reuse_db, str) and reuse_db
            else None
        ),
        extra_args=list(extra_args),
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    return []


def build_locomo_command(
    job: RetainedLoCoMoJob,
    *,
    base_args: list[str],
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the subprocess command for one retained LoCoMo shard."""
    command = [
        python_executable,
        "-m",
        "benchmarks.locomo",
        *base_args,
        "--output",
        str(job.output),
        "--conversations",
        ",".join(job.conversations),
        "--questions",
        ",".join(job.questions),
    ]
    if job.reuse_db is not None:
        command.extend(["--reuse-db", str(job.reuse_db), "--evaluate-only"])
    if job.extra_args:
        command.extend(job.extra_args)
    return command


async def run_jobs(
    config: RetainedLoCoMoRunConfig,
    *,
    max_workers: int,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run configured jobs concurrently and return a shard manifest."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    semaphore = asyncio.Semaphore(max_workers)

    async def run_one(job: RetainedLoCoMoJob) -> dict[str, Any]:
        command = build_locomo_command(job, base_args=config.base_args)
        if dry_run:
            return {
                "name": job.name,
                "command": command,
                "returncode": None,
                "report_path": None,
                "log_path": None,
            }
        job.output.mkdir(parents=True, exist_ok=True)
        log_path = job.output / "retained-slice-runner.log"
        started_at = datetime.now(timezone.utc)
        async with semaphore:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _stderr = await process.communicate()
        log_path.write_bytes(stdout or b"")
        if process.returncode != 0:
            raise RuntimeError(
                f"LoCoMo shard {job.name} failed with exit code {process.returncode}; "
                f"see {log_path}"
            )
        report_path = latest_report_path(job.output)
        return {
            "name": job.name,
            "command": command,
            "returncode": process.returncode,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "report_path": str(report_path),
            "report_sha256": sha256_file_if_exists(report_path),
            "log_path": str(log_path),
            "log_sha256": sha256_file_if_exists(log_path),
        }

    return await asyncio.gather(*(run_one(job) for job in config.jobs))


def latest_report_path(output_dir: str | Path) -> Path:
    """Return the newest LoCoMo report in an output directory."""
    reports = sorted(Path(output_dir).expanduser().glob("locomo-report-*.json"))
    if not reports:
        raise FileNotFoundError(f"No locomo-report-*.json found in {output_dir}")
    return reports[-1]


def aggregate_reports(
    report_paths: list[str | Path],
    *,
    output_path: str | Path,
    duplicate_strategy: str,
) -> dict[str, Any]:
    """Aggregate shard reports and persist report, custody, and manifest artifacts."""
    reports = [load_benchmark_report(path) for path in report_paths]
    report_path_strings = [str(Path(path).expanduser()) for path in report_paths]
    combined = build_combined_report(
        reports,
        source_paths=report_path_strings,
        source_hashes=[sha256_file_if_exists(path) for path in report_paths],
        duplicate_strategy=duplicate_strategy,  # type: ignore[arg-type]
    )
    output = save_combined_report(combined, output_path)
    custody_path = save_failed_question_custody_report(
        build_failed_question_custody_report(combined, source_report=str(output)),
        output.with_name(f"{output.stem}-failed-custody.json"),
    )
    manifest_path = save_combined_run_manifest(
        combined,
        report_path=output,
        custody_path=custody_path,
    )
    return {
        "report_path": str(output),
        "report_sha256": sha256_file_if_exists(output),
        "custody_path": str(custody_path),
        "custody_sha256": sha256_file_if_exists(custody_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file_if_exists(manifest_path),
        "total_questions": combined.total_questions,
        "total_correct": combined.total_correct,
        "overall_accuracy": combined.overall_accuracy,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Retained slice runner JSON config")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent shards")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--aggregate-output",
        default=None,
        help="Override aggregate report output path",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Path for this runner's own manifest JSON",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.aggregate_output:
        config = RetainedLoCoMoRunConfig(
            base_args=config.base_args,
            jobs=config.jobs,
            aggregate_output=Path(args.aggregate_output).expanduser(),
            duplicate_strategy=config.duplicate_strategy,
        )
    shard_results = await run_jobs(
        config,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    aggregate_result = None
    if not args.dry_run and config.aggregate_output is not None:
        aggregate_result = aggregate_reports(
            [result["report_path"] for result in shard_results],
            output_path=config.aggregate_output,
            duplicate_strategy=config.duplicate_strategy,
        )
    manifest = {
        "manifest_kind": "locomo_retained_slice_runner",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).expanduser()),
        "max_workers": args.max_workers,
        "dry_run": bool(args.dry_run),
        "base_args": config.base_args,
        "shards": shard_results,
        "aggregate": aggregate_result,
    }
    manifest_path = (
        Path(args.manifest_output).expanduser()
        if args.manifest_output
        else (
            config.aggregate_output.with_name(
                f"{config.aggregate_output.stem}-runner-manifest.json"
            )
            if config.aggregate_output is not None
            else Path("locomo-retained-slice-runner-manifest.json")
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest)
    print(f"Runner manifest saved to: {manifest_path}", flush=True)
    if aggregate_result is not None:
        print(
            "Aggregate: "
            f"{aggregate_result['total_correct']}/{aggregate_result['total_questions']} "
            f"({aggregate_result['overall_accuracy']:.1%})",
            flush=True,
        )
    return 0


def main() -> None:
    """Run retained LoCoMo shards."""
    args = _build_parser().parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    raise SystemExit(asyncio.run(_run_async(args)))


if __name__ == "__main__":
    main()
