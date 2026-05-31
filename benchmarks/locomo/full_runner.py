"""Two-phase LoCoMo benchmark launcher."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.json_artifacts import write_json_atomic


DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS = 50
_PHASE_CANCEL_WAIT_SECONDS = 10.0


@dataclass(frozen=True)
class FullLoCoMoRunConfig:
    data_path: Path
    output_dir: Path
    db_dir: Path
    provider: str
    api_key: str | None = None
    answer_model: str | None = None
    ingest_model: str | None = None
    retrieval_model: str | None = None
    judge_model: str | None = None
    forced_global_model: str | None = None
    chat_model: str | None = None
    component_models: tuple[str, ...] = ()
    corrections: str | None = None
    community_corrections: str | None = None
    privacy_enforcement: str = "off"
    embedding_backend: str = "none"
    embedding_model: str | None = None
    conversations: str | None = None
    categories: str | None = None
    questions: str | None = None
    max_questions: int | None = None
    max_turns: int | None = None
    parallel_conversations: int = 10
    parallel_questions: int = 3
    adaptive_parallel_questions: bool = False
    adaptive_parallel_min: int = 1
    adaptive_parallel_retries: int = 1
    ingest_mode: str = "online_batch"
    flush_every_turns: int | None = DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS
    diff_against: str | None = None
    skip_ingest: bool = False
    skip_eval: bool = False
    dry_run: bool = False
    locomo_args: tuple[str, ...] = ()
    ingest_args: tuple[str, ...] = ()
    eval_args: tuple[str, ...] = ()


def build_common_locomo_args(config: FullLoCoMoRunConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.locomo",
        "--data-path",
        str(config.data_path),
        "--provider",
        config.provider,
        "--privacy-enforcement",
        config.privacy_enforcement,
        "--embedding-backend",
        config.embedding_backend,
    ]
    optional_flags = {
        "--api-key": config.api_key,
        "--answer-model": config.answer_model,
        "--ingest-model": config.ingest_model,
        "--retrieval-model": config.retrieval_model,
        "--judge-model": config.judge_model,
        "--forced-global-model": config.forced_global_model,
        "--chat-model": config.chat_model,
        "--corrections": config.corrections,
        "--community-corrections": config.community_corrections,
        "--embedding-model": config.embedding_model,
        "--conversations": config.conversations,
        "--categories": config.categories,
        "--questions": config.questions,
    }
    for flag, value in optional_flags.items():
        if value is not None:
            command.extend([flag, str(value)])
    for component_model in config.component_models:
        command.extend(["--component-model", component_model])
    if config.max_questions is not None:
        command.extend(["--max-questions", str(config.max_questions)])
    if config.max_turns is not None:
        command.extend(["--max-turns", str(config.max_turns)])
    command.extend(config.locomo_args)
    return command


def build_ingest_command(config: FullLoCoMoRunConfig) -> list[str]:
    flush_every_turns = _effective_flush_every_turns(config)
    command = [
        *build_common_locomo_args(config),
        "--output",
        str(config.output_dir / "ingest"),
        "--checkpoint-output",
        str(config.output_dir / "ingest" / "locomo-checkpoint.json"),
        "--ingest-only",
        "--keep-db",
        "--benchmark-db-dir",
        str(config.db_dir),
        "--parallel-conversations",
        str(config.parallel_conversations),
        "--ingest-mode",
        config.ingest_mode,
    ]
    if flush_every_turns is not None:
        command.extend(["--flush-every-turns", str(flush_every_turns)])
    command.extend(config.ingest_args)
    return command


def build_evaluate_command(config: FullLoCoMoRunConfig) -> list[str]:
    command = [
        *build_common_locomo_args(config),
        "--output",
        str(config.output_dir / "evaluate"),
        "--checkpoint-output",
        str(config.output_dir / "evaluate" / "locomo-checkpoint.json"),
        "--evaluate-only",
        "--reuse-db-dir",
        str(config.db_dir),
        "--parallel-conversations",
        str(config.parallel_conversations),
        "--parallel-questions",
        str(config.parallel_questions),
    ]
    if config.adaptive_parallel_questions:
        command.extend(
            [
                "--adaptive-parallel-questions",
                "--adaptive-parallel-min",
                str(config.adaptive_parallel_min),
                "--adaptive-parallel-retries",
                str(config.adaptive_parallel_retries),
            ]
        )
    if config.diff_against is not None:
        command.extend(
            [
                "--diff-against",
                config.diff_against,
                "--diff-output",
                str(config.output_dir / "evaluate" / "locomo-diff.json"),
            ]
        )
    command.extend(config.eval_args)
    return command


def run_full_locomo(config: FullLoCoMoRunConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.db_dir.mkdir(parents=True, exist_ok=True)
    phases: list[dict[str, Any]] = []
    if not config.skip_ingest:
        ingest_phase = _run_phase("ingest", build_ingest_command(config), config)
        phases.append(ingest_phase)
    else:
        ingest_phase = None
    if (
        not config.skip_eval
        and (
            ingest_phase is None
            or ingest_phase.get("returncode") in (0, None)
        )
    ):
        phases.append(_run_phase("evaluate", build_evaluate_command(config), config))
    elif not config.skip_eval:
        phases.append(
            _skipped_phase(
                "evaluate",
                build_evaluate_command(config),
                reason="ingest_failed",
            )
        )
    manifest_path = config.output_dir / "locomo-full-run-manifest.json"
    manifest = {
        "manifest_kind": "locomo_full_run_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(config.data_path),
        "output_dir": str(config.output_dir),
        "db_dir": str(config.db_dir),
        "dry_run": config.dry_run,
        "phases": phases,
        "artifacts": _collect_phase_artifacts(config.output_dir),
    }
    write_json_atomic(manifest_path, manifest)
    failed = [
        phase
        for phase in phases
        if phase["returncode"] not in (0, None) and not phase.get("skipped")
    ]
    if failed:
        names = ", ".join(str(phase["phase"]) for phase in failed)
        raise SystemExit(f"LoCoMo full runner failed phase(s): {names}")
    return manifest_path


def _run_phase(
    phase: str,
    command: list[str],
    config: FullLoCoMoRunConfig,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    phase_output_dir = config.output_dir / phase
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    if config.dry_run:
        return {
            "phase": phase,
            "command": command,
            "command_text": shlex.join(command),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "returncode": None,
            "dry_run": True,
        }
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2])
    cancelled = False
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        cancelled = True
        process.terminate()
        deadline = time.monotonic() + _PHASE_CANCEL_WAIT_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()
        returncode = process.wait()
    return {
        "phase": phase,
        "command": command,
        "command_text": shlex.join(command),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "dry_run": False,
        "cancelled": cancelled,
    }


def _skipped_phase(phase: str, command: list[str], *, reason: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "phase": phase,
        "command": command,
        "command_text": shlex.join(command),
        "started_at": now,
        "finished_at": now,
        "returncode": None,
        "dry_run": False,
        "skipped": True,
        "skipped_reason": reason,
    }


def _effective_flush_every_turns(config: FullLoCoMoRunConfig) -> int | None:
    if config.ingest_mode == "online_batch":
        return config.flush_every_turns or DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS
    return config.flush_every_turns


def _collect_phase_artifacts(output_dir: Path) -> dict[str, list[dict[str, str | None]]]:
    artifacts: dict[str, list[dict[str, str | None]]] = {}
    for phase_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        artifacts[phase_dir.name] = [
            {
                "path": str(path),
                "sha256": sha256_file_if_exists(path),
            }
            for path in sorted(phase_dir.glob("*.json"))
        ]
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full LoCoMo in ingest/evaluate phases.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--answer-model")
    parser.add_argument("--ingest-model")
    parser.add_argument("--retrieval-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--forced-global-model")
    parser.add_argument("--chat-model")
    parser.add_argument("--component-model", action="append", default=[])
    parser.add_argument("--corrections")
    parser.add_argument("--community-corrections")
    parser.add_argument("--privacy-enforcement", default="off")
    parser.add_argument("--embedding-backend", default="none")
    parser.add_argument("--embedding-model")
    parser.add_argument("--conversations")
    parser.add_argument("--categories")
    parser.add_argument("--questions")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--parallel-conversations", type=int, default=10)
    parser.add_argument("--parallel-questions", type=int, default=3)
    parser.add_argument("--adaptive-parallel-questions", action="store_true")
    parser.add_argument("--adaptive-parallel-min", type=int, default=1)
    parser.add_argument("--adaptive-parallel-retries", type=int, default=1)
    parser.add_argument(
        "--ingest-mode",
        choices=("online", "online_async", "online_batch", "bulk"),
        default="online_batch",
    )
    parser.add_argument(
        "--flush-every-turns",
        type=int,
        help=(
            "Turns between worker drains for online_batch. Defaults to "
            f"{DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS} in full-run mode."
        ),
    )
    parser.add_argument("--diff-against")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--locomo-arg", action="append", default=[])
    parser.add_argument("--ingest-arg", action="append", default=[])
    parser.add_argument("--eval-arg", action="append", default=[])
    return parser


def _config_from_args(args: argparse.Namespace) -> FullLoCoMoRunConfig:
    return FullLoCoMoRunConfig(
        data_path=Path(args.data_path).expanduser(),
        output_dir=Path(args.output).expanduser(),
        db_dir=Path(args.db_dir).expanduser(),
        provider=args.provider,
        api_key=args.api_key,
        answer_model=args.answer_model,
        ingest_model=args.ingest_model,
        retrieval_model=args.retrieval_model,
        judge_model=args.judge_model,
        forced_global_model=args.forced_global_model,
        chat_model=args.chat_model,
        component_models=tuple(args.component_model or ()),
        corrections=args.corrections,
        community_corrections=args.community_corrections,
        privacy_enforcement=args.privacy_enforcement,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        conversations=args.conversations,
        categories=args.categories,
        questions=args.questions,
        max_questions=args.max_questions,
        max_turns=args.max_turns,
        parallel_conversations=args.parallel_conversations,
        parallel_questions=args.parallel_questions,
        adaptive_parallel_questions=args.adaptive_parallel_questions,
        adaptive_parallel_min=args.adaptive_parallel_min,
        adaptive_parallel_retries=args.adaptive_parallel_retries,
        ingest_mode=args.ingest_mode,
        flush_every_turns=(
            args.flush_every_turns
            if args.flush_every_turns is not None
            else (
                DEFAULT_FULL_RUN_FLUSH_EVERY_TURNS
                if args.ingest_mode == "online_batch"
                else None
            )
        ),
        diff_against=args.diff_against,
        skip_ingest=args.skip_ingest,
        skip_eval=args.skip_eval,
        dry_run=args.dry_run,
        locomo_args=tuple(args.locomo_arg or ()),
        ingest_args=tuple(args.ingest_arg or ()),
        eval_args=tuple(args.eval_arg or ()),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.parallel_conversations < 1:
        parser.error("--parallel-conversations must be at least 1")
    if args.parallel_questions < 1:
        parser.error("--parallel-questions must be at least 1")
    if args.adaptive_parallel_min < 1:
        parser.error("--adaptive-parallel-min must be at least 1")
    if args.adaptive_parallel_min > args.parallel_questions:
        parser.error("--adaptive-parallel-min cannot exceed --parallel-questions")
    if args.adaptive_parallel_retries < 0:
        parser.error("--adaptive-parallel-retries must be non-negative")
    if args.flush_every_turns is not None and args.flush_every_turns < 1:
        parser.error("--flush-every-turns must be at least 1")
    if args.ingest_mode in {"bulk", "online_async"} and args.flush_every_turns is not None:
        parser.error(f"--flush-every-turns cannot be combined with --ingest-mode {args.ingest_mode}")
    manifest_path = run_full_locomo(_config_from_args(args))
    print(json.dumps({"manifest_path": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
