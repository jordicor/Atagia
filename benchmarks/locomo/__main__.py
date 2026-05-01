"""CLI entry point for running the LoCoMo benchmark."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from benchmarks.base import (
    DEFAULT_SCORED_CATEGORIES,
    BenchmarkReport,
    ConversationReport,
    QuestionResult,
)
from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.custody_report import (
    build_failed_question_custody_report,
    save_failed_question_custody_report,
)
from benchmarks.custody_summary import format_retrieval_custody_summary
from benchmarks.failure_taxonomy import (
    build_failure_taxonomy_report,
    failure_taxonomy_manifest_summary,
    format_failure_taxonomy_summary,
    save_failure_taxonomy_report,
)
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from benchmarks.report_diff import (
    BenchmarkDiffReport,
    build_benchmark_diff,
    format_diff_summary,
    load_benchmark_report,
    save_benchmark_diff,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.model_resolution import COMPONENTS_BY_ID

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
_DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-opus-4-6"
_BENCHMARK_DB_FILENAME = "benchmark.db"
_BENCHMARK_DB_METADATA_FILENAME = "run_metadata.json"
_BENCHMARK_INGESTION_PROGRESS_FILENAME = "ingestion_progress.json"
_CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
}
_VALIDATION_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")


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
        "composer_budgeted_marginal": AblationConfig(composer_strategy="budgeted_marginal"),
    }
    preset = presets.get(raw_value)
    if preset is not None:
        return preset
    return AblationConfig.model_validate(json.loads(raw_value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LoCoMo benchmark against Atagia.")
    parser.add_argument("--data-path", default=None, help="Path to locomo10.json")
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory where the benchmark report JSON should be written",
    )
    parser.add_argument("--provider", default=None, help="LLM provider name")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument(
        "--model",
        default=None,
        help="Legacy alias for --answer-model; used only for benchmark answer generation.",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Model spec for benchmark answer generation, e.g. openrouter/google/gemini-3.1-flash-lite-preview,medium.",
    )
    parser.add_argument(
        "--answer-prompt-variant",
        choices=("default", "grounded_connect", "light_world_knowledge"),
        default="default",
        help=(
            "Benchmark-only answer prompt variant. Use default for the normal "
            "Atagia prompt; grounded_connect and light_world_knowledge add "
            "experimental instructions for answer generation only."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Optional LLM model for scoring; defaults to claude-opus-4-6 for "
            "Anthropic, otherwise the answer model"
        ),
    )
    parser.add_argument(
        "--forced-global-model",
        default=None,
        help=(
            "Force one model for every Atagia internal LLM component. "
            "This is useful for old one-model baselines; prefer phase/component "
            "flags for tuning."
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
        "--chat-model",
        default=None,
        help="Model for Atagia chat component fallback when --answer-model is omitted.",
    )
    parser.add_argument(
        "--component-model",
        action="append",
        default=[],
        metavar="COMPONENT=MODEL",
        help=(
            "Override one Atagia LLM component model. Repeatable. "
            "Examples: extractor=openai/gpt-4o-mini, "
            "need_detector=openrouter/google/gemini-3.1-flash-lite-preview,medium"
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
    parser.add_argument(
        "--parallel-conversations",
        type=int,
        default=1,
        help=(
            "Number of LoCoMo conversations to run concurrently. Each conversation "
            "uses its own SQLite DB and per-conversation checkpoint."
        ),
    )
    parser.add_argument(
        "--parallel-questions",
        type=int,
        default=1,
        help=(
            "Number of questions to evaluate concurrently inside each conversation "
            "after ingestion has completed."
        ),
    )
    parser.add_argument(
        "--ingest-mode",
        choices=("online", "bulk"),
        default="online",
        help=(
            "Conversation ingestion mode. online preserves the per-turn worker drain "
            "baseline; bulk persists turns first and rebuilds memory in phases."
        ),
    )
    parser.add_argument(
        "--benchmark-db-dir",
        default=None,
        help=(
            "Directory for retained benchmark SQLite snapshots. Defaults to "
            "docs/tmp/benchmark_dbs when --keep-db is used."
        ),
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep each conversation's benchmark SQLite DB and write run metadata.",
    )
    parser.add_argument(
        "--reuse-db",
        default=None,
        help=(
            "Evaluate one selected conversation from an existing benchmark.db "
            "or a directory containing benchmark.db. Implies --evaluate-only."
        ),
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Ingest selected conversations, drain workers, keep metadata, and skip question evaluation.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip ingestion and evaluate from --reuse-db.",
    )
    parser.add_argument(
        "--list-benchmark-dbs",
        action="store_true",
        help="List retained benchmark DB snapshots under --benchmark-db-dir and exit.",
    )
    parser.add_argument(
        "--list-benchmark-dbs-json",
        action="store_true",
        help="List retained benchmark DB snapshots as JSON and exit.",
    )
    parser.add_argument(
        "--diff-benchmark-db-list",
        nargs=2,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        default=None,
        help="Diff two --list-benchmark-dbs-json snapshots and exit.",
    )
    parser.add_argument(
        "--diff-benchmark-db-list-json",
        action="store_true",
        help="Render --diff-benchmark-db-list as JSON.",
    )
    parser.add_argument(
        "--summarize-run-log",
        default=None,
        help="Summarize a LoCoMo run log and exit.",
    )
    parser.add_argument(
        "--summarize-run-log-json",
        action="store_true",
        help="Summarize --summarize-run-log as JSON.",
    )
    return parser


def _resolve_judge_model(args: argparse.Namespace) -> str | None:
    if args.judge_model is not None:
        return args.judge_model
    if args.provider == "anthropic":
        return _DEFAULT_ANTHROPIC_JUDGE_MODEL
    return None


def _resolve_answer_model(args: argparse.Namespace) -> str | None:
    return args.answer_model or args.model


def _parse_component_model_overrides(raw_values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_value in raw_values or []:
        if "=" not in raw_value:
            raise ValueError(
                "--component-model must use COMPONENT=MODEL, "
                f"got {raw_value!r}"
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
    timestamp = report_path.stem.removeprefix("locomo-report-")
    return report_path.with_name(f"locomo-diff-{timestamp}.json")


def _default_checkpoint_output_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser() / "locomo-checkpoint.json"


def _default_custody_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("locomo-report-")
    return report_path.with_name(f"locomo-failed-custody-{timestamp}.json")


def _default_taxonomy_output_path(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("locomo-report-")
    return report_path.with_name(f"locomo-failure-taxonomy-{timestamp}.json")


async def _run_async(
    args: argparse.Namespace,
) -> tuple[BenchmarkReport, Path, Path | None, Path, Path, Path, Path, dict[str, object], BenchmarkDiffReport | None]:
    benchmark = LoCoMoBenchmark(
        data_path=args.data_path,
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_model=args.model,
        answer_model=args.answer_model,
        judge_model=_resolve_judge_model(args),
        forced_global_model=args.forced_global_model,
        ingest_model=args.ingest_model,
        retrieval_model=args.retrieval_model,
        chat_model_override=args.chat_model,
        component_models=_parse_component_model_overrides(args.component_model),
        answer_prompt_variant=args.answer_prompt_variant,
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
        benchmark_db_dir=args.benchmark_db_dir,
        keep_db=args.keep_db,
        reuse_db=args.reuse_db,
        ingest_only=args.ingest_only,
        evaluate_only=args.evaluate_only,
        parallel_conversations=args.parallel_conversations,
        parallel_questions=args.parallel_questions,
        ingest_mode=args.ingest_mode,
    )
    report_path = benchmark.save_report(report, args.output)

    diff_path = None
    diff_report = None
    if args.diff_against is not None:
        baseline_report = load_benchmark_report(args.diff_against)
        diff_report = build_benchmark_diff(
            baseline_report,
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
        diff_path = save_benchmark_diff(diff_report, requested_diff_output)
    custody_path = _default_custody_output_path(report_path)
    save_failed_question_custody_report(
        build_failed_question_custody_report(report, source_report=str(report_path)),
        custody_path,
    )
    taxonomy_report = build_failure_taxonomy_report(report, source_report=str(report_path))
    taxonomy_summary = failure_taxonomy_manifest_summary(taxonomy_report)
    taxonomy_path = _default_taxonomy_output_path(report_path)
    save_failure_taxonomy_report(taxonomy_report, taxonomy_path)
    manifest_path = benchmark.save_run_manifest(
        report,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        diff_path=diff_path,
        custody_path=custody_path,
        taxonomy_path=taxonomy_path,
        failure_taxonomy_summary=taxonomy_summary,
    )
    return (
        report,
        report_path,
        diff_path,
        checkpoint_path,
        manifest_path,
        custody_path,
        taxonomy_path,
        taxonomy_summary,
        diff_report,
    )


def _benchmark_db_dir(args: argparse.Namespace) -> Path:
    if args.benchmark_db_dir is not None:
        return Path(args.benchmark_db_dir).expanduser()
    return Path(__file__).resolve().parents[2] / "docs" / "tmp" / "benchmark_dbs"


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _format_ingestion_turns(progress: dict[str, object]) -> str:
    ingested_turns = progress.get("ingested_turns")
    selected_turns = progress.get("selected_turns")
    if ingested_turns is None and selected_turns is None:
        return ""
    if selected_turns is None:
        return str(ingested_turns or "")
    return f"{ingested_turns or 0}/{selected_turns}"


def _retained_db_dir_hints(path: Path) -> tuple[str, str]:
    name = path.name
    if not name.startswith("locomo_"):
        return "", ""
    suffix = name.removeprefix("locomo_")
    if "_" not in suffix:
        return suffix, ""
    conversation_id, timestamp = suffix.rsplit("_", 1)
    return conversation_id, timestamp


def _retained_db_file_state(db_path: Path) -> dict[str, object]:
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    stats = {
        path: path.stat()
        for path in (db_path, wal_path, shm_path)
        if path.exists()
    }
    db_bytes = stats[db_path].st_size if db_path in stats else 0
    wal_bytes = stats[wal_path].st_size if wal_path in stats else 0
    shm_bytes = stats[shm_path].st_size if shm_path in stats else 0
    file_updated_at = (
        datetime.fromtimestamp(
            max(stat.st_mtime for stat in stats.values()),
            tz=timezone.utc,
        ).isoformat()
        if stats
        else None
    )
    return {
        "db_bytes": db_bytes,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "total_bytes": db_bytes + wal_bytes + shm_bytes,
        "file_updated_at": file_updated_at,
    }


def _retained_db_sqlite_counts(db_path: Path) -> dict[str, int | None]:
    table_counts = {
        "message_count": "messages",
        "memory_object_count": "memory_objects",
        "memory_embedding_metadata_count": "memory_embedding_metadata",
        "summary_view_count": "summary_views",
        "retrieval_event_count": "retrieval_events",
        "artifact_count": "artifacts",
        "artifact_chunk_count": "artifact_chunks",
        "conversation_topic_count": "conversation_topics",
        "conversation_topic_event_count": "conversation_topic_events",
        "conversation_topic_source_count": "conversation_topic_sources",
    }
    if not db_path.exists():
        return {field_name: None for field_name in table_counts}
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return {field_name: None for field_name in table_counts}
    try:
        return {
            field_name: _sqlite_table_count(connection, table_name)
            for field_name, table_name in table_counts.items()
        }
    finally:
        connection.close()


def _sqlite_table_count(connection: sqlite3.Connection, table_name: str) -> int | None:
    try:
        row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return int(row[0])


def _collect_benchmark_db_entries(db_dir: Path) -> list[dict[str, object]]:
    if not db_dir.exists():
        return []
    entries: list[dict[str, object]] = []
    seen_dirs: set[Path] = set()
    for metadata_path in sorted(db_dir.rglob(_BENCHMARK_DB_METADATA_FILENAME)):
        metadata = _read_json_object(metadata_path)
        if metadata is None:
            continue
        progress_path = metadata_path.with_name(_BENCHMARK_INGESTION_PROGRESS_FILENAME)
        progress = _read_json_object(progress_path) or {}
        db_path = metadata_path.parent / _BENCHMARK_DB_FILENAME
        seen_dirs.add(metadata_path.parent)
        entries.append(
            {
                "source": "metadata",
                "timestamp": str(metadata.get("created_at") or progress.get("updated_at") or ""),
                "conversation_id": str(metadata.get("conversation_id") or progress.get("conversation_id") or ""),
                "turns": _format_ingestion_turns(progress) or str(metadata.get("turn_count") or ""),
                "turn_count": metadata.get("turn_count"),
                "selected_turns": progress.get("selected_turns"),
                "ingested_turns": progress.get("ingested_turns"),
                "model": str(metadata.get("answer_model") or metadata.get("llm_model") or ""),
                "status": str(progress.get("status") or "metadata_complete"),
                "db_path": str(db_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file_if_exists(metadata_path),
                "progress_path": str(progress_path) if progress else None,
                "progress_sha256": (
                    sha256_file_if_exists(progress_path)
                    if progress
                    else None
                ),
                "has_db": db_path.exists(),
                **_retained_db_file_state(db_path),
                **_retained_db_sqlite_counts(db_path),
            }
        )
    for progress_path in sorted(db_dir.rglob(_BENCHMARK_INGESTION_PROGRESS_FILENAME)):
        if progress_path.parent in seen_dirs:
            continue
        progress = _read_json_object(progress_path)
        if progress is None:
            continue
        db_path = progress_path.parent / _BENCHMARK_DB_FILENAME
        seen_dirs.add(progress_path.parent)
        entries.append(
            {
                "source": "progress",
                "timestamp": str(progress.get("updated_at") or ""),
                "conversation_id": str(progress.get("conversation_id") or ""),
                "turns": _format_ingestion_turns(progress),
                "turn_count": None,
                "selected_turns": progress.get("selected_turns"),
                "ingested_turns": progress.get("ingested_turns"),
                "model": "",
                "status": str(progress.get("status") or ""),
                "db_path": str(db_path),
                "metadata_path": None,
                "metadata_sha256": None,
                "progress_path": str(progress_path),
                "progress_sha256": sha256_file_if_exists(progress_path),
                "has_db": db_path.exists(),
                **_retained_db_file_state(db_path),
                **_retained_db_sqlite_counts(db_path),
            }
        )
    for retained_db_path in sorted(db_dir.rglob(_BENCHMARK_DB_FILENAME)):
        if retained_db_path.parent in seen_dirs:
            continue
        conversation_id, timestamp = _retained_db_dir_hints(retained_db_path.parent)
        entries.append(
            {
                "source": "db",
                "timestamp": timestamp,
                "conversation_id": conversation_id,
                "turns": "",
                "turn_count": None,
                "selected_turns": None,
                "ingested_turns": None,
                "model": "",
                "status": "db_present",
                "db_path": str(retained_db_path),
                "metadata_path": None,
                "metadata_sha256": None,
                "progress_path": None,
                "progress_sha256": None,
                "has_db": True,
                **_retained_db_file_state(retained_db_path),
                **_retained_db_sqlite_counts(retained_db_path),
            }
        )
    return sorted(
        entries,
        key=lambda entry: (
            str(entry["timestamp"]),
            str(entry["conversation_id"]),
            str(entry["db_path"]),
        ),
    )


def _format_benchmark_db_list(db_dir: Path) -> str:
    if not db_dir.exists():
        return f"No retained benchmark DB directory found: {db_dir}"
    entries = _collect_benchmark_db_entries(db_dir)
    if not entries:
        return f"No retained benchmark DBs found under: {db_dir}"
    totals = _benchmark_db_entry_totals(entries)
    rows = [
        " | ".join(
            [
                str(entry["timestamp"]),
                str(entry["conversation_id"]),
                str(entry["turns"]),
                str(entry["model"]),
                str(entry["status"]),
                _format_count_cell(entry, "message_count"),
                _format_count_cell(entry, "memory_object_count"),
                _format_count_cell(entry, "summary_view_count"),
                _format_count_cell(entry, "conversation_topic_count"),
                str(entry.get("file_updated_at") or ""),
                str(entry["db_path"]),
            ]
        )
        for entry in entries
    ]
    return "\n".join(
        [
            "timestamp | conversation_id | turns | model | status | messages | memories | summaries | topics | updated | db_path",
            *rows,
            "",
            _format_benchmark_db_totals_text(totals),
        ]
    )


def _format_count_cell(entry: dict[str, object], field_name: str) -> str:
    value = entry.get(field_name)
    return "" if value is None else str(value)


def _format_benchmark_db_list_json(db_dir: Path) -> str:
    entries = _collect_benchmark_db_entries(db_dir)
    payload = {
        "db_dir": str(db_dir),
        "exists": db_dir.exists(),
        "count": len(entries),
        "totals": _benchmark_db_entry_totals(entries),
        "entries": entries,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


_BENCHMARK_DB_DIFF_FIELDS = (
    "total_bytes",
    "message_count",
    "memory_object_count",
    "memory_embedding_metadata_count",
    "summary_view_count",
    "retrieval_event_count",
    "artifact_count",
    "artifact_chunk_count",
    "conversation_topic_count",
    "conversation_topic_event_count",
    "conversation_topic_source_count",
)


def _load_benchmark_db_snapshot(path: str | Path) -> dict[str, object]:
    snapshot_path = Path(path).expanduser()
    payload = _read_json_object(snapshot_path)
    if payload is None:
        raise ValueError(f"Invalid benchmark DB snapshot JSON: {snapshot_path}")
    entries = payload.get("entries")
    totals = payload.get("totals")
    if not isinstance(entries, list) or not isinstance(totals, dict):
        raise ValueError(f"Benchmark DB snapshot is missing entries/totals: {snapshot_path}")
    return payload


def _diff_benchmark_db_snapshots(
    before_path: str | Path,
    after_path: str | Path,
) -> dict[str, object]:
    before = _load_benchmark_db_snapshot(before_path)
    after = _load_benchmark_db_snapshot(after_path)
    before_entries = {
        _benchmark_db_snapshot_entry_key(entry): entry
        for entry in before["entries"]
        if isinstance(entry, dict)
    }
    after_entries = {
        _benchmark_db_snapshot_entry_key(entry): entry
        for entry in after["entries"]
        if isinstance(entry, dict)
    }
    before_totals = before["totals"]
    after_totals = after["totals"]
    if not isinstance(before_totals, dict) or not isinstance(after_totals, dict):
        raise ValueError("Benchmark DB snapshot totals must be objects")
    common_keys = sorted(set(before_entries) & set(after_entries))
    return {
        "before_path": str(Path(before_path).expanduser()),
        "after_path": str(Path(after_path).expanduser()),
        "before_count": before.get("count"),
        "after_count": after.get("count"),
        "new_db_paths": sorted(set(after_entries) - set(before_entries)),
        "missing_db_paths": sorted(set(before_entries) - set(after_entries)),
        "latest_file_updated_at": {
            "before": before_totals.get("latest_file_updated_at"),
            "after": after_totals.get("latest_file_updated_at"),
        },
        "totals_delta": {
            field_name: _numeric_delta(
                before_totals.get(field_name),
                after_totals.get(field_name),
            )
            for field_name in _BENCHMARK_DB_DIFF_FIELDS
        },
        "entries": [
            _diff_benchmark_db_snapshot_entry(
                before_entries[db_path],
                after_entries[db_path],
            )
            for db_path in common_keys
        ],
    }


def _benchmark_db_snapshot_entry_key(entry: dict[str, object]) -> str:
    db_path = entry.get("db_path")
    if isinstance(db_path, str) and db_path:
        return db_path
    return f"{entry.get('conversation_id', '')}:{entry.get('timestamp', '')}"


def _diff_benchmark_db_snapshot_entry(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    return {
        "db_path": _benchmark_db_snapshot_entry_key(after),
        "conversation_id": after.get("conversation_id") or before.get("conversation_id"),
        "status": {
            "before": before.get("status"),
            "after": after.get("status"),
        },
        "file_updated_at": {
            "before": before.get("file_updated_at"),
            "after": after.get("file_updated_at"),
        },
        "deltas": {
            field_name: _numeric_delta(
                before.get(field_name),
                after.get(field_name),
            )
            for field_name in _BENCHMARK_DB_DIFF_FIELDS
        },
    }


def _numeric_delta(before: object, after: object) -> int | None:
    if not isinstance(before, int) or not isinstance(after, int):
        return None
    return after - before


def _format_benchmark_db_snapshot_diff(
    before_path: str | Path,
    after_path: str | Path,
) -> str:
    diff = _diff_benchmark_db_snapshots(before_path, after_path)
    totals_delta = diff["totals_delta"]
    if not isinstance(totals_delta, dict):
        totals_delta = {}
    total_parts = [
        f"{field_name}={value:+d}"
        for field_name, value in totals_delta.items()
        if isinstance(value, int) and value != 0
    ]
    latest_update = diff["latest_file_updated_at"]
    if not isinstance(latest_update, dict):
        latest_update = {}
    lines = [
        "Benchmark DB snapshot diff",
        f"Before: {diff['before_path']}",
        f"After: {diff['after_path']}",
        f"Entries: {diff['before_count']} -> {diff['after_count']}",
        f"Latest update: {latest_update.get('before')} -> {latest_update.get('after')}",
        "Totals delta: " + (" ".join(total_parts) if total_parts else "none"),
    ]
    for label, key in (("New DBs", "new_db_paths"), ("Missing DBs", "missing_db_paths")):
        values = diff.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{label}: {', '.join(str(value) for value in values)}")
    entry_lines = _format_benchmark_db_snapshot_entry_deltas(diff.get("entries"))
    if entry_lines:
        lines.extend(["", "Per DB:", *entry_lines])
    return "\n".join(lines)


def _format_benchmark_db_snapshot_entry_deltas(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    fields = (
        ("messages", "message_count"),
        ("memories", "memory_object_count"),
        ("summaries", "summary_view_count"),
        ("bytes", "total_bytes"),
    )
    for entry in value:
        if not isinstance(entry, dict):
            continue
        deltas = entry.get("deltas")
        if not isinstance(deltas, dict):
            continue
        parts = [
            f"{label}={amount:+d}"
            for label, field_name in fields
            if isinstance(amount := deltas.get(field_name), int) and amount != 0
        ]
        if not parts:
            continue
        lines.append(
            f"{entry.get('conversation_id') or entry.get('db_path')}: "
            + " ".join(parts)
        )
    return lines


def _format_benchmark_db_snapshot_diff_json(
    before_path: str | Path,
    after_path: str | Path,
) -> str:
    return json.dumps(
        _diff_benchmark_db_snapshots(before_path, after_path),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _summarize_run_log(log_path: str | Path) -> dict[str, object]:
    path = Path(log_path).expanduser()
    generated_at = datetime.now(timezone.utc)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "bytes": 0,
            "generated_at": generated_at.isoformat(),
            "file_updated_at": None,
            "seconds_since_update": None,
            "line_count": 0,
            "counts": {},
            "validation_error_schemas": {},
            "validation_error_fields": {},
            "latest_lines": {},
            "ingestion_started_by_conversation": {},
            "turn_progress_by_conversation": {},
            "turn_progress_totals": _run_log_turn_progress_totals({}, {}),
            "question_progress_by_conversation": {},
        }
    counts: Counter[str] = Counter()
    validation_error_schemas: Counter[str] = Counter()
    validation_error_fields: Counter[str] = Counter()
    validation_context_lines_remaining = 0
    line_count = 0
    latest_lines: dict[str, str] = {}
    ingestion_started_by_conversation: dict[str, dict[str, object]] = {}
    turn_progress_by_conversation: dict[str, dict[str, object]] = {}
    question_progress_by_conversation: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line_count += 1
            stripped = line.strip()
            lowered = stripped.lower()
            if validation_context_lines_remaining > 0:
                if _VALIDATION_FIELD_PATTERN.fullmatch(stripped):
                    validation_error_fields[stripped] += 1
                validation_context_lines_remaining -= 1
                if stripped.startswith("The above exception"):
                    validation_context_lines_remaining = 0
            if stripped.startswith("Traceback (most recent call last):"):
                counts["tracebacks"] += 1
                latest_lines["traceback"] = stripped
            if "StructuredOutputError" in stripped:
                counts["structured_output_errors"] += 1
                latest_lines["structured_output_error"] = stripped
            if "ValidationError:" in stripped:
                counts["validation_errors"] += 1
                latest_lines["validation_error"] = stripped
                schema_name = _validation_error_schema(stripped)
                if schema_name is not None:
                    validation_error_schemas[schema_name] += 1
                validation_context_lines_remaining = 40
            if "Failed to process" in stripped and " job " in stripped:
                counts["failed_worker_jobs"] += 1
                latest_lines["failed_worker_job"] = stripped
            if "Failed to process contract job" in stripped:
                counts["contract_worker_failures"] += 1
                latest_lines["contract_worker_failures"] = stripped
            if "Failed to process extraction job" in stripped:
                counts["extraction_worker_failures"] += 1
                latest_lines["extraction_worker_failures"] = stripped
            if "Failed to process compaction job" in stripped:
                counts["compaction_worker_failures"] += 1
                latest_lines["compaction_worker_failures"] = stripped
            if "Failed to process evaluation job" in stripped:
                counts["evaluation_worker_failures"] += 1
                latest_lines["evaluation_worker_failures"] = stripped
            if "Consequence tendency inference fallback" in stripped:
                counts["consequence_tendency_fallback_lines"] += 1
                latest_lines["consequence_tendency_fallback"] = stripped
            if "rate limit" in lowered or "rate_limit" in lowered:
                counts["provider_rate_limit_lines"] += 1
                latest_lines["provider_rate_limit"] = stripped
            if " 429" in lowered or " 529" in lowered:
                counts["provider_status_error_lines"] += 1
                latest_lines["provider_status_error"] = stripped
            if stripped.startswith("Ingesting ") and " turns for " in stripped:
                counts["ingestion_started_lines"] += 1
                latest_lines["ingestion_started"] = stripped
                marker = _parse_run_log_turn_marker(stripped, prefix="Ingesting ")
                if marker is not None:
                    conversation_id, first_count, second_count = marker
                    ingestion_started_by_conversation[conversation_id] = {
                        "line": stripped,
                        "selected_turns": first_count,
                        "source_turns": second_count,
                    }
            if stripped.startswith("Ingested ") and " turns for " in stripped:
                counts["turn_progress_lines"] += 1
                latest_lines["turn_progress"] = stripped
                marker = _parse_run_log_turn_marker(stripped, prefix="Ingested ")
                if marker is not None:
                    conversation_id, first_count, second_count = marker
                    turn_progress_by_conversation[conversation_id] = {
                        "line": stripped,
                        "completed_turns": first_count,
                        "total_turns": second_count,
                    }
            if stripped.startswith("Conversation ") and ": question " in stripped:
                counts["question_started_lines"] += 1
                latest_lines["question_started"] = stripped
                marker = _parse_run_log_question_marker(stripped)
                if marker is not None:
                    conversation_id, current_question, total_questions = marker
                    question_progress_by_conversation[conversation_id] = {
                        "line": stripped,
                        "current_question": current_question,
                        "total_questions": total_questions,
                    }
    stat = path.stat()
    file_updated_at = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    )
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "generated_at": generated_at.isoformat(),
        "file_updated_at": file_updated_at.isoformat(),
        "seconds_since_update": max(0.0, (generated_at - file_updated_at).total_seconds()),
        "line_count": line_count,
        "counts": dict(sorted(counts.items())),
        "validation_error_schemas": dict(sorted(validation_error_schemas.items())),
        "validation_error_fields": dict(sorted(validation_error_fields.items())),
        "latest_lines": dict(sorted(latest_lines.items())),
        "ingestion_started_by_conversation": dict(
            sorted(ingestion_started_by_conversation.items())
        ),
        "turn_progress_by_conversation": dict(
            sorted(turn_progress_by_conversation.items())
        ),
        "turn_progress_totals": _run_log_turn_progress_totals(
            ingestion_started_by_conversation,
            turn_progress_by_conversation,
        ),
        "question_progress_by_conversation": dict(
            sorted(question_progress_by_conversation.items())
        ),
    }


def _format_run_log_summary(log_path: str | Path) -> str:
    summary = _summarize_run_log(log_path)
    if not summary["exists"]:
        return f"No run log found: {summary['path']}"
    counts = summary["counts"]
    count_text = (
        " ".join(f"{key}={counts[key]}" for key in sorted(counts))
        if isinstance(counts, dict) and counts
        else "none"
    )
    return "\n".join(
        [
            f"Run log: {summary['path']}",
            (
                f"Updated at: {summary['file_updated_at']} "
                f"(age_seconds={summary['seconds_since_update']})"
            ),
            f"Bytes: {summary['bytes']} Lines: {summary['line_count']}",
            f"Counts: {count_text}",
            f"Validation schemas: {_format_count_mapping(summary.get('validation_error_schemas'))}",
            f"Validation fields: {_format_count_mapping(summary.get('validation_error_fields'))}",
            f"Latest lines: {_format_latest_run_log_lines(summary.get('latest_lines'))}",
            (
                "Ingestion started by conversation: "
                f"{_format_run_log_turn_map(summary.get('ingestion_started_by_conversation'))}"
            ),
            (
                "Turn progress by conversation: "
                f"{_format_run_log_turn_map(summary.get('turn_progress_by_conversation'))}"
            ),
            (
                "Turn progress totals: "
                f"{_format_run_log_turn_totals(summary.get('turn_progress_totals'))}"
            ),
            (
                "Question progress by conversation: "
                f"{_format_run_log_question_map(summary.get('question_progress_by_conversation'))}"
            ),
        ]
    )


def _parse_run_log_turn_marker(
    line: str,
    *,
    prefix: str,
) -> tuple[str, int, int] | None:
    if not line.startswith(prefix):
        return None
    try:
        count_text, conversation_text = line.removeprefix(prefix).split(
            " turns for ",
            maxsplit=1,
        )
        first_count_text, second_count_text = count_text.split("/", maxsplit=1)
        first_count = int(first_count_text)
        second_count = int(second_count_text)
    except ValueError:
        return None
    conversation_id = conversation_text.strip().rstrip(".")
    if not conversation_id:
        return None
    return conversation_id, first_count, second_count


def _parse_run_log_question_marker(line: str) -> tuple[str, int, int] | None:
    try:
        conversation_text, question_text = line.split(": question ", maxsplit=1)
        current_question_text, total_questions_text = question_text.split("/", maxsplit=1)
        current_question = int(current_question_text)
        total_questions = int(total_questions_text)
        conversation_id = conversation_text.rsplit("(", maxsplit=1)[1].rstrip(")")
    except (IndexError, ValueError):
        return None
    conversation_id = conversation_id.strip()
    if not conversation_id:
        return None
    return conversation_id, current_question, total_questions


def _run_log_turn_progress_totals(
    ingestion_started_by_conversation: dict[str, dict[str, object]],
    turn_progress_by_conversation: dict[str, dict[str, object]],
) -> dict[str, object]:
    conversation_ids = sorted(
        set(ingestion_started_by_conversation) | set(turn_progress_by_conversation)
    )
    completed_turns = 0
    total_turns = 0
    conversations_with_progress = 0
    completed_conversations = 0
    for conversation_id in conversation_ids:
        progress_entry = turn_progress_by_conversation.get(conversation_id, {})
        completed = progress_entry.get("completed_turns")
        total = progress_entry.get("total_turns")
        if isinstance(completed, int) and isinstance(total, int):
            conversations_with_progress += 1
        else:
            completed = 0
            started_entry = ingestion_started_by_conversation.get(conversation_id, {})
            selected_turns = started_entry.get("selected_turns")
            source_turns = started_entry.get("source_turns")
            if isinstance(selected_turns, int):
                total = selected_turns
            elif isinstance(source_turns, int):
                total = source_turns
            else:
                total = 0
        completed = max(0, completed)
        total = max(0, total)
        completed_turns += completed
        total_turns += total
        if total > 0 and completed >= total:
            completed_conversations += 1

    completion_ratio = (
        round(completed_turns / total_turns, 6)
        if total_turns > 0
        else None
    )
    remaining_turns = max(0, total_turns - completed_turns)
    return {
        "conversation_count": len(conversation_ids),
        "conversations_with_progress": conversations_with_progress,
        "completed_conversations": completed_conversations,
        "completed_turns": completed_turns,
        "total_turns": total_turns,
        "remaining_turns": remaining_turns,
        "completion_ratio": completion_ratio,
    }


def _validation_error_schema(line: str) -> str | None:
    for marker in (" validation error for ", " validation errors for "):
        if marker in line:
            schema = line.rsplit(marker, maxsplit=1)[1].strip()
            return schema or None
    return None


def _format_count_mapping(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return " ".join(
        f"{key}={value[key]}"
        for key in sorted(value)
        if isinstance(value.get(key), int)
    )


def _format_latest_run_log_lines(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return " | ".join(
        f"{key}: {value[key]}"
        for key in sorted(value)
        if isinstance(value.get(key), str)
    )


def _format_run_log_turn_map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for conversation_id in sorted(value):
        entry = value.get(conversation_id)
        if not isinstance(entry, dict):
            continue
        completed = entry.get("completed_turns")
        total = entry.get("total_turns")
        selected = entry.get("selected_turns")
        source = entry.get("source_turns")
        if isinstance(completed, int) and isinstance(total, int):
            parts.append(f"{conversation_id}={completed}/{total}")
        elif isinstance(selected, int) and isinstance(source, int):
            parts.append(f"{conversation_id}={selected}/{source}")
    return " ".join(parts) if parts else "none"


def _format_run_log_turn_totals(value: object) -> str:
    if not isinstance(value, dict):
        return "none"
    completed_turns = value.get("completed_turns")
    total_turns = value.get("total_turns")
    remaining_turns = value.get("remaining_turns")
    completion_ratio = value.get("completion_ratio")
    conversation_count = value.get("conversation_count")
    conversations_with_progress = value.get("conversations_with_progress")
    completed_conversations = value.get("completed_conversations")
    if not isinstance(completed_turns, int) or not isinstance(total_turns, int):
        return "none"
    if total_turns <= 0:
        return "none"
    ratio_text = (
        f"{completion_ratio:.1%}"
        if isinstance(completion_ratio, float)
        else "unknown"
    )
    return (
        f"{completed_turns}/{total_turns} ({ratio_text}) "
        f"remaining={remaining_turns} "
        f"conversations={conversation_count} "
        f"with_progress={conversations_with_progress} "
        f"complete={completed_conversations}"
    )


def _format_run_log_question_map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for conversation_id in sorted(value):
        entry = value.get(conversation_id)
        if not isinstance(entry, dict):
            continue
        current_question = entry.get("current_question")
        total_questions = entry.get("total_questions")
        if isinstance(current_question, int) and isinstance(total_questions, int):
            parts.append(f"{conversation_id}={current_question}/{total_questions}")
    return " ".join(parts) if parts else "none"


def _format_run_log_summary_json(log_path: str | Path) -> str:
    return json.dumps(
        _summarize_run_log(log_path),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _benchmark_db_entry_totals(entries: list[dict[str, object]]) -> dict[str, object]:
    count_fields = [
        "db_bytes",
        "wal_bytes",
        "shm_bytes",
        "total_bytes",
        "message_count",
        "memory_object_count",
        "memory_embedding_metadata_count",
        "summary_view_count",
        "retrieval_event_count",
        "artifact_count",
        "artifact_chunk_count",
        "conversation_topic_count",
        "conversation_topic_event_count",
        "conversation_topic_source_count",
    ]
    totals = {
        "entry_count": len(entries),
        "has_db_count": sum(1 for entry in entries if entry.get("has_db") is True),
    }
    for field_name in count_fields:
        totals[field_name] = sum(
            int(value)
            for entry in entries
            for value in [entry.get(field_name)]
            if isinstance(value, int)
        )
    updated_values = [
        str(value)
        for entry in entries
        for value in [entry.get("file_updated_at")]
        if value is not None
    ]
    totals["latest_file_updated_at"] = max(updated_values) if updated_values else None
    return totals


def _format_benchmark_db_totals_text(totals: dict[str, object]) -> str:
    fields = [
        ("entries", "entry_count"),
        ("dbs", "has_db_count"),
        ("total_bytes", "total_bytes"),
        ("messages", "message_count"),
        ("memory_objects", "memory_object_count"),
        ("embedding_metadata", "memory_embedding_metadata_count"),
        ("summary_views", "summary_view_count"),
        ("retrieval_events", "retrieval_event_count"),
        ("artifacts", "artifact_count"),
        ("artifact_chunks", "artifact_chunk_count"),
        ("conversation_topics", "conversation_topic_count"),
        ("topic_events", "conversation_topic_event_count"),
        ("topic_sources", "conversation_topic_source_count"),
    ]
    summary = "Totals: " + " ".join(
        f"{label}={totals.get(field_name, 0)}"
        for label, field_name in fields
    )
    latest_updated_at = totals.get("latest_file_updated_at")
    if latest_updated_at is not None:
        summary = f"{summary} latest_update={latest_updated_at}"
    return summary


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
    manifest_path: Path | None = None,
    custody_path: Path | None = None,
    taxonomy_path: Path | None = None,
    failure_taxonomy_summary: dict[str, object] | None = None,
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
        _format_phase_models(report.model_info),
        f"Trusted evaluation: {bool(report.model_info.get('trusted_evaluation', False))}",
        f"Parallel conversations: {int(report.model_info.get('parallel_conversations', 1))}",
        _format_warning_counts(report.model_info.get("warning_counts")),
        _format_llm_call_summary(report.model_info.get("llm_call_summary")),
        format_retrieval_custody_summary(
            report.model_info.get("retrieval_custody_summary")
        ),
        format_failure_taxonomy_summary(
            failure_taxonomy_summary
            or report.model_info.get("failure_taxonomy_summary")
        ),
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
            "=" * 40,
        ]
    )
    return "\n".join(lines)


def _format_warning_counts(value: object) -> str:
    if not isinstance(value, dict):
        return "Warning counts: unavailable"
    items: list[str] = []
    for key in sorted(value):
        try:
            amount = int(value[key])
        except (TypeError, ValueError):
            continue
        if amount:
            items.append(f"{key}={amount}")
    if not items:
        return "Warning counts: none"
    return "Warning counts: " + ", ".join(items)


def _format_phase_models(model_info: dict[str, object]) -> str:
    parts = []
    for label, key in (
        ("judge", "judge_model"),
        ("forced", "forced_global_model"),
        ("ingest", "ingest_model"),
        ("retrieval", "retrieval_model"),
        ("chat", "chat_model"),
    ):
        value = model_info.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{label}={value}")
    component_models = model_info.get("component_models")
    if isinstance(component_models, dict) and component_models:
        parts.append(f"components={len(component_models)}")
    return "Phase models: " + (" ".join(parts) if parts else "default internal resolution")


def _format_llm_call_summary(value: object) -> str:
    if not isinstance(value, dict):
        return "LLM calls: unavailable"
    total_calls = value.get("total_calls")
    token_totals = value.get("token_totals")
    if not isinstance(token_totals, dict):
        token_totals = {}
    token_parts = [
        f"{key}={int(amount)}"
        for key, amount in token_totals.items()
        if isinstance(amount, int | float) and amount
    ]
    cost_totals = value.get("cost_totals")
    if not isinstance(cost_totals, dict):
        cost_totals = {}
    cost_parts = [
        f"{key}={amount:.6f}"
        for key, amount in cost_totals.items()
        if isinstance(amount, int | float) and amount
    ]
    failed_calls = value.get("failed_calls")
    failed_text = f" failed={failed_calls}" if failed_calls else ""
    tokens_text = " tokens " + " ".join(token_parts) if token_parts else ""
    cost_text = " cost " + " ".join(cost_parts) if cost_parts else ""
    return f"LLM calls: {total_calls or 0}{failed_text}{tokens_text}{cost_text}"


def main() -> None:
    """Parse CLI args, run the benchmark, and print the report path."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.summarize_run_log is not None:
        if args.summarize_run_log_json:
            print(_format_run_log_summary_json(args.summarize_run_log))
        else:
            print(_format_run_log_summary(args.summarize_run_log))
        return
    if args.summarize_run_log_json:
        parser.error("--summarize-run-log-json requires --summarize-run-log")
    if args.diff_benchmark_db_list is not None:
        before_path, after_path = args.diff_benchmark_db_list
        if args.diff_benchmark_db_list_json:
            print(_format_benchmark_db_snapshot_diff_json(before_path, after_path))
        else:
            print(_format_benchmark_db_snapshot_diff(before_path, after_path))
        return
    if args.diff_benchmark_db_list_json:
        parser.error("--diff-benchmark-db-list-json requires --diff-benchmark-db-list")
    if args.list_benchmark_dbs_json:
        print(_format_benchmark_db_list_json(_benchmark_db_dir(args)))
        return
    if args.list_benchmark_dbs:
        print(_format_benchmark_db_list(_benchmark_db_dir(args)))
        return
    if args.data_path is None:
        parser.error("--data-path is required unless a list-benchmark-dbs option is used")
    if args.provider is None:
        parser.error("--provider is required unless a list-benchmark-dbs option is used")
    if args.model is not None and args.answer_model is not None:
        parser.error("--model is a legacy alias; use either --model or --answer-model, not both")
    if (
        _resolve_answer_model(args) is None
        and args.chat_model is None
        and args.forced_global_model is None
        and not args.ingest_only
    ):
        parser.error(
            "--answer-model, --model, --chat-model, or --forced-global-model "
            "is required unless --ingest-only is used"
        )
    try:
        _parse_component_model_overrides(args.component_model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.embedding_backend == "sqlite_vec" and not args.embedding_model:
        parser.error("--embedding-model is required when --embedding-backend is sqlite_vec")
    if args.ingest_only and (args.evaluate_only or args.reuse_db is not None):
        parser.error("--ingest-only cannot be combined with --evaluate-only or --reuse-db")
    if args.evaluate_only and args.reuse_db is None:
        parser.error("--evaluate-only requires --reuse-db")
    if args.parallel_conversations < 1:
        parser.error("--parallel-conversations must be at least 1")
    (
        report,
        output_path,
        diff_path,
        checkpoint_path,
        manifest_path,
        custody_path,
        taxonomy_path,
        failure_taxonomy_summary,
        diff_report,
    ) = asyncio.run(_run_async(args))
    print(
        _format_report_summary(
            report,
            output_path,
            diff_path=diff_path,
            checkpoint_path=checkpoint_path,
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
