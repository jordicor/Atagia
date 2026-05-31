"""LoCoMo benchmark orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from benchmarks.activation_flags import benchmark_activation_flags
from benchmarks.artifact_hash import (
    sha256_directory,
    sha256_file,
    sha256_file_if_exists,
)
from benchmarks.base import (
    DEFAULT_SCORED_CATEGORIES,
    BenchmarkConversation,
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkRunner,
    ConversationReport,
    QuestionResult,
    ScoreResult,
)
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.ingest_health import audit_benchmark_db, classify_ingest_health
from benchmarks.llm_config import provider_api_key_kwargs
from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    merge_llm_call_summaries,
    summarize_llm_calls,
)
from benchmarks.llm_run_guard import LLMRunGuardConfig
from benchmarks.migration_metadata import benchmark_migration_metadata
from benchmarks.retained_db_paths import validate_retained_benchmark_db_dir
from benchmarks.scorer import LLMJudgeScorer
from benchmarks.source_evidence import source_evidence_from_turns
from benchmarks.trusted_eval import (
    activate_trusted_evaluation_memories,
    trusted_evaluation_ablation,
)
from atagia import Atagia
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
)
from atagia.core.storage_backend import StorageDrainSnapshot
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_replay import AblationConfig
from atagia.services.admin_rebuild_service import AdminRebuildService, RebuildResult
from atagia.services.artifact_service import ArtifactService
from atagia.services.chat_support import chat_model
from atagia.services.errors import LLMUnavailableError
from atagia.services.llm_client import (
    LLMError,
    StructuredOutputError,
)
from atagia.services.model_resolution import COMPONENTS_BY_ID, provider_qualified_model
from atagia.services.run_counters import (
    RunCounterAccumulator,
    normalize_run_counters,
    reset_run_counter_accumulator,
    set_run_counter_accumulator,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_RETRIEVAL_PROFILE_ID = "general_qa"
_BENCHMARK_USER_ID = "benchmark-user"
_BENCHMARK_PLATFORM_ID = "locomo"
_BENCHMARK_USER_PERSONA_ID: str | None = None
_BENCHMARK_CHARACTER_ID: str | None = None
_BENCHMARK_DB_FILENAME = "benchmark.db"
_BENCHMARK_METADATA_FILENAME = "run_metadata.json"
_BENCHMARK_INGESTION_PROGRESS_FILENAME = "ingestion_progress.json"
_CRITICAL_EVIDENCE_ID_SAMPLE_LIMIT = 20
_INGEST_MODE_ONLINE = "online"
_INGEST_MODE_ONLINE_ASYNC = "online_async"
_INGEST_MODE_ONLINE_BATCH = "online_batch"
_INGEST_MODE_BULK = "bulk"
_WORKER_DRAIN_TIMEOUT_SECONDS = 6 * 60 * 60
_WORKER_DRAIN_IDLE_TIMEOUT_SECONDS = 5 * 60
_WORKER_DRAIN_HEARTBEAT_SECONDS = 60
_VALID_INGEST_MODES = {
    _INGEST_MODE_ONLINE,
    _INGEST_MODE_ONLINE_ASYNC,
    _INGEST_MODE_ONLINE_BATCH,
    _INGEST_MODE_BULK,
}
_REUSE_DB_COMPLETE_STATUS = "complete"
_REUSE_DB_UNTRUSTED_STATUS = "complete_untrusted"

_TECHNICAL_FAILURE_DIAGNOSIS_BY_STAGE = {
    "retrieval": "retrieval_failed",
    "answer_generation": "answer_generation_failed",
    "judge": "judge_failed",
}
_TECHNICAL_FAILURE_TRACE_KEY_BY_STAGE = {
    "retrieval": "retrieval_failure",
    "answer_generation": "answer_generation_failure",
    "judge": "judge_failure",
}
_TECHNICAL_FAILURE_REASON_PREFIX_BY_STAGE = {
    "retrieval": "Retrieval failed",
    "answer_generation": "Answer generation failed",
    "judge": "Judge call failed",
}


def _apply_corrections(
    questions: list[BenchmarkQuestion],
    corrections: dict[str, Any],
) -> list[BenchmarkQuestion]:
    """Substitute ground truths from a corrections overlay."""
    if not corrections:
        return questions
    result: list[BenchmarkQuestion] = []
    for question in questions:
        correction = corrections.get(question.question_id)
        if correction is not None:
            question = question.model_copy(
                update={"ground_truth": correction["corrected_ground_truth"]},
            )
        result.append(question)
    return result


class LoCoMoBenchmark(BenchmarkRunner):
    """Run the LoCoMo benchmark against Atagia."""

    def __init__(
        self,
        data_path: str | Path,
        llm_provider: str,
        llm_api_key: str | None,
        llm_model: str | None,
        answer_model: str | None = None,
        judge_model: str | None = None,
        forced_global_model: str | None = None,
        ingest_model: str | None = None,
        retrieval_model: str | None = None,
        chat_model_override: str | None = None,
        component_models: dict[str, str] | None = None,
        manifests_dir: str | Path | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
        corrections_path: str | Path | None = None,
        community_corrections_path: str | Path | None = None,
        answer_postcondition_guard_enabled: bool = False,
    ) -> None:
        self._data_path = Path(data_path).expanduser()
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._answer_model = provider_qualified_model(
            llm_provider,
            answer_model or llm_model,
        )
        self._judge_model = (
            provider_qualified_model(llm_provider, judge_model)
            if judge_model is not None
            else self._answer_model
        )
        self._forced_global_model = provider_qualified_model(
            llm_provider,
            forced_global_model,
        )
        self._ingest_model = provider_qualified_model(llm_provider, ingest_model)
        self._retrieval_model = provider_qualified_model(llm_provider, retrieval_model)
        self._chat_model = provider_qualified_model(llm_provider, chat_model_override)
        unknown_components = sorted(
            set(component_models or {}).difference(COMPONENTS_BY_ID)
        )
        if unknown_components:
            valid = ", ".join(sorted(COMPONENTS_BY_ID))
            raise ValueError(
                "Unknown LoCoMo component model override(s): "
                f"{', '.join(unknown_components)}. Valid component ids: {valid}"
            )
        self._component_models = {
            component_id: provider_qualified_model(llm_provider, model) or model
            for component_id, model in (component_models or {}).items()
        }
        self._manifests_dir = (
            Path(manifests_dir).expanduser()
            if manifests_dir is not None
            else _DEFAULT_MANIFESTS_DIR
        )
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._answer_postcondition_guard_enabled = answer_postcondition_guard_enabled
        self._adapter = LoCoMoAdapter(self._data_path)
        self._corrections: dict[str, Any] = {}
        if community_corrections_path is not None:
            from benchmarks.locomo.corrections import load_community_corrections

            self._corrections = load_community_corrections(
                Path(community_corrections_path).expanduser(),
                self._data_path,
            )
        if corrections_path is not None:
            with open(Path(corrections_path).expanduser()) as fh:
                self._corrections.update(json.load(fh))

    def _atagia_model_kwargs(self) -> dict[str, Any]:
        if self._forced_global_model is not None:
            return {"llm_forced_global_model": self._forced_global_model}
        return {
            "llm_ingest_model": self._ingest_model,
            "llm_retrieval_model": self._retrieval_model,
            "llm_chat_model": self._answer_model or self._chat_model,
            "llm_component_models": dict(self._component_models),
        }

    async def run(
        self,
        ablation: AblationConfig | None = None,
        conversation_ids: list[str] | None = None,
        categories: list[int] | None = None,
        question_ids: list[str] | None = None,
        max_questions: int | None = None,
        max_turns: int | None = None,
        checkpoint_path: str | Path | None = None,
        trusted_evaluation: bool = False,
        benchmark_db_dir: str | Path | None = None,
        requested_benchmark_db_dir: str | Path | None = None,
        allow_temp_benchmark_db_dir: bool = False,
        keep_db: bool = False,
        reuse_db: str | Path | None = None,
        reuse_db_dir: str | Path | None = None,
        resume_db: str | Path | None = None,
        resume_db_dir: str | Path | None = None,
        allow_untrusted_reuse: bool = False,
        require_evidence_packets: bool = True,
        llm_progress_interval: int = 0,
        llm_run_guard_config: LLMRunGuardConfig | None = None,
        ingest_only: bool = False,
        evaluate_only: bool = False,
        resume_checkpoint: bool = False,
        parallel_conversations: int = 1,
        parallel_questions: int = 1,
        adaptive_parallel_questions: bool = False,
        adaptive_parallel_min: int = 1,
        adaptive_parallel_retries: int = 1,
        ingest_mode: str = _INGEST_MODE_ONLINE,
        flush_every_turns: int | None = None,
        stage_sleep_seconds: float = 0.0,
        invocation_args: list[str] | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark and return an aggregated report."""
        dataset = self._adapter.load()
        selected_conversations = self._select_conversations(dataset, conversation_ids)
        ablation = self._benchmark_ablation(ablation)
        if parallel_conversations < 1:
            raise ValueError("parallel_conversations must be at least 1")
        if parallel_questions < 1:
            raise ValueError("parallel_questions must be at least 1")
        if adaptive_parallel_min < 1:
            raise ValueError("adaptive_parallel_min must be at least 1")
        if adaptive_parallel_min > parallel_questions:
            raise ValueError("adaptive_parallel_min cannot exceed parallel_questions")
        if adaptive_parallel_retries < 0:
            raise ValueError("adaptive_parallel_retries must be non-negative")
        if resume_checkpoint and checkpoint_path is None:
            raise ValueError("resume_checkpoint requires checkpoint_path")
        if ingest_mode not in _VALID_INGEST_MODES:
            valid_modes = ", ".join(sorted(_VALID_INGEST_MODES))
            raise ValueError(f"ingest_mode must be one of: {valid_modes}")
        if flush_every_turns is not None and flush_every_turns < 1:
            raise ValueError("flush_every_turns must be at least 1")
        if llm_progress_interval < 0:
            raise ValueError("llm_progress_interval must be non-negative")
        if stage_sleep_seconds < 0:
            raise ValueError("stage_sleep_seconds must be non-negative")
        resolved_llm_run_guard_config = llm_run_guard_config or LLMRunGuardConfig()
        effective_ingest_mode = self._effective_ingest_mode(
            ingest_mode,
            flush_every_turns=flush_every_turns,
        )
        self._validate_flush_configuration(
            ingest_mode,
            flush_every_turns=flush_every_turns,
        )
        if ingest_only:
            keep_db = True
        effective_benchmark_db_dir: Path | None = None
        if keep_db:
            effective_benchmark_db_dir = validate_retained_benchmark_db_dir(
                benchmark_db_dir or (_PROJECT_ROOT / "docs" / "tmp" / "benchmark_dbs"),
                allow_temp_benchmark_db_dir=allow_temp_benchmark_db_dir,
            )
        if reuse_db is not None and reuse_db_dir is not None:
            raise ValueError("reuse_db and reuse_db_dir are mutually exclusive")
        if resume_db is not None and resume_db_dir is not None:
            raise ValueError("resume_db and resume_db_dir are mutually exclusive")
        if (reuse_db is not None or reuse_db_dir is not None) and (
            resume_db is not None or resume_db_dir is not None
        ):
            raise ValueError(
                "reuse_db/reuse_db_dir cannot be combined with resume_db/resume_db_dir"
            )
        reuse_db_plan = self._resolve_reuse_db_plan(
            reuse_db=reuse_db,
            reuse_db_dir=reuse_db_dir,
            selected_conversations=selected_conversations,
            allow_untrusted=allow_untrusted_reuse,
        )
        resume_db_plan = self._resolve_resume_db_plan(
            resume_db=resume_db,
            resume_db_dir=resume_db_dir,
            selected_conversations=selected_conversations,
        )
        effective_evaluate_only = evaluate_only or bool(reuse_db_plan)
        if ingest_only and effective_evaluate_only:
            raise ValueError(
                "ingest_only cannot be combined with evaluate_only, reuse_db, or reuse_db_dir"
            )
        if evaluate_only and not reuse_db_plan:
            raise ValueError("evaluate_only requires reuse_db or reuse_db_dir")
        if effective_evaluate_only and resume_db_plan:
            raise ValueError(
                "resume_db/resume_db_dir cannot be combined with evaluate_only"
            )
        reuse_db_by_conversation_id = {
            conversation_id: entry["db_path"]
            for conversation_id, entry in reuse_db_plan.items()
        }
        resume_db_by_conversation_id = {
            conversation_id: entry["db_path"]
            for conversation_id, entry in resume_db_plan.items()
        }
        db_ingest_mode_by_conversation_id = {
            conversation.conversation_id: (
                self._metadata_ingest_mode(
                    (
                        reuse_db_plan.get(conversation.conversation_id)
                        or resume_db_plan[conversation.conversation_id]
                    )["metadata"],
                    default=effective_ingest_mode,
                )
                if (
                    conversation.conversation_id in reuse_db_plan
                    or conversation.conversation_id in resume_db_plan
                )
                else effective_ingest_mode
            )
            for conversation in selected_conversations
        }
        db_ingest_mode = (
            _common_value(db_ingest_mode_by_conversation_id.values()) or "mixed"
        )
        scored_categories = categories or list(DEFAULT_SCORED_CATEGORIES)
        question_filter = set(question_ids) if question_ids else None
        started_at = perf_counter()
        checkpoint_output = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        conversation_inputs = self._conversation_inputs(
            selected_conversations,
            scored_categories=scored_categories,
            question_filter=question_filter,
        )
        run_counters = RunCounterAccumulator()
        parallel_limit = min(parallel_conversations, len(conversation_inputs) or 1)
        if parallel_limit == 1:
            conversation_reports = await self._run_conversations_sequential(
                conversation_inputs,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_output=checkpoint_output,
                run_started_at=started_at,
                trusted_evaluation=trusted_evaluation,
                benchmark_db_dir=effective_benchmark_db_dir,
                keep_db=keep_db,
                reuse_db_by_conversation_id=reuse_db_by_conversation_id,
                resume_db_by_conversation_id=resume_db_by_conversation_id,
                ingest_only=ingest_only,
                evaluate_only=effective_evaluate_only,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=effective_ingest_mode,
                requested_ingest_mode=ingest_mode,
                db_ingest_mode_by_conversation_id=db_ingest_mode_by_conversation_id,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=resolved_llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )
        else:
            conversation_reports = await self._run_conversations_parallel(
                conversation_inputs,
                parallel_limit=parallel_limit,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_output=checkpoint_output,
                run_started_at=started_at,
                trusted_evaluation=trusted_evaluation,
                benchmark_db_dir=effective_benchmark_db_dir,
                keep_db=keep_db,
                reuse_db_by_conversation_id=reuse_db_by_conversation_id,
                resume_db_by_conversation_id=resume_db_by_conversation_id,
                ingest_only=ingest_only,
                evaluate_only=effective_evaluate_only,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=effective_ingest_mode,
                requested_ingest_mode=ingest_mode,
                db_ingest_mode_by_conversation_id=db_ingest_mode_by_conversation_id,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=resolved_llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )

        model_info_extra = {
            "trusted_evaluation": bool(trusted_evaluation),
            "parallel_conversations": parallel_limit,
            "parallel_questions": parallel_questions,
            "adaptive_parallel_questions": bool(adaptive_parallel_questions),
            "adaptive_parallel_min": adaptive_parallel_min,
            "adaptive_parallel_retries": adaptive_parallel_retries,
            "ingest_mode": db_ingest_mode,
            "requested_ingest_mode": ingest_mode,
            "effective_ingest_mode": effective_ingest_mode,
            "resume_checkpoint": bool(resume_checkpoint),
            "flush_every_turns": flush_every_turns,
            "require_evidence_packets": bool(require_evidence_packets),
            "allow_untrusted_reuse": bool(allow_untrusted_reuse),
            "llm_progress_interval": llm_progress_interval,
            "stage_sleep_seconds": stage_sleep_seconds,
            "selection": _selection_summary(
                conversation_inputs,
                scored_categories=scored_categories,
                question_ids=question_ids,
                max_questions=max_questions,
                max_turns=max_turns,
            ),
            "benchmark_db": {
                "keep_db": bool(keep_db),
                "benchmark_db_dir": (
                    str(Path(benchmark_db_dir).expanduser())
                    if benchmark_db_dir is not None
                    else None
                ),
                "requested_benchmark_db_dir": (
                    str(Path(requested_benchmark_db_dir).expanduser())
                    if requested_benchmark_db_dir is not None
                    else None
                ),
                "effective_benchmark_db_dir": (
                    str(effective_benchmark_db_dir)
                    if effective_benchmark_db_dir is not None
                    else None
                ),
                "allow_temp_benchmark_db_dir": allow_temp_benchmark_db_dir,
                "reuse_db": str(Path(reuse_db).expanduser())
                if reuse_db is not None
                else None,
                "reuse_db_dir": (
                    str(Path(reuse_db_dir).expanduser())
                    if reuse_db_dir is not None
                    else None
                ),
                "reuse_db_count": len(reuse_db_by_conversation_id),
                "resume_db": str(Path(resume_db).expanduser())
                if resume_db is not None
                else None,
                "resume_db_dir": (
                    str(Path(resume_db_dir).expanduser())
                    if resume_db_dir is not None
                    else None
                ),
                "resume_db_count": len(resume_db_by_conversation_id),
                "ingest_only": bool(ingest_only),
                "evaluate_only": bool(effective_evaluate_only),
                "ingest_mode": db_ingest_mode,
                "requested_ingest_mode": ingest_mode,
                "effective_ingest_mode": effective_ingest_mode,
                "flush_every_turns": flush_every_turns,
                "require_evidence_packets": bool(require_evidence_packets),
                "allow_untrusted_reuse": bool(allow_untrusted_reuse),
            },
            "invocation_args": invocation_args or [],
            "run_counters": run_counters.snapshot(),
        }
        report = self._build_report(
            conversation_reports,
            ablation=ablation,
            started_at=started_at,
            model_info_extra=model_info_extra,
        )
        if checkpoint_output is not None:
            self._write_report(report, checkpoint_output)
        return report

    def _conversation_inputs(
        self,
        conversations: Sequence[BenchmarkConversation],
        *,
        scored_categories: list[int],
        question_filter: set[str] | None,
    ) -> list[tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]]:
        result: list[tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]] = []
        for conversation_index, conversation in enumerate(conversations, start=1):
            filtered_questions = _apply_corrections(
                conversation.filtered_questions(scored_categories),
                self._corrections,
            )
            if question_filter is not None:
                filtered_questions = [
                    question
                    for question in filtered_questions
                    if question.question_id in question_filter
                ]
            result.append((conversation_index, conversation, filtered_questions))
        return result

    async def _run_conversations_sequential(
        self,
        conversation_inputs: Sequence[
            tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]
        ],
        *,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_output: Path | None,
        run_started_at: float,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db_by_conversation_id: dict[str, Path],
        resume_db_by_conversation_id: dict[str, Path],
        ingest_only: bool,
        evaluate_only: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode_by_conversation_id: dict[str, str],
        flush_every_turns: int | None,
        require_evidence_packets: bool,
        llm_progress_interval: int,
        llm_run_guard_config: LLMRunGuardConfig,
        stage_sleep_seconds: float,
        run_counters: RunCounterAccumulator,
    ) -> list[ConversationReport]:
        conversation_reports: list[ConversationReport] = []
        for conversation_index, conversation, filtered_questions in conversation_inputs:
            if conversation_reports:
                await self._stage_sleep(
                    stage_sleep_seconds,
                    stage="conversation_start",
                    conversation_id=conversation.conversation_id,
                )
            report = await self._run_conversation(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=len(conversation_inputs),
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_output,
                completed_conversation_reports=tuple(conversation_reports),
                run_started_at=run_started_at,
                trusted_evaluation=trusted_evaluation,
                benchmark_db_dir=benchmark_db_dir,
                keep_db=keep_db,
                reuse_db=reuse_db_by_conversation_id.get(conversation.conversation_id),
                resume_db=resume_db_by_conversation_id.get(
                    conversation.conversation_id
                ),
                ingest_only=ingest_only,
                evaluate_only=evaluate_only,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=ingest_mode,
                requested_ingest_mode=requested_ingest_mode,
                db_ingest_mode=db_ingest_mode_by_conversation_id[
                    conversation.conversation_id
                ],
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )
            conversation_reports.append(report)
        return conversation_reports

    async def _run_conversations_parallel(
        self,
        conversation_inputs: Sequence[
            tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]
        ],
        *,
        parallel_limit: int,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_output: Path | None,
        run_started_at: float,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db_by_conversation_id: dict[str, Path],
        resume_db_by_conversation_id: dict[str, Path],
        ingest_only: bool,
        evaluate_only: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode_by_conversation_id: dict[str, str],
        flush_every_turns: int | None,
        require_evidence_packets: bool,
        llm_progress_interval: int,
        llm_run_guard_config: LLMRunGuardConfig,
        stage_sleep_seconds: float,
        run_counters: RunCounterAccumulator,
    ) -> list[ConversationReport]:
        semaphore = asyncio.Semaphore(parallel_limit)
        start_lock = asyncio.Lock()
        started_count = 0

        async def run_one(
            conversation_index: int,
            conversation: BenchmarkConversation,
            filtered_questions: list[BenchmarkQuestion],
        ) -> ConversationReport:
            nonlocal started_count
            async with start_lock:
                if started_count:
                    await self._stage_sleep(
                        stage_sleep_seconds,
                        stage="conversation_start",
                        conversation_id=conversation.conversation_id,
                    )
                started_count += 1
            async with semaphore:
                checkpoint_path = (
                    self._parallel_checkpoint_path(
                        checkpoint_output,
                        conversation.conversation_id,
                    )
                    if checkpoint_output is not None
                    else None
                )
                return await self._run_conversation(
                    conversation,
                    filtered_questions=filtered_questions,
                    conversation_index=conversation_index,
                    conversation_count=len(conversation_inputs),
                    ablation=ablation,
                    max_questions=max_questions,
                    max_turns=max_turns,
                    checkpoint_path=checkpoint_path,
                    completed_conversation_reports=(),
                    run_started_at=run_started_at,
                    trusted_evaluation=trusted_evaluation,
                    benchmark_db_dir=benchmark_db_dir,
                    keep_db=keep_db,
                    reuse_db=reuse_db_by_conversation_id.get(
                        conversation.conversation_id
                    ),
                    resume_db=resume_db_by_conversation_id.get(
                        conversation.conversation_id
                    ),
                    ingest_only=ingest_only,
                    evaluate_only=evaluate_only,
                    resume_checkpoint=resume_checkpoint,
                    parallel_questions=parallel_questions,
                    adaptive_parallel_questions=adaptive_parallel_questions,
                    adaptive_parallel_min=adaptive_parallel_min,
                    adaptive_parallel_retries=adaptive_parallel_retries,
                    ingest_mode=ingest_mode,
                    requested_ingest_mode=requested_ingest_mode,
                    db_ingest_mode=db_ingest_mode_by_conversation_id[
                        conversation.conversation_id
                    ],
                    flush_every_turns=flush_every_turns,
                    require_evidence_packets=require_evidence_packets,
                    llm_progress_interval=llm_progress_interval,
                    llm_run_guard_config=llm_run_guard_config,
                    stage_sleep_seconds=stage_sleep_seconds,
                    run_counters=run_counters,
                )

        tasks = [
            asyncio.create_task(
                run_one(conversation_index, conversation, filtered_questions)
            )
            for conversation_index, conversation, filtered_questions in conversation_inputs
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    @staticmethod
    def _parallel_checkpoint_path(checkpoint_path: Path, conversation_id: str) -> Path:
        suffix = checkpoint_path.suffix or ".json"
        stem = checkpoint_path.stem
        safe_conversation_id = _safe_path_component(conversation_id)
        return checkpoint_path.with_name(f"{stem}-{safe_conversation_id}{suffix}")

    @staticmethod
    def _parallel_checkpoint_entries(
        report: BenchmarkReport,
        checkpoint_path: str | Path | None,
    ) -> list[dict[str, str | None]]:
        if checkpoint_path is None:
            return []
        parallel_conversations = report.model_info.get("parallel_conversations")
        if not isinstance(parallel_conversations, int) or parallel_conversations <= 1:
            return []
        checkpoint_base = Path(checkpoint_path).expanduser()
        entries: list[dict[str, str | None]] = []
        for conversation in report.conversations:
            partial_path = LoCoMoBenchmark._parallel_checkpoint_path(
                checkpoint_base,
                conversation.conversation_id,
            )
            entries.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "path": str(partial_path),
                    "sha256": sha256_file_if_exists(partial_path),
                }
            )
        return entries

    @staticmethod
    async def _stage_sleep(
        seconds: float,
        *,
        stage: str,
        conversation_id: str | None = None,
        question_id: str | None = None,
    ) -> None:
        if seconds <= 0:
            return
        logger.info(
            "locomo_stage_sleep stage=%s seconds=%s conversation_id=%s question_id=%s",
            stage,
            seconds,
            conversation_id,
            question_id,
        )
        await asyncio.sleep(seconds)

    async def _run_conversation(
        self,
        conversation: BenchmarkConversation,
        *,
        filtered_questions: list[BenchmarkQuestion],
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_path: Path | None,
        completed_conversation_reports: Sequence[ConversationReport],
        run_started_at: float,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db: str | Path | None,
        resume_db: str | Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode: str,
        flush_every_turns: int | None,
        require_evidence_packets: bool,
        llm_progress_interval: int,
        llm_run_guard_config: LLMRunGuardConfig,
        stage_sleep_seconds: float,
        run_counters: RunCounterAccumulator,
    ) -> ConversationReport:
        if reuse_db is not None:
            db_path = self._resolve_reuse_db(reuse_db)
            return await self._run_conversation_with_db(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_path,
                completed_conversation_reports=completed_conversation_reports,
                run_started_at=run_started_at,
                trusted_evaluation=trusted_evaluation,
                db_path=db_path,
                metadata_dir=db_path.parent,
                ingest_only=ingest_only,
                evaluate_only=True,
                resume_ingest=False,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=ingest_mode,
                requested_ingest_mode=requested_ingest_mode,
                db_ingest_mode=db_ingest_mode,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )
        if resume_db is not None:
            db_path = self._resolve_reuse_db(resume_db)
            return await self._run_conversation_with_db(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_path,
                completed_conversation_reports=completed_conversation_reports,
                run_started_at=run_started_at,
                trusted_evaluation=trusted_evaluation,
                db_path=db_path,
                metadata_dir=db_path.parent,
                ingest_only=ingest_only,
                evaluate_only=False,
                resume_ingest=True,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=ingest_mode,
                requested_ingest_mode=requested_ingest_mode,
                db_ingest_mode=db_ingest_mode,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )
        if keep_db:
            metadata_dir = self._new_persistent_db_dir(
                benchmark_db_dir=benchmark_db_dir,
                conversation_id=conversation.conversation_id,
            )
            db_path = metadata_dir / _BENCHMARK_DB_FILENAME
            return await self._run_conversation_with_db(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_path,
                completed_conversation_reports=completed_conversation_reports,
                run_started_at=run_started_at,
                trusted_evaluation=trusted_evaluation,
                db_path=db_path,
                metadata_dir=metadata_dir,
                ingest_only=ingest_only,
                evaluate_only=evaluate_only,
                resume_ingest=False,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=ingest_mode,
                requested_ingest_mode=requested_ingest_mode,
                db_ingest_mode=db_ingest_mode,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )
        with TemporaryDirectory(
            prefix=f"atagia-locomo-{conversation.conversation_id}-"
        ) as temp_dir:
            db_path = Path(temp_dir) / _BENCHMARK_DB_FILENAME
            return await self._run_conversation_with_db(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_path,
                completed_conversation_reports=completed_conversation_reports,
                run_started_at=run_started_at,
                trusted_evaluation=trusted_evaluation,
                db_path=db_path,
                metadata_dir=None,
                ingest_only=ingest_only,
                evaluate_only=evaluate_only,
                resume_ingest=False,
                resume_checkpoint=resume_checkpoint,
                parallel_questions=parallel_questions,
                adaptive_parallel_questions=adaptive_parallel_questions,
                adaptive_parallel_min=adaptive_parallel_min,
                adaptive_parallel_retries=adaptive_parallel_retries,
                ingest_mode=ingest_mode,
                requested_ingest_mode=requested_ingest_mode,
                db_ingest_mode=db_ingest_mode,
                flush_every_turns=flush_every_turns,
                require_evidence_packets=require_evidence_packets,
                llm_progress_interval=llm_progress_interval,
                llm_run_guard_config=llm_run_guard_config,
                stage_sleep_seconds=stage_sleep_seconds,
                run_counters=run_counters,
            )

    async def _run_conversation_with_db(
        self,
        conversation: BenchmarkConversation,
        *,
        filtered_questions: list[BenchmarkQuestion],
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_path: Path | None,
        completed_conversation_reports: Sequence[ConversationReport],
        run_started_at: float,
        trusted_evaluation: bool,
        db_path: Path,
        metadata_dir: Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        resume_ingest: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode: str,
        flush_every_turns: int | None,
        require_evidence_packets: bool,
        llm_progress_interval: int,
        llm_run_guard_config: LLMRunGuardConfig,
        stage_sleep_seconds: float,
        run_counters: RunCounterAccumulator,
    ) -> ConversationReport:
        llm_recorder = LLMCallRecorder(progress_interval=llm_progress_interval)
        conversation_started_at = perf_counter()
        engine: Atagia | None = None
        try:
            counter_token = set_run_counter_accumulator(run_counters)
            try:
                async with Atagia(
                    db_path=db_path,
                    manifests_dir=self._manifests_dir,
                    **self._atagia_model_kwargs(),
                    **provider_api_key_kwargs(self._llm_provider, self._llm_api_key),
                    embedding_backend=self._embedding_backend,
                    embedding_model=self._embedding_model,
                    skip_belief_revision=ablation.skip_belief_revision
                    if ablation
                    else False,
                    skip_compaction=ablation.skip_compaction if ablation else False,
                    answer_postcondition_guard_enabled=(
                        self._answer_postcondition_guard_enabled
                    ),
                ) as open_engine:
                    engine = open_engine
                    report = await self._run_conversation_with_open_engine(
                        engine,
                        conversation,
                        filtered_questions=filtered_questions,
                        conversation_index=conversation_index,
                        conversation_count=conversation_count,
                        ablation=ablation,
                        max_questions=max_questions,
                        max_turns=max_turns,
                        checkpoint_path=checkpoint_path,
                        completed_conversation_reports=completed_conversation_reports,
                        run_started_at=run_started_at,
                        trusted_evaluation=trusted_evaluation,
                        db_path=db_path,
                        metadata_dir=metadata_dir,
                        ingest_only=ingest_only,
                        evaluate_only=evaluate_only,
                        resume_ingest=resume_ingest,
                        resume_checkpoint=resume_checkpoint,
                        parallel_questions=parallel_questions,
                        adaptive_parallel_questions=adaptive_parallel_questions,
                        adaptive_parallel_min=adaptive_parallel_min,
                        adaptive_parallel_retries=adaptive_parallel_retries,
                        ingest_mode=ingest_mode,
                        requested_ingest_mode=requested_ingest_mode,
                        db_ingest_mode=db_ingest_mode,
                        flush_every_turns=flush_every_turns,
                        require_evidence_packets=require_evidence_packets,
                        llm_run_guard_config=llm_run_guard_config,
                        llm_recorder=llm_recorder,
                        conversation_started_at=conversation_started_at,
                        stage_sleep_seconds=stage_sleep_seconds,
                    )
            finally:
                reset_run_counter_accumulator(counter_token)
            return report
        except BaseException as exc:
            if (
                metadata_dir is not None
                and not evaluate_only
                and not self._metadata_is_complete(metadata_dir)
            ):
                self._mark_interrupted_ingestion(
                    metadata_dir=metadata_dir,
                    db_path=db_path,
                    conversation=conversation,
                    ablation=ablation,
                    max_turns=max_turns,
                    trusted_evaluation=trusted_evaluation,
                    ingest_mode=ingest_mode,
                    flush_every_turns=flush_every_turns,
                    exc=exc,
                    llm_call_summary=llm_recorder.summary(),
                    llm_run_guard_snapshot=self._llm_run_guard_snapshot(engine),
                )
            raise

    async def _run_conversation_with_open_engine(
        self,
        engine: Atagia,
        conversation: BenchmarkConversation,
        *,
        filtered_questions: list[BenchmarkQuestion],
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_path: Path | None,
        completed_conversation_reports: Sequence[ConversationReport],
        run_started_at: float,
        trusted_evaluation: bool,
        db_path: Path,
        metadata_dir: Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        resume_ingest: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode: str,
        flush_every_turns: int | None,
        require_evidence_packets: bool,
        llm_run_guard_config: LLMRunGuardConfig,
        llm_recorder: LLMCallRecorder,
        conversation_started_at: float,
        stage_sleep_seconds: float,
    ) -> ConversationReport:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        install_llm_call_recorder(runtime.llm_client, llm_recorder)

        def check_llm_health(stage: str) -> None:
            if evaluate_only:
                return
            try:
                llm_recorder.raise_if_unhealthy(
                    llm_run_guard_config,
                    elapsed_seconds=perf_counter() - conversation_started_at,
                )
            except Exception:
                logger.exception(
                    "LoCoMo LLM run guard failed conversation_id=%s stage=%s",
                    conversation.conversation_id,
                    stage,
                )
                raise

        user_id = _BENCHMARK_USER_ID
        rebuild_result: RebuildResult | None = None
        ingest_health: dict[str, Any] | None = None
        if evaluate_only:
            turn_message_ids = await self._load_turn_message_ids(
                engine,
                user_id=user_id,
                conversation=conversation,
                max_turns=max_turns,
            )
            if metadata_dir is not None:
                reuse_metadata = self._read_metadata_file(
                    metadata_dir / _BENCHMARK_METADATA_FILENAME
                )
                loaded_ingest_health = reuse_metadata.get("ingest_health")
                if isinstance(loaded_ingest_health, dict):
                    ingest_health = loaded_ingest_health
        else:
            selected_turns = (
                conversation.turns[:max_turns]
                if max_turns is not None
                else conversation.turns
            )
            await engine.create_user(user_id)
            await engine.create_conversation(
                user_id,
                conversation.conversation_id,
                assistant_mode_id=_DEFAULT_RETRIEVAL_PROFILE_ID,
                user_persona_id=_BENCHMARK_USER_PERSONA_ID,
                platform_id=_BENCHMARK_PLATFORM_ID,
                character_id=_BENCHMARK_CHARACTER_ID,
                mode=_DEFAULT_RETRIEVAL_PROFILE_ID,
            )
            resume_start_turn_index = (
                await self._resume_start_turn_index(
                    engine,
                    user_id=user_id,
                    conversation=conversation,
                    selected_turns=selected_turns,
                )
                if resume_ingest
                else 0
            )
            if metadata_dir is not None:
                self._write_ingestion_metadata(
                    metadata_dir=metadata_dir,
                    db_path=db_path,
                    conversation=conversation,
                    ablation=ablation,
                    max_turns=max_turns,
                    trusted_evaluation=trusted_evaluation,
                    ingest_mode=ingest_mode,
                    flush_every_turns=flush_every_turns,
                    status="resuming" if resume_ingest else "started",
                )
            progress_path = (
                metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME
                if metadata_dir is not None
                else None
            )
            if resume_start_turn_index >= len(selected_turns):
                turn_message_ids = await self._load_turn_message_ids(
                    engine,
                    user_id=user_id,
                    conversation=conversation,
                    max_turns=max_turns,
                )
                if (
                    resume_ingest
                    and db_ingest_mode == _INGEST_MODE_BULK
                    and self._bulk_resume_needs_rebuild(
                        metadata_dir=metadata_dir,
                        db_path=db_path,
                        expected_message_count=len(selected_turns),
                        skip_compaction=bool(ablation and ablation.skip_compaction),
                    )
                ):
                    rebuild_result = await self._rebuild_existing_bulk_conversation(
                        engine,
                        user_id,
                        conversation,
                        ablation=ablation,
                        max_turns=max_turns,
                        progress_path=progress_path,
                        llm_health_check=check_llm_health,
                    )
            elif ingest_mode == _INGEST_MODE_BULK:
                turn_message_ids, rebuild_result = await self._bulk_ingest_conversation(
                    engine,
                    user_id,
                    conversation,
                    ablation=ablation,
                    max_turns=max_turns,
                    progress_path=progress_path,
                    start_turn_index=resume_start_turn_index,
                    llm_health_check=check_llm_health,
                )
            else:
                turn_message_ids = await self._ingest_conversation(
                    engine,
                    user_id,
                    conversation,
                    ablation=ablation,
                    max_turns=max_turns,
                    progress_path=progress_path,
                    ingest_mode=ingest_mode,
                    flush_every_turns=flush_every_turns,
                    start_turn_index=resume_start_turn_index,
                    llm_health_check=check_llm_health,
                )
            check_llm_health("ingest")
            drain_extra = await self._drain_workers_for_ingestion(
                engine,
                db_path=db_path,
                conversation=conversation,
                selected_turns=selected_turns,
                ingested_turns=len(selected_turns),
                progress_path=(
                    metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME
                    if metadata_dir is not None
                    else None
                ),
                last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
                stage="final",
            )
            check_llm_health("worker_drain")
            if metadata_dir is not None:
                self._write_ingestion_progress(
                    metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME,
                    conversation=conversation,
                    selected_turns=selected_turns,
                    ingested_turns=len(selected_turns),
                    status="workers_drained",
                    last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
                    extra=drain_extra,
                )
            llm_call_summary = llm_recorder.summary()
            ingest_health = self._classify_current_ingest_health(
                db_path=db_path,
                llm_call_summary=llm_call_summary,
                require_evidence_packets=require_evidence_packets,
                llm_run_guard_config=llm_run_guard_config,
                rebuild_result=rebuild_result,
                require_rebuild_result=db_ingest_mode == _INGEST_MODE_BULK,
                require_summary_views=(
                    db_ingest_mode == _INGEST_MODE_BULK
                    and not bool(ablation and ablation.skip_compaction)
                ),
                expected_message_count=len(selected_turns),
            )
            status = (
                _REUSE_DB_COMPLETE_STATUS
                if ingest_health["trusted_ingest"]
                else _REUSE_DB_UNTRUSTED_STATUS
            )
            if metadata_dir is not None:
                self._write_ingestion_metadata(
                    metadata_dir=metadata_dir,
                    db_path=db_path,
                    conversation=conversation,
                    ablation=ablation,
                    max_turns=max_turns,
                    trusted_evaluation=trusted_evaluation,
                    ingest_mode=ingest_mode,
                    flush_every_turns=flush_every_turns,
                    status=status,
                    rebuild_result=rebuild_result,
                    llm_call_summary=llm_call_summary,
                    ingest_health=ingest_health,
                )
            if not ingest_health["trusted_ingest"] and not ingest_only:
                raise RuntimeError(
                    "Benchmark ingestion is untrusted; refusing to evaluate. "
                    f"Reasons: {'; '.join(ingest_health['reasons'])}"
                )
        if ingest_only:
            return self._build_conversation_report(
                conversation.conversation_id,
                [],
                metadata=self._conversation_metadata(
                    db_path=db_path,
                    metadata_dir=metadata_dir,
                    evaluate_only=evaluate_only,
                    ingest_only=ingest_only,
                    ingest_mode=db_ingest_mode,
                    rebuild_result=rebuild_result,
                    llm_call_summary=llm_recorder.summary(),
                    ingest_health=ingest_health,
                ),
            )
        trusted_activation_count = (
            await activate_trusted_evaluation_memories(engine.runtime, user_id)
            if trusted_evaluation and engine.runtime is not None
            else 0
        )

        selected_questions = (
            filtered_questions[:max_questions]
            if max_questions is not None
            else filtered_questions
        )
        question_results = await self._score_questions_for_conversation(
            engine,
            user_id=user_id,
            conversation=conversation,
            selected_questions=selected_questions,
            conversation_index=conversation_index,
            conversation_count=conversation_count,
            ablation=ablation,
            checkpoint_path=checkpoint_path,
            completed_conversation_reports=completed_conversation_reports,
            run_started_at=run_started_at,
            trusted_evaluation=trusted_evaluation,
            trusted_activation_count=trusted_activation_count,
            turn_message_ids=turn_message_ids,
            db_path=db_path,
            metadata_dir=metadata_dir,
            ingest_only=ingest_only,
            evaluate_only=evaluate_only,
            resume_checkpoint=resume_checkpoint,
            parallel_questions=parallel_questions,
            adaptive_parallel_questions=adaptive_parallel_questions,
            adaptive_parallel_min=adaptive_parallel_min,
            adaptive_parallel_retries=adaptive_parallel_retries,
            ingest_mode=ingest_mode,
            requested_ingest_mode=requested_ingest_mode,
            db_ingest_mode=db_ingest_mode,
            llm_recorder=llm_recorder,
            stage_sleep_seconds=stage_sleep_seconds,
        )

        return self._build_conversation_report(
            conversation.conversation_id,
            question_results,
            metadata=self._conversation_metadata(
                db_path=db_path,
                metadata_dir=metadata_dir,
                evaluate_only=evaluate_only,
                ingest_only=ingest_only,
                ingest_mode=db_ingest_mode,
                rebuild_result=rebuild_result,
                llm_call_summary=llm_recorder.summary(),
                ingest_health=ingest_health,
            ),
        )

    async def _ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        ablation: AblationConfig | None,
        max_turns: int | None = None,
        progress_path: Path | None = None,
        ingest_mode: str,
        flush_every_turns: int | None,
        start_turn_index: int = 0,
        llm_health_check: Any | None = None,
    ) -> dict[str, str]:
        total_turns = len(conversation.turns)
        turns = (
            conversation.turns[:max_turns]
            if max_turns is not None
            else conversation.turns
        )
        if start_turn_index < 0 or start_turn_index > len(turns):
            raise ValueError("start_turn_index is outside the selected turn range")
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        progress_status = self._online_ingest_progress_status(
            ingest_mode,
            flush_every_turns=flush_every_turns,
        )
        flush_interval = self._online_flush_interval(
            ingest_mode,
            flush_every_turns=flush_every_turns,
        )
        remaining_turns = turns[start_turn_index:]
        print(
            f"Ingesting {len(remaining_turns)}/{total_turns} turns for "
            f"{conversation.conversation_id} from turn {start_turn_index + 1}...",
            flush=True,
        )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=start_turn_index,
            status=progress_status,
            last_turn_id=(
                turns[start_turn_index - 1].turn_id if start_turn_index > 0 else None
            ),
        )
        for turn_index, turn in enumerate(remaining_turns, start=start_turn_index + 1):
            await engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                role=turn.role,
                text=f"{turn.speaker}: {turn.text}",
                mode=_DEFAULT_RETRIEVAL_PROFILE_ID,
                user_persona_id=_BENCHMARK_USER_PERSONA_ID,
                platform_id=_BENCHMARK_PLATFORM_ID,
                character_id=_BENCHMARK_CHARACTER_ID,
                occurred_at=turn.timestamp or None,
                attachments=turn.attachments or None,
                privacy_enforcement=self._benchmark_privacy_enforcement(ablation),
                authenticated_user_privilege_level=(
                    "atagia_master"
                    if self._benchmark_privacy_enforcement(ablation) == "off"
                    else None
                ),
                authenticated_user_is_atagia_master=(
                    self._benchmark_privacy_enforcement(ablation) == "off"
                ),
            )
            if llm_health_check is not None:
                llm_health_check(f"turn_{turn_index}")
            should_flush = (
                runtime.settings.workers_enabled
                and flush_interval is not None
                and turn_index % flush_interval == 0
                and turn_index < len(turns)
            )
            if should_flush:
                await self._drain_workers_for_ingestion(
                    engine,
                    db_path=Path(runtime.database_path),
                    conversation=conversation,
                    selected_turns=turns,
                    ingested_turns=turn_index,
                    progress_path=progress_path,
                    last_turn_id=turn.turn_id,
                    stage=f"turn_{turn_index}_batch",
                )
                if llm_health_check is not None:
                    llm_health_check(f"turn_{turn_index}_flush")
            if turn_index == len(turns) or turn_index % 25 == 0:
                self._write_ingestion_progress(
                    progress_path,
                    conversation=conversation,
                    selected_turns=turns,
                    ingested_turns=turn_index,
                    status=progress_status,
                    last_turn_id=turn.turn_id,
                )
                print(
                    f"Ingested {turn_index}/{len(turns)} turns for {conversation.conversation_id}.",
                    flush=True,
                )
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation(conversation.conversation_id, user_id)
        finally:
            await connection.close()
        if len(messages) != len(turns):
            raise RuntimeError(
                "Benchmark turn/message mapping failed for "
                f"{conversation.conversation_id}: expected {len(turns)} "
                f"messages, found {len(messages)}"
            )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=len(messages),
            status="ingestion_complete",
            last_turn_id=turns[-1].turn_id if turns else None,
        )
        return {
            str(turn.turn_id): str(message["id"])
            for turn, message in zip(turns, messages, strict=True)
            if turn.turn_id is not None
        }

    async def _drain_workers_for_ingestion(
        self,
        engine: Atagia,
        *,
        db_path: Path,
        conversation: BenchmarkConversation,
        selected_turns: Sequence[Any],
        ingested_turns: int,
        progress_path: Path | None,
        last_turn_id: str | None,
        stage: str,
    ) -> dict[str, Any]:
        last_db_counts: dict[str, int] | None = None
        last_extra: dict[str, Any] = {}
        drain_started_at = perf_counter()
        last_heartbeat_at: float | None = None
        last_acked: int | None = None

        async def heartbeat(snapshot: StorageDrainSnapshot) -> bool:
            nonlocal last_acked, last_db_counts, last_extra, last_heartbeat_at
            heartbeat_at = perf_counter()
            heartbeat_interval = (
                None
                if last_heartbeat_at is None
                else max(0.001, heartbeat_at - last_heartbeat_at)
            )
            db_counts = self._worker_drain_db_counts(db_path)
            db_count_delta = (
                self._worker_drain_count_delta(db_counts, last_db_counts)
                if last_db_counts is not None
                else {key: 0 for key in db_counts}
            )
            acked_delta = (
                0
                if last_acked is None
                else max(0, snapshot.total_acked - last_acked)
            )
            acked_per_second = (
                None
                if heartbeat_interval is None
                else acked_delta / heartbeat_interval
            )
            db_progressed = self._worker_drain_counts_progressed(db_count_delta)
            last_db_counts = db_counts
            last_acked = snapshot.total_acked
            last_heartbeat_at = heartbeat_at
            last_extra = self._worker_drain_progress_extra(
                stage=stage,
                snapshot=snapshot,
                db_counts=db_counts,
                db_count_delta=db_count_delta,
                acked_delta=acked_delta,
                acked_per_second=acked_per_second,
            )
            self._write_ingestion_progress(
                progress_path,
                conversation=conversation,
                selected_turns=selected_turns,
                ingested_turns=ingested_turns,
                status="draining_workers",
                last_turn_id=last_turn_id,
                extra=last_extra,
            )
            print(
                self._format_worker_drain_heartbeat(
                    conversation_id=conversation.conversation_id,
                    stage=stage,
                    snapshot=snapshot,
                    db_counts=db_counts,
                    db_count_delta=db_count_delta,
                    acked_per_second=acked_per_second,
                ),
                flush=True,
            )
            return db_progressed

        initial_snapshot = (
            (await engine.runtime.storage_backend.drain_snapshot())
            if engine.runtime is not None
            else StorageDrainSnapshot()
        )
        await heartbeat(
            initial_snapshot.with_timing(
                elapsed_seconds=0.0,
                idle_seconds=0.0,
                timeout_seconds=float(_WORKER_DRAIN_TIMEOUT_SECONDS),
                idle_timeout_seconds=float(_WORKER_DRAIN_IDLE_TIMEOUT_SECONDS),
            )
        )
        drained = await engine.flush(
            timeout_seconds=float(_WORKER_DRAIN_TIMEOUT_SECONDS),
            idle_timeout_seconds=float(_WORKER_DRAIN_IDLE_TIMEOUT_SECONDS),
            progress_interval_seconds=float(_WORKER_DRAIN_HEARTBEAT_SECONDS),
            progress_callback=heartbeat,
        )
        if drained:
            final_snapshot = (
                await engine.runtime.storage_backend.drain_snapshot()
                if engine.runtime is not None
                else StorageDrainSnapshot()
            )
            final_snapshot = final_snapshot.with_timing(
                elapsed_seconds=perf_counter() - drain_started_at,
                idle_seconds=0.0,
                timeout_seconds=float(_WORKER_DRAIN_TIMEOUT_SECONDS),
                idle_timeout_seconds=float(_WORKER_DRAIN_IDLE_TIMEOUT_SECONDS),
            )
            final_counts = self._worker_drain_db_counts(db_path)
            return self._worker_drain_progress_extra(
                stage=stage,
                snapshot=final_snapshot,
                db_counts=final_counts,
                db_count_delta=(
                    self._worker_drain_count_delta(final_counts, last_db_counts)
                    if last_db_counts is not None
                    else {key: 0 for key in final_counts}
                ),
                acked_delta=(
                    0
                    if last_acked is None
                    else max(0, final_snapshot.total_acked - last_acked)
                ),
                acked_per_second=None,
            )
        timeout_extra = last_extra or self._worker_drain_progress_extra(
            stage=stage,
            snapshot=StorageDrainSnapshot(),
            db_counts=self._worker_drain_db_counts(db_path),
            db_count_delta={},
            acked_delta=0,
            acked_per_second=None,
        )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=selected_turns,
            ingested_turns=ingested_turns,
            status="worker_drain_timeout",
            last_turn_id=last_turn_id,
            extra=timeout_extra,
        )
        raise RuntimeError(
            "Timed out while draining workers for "
            f"{conversation.conversation_id} stage={stage}"
        )

    @staticmethod
    def _worker_drain_counts_progressed(delta: dict[str, int]) -> bool:
        return any(int(value) > 0 for value in delta.values())

    @staticmethod
    def _worker_drain_count_delta(
        current: dict[str, int],
        previous: dict[str, int] | None,
    ) -> dict[str, int]:
        if previous is None:
            return {key: 0 for key in current}
        return {
            key: int(current.get(key, 0)) - int(previous.get(key, 0))
            for key in sorted(set(current) | set(previous))
        }

    @staticmethod
    def _worker_drain_db_counts(db_path: Path) -> dict[str, int]:
        table_names = (
            "messages",
            "memory_objects",
            "summary_views",
            "memory_support_edges",
            "memory_evidence_spans",
            "conversation_topics",
            "conversation_topic_events",
            "conversation_topic_sources",
        )
        counts = {table_name: 0 for table_name in table_names}
        if not db_path.is_file():
            return counts
        try:
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
                existing_tables = {str(row[0]) for row in rows}
                for table_name in table_names:
                    if table_name not in existing_tables:
                        continue
                    value = connection.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()
                    counts[table_name] = int(value[0] if value else 0)
        except sqlite3.Error:
            logger.exception("Failed to collect worker drain DB counts path=%s", db_path)
        return counts

    @staticmethod
    def _worker_drain_progress_extra(
        *,
        stage: str,
        snapshot: StorageDrainSnapshot,
        db_counts: dict[str, int],
        db_count_delta: dict[str, int],
        acked_delta: int,
        acked_per_second: float | None,
    ) -> dict[str, Any]:
        return {
            "worker_drain": {
                "stage": stage,
                "timeout_seconds": _WORKER_DRAIN_TIMEOUT_SECONDS,
                "idle_timeout_seconds": _WORKER_DRAIN_IDLE_TIMEOUT_SECONDS,
                "heartbeat_seconds": _WORKER_DRAIN_HEARTBEAT_SECONDS,
                "storage": snapshot.to_dict(),
                "db_counts": dict(sorted(db_counts.items())),
                "db_count_delta": dict(sorted(db_count_delta.items())),
                "acked_delta": acked_delta,
                "acked_per_second": (
                    round(acked_per_second, 3)
                    if acked_per_second is not None
                    else None
                ),
            }
        }

    @staticmethod
    def _format_worker_drain_heartbeat(
        *,
        conversation_id: str,
        stage: str,
        snapshot: StorageDrainSnapshot,
        db_counts: dict[str, int],
        db_count_delta: dict[str, int],
        acked_per_second: float | None,
    ) -> str:
        job_types = ",".join(
            f"{key}:{value}" for key, value in sorted(snapshot.pending_job_types.items())
        ) or "none"
        active = ",".join(
            f"{job.get('job_type')}:{job.get('seconds_pending')}s"
            for job in snapshot.active_jobs[:3]
        ) or "none"
        count_parts = " ".join(
            f"{key}={value}" for key, value in sorted(db_counts.items()) if value
        ) or "none"
        delta_parts = " ".join(
            f"{key}=+{value}"
            for key, value in sorted(db_count_delta.items())
            if value > 0
        ) or "none"
        ack_rate = "n/a" if acked_per_second is None else f"{acked_per_second:.3f}/s"
        return (
            "worker_drain_heartbeat "
            f"conversation_id={conversation_id} stage={stage} "
            f"elapsed={snapshot.elapsed_seconds:.1f}s idle={snapshot.idle_seconds:.1f}s "
            f"queued={snapshot.total_queued} pending={snapshot.total_pending} "
            f"acked={snapshot.total_acked} ack_rate={ack_rate} "
            f"job_types={job_types} active={active} "
            f"db_counts={count_parts} db_delta={delta_parts}"
        )

    async def _bulk_ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        ablation: AblationConfig | None,
        max_turns: int | None = None,
        progress_path: Path | None = None,
        start_turn_index: int = 0,
        llm_health_check: Any | None = None,
    ) -> tuple[dict[str, str], RebuildResult]:
        total_turns = len(conversation.turns)
        turns = (
            conversation.turns[:max_turns]
            if max_turns is not None
            else conversation.turns
        )
        if start_turn_index < 0 or start_turn_index > len(turns):
            raise ValueError("start_turn_index is outside the selected turn range")
        remaining_turns = turns[start_turn_index:]
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        print(
            f"Bulk ingesting {len(remaining_turns)}/{total_turns} turns for "
            f"{conversation.conversation_id} from turn {start_turn_index + 1}...",
            flush=True,
        )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=start_turn_index,
            status="bulk_ingesting",
            last_turn_id=(
                turns[start_turn_index - 1].turn_id if start_turn_index > 0 else None
            ),
        )
        connection = await runtime.open_connection()
        try:
            messages_repo = MessageRepository(connection, runtime.clock)
            conversations_repo = ConversationRepository(connection, runtime.clock)
            artifacts = ArtifactService(
                connection,
                runtime.clock,
                blob_store=runtime.artifact_blob_store,
            )
            conversation_row = await conversations_repo.get_conversation(
                conversation.conversation_id,
                user_id,
            )
            if conversation_row is None:
                raise RuntimeError(
                    f"Benchmark conversation {conversation.conversation_id} was not initialized"
                )
            try:
                await connection.execute("BEGIN")
                for turn in remaining_turns:
                    resolved_occurred_at = (
                        normalize_optional_timestamp(turn.timestamp)
                        or runtime.clock.now().isoformat()
                    )
                    message_text = f"{turn.speaker}: {turn.text}"
                    attachment_bundle = artifacts.prepare_attachments(
                        message_text=message_text,
                        attachments=turn.attachments or None,
                        user_id=user_id,
                        conversation=conversation_row,
                    )
                    metadata = self._message_metadata_for_attachments(attachment_bundle)
                    created_message = await messages_repo.create_message(
                        message_id=None,
                        conversation_id=conversation.conversation_id,
                        role=turn.role,
                        seq=None,
                        text=attachment_bundle.prompt_text,
                        token_count=None,
                        metadata=metadata,
                        occurred_at=resolved_occurred_at,
                        commit=False,
                    )
                    if attachment_bundle.artifacts:
                        await artifacts.persist_prepared_attachments(
                            bundle=attachment_bundle,
                            message_id=str(created_message["id"]),
                            commit=False,
                        )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

            messages = await messages_repo.list_messages_for_conversation(
                conversation.conversation_id,
                user_id,
            )
            if len(messages) != len(turns):
                raise RuntimeError(
                    "Benchmark bulk turn/message mapping failed for "
                    f"{conversation.conversation_id}: expected {len(turns)} "
                    f"messages, found {len(messages)}"
                )
            self._write_ingestion_progress(
                progress_path,
                conversation=conversation,
                selected_turns=turns,
                ingested_turns=len(messages),
                status="bulk_messages_persisted",
                last_turn_id=turns[-1].turn_id if turns else None,
            )
            self._write_ingestion_progress(
                progress_path,
                conversation=conversation,
                selected_turns=turns,
                ingested_turns=len(messages),
                status="bulk_rebuilding_memory",
                last_turn_id=turns[-1].turn_id if turns else None,
            )
            rebuild_result = await AdminRebuildService(
                connection=connection,
                llm_client=runtime.llm_client,
                embedding_index=runtime.embedding_index,
                clock=runtime.clock,
                manifest_loader=runtime.manifest_loader,
                settings=runtime.settings,
                storage_backend=runtime.storage_backend,
            ).rebuild_conversation(
                user_id,
                conversation.conversation_id,
                skip_final_compaction=bool(ablation and ablation.skip_compaction),
                llm_health_check=llm_health_check,
            )
            if rebuild_result.processed_messages != len(messages):
                raise RuntimeError(
                    "Benchmark bulk rebuild processed an unexpected number of "
                    f"messages for {conversation.conversation_id}: expected "
                    f"{len(messages)}, processed {rebuild_result.processed_messages}"
                )
        finally:
            await connection.close()

        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=len(turns),
            status="ingestion_complete",
            last_turn_id=turns[-1].turn_id if turns else None,
        )
        print(
            f"Bulk ingested {len(turns)}/{len(turns)} turns for {conversation.conversation_id}.",
            flush=True,
        )
        return (
            {
                str(turn.turn_id): str(message["id"])
                for turn, message in zip(turns, messages, strict=True)
                if turn.turn_id is not None
            },
            rebuild_result,
        )

    async def _rebuild_existing_bulk_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        ablation: AblationConfig | None,
        max_turns: int | None,
        progress_path: Path | None,
        llm_health_check: Any,
    ) -> RebuildResult:
        selected_turns = (
            conversation.turns[:max_turns]
            if max_turns is not None
            else conversation.turns
        )
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation(conversation.conversation_id, user_id)
            if len(messages) != len(selected_turns):
                raise RuntimeError(
                    "Benchmark bulk resume cannot rebuild an incomplete message set "
                    f"for {conversation.conversation_id}: expected "
                    f"{len(selected_turns)}, found {len(messages)}"
                )
            self._write_ingestion_progress(
                progress_path,
                conversation=conversation,
                selected_turns=selected_turns,
                ingested_turns=len(messages),
                status="bulk_rebuilding_memory",
                last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
                extra={"resume_rebuild": True},
            )
            rebuild_result = await AdminRebuildService(
                connection=connection,
                llm_client=runtime.llm_client,
                embedding_index=runtime.embedding_index,
                clock=runtime.clock,
                manifest_loader=runtime.manifest_loader,
                settings=runtime.settings,
                storage_backend=runtime.storage_backend,
            ).rebuild_conversation(
                user_id,
                conversation.conversation_id,
                skip_final_compaction=bool(ablation and ablation.skip_compaction),
                llm_health_check=llm_health_check,
            )
            if rebuild_result.processed_messages != len(messages):
                raise RuntimeError(
                    "Benchmark bulk resume rebuild processed an unexpected number "
                    f"of messages for {conversation.conversation_id}: expected "
                    f"{len(messages)}, processed {rebuild_result.processed_messages}"
                )
            return rebuild_result
        finally:
            await connection.close()

    @staticmethod
    def _bulk_resume_needs_rebuild(
        *,
        metadata_dir: Path | None,
        db_path: Path,
        expected_message_count: int,
        skip_compaction: bool,
    ) -> bool:
        metadata = (
            LoCoMoBenchmark._read_metadata_file(
                metadata_dir / _BENCHMARK_METADATA_FILENAME
            )
            if metadata_dir is not None
            else {}
        )
        rebuild_result = metadata.get("rebuild_result")
        if not isinstance(rebuild_result, dict):
            return True
        if rebuild_result.get("status") != "rebuilt":
            return True
        if int(rebuild_result.get("processed_messages") or 0) != expected_message_count:
            return True
        if skip_compaction or expected_message_count < 10:
            return False
        audit = audit_benchmark_db(db_path)
        counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
        return int(counts.get("summary_views") or 0) == 0

    @staticmethod
    def _message_metadata_for_attachments(attachment_bundle: Any) -> dict[str, Any]:
        if not attachment_bundle.artifacts:
            return {}
        return {
            "attachments": attachment_bundle.attachments,
            "attachment_count": len(attachment_bundle.artifacts),
            "attachment_artifact_ids": [
                str(prepared.artifact["id"]) for prepared in attachment_bundle.artifacts
            ],
            "artifact_backed": True,
            "skip_by_default": True,
            "include_raw": False,
            "requires_explicit_request": True,
            "content_kind": "artifact",
            "context_placeholder": attachment_bundle.context_placeholder,
        }

    async def _load_turn_message_ids(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: BenchmarkConversation,
        max_turns: int | None = None,
    ) -> dict[str, str]:
        turns = (
            conversation.turns[:max_turns]
            if max_turns is not None
            else conversation.turns
        )
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation(conversation.conversation_id, user_id)
        finally:
            await connection.close()
        if len(messages) != len(turns):
            raise RuntimeError(
                "Reusable benchmark DB does not match requested ingestion state for "
                f"{conversation.conversation_id}: expected {len(turns)} messages, "
                f"found {len(messages)}"
            )
        return {
            str(turn.turn_id): str(message["id"])
            for turn, message in zip(turns, messages, strict=True)
            if turn.turn_id is not None
        }

    async def _resume_start_turn_index(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: BenchmarkConversation,
        selected_turns: Sequence[Any],
    ) -> int:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation(conversation.conversation_id, user_id)
        finally:
            await connection.close()
        if len(messages) > len(selected_turns):
            raise RuntimeError(
                "Resumable benchmark DB has more messages than the requested "
                f"turn selection for {conversation.conversation_id}: expected at "
                f"most {len(selected_turns)}, found {len(messages)}"
            )
        return len(messages)

    async def _score_questions_for_conversation(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: BenchmarkConversation,
        selected_questions: Sequence[BenchmarkQuestion],
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        checkpoint_path: Path | None,
        completed_conversation_reports: Sequence[ConversationReport],
        run_started_at: float,
        trusted_evaluation: bool,
        trusted_activation_count: int,
        turn_message_ids: dict[str, str],
        db_path: Path,
        metadata_dir: Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        resume_checkpoint: bool,
        parallel_questions: int,
        adaptive_parallel_questions: bool,
        adaptive_parallel_min: int,
        adaptive_parallel_retries: int,
        ingest_mode: str,
        requested_ingest_mode: str,
        db_ingest_mode: str,
        llm_recorder: LLMCallRecorder,
        stage_sleep_seconds: float,
    ) -> list[QuestionResult]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        results_by_index: dict[int, QuestionResult] = {}
        existing_results = (
            self._checkpoint_results_for_conversation(
                checkpoint_path,
                conversation.conversation_id,
            )
            if resume_checkpoint and checkpoint_path is not None
            else {}
        )
        for question_index, question in enumerate(selected_questions, start=1):
            existing_result = existing_results.get(question.question_id)
            if existing_result is not None:
                results_by_index[question_index] = existing_result
        checkpoint_lock = asyncio.Lock()
        question_start_lock = asyncio.Lock()

        async def write_checkpoint() -> None:
            if checkpoint_path is None:
                return
            ordered_results = [
                results_by_index[index] for index in sorted(results_by_index)
            ]
            checkpoint_metadata = {
                "partial": True,
                "conversation_id": conversation.conversation_id,
                "conversation_index": conversation_index,
                "conversation_count": conversation_count,
                "completed_questions": len(ordered_results),
                "selected_questions": len(selected_questions),
                "trusted_evaluation": trusted_evaluation,
            }
            if resume_checkpoint:
                checkpoint_metadata["resumed_questions"] = len(existing_results)
            checkpoint_report = self._build_report(
                [
                    *completed_conversation_reports,
                    self._build_conversation_report(
                        conversation.conversation_id,
                        ordered_results,
                        metadata=self._conversation_metadata(
                            db_path=db_path,
                            metadata_dir=metadata_dir,
                            evaluate_only=evaluate_only,
                            ingest_only=ingest_only,
                            ingest_mode=db_ingest_mode,
                        ),
                    ),
                ],
                ablation=ablation,
                started_at=run_started_at,
                model_info_extra={
                    "parallel_questions": parallel_questions,
                    "adaptive_parallel_questions": bool(adaptive_parallel_questions),
                    "adaptive_parallel_min": adaptive_parallel_min,
                    "adaptive_parallel_retries": adaptive_parallel_retries,
                    "ingest_mode": db_ingest_mode,
                    "requested_ingest_mode": requested_ingest_mode,
                    "effective_ingest_mode": ingest_mode,
                    "resume_checkpoint": bool(resume_checkpoint),
                    "stage_sleep_seconds": stage_sleep_seconds,
                    "checkpoint": checkpoint_metadata,
                },
            )
            self._write_report(checkpoint_report, checkpoint_path)

        async def score_one(
            question_index: int, question: BenchmarkQuestion
        ) -> QuestionResult:
            async with question_start_lock:
                await self._stage_sleep(
                    stage_sleep_seconds,
                    stage="question_start",
                    conversation_id=conversation.conversation_id,
                    question_id=question.question_id,
                )
            result = await self._score_question_on_db_snapshot(
                base_db_path=db_path,
                user_id=user_id,
                conversation=conversation,
                question=question,
                question_index=question_index,
                question_count=len(selected_questions),
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                trusted_evaluation=trusted_evaluation,
                trusted_activation_count=trusted_activation_count,
                turn_message_ids=turn_message_ids,
                llm_recorder=llm_recorder,
            )
            async with checkpoint_lock:
                results_by_index[question_index] = result
                await write_checkpoint()
            return result

        async def discard_result(question_index: int) -> None:
            async with checkpoint_lock:
                results_by_index.pop(question_index, None)
                await write_checkpoint()

        pending_questions = [
            (question_index, question)
            for question_index, question in enumerate(selected_questions, start=1)
            if question_index not in results_by_index
        ]

        if parallel_questions == 1:
            for question_index, question in pending_questions:
                await score_one(question_index, question)
        elif adaptive_parallel_questions:
            await self._score_questions_adaptively(
                pending_questions,
                score_one=score_one,
                discard_result=discard_result,
                initial_parallel_questions=parallel_questions,
                min_parallel_questions=adaptive_parallel_min,
                max_retries=adaptive_parallel_retries,
            )
        else:
            semaphore = asyncio.Semaphore(parallel_questions)

            async def score_bounded(
                question_index: int, question: BenchmarkQuestion
            ) -> None:
                async with semaphore:
                    await score_one(question_index, question)

            tasks = [
                asyncio.create_task(score_bounded(question_index, question))
                for question_index, question in pending_questions
            ]
            try:
                await asyncio.gather(*tasks)
            except Exception:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return [
            results_by_index[index] for index in range(1, len(selected_questions) + 1)
        ]

    async def _score_questions_adaptively(
        self,
        pending_questions: Sequence[tuple[int, BenchmarkQuestion]],
        *,
        score_one: Any,
        discard_result: Any,
        initial_parallel_questions: int,
        min_parallel_questions: int,
        max_retries: int,
    ) -> None:
        current_parallel_questions = initial_parallel_questions
        queue = list(pending_questions)
        attempts_by_question_id: Counter[str] = Counter()
        while queue:
            wave = queue[:current_parallel_questions]
            del queue[:current_parallel_questions]
            results = await asyncio.gather(
                *[
                    score_one(question_index, question)
                    for question_index, question in wave
                ]
            )
            retry_wave: list[tuple[int, BenchmarkQuestion]] = []
            for item, result in zip(wave, results, strict=True):
                question_index, question = item
                question_id = question.question_id
                if attempts_by_question_id[
                    question_id
                ] < max_retries and self._is_adaptive_retryable_failure(result):
                    attempts_by_question_id[question_id] += 1
                    await discard_result(question_index)
                    retry_wave.append(item)
            if retry_wave:
                if current_parallel_questions > min_parallel_questions:
                    current_parallel_questions = max(
                        min_parallel_questions,
                        current_parallel_questions // 2,
                    )
                    logger.warning(
                        "LoCoMo adaptive question concurrency reduced to %s after "
                        "retryable provider failures.",
                        current_parallel_questions,
                    )
                queue = retry_wave + queue

    @staticmethod
    def _is_adaptive_retryable_failure(result: QuestionResult) -> bool:
        trace = result.trace if isinstance(result.trace, dict) else {}
        failure_stage = str(trace.get("failure_stage") or "")
        if not failure_stage:
            return False
        for trace_key in _TECHNICAL_FAILURE_TRACE_KEY_BY_STAGE.values():
            failure = trace.get(trace_key)
            if not isinstance(failure, dict):
                continue
            failure_text = " ".join(
                str(failure.get(field) or "")
                for field in ("exception_class", "message")
            ).lower()
            if any(
                marker in failure_text
                for marker in (
                    "ratelimit",
                    "rate limit",
                    "429",
                    "overload",
                    "overloaded",
                    "timeout",
                    "temporarily unavailable",
                    "service unavailable",
                )
            ):
                return True
        return False

    async def _score_question_on_db_snapshot(
        self,
        *,
        base_db_path: Path,
        user_id: str,
        conversation: BenchmarkConversation,
        question: BenchmarkQuestion,
        question_index: int,
        question_count: int,
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
        trusted_activation_count: int,
        turn_message_ids: dict[str, str],
        llm_recorder: LLMCallRecorder,
    ) -> QuestionResult:
        """Run one question through the real Atagia chat path on an isolated DB."""
        safe_conversation_id = _safe_path_component(conversation.conversation_id)
        with TemporaryDirectory(
            prefix=f"atagia-locomo-question-{safe_conversation_id}-"
        ) as temp_dir:
            question_db_path = Path(temp_dir) / _BENCHMARK_DB_FILENAME
            self._copy_sqlite_db(base_db_path, question_db_path)
            async with Atagia(
                db_path=question_db_path,
                manifests_dir=self._manifests_dir,
                **self._atagia_model_kwargs(),
                **provider_api_key_kwargs(self._llm_provider, self._llm_api_key),
                embedding_backend=self._embedding_backend,
                embedding_model=self._embedding_model,
                skip_belief_revision=ablation.skip_belief_revision
                if ablation
                else False,
                skip_compaction=ablation.skip_compaction if ablation else False,
                answer_postcondition_guard_enabled=self._answer_postcondition_guard_enabled,
            ) as question_engine:
                runtime = question_engine.runtime
                if runtime is None:
                    raise RuntimeError("Atagia runtime was unexpectedly unavailable")
                install_llm_call_recorder(runtime.llm_client, llm_recorder)
                judge = LLMJudgeScorer(
                    runtime.llm_client,
                    self._judge_model or chat_model(runtime.settings),
                )
                return await self._score_question(
                    question_engine,
                    user_id=user_id,
                    conversation=conversation,
                    question=question,
                    question_index=question_index,
                    question_count=question_count,
                    conversation_index=conversation_index,
                    conversation_count=conversation_count,
                    ablation=ablation,
                    judge=judge,
                    trusted_evaluation=trusted_evaluation,
                    trusted_activation_count=trusted_activation_count,
                    turn_message_ids=turn_message_ids,
                    llm_recorder=llm_recorder,
                )

    async def _score_question(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: BenchmarkConversation,
        question: BenchmarkQuestion,
        question_index: int,
        question_count: int,
        conversation_index: int,
        conversation_count: int,
        ablation: AblationConfig | None,
        judge: LLMJudgeScorer,
        trusted_evaluation: bool,
        trusted_activation_count: int,
        turn_message_ids: dict[str, str],
        llm_recorder: LLMCallRecorder,
    ) -> QuestionResult:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        print(
            f"Conversation {conversation_index}/{conversation_count} "
            f"({conversation.conversation_id}): question "
            f"{question_index}/{question_count}",
            flush=True,
        )
        with llm_recorder.context(
            conversation_id=conversation.conversation_id,
            question_id=question.question_id,
            question_index=question_index,
            category=question.category,
        ):
            chat_started_at = perf_counter()
            try:
                chat_result = await engine.chat(
                    user_id=user_id,
                    conversation_id=conversation.conversation_id,
                    message=question.question_text,
                    mode=_DEFAULT_RETRIEVAL_PROFILE_ID,
                    ablation=(
                        trusted_evaluation_ablation(ablation)
                        if trusted_evaluation
                        else ablation
                    ),
                    debug=True,
                    user_persona_id=_BENCHMARK_USER_PERSONA_ID,
                    platform_id=_BENCHMARK_PLATFORM_ID,
                    character_id=_BENCHMARK_CHARACTER_ID,
                    privacy_enforcement=self._benchmark_privacy_enforcement(
                        ablation,
                    ),
                    authenticated_user_privilege_level=(
                        "atagia_master"
                        if self._benchmark_privacy_enforcement(ablation) == "off"
                        else None
                    ),
                    authenticated_user_is_atagia_master=(
                        self._benchmark_privacy_enforcement(ablation) == "off"
                    ),
                )
                prediction = chat_result.response_text
            except (LLMError, LLMUnavailableError, StructuredOutputError) as exc:
                retrieval_time_ms = (perf_counter() - chat_started_at) * 1000.0
                trace_payload = self._basic_question_trace(
                    question=question,
                    user_id=user_id,
                    conversation_id=conversation.conversation_id,
                    assistant_mode_id=_DEFAULT_RETRIEVAL_PROFILE_ID,
                    timestamp_iso=runtime.clock.now().isoformat(),
                    turn_message_ids=turn_message_ids,
                    trusted_evaluation=trusted_evaluation,
                    trusted_activation_count=trusted_activation_count,
                )
                llm_calls = llm_recorder.records_for_context(
                    conversation_id=conversation.conversation_id,
                    question_id=question.question_id,
                )
                trace_payload["llm_calls"] = llm_calls
                trace_payload["llm_call_summary"] = summarize_llm_calls(llm_calls)
                self._annotate_benchmark_privacy_mode(trace_payload, ablation)
                return self._technical_failure_result(
                    question=question,
                    stage="retrieval",
                    exc=exc,
                    judge_model=judge.judge_model,
                    conversation=conversation,
                    question_index=question_index,
                    retrieval_time_ms=retrieval_time_ms,
                    trace=trace_payload,
                )
            retrieval_time_ms = (perf_counter() - chat_started_at) * 1000.0
            selected_memory_count = len(
                chat_result.composed_context.selected_memory_ids
                if chat_result.composed_context is not None
                else []
            )
            try:
                source_evidence = self._source_evidence_for_question(
                    conversation,
                    question,
                )
                score_result = await judge.score(
                    question=question.question_text,
                    prediction=prediction,
                    ground_truth=question.ground_truth,
                    source_evidence=source_evidence,
                )
            except (LLMError, StructuredOutputError) as exc:
                trace_payload = await self._build_question_trace_from_chat_result(
                    engine,
                    user_id=user_id,
                    question=question,
                    chat_result=chat_result,
                    retrieval_profile_id=_DEFAULT_RETRIEVAL_PROFILE_ID,
                    conversation_id=conversation.conversation_id,
                    turn_message_ids=turn_message_ids,
                    passed=False,
                    trusted_evaluation=trusted_evaluation,
                    trusted_activation_count=trusted_activation_count,
                )
                llm_calls = llm_recorder.records_for_context(
                    conversation_id=conversation.conversation_id,
                    question_id=question.question_id,
                )
                trace_payload["llm_calls"] = llm_calls
                trace_payload["llm_call_summary"] = summarize_llm_calls(llm_calls)
                self._annotate_benchmark_privacy_mode(trace_payload, ablation)
                return self._technical_failure_result(
                    question=question,
                    stage="judge",
                    exc=exc,
                    judge_model=judge.judge_model,
                    conversation=conversation,
                    question_index=question_index,
                    prediction=prediction,
                    memories_used=selected_memory_count,
                    retrieval_time_ms=retrieval_time_ms,
                    trace=trace_payload,
                )
        trace_payload = await self._build_question_trace_from_chat_result(
            engine,
            user_id=user_id,
            question=question,
            chat_result=chat_result,
            retrieval_profile_id=_DEFAULT_RETRIEVAL_PROFILE_ID,
            conversation_id=conversation.conversation_id,
            turn_message_ids=turn_message_ids,
            passed=bool(score_result.score),
            trusted_evaluation=trusted_evaluation,
            trusted_activation_count=trusted_activation_count,
        )
        trace_payload["grade_context"] = self._grade_context_for_question(
            question,
            source_evidence,
        )
        llm_calls = llm_recorder.records_for_context(
            conversation_id=conversation.conversation_id,
            question_id=question.question_id,
        )
        trace_payload["llm_calls"] = llm_calls
        trace_payload["llm_call_summary"] = summarize_llm_calls(llm_calls)
        self._annotate_benchmark_privacy_mode(trace_payload, ablation)
        return QuestionResult(
            question=question,
            prediction=prediction,
            score_result=score_result,
            memories_used=selected_memory_count,
            retrieval_time_ms=retrieval_time_ms,
            trace=trace_payload,
        )

    @staticmethod
    def _technical_failure_result(
        *,
        question: BenchmarkQuestion,
        stage: str,
        exc: Exception,
        judge_model: str,
        conversation: BenchmarkConversation,
        question_index: int,
        prediction: str = "",
        memories_used: int = 0,
        retrieval_time_ms: float = 0.0,
        trace: dict[str, Any] | None = None,
    ) -> QuestionResult:
        exc_class = type(exc).__name__
        message = str(exc)
        truncated = message if len(message) <= 500 else f"{message[:500]}..."
        diagnosis_bucket = _TECHNICAL_FAILURE_DIAGNOSIS_BY_STAGE.get(
            stage,
            "technical_failure",
        )
        trace_key = _TECHNICAL_FAILURE_TRACE_KEY_BY_STAGE.get(
            stage, "technical_failure"
        )
        reason_prefix = _TECHNICAL_FAILURE_REASON_PREFIX_BY_STAGE.get(
            stage,
            "Question execution failed",
        )
        logger.warning(
            "LoCoMo question %s failed for conversation_id=%s question_index=%s: %s: %s",
            stage,
            conversation.conversation_id,
            question_index,
            exc_class,
            truncated,
        )
        trace_payload = dict(trace or {})
        trace_payload["failure_stage"] = stage
        trace_payload["diagnosis_bucket"] = diagnosis_bucket
        trace_payload["sufficiency_diagnostic"] = diagnosis_bucket
        trace_payload[trace_key] = {
            "exception_class": exc_class,
            "message": truncated,
        }
        return QuestionResult(
            question=question,
            prediction=prediction,
            score_result=ScoreResult(
                score=0,
                reasoning=f"{reason_prefix}: {exc_class}: {truncated}",
                judge_model=judge_model,
            ),
            memories_used=memories_used,
            retrieval_time_ms=retrieval_time_ms,
            trace=trace_payload,
        )

    async def _build_question_trace_from_chat_result(
        self,
        engine: Atagia,
        *,
        user_id: str,
        question: BenchmarkQuestion,
        chat_result: Any,
        retrieval_profile_id: str,
        conversation_id: str,
        turn_message_ids: dict[str, str],
        passed: bool,
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> dict[str, Any]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")

        debug = chat_result.debug if isinstance(chat_result.debug, dict) else {}
        context = chat_result.composed_context
        selected_memory_ids = (
            list(context.selected_memory_ids)
            if context is not None
            else list(debug.get("selected_memory_ids") or [])
        )
        evidence_message_ids = [
            turn_message_ids[turn_id]
            for turn_id in question.evidence_turn_ids
            if turn_id in turn_message_ids
        ]
        missing_evidence_turn_ids = [
            turn_id
            for turn_id in question.evidence_turn_ids
            if turn_id not in turn_message_ids
        ]

        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            selected_rows = await memories.list_memory_objects_by_ids(
                user_id,
                selected_memory_ids,
            )
            evidence_rows_by_id: dict[str, dict[str, Any]] = {}
            for message_id in evidence_message_ids:
                for row in await memories.list_for_source_message(
                    user_id=user_id,
                    source_message_id=message_id,
                    statuses=None,
                ):
                    evidence_rows_by_id[str(row["id"])] = row
        finally:
            await connection.close()

        evidence_rows = list(evidence_rows_by_id.values())
        active_evidence_count = sum(
            1 for row in evidence_rows if str(row.get("status")) == "active"
        )
        selected_evidence_ids = sorted(
            set(selected_memory_ids).intersection(evidence_rows_by_id)
        )
        retrieval_custody = list(debug.get("retrieval_custody_v2") or [])
        critical_evidence_custody = self._critical_evidence_custody(
            evidence_rows_by_id,
            retrieval_custody,
        )
        scored_candidates = debug.get("scored_candidates")
        candidate_count = (
            len(scored_candidates) if isinstance(scored_candidates, list) else 0
        )
        diagnosis_bucket = self._diagnosis_bucket(
            passed=passed,
            has_evidence_turns=bool(question.evidence_turn_ids),
            evidence_message_count=len(evidence_message_ids),
            evidence_memory_count=len(evidence_rows),
            active_evidence_count=active_evidence_count,
            candidate_count=candidate_count,
            selected_memory_count=len(selected_memory_ids),
            selected_evidence_count=len(selected_evidence_ids),
        )
        context_payload = debug.get("context_view")
        debug_retrieval_trace = debug.get("retrieval_trace")
        if isinstance(debug_retrieval_trace, dict):
            retrieval_trace = dict(debug_retrieval_trace)
            retrieval_trace.setdefault("source", "real_chat_debug")
        else:
            retrieval_trace = {
                "query_text": question.question_text,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "timestamp_iso": runtime.clock.now().isoformat(),
                "source": "real_chat_debug",
                "retrieval_sufficiency": debug.get("retrieval_sufficiency"),
                "candidate_search_summary": debug.get("candidate_search_summary"),
                "retrieval_diagnostics_for_guard": debug.get(
                    "retrieval_diagnostics_for_guard"
                ),
            }
        return {
            "diagnosis_bucket": diagnosis_bucket,
            "sufficiency_diagnostic": self._sufficiency_diagnostic_from_debug(
                diagnosis_bucket,
                debug=debug,
            ),
            "shadow_sufficiency_diagnostics": debug.get("retrieval_sufficiency"),
            "trusted_evaluation": trusted_evaluation,
            "trusted_activation_count": trusted_activation_count,
            "mode": retrieval_profile_id,
            "retrieval_profile_id": retrieval_profile_id,
            "retrieval_conversation_id": conversation_id,
            "retrieval_event_id": chat_result.retrieval_event_id,
            "real_chat_path": True,
            "evidence_turn_ids": list(question.evidence_turn_ids),
            "evidence_message_ids": evidence_message_ids,
            "missing_evidence_turn_ids": missing_evidence_turn_ids,
            "evidence_memory_count": len(evidence_rows),
            "active_evidence_count": active_evidence_count,
            "evidence_memory_status_counts": self._count_field(evidence_rows, "status"),
            "evidence_memory_privacy_counts": self._count_int_field(
                evidence_rows,
                "privacy_level",
            ),
            "evidence_memory_ids": sorted(evidence_rows_by_id),
            "evidence_memory_summaries": self._memory_summaries(evidence_rows),
            "critical_evidence_custody": critical_evidence_custody,
            "selected_memory_ids": selected_memory_ids,
            "selected_memory_rows_found": len(selected_rows),
            "selected_evidence_memory_ids": selected_evidence_ids,
            "selected_memory_summaries": self._memory_summaries(selected_rows),
            "context": self._context_summary_from_chat_context(
                context,
                context_payload,
            ),
            "context_envelope": debug.get("context_envelope"),
            "retrieval_custody": retrieval_custody,
            "retrieval_trace": retrieval_trace,
            "answer_postcondition_guard": debug.get("answer_postcondition_guard"),
            "debug_authority": debug.get("authority"),
        }

    @staticmethod
    def _source_evidence_for_question(
        conversation: BenchmarkConversation,
        question: BenchmarkQuestion,
    ) -> list[dict[str, str]]:
        return source_evidence_from_turns(
            evidence_turn_ids=question.evidence_turn_ids,
            turns=conversation.turns,
            conversation_id=conversation.conversation_id,
        )

    @staticmethod
    def _grade_context_for_question(
        question: BenchmarkQuestion,
        source_evidence: list[dict[str, str]],
    ) -> dict[str, Any]:
        source_turn_ids = [
            str(item.get("turn_id") or "")
            for item in source_evidence
            if item.get("turn_id")
        ]
        source_timestamps = [
            str(item.get("timestamp") or "")
            for item in source_evidence
            if item.get("timestamp")
        ]
        return {
            "grader": "llm_judge",
            "judge_mode": (
                "source_aware_llm_judge"
                if source_evidence
                else "llm_judge"
            ),
            "source_evidence_used": bool(source_evidence),
            "source_evidence_source": (
                "official_benchmark_dataset" if source_evidence else None
            ),
            "source_turn_ids": source_turn_ids,
            "source_timestamps": source_timestamps,
            "abstention_kind": None,
            "question_id": question.question_id,
        }

    @staticmethod
    def _basic_question_trace(
        *,
        question: BenchmarkQuestion,
        user_id: str,
        conversation_id: str,
        assistant_mode_id: str,
        timestamp_iso: str,
        turn_message_ids: dict[str, str],
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> dict[str, Any]:
        evidence_message_ids = [
            turn_message_ids[turn_id]
            for turn_id in question.evidence_turn_ids
            if turn_id in turn_message_ids
        ]
        missing_evidence_turn_ids = [
            turn_id
            for turn_id in question.evidence_turn_ids
            if turn_id not in turn_message_ids
        ]
        return {
            "trusted_evaluation": trusted_evaluation,
            "trusted_activation_count": trusted_activation_count,
            "mode": assistant_mode_id,
            "retrieval_profile_id": assistant_mode_id,
            "retrieval_conversation_id": conversation_id,
            "real_chat_path": True,
            "evidence_turn_ids": list(question.evidence_turn_ids),
            "evidence_message_ids": evidence_message_ids,
            "missing_evidence_turn_ids": missing_evidence_turn_ids,
            "evidence_memory_ids": [],
            "critical_evidence_custody": {
                "counts": {
                    "critical_evidence_count": 0,
                    "raw_candidate_count": 0,
                    "scored_count": 0,
                    "selected_count": 0,
                    "absent_count": 0,
                },
                "survival_stage_counts": {},
                "items": [],
            },
            "selected_memory_ids": [],
            "selected_evidence_memory_ids": [],
            "retrieval_custody": [],
            "retrieval_trace": {
                "query_text": question.question_text,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "timestamp_iso": timestamp_iso,
                "source": "real_chat_debug",
            },
        }

    @staticmethod
    def _benchmark_ablation(ablation: AblationConfig | None) -> AblationConfig:
        if ablation is None:
            return AblationConfig(privacy_enforcement="off")
        if "privacy_enforcement" in ablation.model_fields_set:
            return ablation
        return ablation.model_copy(update={"privacy_enforcement": "off"})

    @staticmethod
    def _benchmark_privacy_enforcement(ablation: AblationConfig | None) -> str:
        if ablation is None:
            return "off"
        return ablation.privacy_enforcement

    @classmethod
    def _annotate_benchmark_privacy_mode(
        cls,
        trace: dict[str, Any],
        ablation: AblationConfig | None,
    ) -> None:
        privacy_enforcement = cls._benchmark_privacy_enforcement(ablation)
        trace["benchmark_privacy_enforcement"] = privacy_enforcement
        trace["benchmark_answer_privacy_override"] = privacy_enforcement == "off"
        trace["benchmark_high_risk_secret_redaction_disabled"] = (
            privacy_enforcement == "off"
        )

    @staticmethod
    def _context_summary_from_chat_context(
        context: Any,
        context_payload: Any,
    ) -> dict[str, Any]:
        if context is not None:
            return {
                "items_included": context.items_included,
                "items_dropped": context.items_dropped,
                "budget_tokens": context.budget_tokens,
                "total_tokens_estimate": context.total_tokens_estimate,
                "contract_block_chars": len(context.contract_block),
                "workspace_block_chars": len(context.workspace_block),
                "memory_block_chars": len(context.memory_block),
                "state_block_chars": len(context.state_block),
                "answer_evidence_memory_ids": list(
                    getattr(context, "answer_evidence_memory_ids", [])
                ),
                "answer_evidence_items": list(
                    getattr(context, "answer_evidence_items", [])
                ),
                "answer_evidence_sufficiency": dict(
                    getattr(context, "answer_evidence_sufficiency", {}) or {}
                ),
            }
        if isinstance(context_payload, dict):
            return {
                "items_included": context_payload.get("items_included", 0),
                "items_dropped": context_payload.get("items_dropped", 0),
                "budget_tokens": context_payload.get("budget_tokens", 0),
                "total_tokens_estimate": context_payload.get(
                    "total_tokens_estimate",
                    0,
                ),
                "contract_block_chars": len(
                    str(context_payload.get("contract_block") or "")
                ),
                "workspace_block_chars": len(
                    str(context_payload.get("workspace_block") or "")
                ),
                "memory_block_chars": len(
                    str(context_payload.get("memory_block") or "")
                ),
                "state_block_chars": len(str(context_payload.get("state_block") or "")),
                "answer_evidence_memory_ids": list(
                    context_payload.get("answer_evidence_memory_ids") or []
                ),
                "answer_evidence_items": list(
                    context_payload.get("answer_evidence_items") or []
                ),
                "answer_evidence_sufficiency": dict(
                    context_payload.get("answer_evidence_sufficiency") or {}
                ),
            }
        return {
            "items_included": 0,
            "items_dropped": 0,
            "budget_tokens": 0,
            "total_tokens_estimate": 0,
            "contract_block_chars": 0,
            "workspace_block_chars": 0,
            "memory_block_chars": 0,
            "state_block_chars": 0,
            "answer_evidence_memory_ids": [],
            "answer_evidence_items": [],
            "answer_evidence_sufficiency": {},
        }

    @staticmethod
    def _select_conversations(
        dataset: Any,
        conversation_ids: list[str] | None,
    ) -> list[BenchmarkConversation]:
        conversations = list(dataset.conversations)
        if conversation_ids is None:
            return conversations
        requested = set(conversation_ids)
        selected = [
            conversation
            for conversation in conversations
            if conversation.conversation_id in requested
        ]
        missing = requested.difference(
            conversation.conversation_id for conversation in selected
        )
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Unknown LoCoMo conversation ids: {missing_list}")
        return selected

    @staticmethod
    def _accuracy(total_correct: int, total_questions: int) -> float:
        if total_questions <= 0:
            return 0.0
        return total_correct / total_questions

    @staticmethod
    def _category_breakdown(results: Any) -> dict[int, float]:
        stats: dict[int, dict[str, int]] = {}
        for result in results:
            category = result.question.category
            bucket = stats.setdefault(category, {"correct": 0, "total": 0})
            bucket["correct"] += result.score_result.score
            bucket["total"] += 1
        return {
            category: (bucket["correct"] / bucket["total"])
            for category, bucket in sorted(stats.items())
            if bucket["total"] > 0
        }

    @staticmethod
    def _diagnosis_bucket(
        *,
        passed: bool,
        has_evidence_turns: bool,
        evidence_message_count: int,
        evidence_memory_count: int,
        active_evidence_count: int,
        candidate_count: int,
        selected_memory_count: int,
        selected_evidence_count: int,
    ) -> str:
        if passed:
            return "passed"
        if has_evidence_turns and evidence_message_count == 0:
            return "evidence_mapping_missing"
        if has_evidence_turns and evidence_memory_count == 0:
            return "missing_extraction"
        if evidence_memory_count > 0 and active_evidence_count == 0:
            return "memory_not_active"
        if candidate_count == 0:
            return "retrieval_no_candidates"
        if selected_memory_count == 0:
            return "composition_selected_none"
        if evidence_memory_count > 0 and selected_evidence_count == 0:
            return "retrieval_or_ranking_miss"
        return "answer_policy_or_grading"

    @staticmethod
    def _sufficiency_diagnostic_from_debug(
        diagnosis_bucket: str,
        *,
        debug: dict[str, Any],
    ) -> str:
        if diagnosis_bucket == "passed":
            return "retrieval_sufficient"
        if diagnosis_bucket == "evidence_mapping_missing":
            return "missing_raw_evidence"
        if diagnosis_bucket == "missing_extraction":
            return "missing_memory_extraction"
        if diagnosis_bucket == "memory_not_active":
            return "unsafe_or_requires_confirmation"
        if diagnosis_bucket == "retrieval_no_candidates":
            retrieval_plan = debug.get("retrieval_plan")
            if (
                isinstance(retrieval_plan, dict)
                and retrieval_plan.get("raw_context_access_mode") == "artifact"
            ):
                return "missing_artifact_support"
            return "missing_raw_evidence"
        if diagnosis_bucket in {
            "composition_selected_none",
            "retrieval_or_ranking_miss",
        }:
            return "retrieval_insufficient"
        if diagnosis_bucket in {
            "retrieval_failed",
            "answer_generation_failed",
            "judge_failed",
        }:
            return diagnosis_bucket
        return "answer_or_judge_issue"

    @staticmethod
    def _count_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
        return dict(Counter(str(row.get(field_name) or "") for row in rows))

    @staticmethod
    def _count_int_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
        return dict(Counter(str(int(row.get(field_name) or 0)) for row in rows))

    @classmethod
    def _critical_evidence_custody(
        cls,
        evidence_rows_by_id: dict[str, dict[str, Any]],
        retrieval_custody: list[dict[str, Any]],
    ) -> dict[str, Any]:
        records_by_id: dict[str, dict[str, Any]] = {}
        for record in retrieval_custody:
            candidate_id = str(record.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in records_by_id:
                records_by_id[candidate_id] = record

        items: list[dict[str, Any]] = []
        stage_counts: Counter[str] = Counter()
        raw_candidate_count = 0
        scored_count = 0
        selected_count = 0
        for evidence_id in sorted(evidence_rows_by_id):
            row = evidence_rows_by_id[evidence_id]
            record = records_by_id.get(evidence_id)
            in_custody = record is not None
            scored = bool(record.get("scored")) if record is not None else False
            selected = bool(record.get("selected")) if record is not None else False
            if in_custody:
                raw_candidate_count += 1
            if scored:
                scored_count += 1
            if selected:
                selected_count += 1
            survival_stage = cls._critical_evidence_survival_stage(record)
            stage_counts[survival_stage] += 1
            source_message_ids = cls._payload_id_list(
                row,
                "source_message_ids",
            )
            source_object_ids = cls._payload_id_list(
                row,
                "source_object_ids",
            )
            item: dict[str, Any] = {
                "memory_id": evidence_id,
                "object_type": str(row.get("object_type") or ""),
                "scope": str(row.get("scope") or ""),
                "status": str(row.get("status") or ""),
                "privacy_level": int(row.get("privacy_level") or 0),
                "source_kind": str(row.get("source_kind") or ""),
                "source_message_id_count": len(source_message_ids),
                "source_message_ids": source_message_ids[
                    :_CRITICAL_EVIDENCE_ID_SAMPLE_LIMIT
                ],
                "source_object_id_count": len(source_object_ids),
                "source_object_ids": source_object_ids[
                    :_CRITICAL_EVIDENCE_ID_SAMPLE_LIMIT
                ],
                "in_raw_candidates": in_custody,
                "scored": scored,
                "selected": selected,
                "survival_stage": survival_stage,
            }
            if record is not None:
                item.update(
                    {
                        "candidate_kind": str(record.get("candidate_kind") or ""),
                        "channels": [
                            value
                            for channel in record.get("channels") or []
                            if (value := str(channel).strip())
                        ],
                        "score_rank": record.get("score_rank"),
                        "selection_rank": record.get("selection_rank"),
                        "drop_stage": record.get("drop_stage"),
                        "drop_reason": record.get("drop_reason"),
                        "composer_decision": record.get("composer_decision"),
                        "matched_subquery_indexes": [
                            int(index)
                            for index in record.get("matched_subquery_indexes") or []
                            if isinstance(index, int)
                        ],
                    }
                )
            items.append(item)

        return {
            "counts": {
                "critical_evidence_count": len(evidence_rows_by_id),
                "raw_candidate_count": raw_candidate_count,
                "scored_count": scored_count,
                "selected_count": selected_count,
                "absent_count": len(evidence_rows_by_id) - raw_candidate_count,
            },
            "survival_stage_counts": dict(sorted(stage_counts.items())),
            "items": items,
        }

    @staticmethod
    def _critical_evidence_survival_stage(record: dict[str, Any] | None) -> str:
        if record is None:
            return "absent_from_raw_candidates"
        if record.get("selected") is True:
            return "selected"
        drop_stage = str(record.get("drop_stage") or "").strip()
        if drop_stage:
            return drop_stage
        if record.get("scored") is True:
            return "composer"
        if record.get("shortlisted") is True:
            return "post_applicability_rerank"
        return "shortlist"

    @staticmethod
    def _payload_id_list(row: dict[str, Any], key: str) -> list[str]:
        payload = row.get("payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            return []
        raw_ids = payload.get(key)
        if not isinstance(raw_ids, list):
            return []
        normalized: set[str] = set()
        for item in raw_ids:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                normalized.add(value)
        return sorted(normalized)

    @staticmethod
    def _memory_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for row in rows:
            summaries.append(
                {
                    "memory_id": str(row.get("id") or ""),
                    "object_type": str(row.get("object_type") or ""),
                    "scope": str(row.get("scope") or ""),
                    "status": str(row.get("status") or ""),
                    "privacy_level": int(row.get("privacy_level") or 0),
                    "memory_category": str(row.get("memory_category") or ""),
                    "source_kind": str(row.get("source_kind") or ""),
                    "canonical_preview": " ".join(
                        str(row.get("canonical_text") or "").split()
                    )[:180],
                }
            )
        return summaries

    def _base_model_info(self) -> dict[str, Any]:
        effective_answer_model = (
            self._answer_model or self._chat_model or self._forced_global_model or ""
        )
        return {
            "provider": self._llm_provider,
            "answer_model": effective_answer_model,
            "judge_model": self._judge_model or effective_answer_model,
            "forced_global_model": self._forced_global_model,
            "ingest_model": self._ingest_model,
            "retrieval_model": self._retrieval_model,
            "chat_model": self._chat_model,
            "component_models": dict(sorted(self._component_models.items())),
            "activation_flags": self._activation_flags(),
            "real_chat_path": True,
            "mode": _DEFAULT_RETRIEVAL_PROFILE_ID,
            "retrieval_profile_id": _DEFAULT_RETRIEVAL_PROFILE_ID,
            "platform_id": _BENCHMARK_PLATFORM_ID,
            "user_persona_id": _BENCHMARK_USER_PERSONA_ID,
            "character_id": _BENCHMARK_CHARACTER_ID,
        }

    def _activation_flags(self) -> dict[str, Any]:
        return benchmark_activation_flags(
            embedding_backend=self._embedding_backend,
            answer_postcondition_guard_enabled=(
                self._answer_postcondition_guard_enabled
            ),
        )

    def _build_report(
        self,
        conversation_reports: Sequence[ConversationReport],
        *,
        ablation: AblationConfig | None,
        started_at: float,
        model_info_extra: dict[str, Any] | None = None,
    ) -> BenchmarkReport:
        total_questions = sum(len(report.results) for report in conversation_reports)
        total_correct = sum(
            result.score_result.score
            for report in conversation_reports
            for result in report.results
        )
        model_info = self._base_model_info()
        if model_info_extra is not None:
            model_info.update(model_info_extra)
        model_info.setdefault(
            "warning_counts",
            self._aggregate_warning_counts(conversation_reports),
        )
        model_info.setdefault(
            "failure_stage_counts",
            self._aggregate_failure_stage_counts(conversation_reports),
        )
        model_info.setdefault(
            "retrieval_custody_summary",
            self._aggregate_retrieval_custody_summary(conversation_reports),
        )
        model_info.setdefault(
            "llm_call_summary",
            self._aggregate_llm_call_summary(conversation_reports),
        )
        model_info.setdefault("run_counters", normalize_run_counters(None))
        return BenchmarkReport(
            benchmark_name="LoCoMo",
            overall_accuracy=self._accuracy(total_correct, total_questions),
            category_breakdown=self._category_breakdown(
                result for report in conversation_reports for result in report.results
            ),
            conversations=list(conversation_reports),
            total_questions=total_questions,
            total_correct=total_correct,
            ablation_config=(
                ablation.model_dump(mode="json", exclude_none=True)
                if ablation is not None
                else None
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_info=model_info,
            duration_seconds=perf_counter() - started_at,
        )

    def _build_conversation_report(
        self,
        conversation_id: str,
        question_results: Sequence[QuestionResult],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationReport:
        total_correct = sum(result.score_result.score for result in question_results)
        return ConversationReport(
            conversation_id=conversation_id,
            results=list(question_results),
            accuracy=self._accuracy(total_correct, len(question_results)),
            category_breakdown=self._category_breakdown(question_results),
            metadata=metadata or {},
        )

    @staticmethod
    def _conversation_metadata(
        *,
        db_path: Path,
        metadata_dir: Path | None,
        evaluate_only: bool,
        ingest_only: bool,
        ingest_mode: str | None = None,
        rebuild_result: RebuildResult | None = None,
        llm_call_summary: dict[str, Any] | None = None,
        ingest_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "benchmark_db_path": str(db_path),
            "evaluate_only": bool(evaluate_only),
            "ingest_only": bool(ingest_only),
        }
        if ingest_mode is not None:
            metadata["ingest_mode"] = ingest_mode
        if rebuild_result is not None:
            metadata["rebuild_result"] = rebuild_result.model_dump(mode="json")
        if llm_call_summary is not None:
            metadata["llm_call_summary"] = llm_call_summary
        if ingest_health is not None:
            metadata["ingest_health"] = ingest_health
            metadata["trusted_ingest"] = bool(ingest_health.get("trusted_ingest"))
        if metadata_dir is not None:
            metadata["benchmark_metadata_dir"] = str(metadata_dir)
            metadata["benchmark_run_metadata_path"] = str(
                metadata_dir / _BENCHMARK_METADATA_FILENAME
            )
        return metadata

    @staticmethod
    def _aggregate_warning_counts(
        conversation_reports: Sequence[ConversationReport],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter(
            {
                "failed_questions": 0,
                "degraded_retrievals": 0,
                "need_detection_degraded": 0,
                "retrieval_no_candidates": 0,
                "composition_selected_none": 0,
                "missing_extraction": 0,
                "memory_not_active": 0,
                "retrieval_or_ranking_miss": 0,
                "answer_policy_or_grading": 0,
                "retrieval_failed": 0,
                "answer_generation_failed": 0,
                "judge_failed": 0,
                "structured_output_retries": 0,
                "synthesis_preservation": 0,
                "applicability_fallback": 0,
                "provider_rate_limits": 0,
                "tracebacks": 0,
                "failed_worker_jobs": 0,
            }
        )
        for report in conversation_reports:
            for result in report.results:
                if result.score_result.score == 0:
                    counts["failed_questions"] += 1
                trace = result.trace or {}
                diagnosis = str(trace.get("diagnosis_bucket") or "")
                if diagnosis in counts:
                    counts[diagnosis] += 1
                retrieval_trace = trace.get("retrieval_trace")
                if isinstance(retrieval_trace, dict):
                    if retrieval_trace.get("degraded_mode"):
                        counts["degraded_retrievals"] += 1
                    need_detection = retrieval_trace.get("need_detection")
                    if isinstance(need_detection, dict) and need_detection.get(
                        "degraded_mode"
                    ):
                        counts["need_detection_degraded"] += 1
        return dict(counts)

    @staticmethod
    def _aggregate_retrieval_custody_summary(
        conversation_reports: Sequence[ConversationReport],
    ) -> dict[str, object]:
        return summarize_retrieval_custody(
            result.trace.get("retrieval_custody", [])
            if isinstance(result.trace, dict)
            else []
            for report in conversation_reports
            for result in report.results
        )

    @staticmethod
    def _aggregate_failure_stage_counts(
        conversation_reports: Sequence[ConversationReport],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for report in conversation_reports:
            for result in report.results:
                trace = result.trace if isinstance(result.trace, dict) else {}
                value = str(trace.get("failure_stage") or "").strip()
                if value:
                    counts[value] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _aggregate_llm_call_summary(
        conversation_reports: Sequence[ConversationReport],
    ) -> dict[str, Any]:
        summaries = [
            report.metadata.get("llm_call_summary")
            for report in conversation_reports
            if isinstance(report.metadata.get("llm_call_summary"), dict)
        ]
        return merge_llm_call_summaries(summaries)

    @staticmethod
    def _write_report(report: BenchmarkReport, report_path: str | Path) -> Path:
        return write_json_atomic(
            Path(report_path).expanduser(),
            report.model_dump(mode="json"),
        )

    @staticmethod
    def _checkpoint_results_for_conversation(
        checkpoint_path: Path | None,
        conversation_id: str,
    ) -> dict[str, QuestionResult]:
        if checkpoint_path is None or not checkpoint_path.exists():
            return {}
        try:
            report = BenchmarkReport.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8"),
            )
        except (OSError, ValueError):
            return {}
        for conversation_report in report.conversations:
            if conversation_report.conversation_id != conversation_id:
                continue
            return {
                result.question.question_id: result
                for result in conversation_report.results
            }
        return {}

    @staticmethod
    def save_report(report: BenchmarkReport, output_dir: str | Path) -> Path:
        """Persist a benchmark report as JSON and return its path."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = output_path / f"locomo-report-{timestamp}.json"
        return LoCoMoBenchmark._write_report(report, report_path)

    def build_run_manifest(
        self,
        report: BenchmarkReport,
        *,
        report_path: str | Path,
        checkpoint_path: str | Path | None = None,
        diff_path: str | Path | None = None,
        custody_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        readout_path: str | Path | None = None,
        failure_taxonomy_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an auditable manifest for one saved benchmark report."""
        report_file = Path(report_path).expanduser()
        checkpoint_file = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        diff_file = Path(diff_path).expanduser() if diff_path is not None else None
        custody_file = (
            Path(custody_path).expanduser() if custody_path is not None else None
        )
        taxonomy_file = (
            Path(taxonomy_path).expanduser() if taxonomy_path is not None else None
        )
        readout_file = (
            Path(readout_path).expanduser() if readout_path is not None else None
        )
        retained_db_paths = [
            str(conversation.metadata["benchmark_db_path"])
            for conversation in report.conversations
            if conversation.metadata.get("benchmark_db_path")
        ]
        return {
            "manifest_kind": "benchmark_run_manifest",
            "benchmark_name": report.benchmark_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_timestamp": report.timestamp,
            "report_path": str(report_file),
            "report_sha256": sha256_file_if_exists(report_file),
            "checkpoint_path": (
                str(checkpoint_file) if checkpoint_file is not None else None
            ),
            "checkpoint_sha256": (
                sha256_file_if_exists(checkpoint_file)
                if checkpoint_file is not None
                else None
            ),
            "partial_checkpoint_paths": self._parallel_checkpoint_entries(
                report,
                checkpoint_path,
            ),
            "diff_path": str(diff_file) if diff_file is not None else None,
            "diff_sha256": (
                sha256_file_if_exists(diff_file) if diff_file is not None else None
            ),
            "custody_path": (str(custody_file) if custody_file is not None else None),
            "custody_sha256": (
                sha256_file_if_exists(custody_file)
                if custody_file is not None
                else None
            ),
            "taxonomy_path": (
                str(taxonomy_file) if taxonomy_file is not None else None
            ),
            "taxonomy_sha256": (
                sha256_file_if_exists(taxonomy_file)
                if taxonomy_file is not None
                else None
            ),
            "readout_path": str(readout_file) if readout_file is not None else None,
            "readout_sha256": (
                sha256_file_if_exists(readout_file)
                if readout_file is not None
                else None
            ),
            "dataset": {
                "path": str(self._data_path),
                "sha256": sha256_file(self._data_path),
            },
            "migrations": benchmark_migration_metadata(),
            "git": _git_state(),
            "activation_flags": self._activation_flags(),
            "model_info": report.model_info,
            "run_counters": normalize_run_counters(
                report.model_info.get("run_counters")
            ),
            "ablation_config": report.ablation_config,
            "result_summary": {
                "overall_accuracy": report.overall_accuracy,
                "total_correct": report.total_correct,
                "total_questions": report.total_questions,
                "category_breakdown": report.category_breakdown,
                "duration_seconds": report.duration_seconds,
                "retrieval_time_ms": _question_result_numeric_summary(
                    report,
                    "retrieval_time_ms",
                ),
                "memories_used": _question_result_numeric_summary(
                    report,
                    "memories_used",
                ),
            },
            "conversation_ids": [
                conversation.conversation_id for conversation in report.conversations
            ],
            "selection": _manifest_selection_summary(report),
            "retained_db_paths": retained_db_paths,
            "warning_counts": report.model_info.get("warning_counts", {}),
            "failure_stage_counts": report.model_info.get("failure_stage_counts", {}),
            "retrieval_custody_summary": report.model_info.get(
                "retrieval_custody_summary",
                {},
            ),
            "failure_taxonomy_summary": failure_taxonomy_summary or {},
            "duplicate_selection_rule": "conversation_id/question_id uniqueness; later aggregate helpers may replace duplicates explicitly",
            "benchmark_questions_persisted_as_messages": False,
        }

    def save_run_manifest(
        self,
        report: BenchmarkReport,
        *,
        report_path: str | Path,
        checkpoint_path: str | Path | None = None,
        diff_path: str | Path | None = None,
        custody_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        readout_path: str | Path | None = None,
        failure_taxonomy_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a benchmark run manifest beside the saved report."""
        manifest = self.build_run_manifest(
            report,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            diff_path=diff_path,
            custody_path=custody_path,
            taxonomy_path=taxonomy_path,
            readout_path=readout_path,
            failure_taxonomy_summary=failure_taxonomy_summary,
        )
        destination = _manifest_path_for_report(Path(report_path).expanduser())
        return write_json_atomic(destination, manifest)

    @staticmethod
    def _effective_ingest_mode(
        ingest_mode: str,
        *,
        flush_every_turns: int | None,
    ) -> str:
        if ingest_mode == _INGEST_MODE_ONLINE and flush_every_turns not in (None, 1):
            return _INGEST_MODE_ONLINE_BATCH
        return ingest_mode

    @staticmethod
    def _validate_flush_configuration(
        ingest_mode: str,
        *,
        flush_every_turns: int | None,
    ) -> None:
        if ingest_mode == _INGEST_MODE_ONLINE_BATCH and flush_every_turns is None:
            raise ValueError("online_batch ingest requires flush_every_turns")
        if ingest_mode in {_INGEST_MODE_BULK, _INGEST_MODE_ONLINE_ASYNC} and (
            flush_every_turns is not None
        ):
            raise ValueError(
                f"flush_every_turns cannot be combined with ingest_mode={ingest_mode}"
            )

    @staticmethod
    def _online_flush_interval(
        ingest_mode: str,
        *,
        flush_every_turns: int | None,
    ) -> int | None:
        if ingest_mode == _INGEST_MODE_ONLINE_ASYNC:
            return None
        if ingest_mode == _INGEST_MODE_ONLINE_BATCH:
            return flush_every_turns
        return 1

    @staticmethod
    def _online_ingest_progress_status(
        ingest_mode: str,
        *,
        flush_every_turns: int | None,
    ) -> str:
        if ingest_mode == _INGEST_MODE_ONLINE_ASYNC:
            return "online_async_ingesting"
        if ingest_mode == _INGEST_MODE_ONLINE_BATCH or (
            ingest_mode == _INGEST_MODE_ONLINE and flush_every_turns not in (None, 1)
        ):
            return "online_batch_ingesting"
        return "ingesting"

    @staticmethod
    def _metadata_ingest_mode(metadata: dict[str, Any], *, default: str) -> str:
        metadata_ingest_mode = metadata.get("ingest_mode")
        if metadata_ingest_mode in _VALID_INGEST_MODES:
            return str(metadata_ingest_mode)
        return default

    def _resolve_reuse_db_plan(
        self,
        *,
        reuse_db: str | Path | None,
        reuse_db_dir: str | Path | None,
        selected_conversations: Sequence[BenchmarkConversation],
        allow_untrusted: bool,
    ) -> dict[str, dict[str, Any]]:
        if reuse_db is None and reuse_db_dir is None:
            return {}
        if reuse_db is not None:
            if len(selected_conversations) != 1:
                raise ValueError(
                    "reuse_db can only be used with exactly one selected conversation"
                )
            conversation_id = selected_conversations[0].conversation_id
            db_path = self._resolve_reuse_db(reuse_db)
            metadata = self._read_reuse_db_metadata(db_path)
            self._validate_reuse_db_metadata(
                metadata,
                expected_conversation_id=conversation_id,
                source=str(db_path),
                allow_untrusted=allow_untrusted,
            )
            return {
                conversation_id: {
                    "db_path": db_path,
                    "metadata": metadata,
                }
            }
        assert reuse_db_dir is not None
        return self._resolve_reuse_db_dir_plan(
            reuse_db_dir,
            selected_conversations=selected_conversations,
            allow_untrusted=allow_untrusted,
        )

    def _resolve_reuse_db_dir_plan(
        self,
        reuse_db_dir: str | Path,
        *,
        selected_conversations: Sequence[BenchmarkConversation],
        allow_untrusted: bool,
    ) -> dict[str, dict[str, Any]]:
        base_dir = Path(reuse_db_dir).expanduser()
        if not base_dir.exists():
            raise ValueError(
                f"Reusable benchmark DB directory does not exist: {base_dir}"
            )
        if not base_dir.is_dir():
            raise ValueError(
                f"Reusable benchmark DB directory is not a directory: {base_dir}"
            )
        selected_ids = {
            conversation.conversation_id for conversation in selected_conversations
        }
        valid_entries: dict[str, dict[str, Any]] = {}
        unusable_entries: dict[str, list[str]] = {}
        for metadata_path in sorted(base_dir.rglob(_BENCHMARK_METADATA_FILENAME)):
            metadata = self._read_metadata_file(metadata_path)
            if not metadata:
                continue
            conversation_id = metadata.get("conversation_id")
            if (
                not isinstance(conversation_id, str)
                or conversation_id not in selected_ids
            ):
                continue
            db_path = metadata_path.parent / _BENCHMARK_DB_FILENAME
            unusable_reason = self._reuse_db_metadata_error(
                metadata,
                expected_conversation_id=conversation_id,
                source=str(metadata_path),
                allow_untrusted=allow_untrusted,
            )
            if unusable_reason is None and not db_path.is_file():
                unusable_reason = f"missing benchmark DB file: {db_path}"
            if unusable_reason is not None:
                unusable_entries.setdefault(conversation_id, []).append(unusable_reason)
                continue
            if conversation_id in valid_entries:
                raise ValueError(
                    "Reusable benchmark DB directory has multiple usable entries for "
                    f"{conversation_id}: {valid_entries[conversation_id]['db_path']} and {db_path}"
                )
            valid_entries[conversation_id] = {
                "db_path": db_path,
                "metadata": metadata,
            }
        missing = sorted(selected_ids.difference(valid_entries))
        if missing:
            details = []
            for conversation_id in missing:
                reasons = unusable_entries.get(conversation_id)
                if reasons:
                    details.append(f"{conversation_id}: {'; '.join(reasons)}")
            detail_text = f" Unusable entries: {' | '.join(details)}" if details else ""
            raise ValueError(
                "reuse_db_dir is missing usable benchmark DBs for selected "
                f"conversation(s): {', '.join(missing)}.{detail_text}"
            )
        return valid_entries

    def _resolve_resume_db_plan(
        self,
        *,
        resume_db: str | Path | None,
        resume_db_dir: str | Path | None,
        selected_conversations: Sequence[BenchmarkConversation],
    ) -> dict[str, dict[str, Any]]:
        if resume_db is None and resume_db_dir is None:
            return {}
        if resume_db is not None:
            if len(selected_conversations) != 1:
                raise ValueError(
                    "resume_db can only be used with exactly one selected conversation"
                )
            conversation_id = selected_conversations[0].conversation_id
            db_path = self._resolve_reuse_db(resume_db)
            metadata = self._read_reuse_db_metadata(db_path)
            self._validate_resume_db_metadata(
                metadata,
                expected_conversation_id=conversation_id,
                source=str(db_path),
            )
            return {
                conversation_id: {
                    "db_path": db_path,
                    "metadata": metadata,
                }
            }
        assert resume_db_dir is not None
        return self._resolve_resume_db_dir_plan(
            resume_db_dir,
            selected_conversations=selected_conversations,
        )

    def _resolve_resume_db_dir_plan(
        self,
        resume_db_dir: str | Path,
        *,
        selected_conversations: Sequence[BenchmarkConversation],
    ) -> dict[str, dict[str, Any]]:
        base_dir = Path(resume_db_dir).expanduser()
        if not base_dir.exists():
            raise ValueError(
                f"Resumable benchmark DB directory does not exist: {base_dir}"
            )
        if not base_dir.is_dir():
            raise ValueError(
                f"Resumable benchmark DB directory is not a directory: {base_dir}"
            )
        selected_ids = {
            conversation.conversation_id for conversation in selected_conversations
        }
        valid_entries: dict[str, dict[str, Any]] = {}
        unusable_entries: dict[str, list[str]] = {}
        for metadata_path in sorted(base_dir.rglob(_BENCHMARK_METADATA_FILENAME)):
            metadata = self._read_metadata_file(metadata_path)
            if not metadata:
                continue
            conversation_id = metadata.get("conversation_id")
            if (
                not isinstance(conversation_id, str)
                or conversation_id not in selected_ids
            ):
                continue
            db_path = metadata_path.parent / _BENCHMARK_DB_FILENAME
            unusable_reason = self._resume_db_metadata_error(
                metadata,
                expected_conversation_id=conversation_id,
                source=str(metadata_path),
            )
            if unusable_reason is None and not db_path.is_file():
                unusable_reason = f"missing benchmark DB file: {db_path}"
            if unusable_reason is not None:
                unusable_entries.setdefault(conversation_id, []).append(unusable_reason)
                continue
            if conversation_id in valid_entries:
                raise ValueError(
                    "Resumable benchmark DB directory has multiple usable entries for "
                    f"{conversation_id}: {valid_entries[conversation_id]['db_path']} and {db_path}"
                )
            valid_entries[conversation_id] = {
                "db_path": db_path,
                "metadata": metadata,
            }
        missing = sorted(selected_ids.difference(valid_entries))
        if missing:
            details = []
            for conversation_id in missing:
                reasons = unusable_entries.get(conversation_id)
                if reasons:
                    details.append(f"{conversation_id}: {'; '.join(reasons)}")
            detail_text = f" Unusable entries: {' | '.join(details)}" if details else ""
            raise ValueError(
                "resume_db_dir is missing resumable benchmark DBs for selected "
                f"conversation(s): {', '.join(missing)}.{detail_text}"
            )
        return valid_entries

    @classmethod
    def _validate_resume_db_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        expected_conversation_id: str,
        source: str,
    ) -> None:
        error = cls._resume_db_metadata_error(
            metadata,
            expected_conversation_id=expected_conversation_id,
            source=source,
        )
        if error is not None:
            raise ValueError(error)

    @staticmethod
    def _resume_db_metadata_error(
        metadata: dict[str, Any],
        *,
        expected_conversation_id: str,
        source: str,
    ) -> str | None:
        if not metadata:
            return None
        metadata_conversation_id = metadata.get("conversation_id")
        if (
            isinstance(metadata_conversation_id, str)
            and metadata_conversation_id
            and metadata_conversation_id != expected_conversation_id
        ):
            return (
                "Resumable benchmark DB conversation mismatch: metadata has "
                f"{metadata_conversation_id}, selected {expected_conversation_id}"
            )
        return None

    @staticmethod
    def _read_metadata_file(metadata_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    @classmethod
    def _validate_reuse_db_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        expected_conversation_id: str,
        source: str,
        allow_untrusted: bool = False,
    ) -> None:
        error = cls._reuse_db_metadata_error(
            metadata,
            expected_conversation_id=expected_conversation_id,
            source=source,
            allow_untrusted=allow_untrusted,
        )
        if error is not None:
            raise ValueError(error)

    @staticmethod
    def _reuse_db_metadata_error(
        metadata: dict[str, Any],
        *,
        expected_conversation_id: str,
        source: str,
        allow_untrusted: bool = False,
    ) -> str | None:
        if not metadata:
            return None
        metadata_conversation_id = metadata.get("conversation_id")
        if (
            isinstance(metadata_conversation_id, str)
            and metadata_conversation_id
            and metadata_conversation_id != expected_conversation_id
        ):
            return (
                "Reusable benchmark DB conversation mismatch: metadata has "
                f"{metadata_conversation_id}, selected {expected_conversation_id}"
            )
        trusted_ingest = metadata.get("trusted_ingest")
        status = metadata.get("status")
        if (
            trusted_ingest is False
            and status == _REUSE_DB_UNTRUSTED_STATUS
            and not allow_untrusted
        ):
            return f"Reusable benchmark DB is untrusted for evaluate-only: {source}"
        if metadata.get("usable_for_evaluate_only") is False and not (
            allow_untrusted
            and trusted_ingest is False
            and status == _REUSE_DB_UNTRUSTED_STATUS
        ):
            return f"Reusable benchmark DB is not usable for evaluate-only: {source}"
        if trusted_ingest is False and not allow_untrusted:
            return f"Reusable benchmark DB is untrusted for evaluate-only: {source}"
        if (
            isinstance(status, str)
            and status
            and status not in {_REUSE_DB_COMPLETE_STATUS, _REUSE_DB_UNTRUSTED_STATUS}
        ):
            return f"Reusable benchmark DB is not complete: {source} status={status}"
        if status == _REUSE_DB_UNTRUSTED_STATUS and not allow_untrusted:
            return f"Reusable benchmark DB is untrusted for evaluate-only: {source}"
        integrity_error = LoCoMoBenchmark._reuse_db_metadata_integrity_error(metadata)
        if integrity_error is not None:
            return f"Reusable benchmark DB failed integrity checks: {source} {integrity_error}"
        return None

    @staticmethod
    def _reuse_db_metadata_integrity_error(metadata: dict[str, Any]) -> str | None:
        if metadata.get("ingest_mode") != _INGEST_MODE_BULK:
            return None
        rebuild_result = metadata.get("rebuild_result")
        if not isinstance(rebuild_result, dict):
            return "bulk metadata is missing required rebuild_result"
        if rebuild_result.get("status") != "rebuilt":
            return (
                "bulk rebuild_result did not report rebuilt status: "
                f"{rebuild_result.get('status')}"
            )
        turn_count = metadata.get("turn_count")
        if isinstance(turn_count, int):
            processed = int(rebuild_result.get("processed_messages") or 0)
            if processed != turn_count:
                return (
                    "bulk rebuild_result processed message count mismatch: "
                    f"{processed}/{turn_count}"
                )
            if turn_count >= 10 and not LoCoMoBenchmark._metadata_skips_compaction(metadata):
                summary_count = LoCoMoBenchmark._metadata_ingest_health_count(
                    metadata,
                    "summary_views",
                )
                if summary_count == 0:
                    return "bulk compaction produced no summary_views"
        return None

    @staticmethod
    def _metadata_skips_compaction(metadata: dict[str, Any]) -> bool:
        ablation = metadata.get("ablation_config")
        return isinstance(ablation, dict) and bool(ablation.get("skip_compaction"))

    @staticmethod
    def _metadata_ingest_health_count(
        metadata: dict[str, Any],
        table_name: str,
    ) -> int | None:
        ingest_health = metadata.get("ingest_health")
        if not isinstance(ingest_health, dict):
            return None
        db_audit = ingest_health.get("db_audit")
        if not isinstance(db_audit, dict):
            return None
        counts = db_audit.get("counts")
        if not isinstance(counts, dict):
            return None
        value = counts.get(table_name)
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _resolve_reuse_db(reuse_db: str | Path) -> Path:
        path = Path(reuse_db).expanduser()
        db_path = path / _BENCHMARK_DB_FILENAME if path.is_dir() else path
        if not db_path.exists():
            raise ValueError(f"Reusable benchmark DB does not exist: {db_path}")
        if not db_path.is_file():
            raise ValueError(f"Reusable benchmark DB is not a file: {db_path}")
        return db_path

    @staticmethod
    def _copy_sqlite_db(source_db_path: Path, destination_db_path: Path) -> None:
        destination_db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_db_path, destination_db_path)
        for suffix in ("-wal", "-shm"):
            sidecar = source_db_path.with_name(f"{source_db_path.name}{suffix}")
            if sidecar.exists():
                shutil.copy2(
                    sidecar,
                    destination_db_path.with_name(
                        f"{destination_db_path.name}{suffix}"
                    ),
                )

    @staticmethod
    def _read_reuse_db_metadata(reuse_db: str | Path) -> dict[str, Any]:
        metadata_path = (
            LoCoMoBenchmark._resolve_reuse_db(reuse_db).parent
            / _BENCHMARK_METADATA_FILENAME
        )
        return LoCoMoBenchmark._read_metadata_file(metadata_path)

    @staticmethod
    def _new_persistent_db_dir(
        *,
        benchmark_db_dir: str | Path | None,
        conversation_id: str,
    ) -> Path:
        base_dir = (
            Path(benchmark_db_dir).expanduser()
            if benchmark_db_dir is not None
            else _PROJECT_ROOT / "docs" / "tmp" / "benchmark_dbs"
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"locomo_{_safe_path_component(conversation_id)}_{timestamp}"
        candidate = base_dir / stem
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base_dir / f"{stem}_{suffix}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    @staticmethod
    def _metadata_is_complete(metadata_dir: Path) -> bool:
        metadata = LoCoMoBenchmark._read_metadata_file(
            metadata_dir / _BENCHMARK_METADATA_FILENAME
        )
        return (
            metadata.get("status")
            in {_REUSE_DB_COMPLETE_STATUS, _REUSE_DB_UNTRUSTED_STATUS}
        )

    @staticmethod
    def _classify_current_ingest_health(
        *,
        db_path: Path,
        llm_call_summary: dict[str, Any],
        require_evidence_packets: bool,
        llm_run_guard_config: LLMRunGuardConfig,
        rebuild_result: RebuildResult | None,
        require_rebuild_result: bool = False,
        require_summary_views: bool = False,
        expected_message_count: int | None = None,
    ) -> dict[str, Any]:
        return classify_ingest_health(
            db_audit=audit_benchmark_db(db_path),
            llm_call_summary=llm_call_summary,
            require_evidence_packets=require_evidence_packets,
            llm_guard_config=llm_run_guard_config,
            rebuild_result=(
                rebuild_result.model_dump(mode="json")
                if rebuild_result is not None
                else None
            ),
            require_rebuild_result=require_rebuild_result,
            require_summary_views=require_summary_views,
            expected_message_count=expected_message_count,
        )

    def _mark_interrupted_ingestion(
        self,
        *,
        metadata_dir: Path,
        db_path: Path,
        conversation: BenchmarkConversation,
        ablation: AblationConfig | None,
        max_turns: int | None,
        trusted_evaluation: bool,
        ingest_mode: str,
        flush_every_turns: int | None,
        exc: BaseException,
        llm_call_summary: dict[str, Any] | None = None,
        llm_run_guard_snapshot: dict[str, Any] | None = None,
    ) -> None:
        selected_turns = (
            conversation.turns[:max_turns]
            if max_turns is not None
            else conversation.turns
        )
        progress_path = metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME
        metadata_path = metadata_dir / _BENCHMARK_METADATA_FILENAME
        progress = self._read_metadata_file(progress_path)
        previous_metadata = self._read_metadata_file(metadata_path)
        if llm_call_summary is None and isinstance(
            previous_metadata.get("llm_call_summary"),
            dict,
        ):
            llm_call_summary = previous_metadata["llm_call_summary"]
        if llm_run_guard_snapshot is None and isinstance(
            previous_metadata.get("llm_run_guard_snapshot"),
            dict,
        ):
            llm_run_guard_snapshot = previous_metadata["llm_run_guard_snapshot"]
        interruption_db_audit: dict[str, Any] | None = None
        try:
            interruption_db_audit = audit_benchmark_db(db_path)
        except Exception:
            logger.exception(
                "Failed to audit interrupted LoCoMo benchmark DB path=%s",
                db_path,
            )
        ingested_turns = _coerce_nonnegative_int(progress.get("ingested_turns"))
        last_turn_id = progress.get("last_turn_id")
        status = self._ingestion_interruption_status(exc)
        interruption = {
            "exception_class": exc.__class__.__name__,
            "message": str(exc),
            "interrupted_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=selected_turns,
            ingested_turns=ingested_turns,
            status=status,
            last_turn_id=last_turn_id if isinstance(last_turn_id, str) else None,
            extra={"interruption": interruption},
        )
        self._write_ingestion_metadata(
            metadata_dir=metadata_dir,
            db_path=db_path,
            conversation=conversation,
            ablation=ablation,
            max_turns=max_turns,
            trusted_evaluation=trusted_evaluation,
            ingest_mode=ingest_mode,
            flush_every_turns=flush_every_turns,
            status=status,
            interruption=interruption,
            llm_call_summary=llm_call_summary,
            llm_run_guard_snapshot=llm_run_guard_snapshot,
            interruption_db_audit=interruption_db_audit,
        )

    @staticmethod
    def _ingestion_interruption_status(exc: BaseException) -> str:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            return "cancelled"
        return "interrupted"

    def _write_ingestion_metadata(
        self,
        *,
        metadata_dir: Path,
        db_path: Path,
        conversation: BenchmarkConversation,
        ablation: AblationConfig | None,
        max_turns: int | None,
        trusted_evaluation: bool,
        ingest_mode: str,
        flush_every_turns: int | None,
        status: str,
        rebuild_result: RebuildResult | None = None,
        llm_call_summary: dict[str, Any] | None = None,
        llm_run_guard_snapshot: dict[str, Any] | None = None,
        interruption_db_audit: dict[str, Any] | None = None,
        ingest_health: dict[str, Any] | None = None,
        interruption: dict[str, Any] | None = None,
    ) -> None:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / _BENCHMARK_METADATA_FILENAME
        now = datetime.now(timezone.utc).isoformat()
        previous_created_at = _read_json_field(metadata_path, "created_at")
        created_at = (
            previous_created_at if isinstance(previous_created_at, str) else now
        )
        trusted_ingest = (
            bool(ingest_health.get("trusted_ingest"))
            if isinstance(ingest_health, dict)
            else (
                True
                if status == _REUSE_DB_COMPLETE_STATUS
                else False
                if status in {_REUSE_DB_UNTRUSTED_STATUS, "interrupted"}
                else None
            )
        )
        usable_for_evaluate_only = (
            status == _REUSE_DB_COMPLETE_STATUS and trusted_ingest is not False
        )
        metadata = {
            "benchmark_name": "LoCoMo",
            "created_at": created_at,
            "updated_at": now,
            "db_path": str(db_path),
            "effective_benchmark_db_dir": str(metadata_dir.parent),
            "data_path": str(self._data_path),
            "data_sha256": sha256_file(self._data_path),
            "conversation_id": conversation.conversation_id,
            "turn_count": len(
                conversation.turns[:max_turns]
                if max_turns is not None
                else conversation.turns
            ),
            "max_turns": max_turns,
            "mode": _DEFAULT_RETRIEVAL_PROFILE_ID,
            "retrieval_profile_id": _DEFAULT_RETRIEVAL_PROFILE_ID,
            "platform_id": _BENCHMARK_PLATFORM_ID,
            "user_persona_id": _BENCHMARK_USER_PERSONA_ID,
            "character_id": _BENCHMARK_CHARACTER_ID,
            "provider": self._llm_provider,
            "llm_model": self._answer_model
            or self._chat_model
            or self._forced_global_model,
            "answer_model": self._answer_model
            or self._chat_model
            or self._forced_global_model,
            "judge_model": self._judge_model
            or self._answer_model
            or self._chat_model
            or self._forced_global_model,
            "forced_global_model": self._forced_global_model,
            "ingest_model": self._ingest_model,
            "retrieval_model": self._retrieval_model,
            "chat_model": self._chat_model,
            "component_models": dict(sorted(self._component_models.items())),
            "activation_flags": self._activation_flags(),
            "real_chat_path": True,
            "embedding_model": self._embedding_model,
            "ablation_config": (
                ablation.model_dump(mode="json", exclude_none=True)
                if ablation is not None
                else None
            ),
            "trusted_evaluation": bool(trusted_evaluation),
            "ingest_mode": ingest_mode,
            "flush_every_turns": flush_every_turns,
            "status": status,
            "trusted_ingest": trusted_ingest,
            "ingestion_complete": usable_for_evaluate_only,
            "usable_for_evaluate_only": usable_for_evaluate_only,
            "partial": not usable_for_evaluate_only,
            "interrupted": status == "interrupted",
            "cancelled": status == "cancelled",
            "interruption": interruption,
            "rebuild_result": (
                rebuild_result.model_dump(mode="json")
                if rebuild_result is not None
                else None
            ),
            "llm_call_summary": llm_call_summary,
            "llm_run_guard_snapshot": llm_run_guard_snapshot,
            "interruption_db_audit": interruption_db_audit,
            "ingest_health": ingest_health,
            "manifests_dir": str(self._manifests_dir),
            "manifests_sha256": sha256_directory(self._manifests_dir),
            **_benchmark_migration_metadata(),
            "git": _git_state(),
        }
        write_json_atomic(metadata_path, metadata)

    @staticmethod
    def _llm_run_guard_snapshot(engine: Atagia | None) -> dict[str, Any] | None:
        if engine is None or engine.runtime is None:
            return None
        snapshot = engine.runtime.llm_client.llm_run_guard_snapshot()
        return snapshot if isinstance(snapshot, dict) else None

    @staticmethod
    def _write_ingestion_progress(
        progress_path: Path | None,
        *,
        conversation: BenchmarkConversation,
        selected_turns: Sequence[Any],
        ingested_turns: int,
        status: str,
        last_turn_id: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "conversation_id": conversation.conversation_id,
            "total_turns": len(conversation.turns),
            "selected_turns": len(selected_turns),
            "ingested_turns": ingested_turns,
            "last_turn_id": last_turn_id,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(progress_path, payload)


def _common_value(values: Iterable[str]) -> str | None:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return None
    return first if all(value == first for value in iterator) else None


def _coerce_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _read_json_field(path: Path, field_name: str) -> object:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get(field_name)


def _safe_path_component(value: str) -> str:
    return (
        "".join(
            char if char.isalnum() or char in {"-", "_"} else "-" for char in value
        ).strip("-_")
        or "conversation"
    )


def _benchmark_migration_metadata() -> dict[str, object]:
    metadata = benchmark_migration_metadata()
    return {
        "migrations_dir": metadata["path"],
        "migrations_sha256": metadata["sha256"],
        "migration_versions": metadata["versions"],
        "latest_migration_version": metadata["latest_version"],
    }


def _question_result_numeric_summary(
    report: BenchmarkReport, field_name: str
) -> dict[str, float | int | None]:
    values = [
        float(getattr(result, field_name))
        for conversation in report.conversations
        for result in conversation.results
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _selection_summary(
    conversation_inputs: Sequence[
        tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]
    ],
    *,
    scored_categories: list[int],
    question_ids: list[str] | None,
    max_questions: int | None,
    max_turns: int | None,
) -> dict[str, object]:
    planned_question_count = sum(
        len(filtered_questions[:max_questions])
        if max_questions is not None
        else len(filtered_questions)
        for _, _, filtered_questions in conversation_inputs
    )
    return {
        "conversation_ids": [
            conversation.conversation_id for _, conversation, _ in conversation_inputs
        ],
        "scored_categories": list(scored_categories),
        "question_filter": list(question_ids) if question_ids is not None else None,
        "max_questions": max_questions,
        "max_turns": max_turns,
        "selected_conversation_count": len(conversation_inputs),
        "planned_question_count": planned_question_count,
    }


def _manifest_selection_summary(report: BenchmarkReport) -> dict[str, object]:
    selection = report.model_info.get("selection")
    if isinstance(selection, dict):
        return dict(selection)
    return {
        "conversation_ids": [
            conversation.conversation_id for conversation in report.conversations
        ],
        "selected_conversation_count": len(report.conversations),
        "completed_question_count": report.total_questions,
    }


def _manifest_path_for_report(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("locomo-report-")
    return report_path.with_name(f"locomo-run-manifest-{timestamp}.json")


def _git_state() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=_PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }
