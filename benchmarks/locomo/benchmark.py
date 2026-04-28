"""LoCoMo benchmark orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

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
from benchmarks.retrieval_custody import build_retrieval_custody
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.migration_metadata import benchmark_migration_metadata
from benchmarks.scorer import LLMJudgeScorer
from benchmarks.trusted_eval import (
    TRUSTED_EVALUATION_PROMPT_NOTE,
    activate_trusted_evaluation_memories,
    trusted_evaluation_ablation,
)
from atagia import Atagia
from atagia.core.llm_output_limits import LOCOMO_ANSWER_MAX_OUTPUT_TOKENS
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_memory import RetrievalTrace
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.admin_rebuild_service import AdminRebuildService, RebuildResult
from atagia.services.chat_support import build_system_prompt, chat_model, resolve_policy
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMError,
    LLMMessage,
    StructuredOutputError,
)
from atagia.services.model_resolution import provider_qualified_model
from atagia.services.retrieval_service import RetrievalService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_ASSISTANT_MODE_ID = "general_qa"
_BENCHMARK_DB_FILENAME = "benchmark.db"
_BENCHMARK_METADATA_FILENAME = "run_metadata.json"
_BENCHMARK_INGESTION_PROGRESS_FILENAME = "ingestion_progress.json"
_INGEST_MODE_ONLINE = "online"
_INGEST_MODE_BULK = "bulk"
_VALID_INGEST_MODES = {_INGEST_MODE_ONLINE, _INGEST_MODE_BULK}


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
        judge_model: str | None = None,
        manifests_dir: str | Path | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
        corrections_path: str | Path | None = None,
        community_corrections_path: str | Path | None = None,
    ) -> None:
        self._data_path = Path(data_path).expanduser()
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = provider_qualified_model(llm_provider, llm_model)
        self._judge_model = (
            provider_qualified_model(llm_provider, judge_model)
            if judge_model is not None
            else self._llm_model
        )
        self._manifests_dir = (
            Path(manifests_dir).expanduser()
            if manifests_dir is not None
            else _DEFAULT_MANIFESTS_DIR
        )
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
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
        keep_db: bool = False,
        reuse_db: str | Path | None = None,
        ingest_only: bool = False,
        evaluate_only: bool = False,
        parallel_conversations: int = 1,
        parallel_questions: int = 1,
        ingest_mode: str = _INGEST_MODE_ONLINE,
    ) -> BenchmarkReport:
        """Run the benchmark and return an aggregated report."""
        dataset = self._adapter.load()
        selected_conversations = self._select_conversations(dataset, conversation_ids)
        if parallel_conversations < 1:
            raise ValueError("parallel_conversations must be at least 1")
        if parallel_questions < 1:
            raise ValueError("parallel_questions must be at least 1")
        if ingest_mode not in _VALID_INGEST_MODES:
            valid_modes = ", ".join(sorted(_VALID_INGEST_MODES))
            raise ValueError(f"ingest_mode must be one of: {valid_modes}")
        if ingest_only:
            keep_db = True
        effective_evaluate_only = evaluate_only or reuse_db is not None
        if ingest_only and effective_evaluate_only:
            raise ValueError("ingest_only cannot be combined with evaluate_only or reuse_db")
        if reuse_db is not None and len(selected_conversations) != 1:
            raise ValueError("reuse_db can only be used with exactly one selected conversation")
        if evaluate_only and reuse_db is None:
            raise ValueError("evaluate_only requires reuse_db")
        reuse_db_metadata = self._read_reuse_db_metadata(reuse_db) if reuse_db is not None else {}
        if reuse_db_metadata:
            metadata_conversation_id = reuse_db_metadata.get("conversation_id")
            if (
                isinstance(metadata_conversation_id, str)
                and metadata_conversation_id
                and metadata_conversation_id != selected_conversations[0].conversation_id
            ):
                raise ValueError(
                    "Reusable benchmark DB conversation mismatch: metadata has "
                    f"{metadata_conversation_id}, selected "
                    f"{selected_conversations[0].conversation_id}"
                )
        metadata_ingest_mode = reuse_db_metadata.get("ingest_mode")
        db_ingest_mode = (
            str(metadata_ingest_mode)
            if metadata_ingest_mode in _VALID_INGEST_MODES
            else ingest_mode
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
                benchmark_db_dir=benchmark_db_dir,
                keep_db=keep_db,
                reuse_db=reuse_db,
                ingest_only=ingest_only,
                evaluate_only=effective_evaluate_only,
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
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
                benchmark_db_dir=benchmark_db_dir,
                keep_db=keep_db,
                reuse_db=reuse_db,
                ingest_only=ingest_only,
                evaluate_only=effective_evaluate_only,
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
            )

        model_info_extra = {
            "trusted_evaluation": bool(trusted_evaluation),
            "parallel_conversations": parallel_limit,
            "parallel_questions": parallel_questions,
            "ingest_mode": db_ingest_mode,
            "requested_ingest_mode": ingest_mode,
            "selection": _selection_summary(
                conversation_inputs,
                scored_categories=scored_categories,
                question_ids=question_ids,
                max_questions=max_questions,
                max_turns=max_turns,
            ),
            "benchmark_db": {
                "keep_db": bool(keep_db),
                "reuse_db": str(Path(reuse_db).expanduser()) if reuse_db is not None else None,
                "ingest_only": bool(ingest_only),
                "evaluate_only": bool(effective_evaluate_only),
                "ingest_mode": db_ingest_mode,
                "requested_ingest_mode": ingest_mode,
            },
        }
        if benchmark_db_dir is not None:
            model_info_extra["benchmark_db"]["benchmark_db_dir"] = str(
                Path(benchmark_db_dir).expanduser()
            )
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
        conversation_inputs: Sequence[tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]],
        *,
        ablation: AblationConfig | None,
        max_questions: int | None,
        max_turns: int | None,
        checkpoint_output: Path | None,
        run_started_at: float,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db: str | Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        parallel_questions: int,
        ingest_mode: str,
        db_ingest_mode: str,
    ) -> list[ConversationReport]:
        conversation_reports: list[ConversationReport] = []
        for conversation_index, conversation, filtered_questions in conversation_inputs:
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
                reuse_db=reuse_db,
                ingest_only=ingest_only,
                evaluate_only=evaluate_only,
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
            )
            conversation_reports.append(report)
        return conversation_reports

    async def _run_conversations_parallel(
        self,
        conversation_inputs: Sequence[tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]],
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
        reuse_db: str | Path | None,
        ingest_only: bool,
        evaluate_only: bool,
        parallel_questions: int,
        ingest_mode: str,
        db_ingest_mode: str,
    ) -> list[ConversationReport]:
        semaphore = asyncio.Semaphore(parallel_limit)

        async def run_one(
            conversation_index: int,
            conversation: BenchmarkConversation,
            filtered_questions: list[BenchmarkQuestion],
        ) -> ConversationReport:
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
                    reuse_db=reuse_db,
                    ingest_only=ingest_only,
                    evaluate_only=evaluate_only,
                    parallel_questions=parallel_questions,
                    ingest_mode=ingest_mode,
                    db_ingest_mode=db_ingest_mode,
                )

        tasks = [
            asyncio.create_task(run_one(conversation_index, conversation, filtered_questions))
            for conversation_index, conversation, filtered_questions in conversation_inputs
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except Exception:
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
        ingest_only: bool,
        evaluate_only: bool,
        parallel_questions: int,
        ingest_mode: str,
        db_ingest_mode: str,
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
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
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
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
            )
        with TemporaryDirectory(prefix=f"atagia-locomo-{conversation.conversation_id}-") as temp_dir:
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
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
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
        parallel_questions: int,
        ingest_mode: str,
        db_ingest_mode: str,
    ) -> ConversationReport:
        async with Atagia(
            db_path=db_path,
            manifests_dir=self._manifests_dir,
            llm_provider=self._llm_provider,
            llm_api_key=self._llm_api_key,
            llm_model=self._llm_model,
            embedding_backend=self._embedding_backend,
            embedding_model=self._embedding_model,
            skip_belief_revision=ablation.skip_belief_revision if ablation else False,
            skip_compaction=ablation.skip_compaction if ablation else False,
        ) as engine:
            user_id = "benchmark-user"
            rebuild_result: RebuildResult | None = None
            if evaluate_only:
                turn_message_ids = await self._load_turn_message_ids(
                    engine,
                    user_id=user_id,
                    conversation=conversation,
                    max_turns=max_turns,
                )
            else:
                await engine.create_user(user_id)
                await engine.create_conversation(
                    user_id,
                    conversation.conversation_id,
                    assistant_mode_id=_DEFAULT_ASSISTANT_MODE_ID,
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
                        status="started",
                    )
                progress_path = (
                    metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME
                    if metadata_dir is not None
                    else None
                )
                if ingest_mode == _INGEST_MODE_BULK:
                    turn_message_ids, rebuild_result = await self._bulk_ingest_conversation(
                        engine,
                        user_id,
                        conversation,
                        ablation=ablation,
                        max_turns=max_turns,
                        progress_path=progress_path,
                    )
                else:
                    turn_message_ids = await self._ingest_conversation(
                        engine,
                        user_id,
                        conversation,
                        max_turns=max_turns,
                        progress_path=progress_path,
                    )
                selected_turns = (
                    conversation.turns[:max_turns]
                    if max_turns is not None
                    else conversation.turns
                )
                if metadata_dir is not None:
                    self._write_ingestion_progress(
                        metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME,
                        conversation=conversation,
                        selected_turns=selected_turns,
                        ingested_turns=len(selected_turns),
                        status="draining_workers",
                        last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
                    )
                drained = await engine.flush(timeout_seconds=1800.0)
                if not drained:
                    if metadata_dir is not None:
                        self._write_ingestion_progress(
                            metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME,
                            conversation=conversation,
                            selected_turns=selected_turns,
                            ingested_turns=len(selected_turns),
                            status="worker_drain_timeout",
                            last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
                        )
                    raise RuntimeError(
                        f"Timed out while draining workers for {conversation.conversation_id}"
                    )
                if metadata_dir is not None:
                    self._write_ingestion_progress(
                        metadata_dir / _BENCHMARK_INGESTION_PROGRESS_FILENAME,
                        conversation=conversation,
                        selected_turns=selected_turns,
                        ingested_turns=len(selected_turns),
                        status="workers_drained",
                        last_turn_id=selected_turns[-1].turn_id if selected_turns else None,
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
                        status="complete",
                        rebuild_result=rebuild_result,
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
                parallel_questions=parallel_questions,
                ingest_mode=ingest_mode,
                db_ingest_mode=db_ingest_mode,
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
            ),
        )

    async def _ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        max_turns: int | None = None,
        progress_path: Path | None = None,
    ) -> dict[str, str]:
        total_turns = len(conversation.turns)
        turns = conversation.turns[:max_turns] if max_turns is not None else conversation.turns
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        print(
            f"Ingesting {len(turns)}/{total_turns} turns for {conversation.conversation_id}...",
            flush=True,
        )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=0,
            status="ingesting",
            last_turn_id=None,
        )
        for turn_index, turn in enumerate(turns, start=1):
            await engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                role=turn.role,
                text=f"{turn.speaker}: {turn.text}",
                occurred_at=turn.timestamp or None,
            )
            if runtime.settings.workers_enabled:
                drained = await engine.flush(timeout_seconds=1800.0)
                if not drained:
                    raise RuntimeError(
                        "Timed out while draining workers during benchmark ingestion "
                        f"for {conversation.conversation_id}"
                    )
            if turn_index == len(turns) or turn_index % 25 == 0:
                self._write_ingestion_progress(
                    progress_path,
                    conversation=conversation,
                    selected_turns=turns,
                    ingested_turns=turn_index,
                    status="ingesting",
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

    async def _bulk_ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        ablation: AblationConfig | None,
        max_turns: int | None = None,
        progress_path: Path | None = None,
    ) -> tuple[dict[str, str], RebuildResult]:
        total_turns = len(conversation.turns)
        turns = conversation.turns[:max_turns] if max_turns is not None else conversation.turns
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        print(
            f"Bulk ingesting {len(turns)}/{total_turns} turns for {conversation.conversation_id}...",
            flush=True,
        )
        self._write_ingestion_progress(
            progress_path,
            conversation=conversation,
            selected_turns=turns,
            ingested_turns=0,
            status="bulk_ingesting",
            last_turn_id=None,
        )
        connection = await runtime.open_connection()
        try:
            messages_repo = MessageRepository(connection, runtime.clock)
            try:
                await connection.execute("BEGIN")
                for turn in turns:
                    resolved_occurred_at = (
                        normalize_optional_timestamp(turn.timestamp)
                        or runtime.clock.now().isoformat()
                    )
                    await messages_repo.create_message(
                        message_id=None,
                        conversation_id=conversation.conversation_id,
                        role=turn.role,
                        seq=None,
                        text=f"{turn.speaker}: {turn.text}",
                        token_count=None,
                        metadata={},
                        occurred_at=resolved_occurred_at,
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

    async def _load_turn_message_ids(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: BenchmarkConversation,
        max_turns: int | None = None,
    ) -> dict[str, str]:
        turns = conversation.turns[:max_turns] if max_turns is not None else conversation.turns
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
        parallel_questions: int,
        ingest_mode: str,
        db_ingest_mode: str,
    ) -> list[QuestionResult]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        judge = LLMJudgeScorer(
            runtime.llm_client,
            self._judge_model or chat_model(runtime.settings),
        )
        results_by_index: dict[int, QuestionResult] = {}
        checkpoint_lock = asyncio.Lock()

        async def score_one(question_index: int, question: BenchmarkQuestion) -> None:
            result = await self._score_question(
                engine,
                user_id=user_id,
                conversation=conversation,
                question=question,
                question_index=question_index,
                question_count=len(selected_questions),
                conversation_index=conversation_index,
                conversation_count=conversation_count,
                ablation=ablation,
                judge=judge,
                trusted_evaluation=trusted_evaluation,
                trusted_activation_count=trusted_activation_count,
                turn_message_ids=turn_message_ids,
            )
            async with checkpoint_lock:
                results_by_index[question_index] = result
                if checkpoint_path is not None:
                    ordered_results = [
                        results_by_index[index]
                        for index in sorted(results_by_index)
                    ]
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
                            "ingest_mode": db_ingest_mode,
                            "requested_ingest_mode": ingest_mode,
                            "checkpoint": {
                                "partial": True,
                                "conversation_id": conversation.conversation_id,
                                "conversation_index": conversation_index,
                                "conversation_count": conversation_count,
                                "completed_questions": len(ordered_results),
                                "selected_questions": len(selected_questions),
                                "trusted_evaluation": trusted_evaluation,
                            },
                        },
                    )
                    self._write_report(checkpoint_report, checkpoint_path)

        if parallel_questions == 1:
            for question_index, question in enumerate(selected_questions, start=1):
                await score_one(question_index, question)
        else:
            semaphore = asyncio.Semaphore(parallel_questions)

            async def score_bounded(question_index: int, question: BenchmarkQuestion) -> None:
                async with semaphore:
                    await score_one(question_index, question)

            tasks = [
                asyncio.create_task(score_bounded(question_index, question))
                for question_index, question in enumerate(selected_questions, start=1)
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
            results_by_index[index]
            for index in range(1, len(selected_questions) + 1)
        ]

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
        # LLM errors during answer generation or judging must not cancel sibling
        # questions in the same parallel batch (asyncio.gather would propagate
        # and cancel every in-flight task across all conversations otherwise).
        try:
            retrieval_trace = RetrievalTrace(
                query_text=question.question_text,
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                timestamp_iso=runtime.clock.now().isoformat(),
            )
            retrieval_started_at = perf_counter()
            pipeline_result, assistant_mode_id = await self._retrieve_question_context(
                engine,
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                question_text=question.question_text,
                ablation=(
                    trusted_evaluation_ablation(ablation)
                    if trusted_evaluation
                    else ablation
                ),
                trace=retrieval_trace,
            )
            retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000.0
            prediction = await self._generate_answer(
                runtime=runtime,
                assistant_mode_id=assistant_mode_id,
                pipeline_result=pipeline_result,
                question_text=question.question_text,
                trusted_evaluation=trusted_evaluation,
            )
            score_result = await judge.score(
                question=question.question_text,
                prediction=prediction,
                ground_truth=question.ground_truth,
            )
            trace_payload = await self._build_question_trace(
                engine,
                user_id=user_id,
                question=question,
                pipeline_result=pipeline_result,
                retrieval_trace=retrieval_trace,
                assistant_mode_id=assistant_mode_id,
                conversation_id=conversation.conversation_id,
                turn_message_ids=turn_message_ids,
                passed=bool(score_result.score),
                trusted_evaluation=trusted_evaluation,
                trusted_activation_count=trusted_activation_count,
            )
            return QuestionResult(
                question=question,
                prediction=prediction,
                score_result=score_result,
                memories_used=len(pipeline_result.composed_context.selected_memory_ids),
                retrieval_time_ms=retrieval_time_ms,
                trace=trace_payload,
            )
        except (LLMError, StructuredOutputError) as exc:
            exc_class = type(exc).__name__
            message = str(exc)
            truncated = message if len(message) <= 500 else f"{message[:500]}..."
            logger.warning(
                "LoCoMo question scoring failed for conversation_id=%s question_index=%s: %s: %s",
                conversation.conversation_id,
                question_index,
                exc_class,
                truncated,
            )
            return QuestionResult(
                question=question,
                prediction="",
                score_result=ScoreResult(
                    score=0,
                    reasoning=f"Judge call failed: {exc_class}: {truncated}",
                    judge_model=judge.judge_model,
                ),
                memories_used=0,
                retrieval_time_ms=0.0,
                trace={"judge_failure": {"exception_class": exc_class, "message": truncated}},
            )

    async def _retrieve_question_context(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation_id: str,
        question_text: str,
        ablation: AblationConfig | None,
        trace: RetrievalTrace | None = None,
    ) -> tuple[PipelineResult, str]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")
        connection = await runtime.open_connection()
        try:
            conversations = ConversationRepository(connection, runtime.clock)
            conversation = await conversations.get_conversation(conversation_id, user_id)
            if conversation is None:
                raise ValueError(f"Conversation {conversation_id} was not found for benchmark user")
            assistant_mode_id = str(conversation["assistant_mode_id"])
        finally:
            await connection.close()

        pipeline_result = await RetrievalService(runtime).retrieve(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=question_text,
            mode=assistant_mode_id,
            ablation=ablation,
            trace=trace,
        )
        return pipeline_result, assistant_mode_id

    async def _build_question_trace(
        self,
        engine: Atagia,
        *,
        user_id: str,
        question: BenchmarkQuestion,
        pipeline_result: PipelineResult,
        retrieval_trace: RetrievalTrace,
        assistant_mode_id: str,
        conversation_id: str,
        turn_message_ids: dict[str, str],
        passed: bool,
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> dict[str, Any]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")

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
        selected_memory_ids = list(pipeline_result.composed_context.selected_memory_ids)

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
        context = pipeline_result.composed_context
        diagnosis_bucket = self._diagnosis_bucket(
            passed=passed,
            has_evidence_turns=bool(question.evidence_turn_ids),
            evidence_message_count=len(evidence_message_ids),
            evidence_memory_count=len(evidence_rows),
            active_evidence_count=active_evidence_count,
            candidate_count=(
                retrieval_trace.candidate_search.total_after_fusion
                if retrieval_trace.candidate_search is not None
                else 0
            ),
            selected_memory_count=len(selected_memory_ids),
            selected_evidence_count=len(selected_evidence_ids),
        )
        return {
            "diagnosis_bucket": diagnosis_bucket,
            "sufficiency_diagnostic": self._sufficiency_diagnostic(
                diagnosis_bucket,
                retrieval_trace=retrieval_trace,
            ),
            "shadow_sufficiency_diagnostics": (
                pipeline_result.retrieval_sufficiency.model_dump(mode="json")
                if pipeline_result.retrieval_sufficiency is not None
                else None
            ),
            "trusted_evaluation": trusted_evaluation,
            "trusted_activation_count": trusted_activation_count,
            "assistant_mode_id": assistant_mode_id,
            "retrieval_conversation_id": conversation_id,
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
            "selected_memory_ids": selected_memory_ids,
            "selected_memory_rows_found": len(selected_rows),
            "selected_evidence_memory_ids": selected_evidence_ids,
            "selected_memory_summaries": self._memory_summaries(selected_rows),
            "context": {
                "items_included": context.items_included,
                "items_dropped": context.items_dropped,
                "budget_tokens": context.budget_tokens,
                "total_tokens_estimate": context.total_tokens_estimate,
                "contract_block_chars": len(context.contract_block),
                "workspace_block_chars": len(context.workspace_block),
                "memory_block_chars": len(context.memory_block),
                "state_block_chars": len(context.state_block),
            },
            "retrieval_custody": build_retrieval_custody(
                pipeline_result=pipeline_result,
                selected_memory_ids=selected_memory_ids,
                user_id=user_id,
            ),
            "retrieval_trace": retrieval_trace.model_dump(mode="json"),
        }

    async def _generate_answer(
        self,
        *,
        runtime: Any,
        assistant_mode_id: str,
        pipeline_result: PipelineResult,
        question_text: str,
        trusted_evaluation: bool = False,
    ) -> str:
        resolved_policy = resolve_policy(
            runtime.manifests,
            assistant_mode_id,
            runtime.policy_resolver,
        )
        system_prompt = build_system_prompt(
            assistant_mode_id,
            resolved_policy,
            pipeline_result.composed_context.contract_block,
            pipeline_result.composed_context.workspace_block,
            pipeline_result.composed_context.memory_block,
            pipeline_result.composed_context.state_block,
        )
        if trusted_evaluation:
            system_prompt = f"{system_prompt}\n\n{TRUSTED_EVALUATION_PROMPT_NOTE}"
        response = await runtime.llm_client.complete(
            LLMCompletionRequest(
                model=self._llm_model or chat_model(runtime.settings),
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=question_text),
                ],
                temperature=0.0,
                max_output_tokens=LOCOMO_ANSWER_MAX_OUTPUT_TOKENS,
                metadata={
                    "purpose": "benchmark_answer_generation",
                    "question": question_text,
                },
            )
        )
        return response.output_text

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
    def _sufficiency_diagnostic(
        diagnosis_bucket: str,
        *,
        retrieval_trace: RetrievalTrace,
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
            if retrieval_trace.raw_context_access_mode == "artifact":
                return "missing_artifact_support"
            return "missing_raw_evidence"
        if diagnosis_bucket in {"composition_selected_none", "retrieval_or_ranking_miss"}:
            return "retrieval_insufficient"
        return "answer_or_judge_issue"

    @staticmethod
    def _count_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
        return dict(Counter(str(row.get(field_name) or "") for row in rows))

    @staticmethod
    def _count_int_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
        return dict(Counter(str(int(row.get(field_name) or 0)) for row in rows))

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
        return {
            "provider": self._llm_provider,
            "answer_model": self._llm_model or "",
            "judge_model": self._judge_model or self._llm_model or "",
            "assistant_mode_id": _DEFAULT_ASSISTANT_MODE_ID,
        }

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
            "retrieval_custody_summary",
            self._aggregate_retrieval_custody_summary(conversation_reports),
        )
        return BenchmarkReport(
            benchmark_name="LoCoMo",
            overall_accuracy=self._accuracy(total_correct, total_questions),
            category_breakdown=self._category_breakdown(
                result
                for report in conversation_reports
                for result in report.results
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
                    if isinstance(need_detection, dict) and need_detection.get("degraded_mode"):
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
    def _write_report(report: BenchmarkReport, report_path: str | Path) -> Path:
        return write_json_atomic(
            Path(report_path).expanduser(),
            report.model_dump(mode="json"),
        )

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
        failure_taxonomy_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an auditable manifest for one saved benchmark report."""
        report_file = Path(report_path).expanduser()
        checkpoint_file = (
            Path(checkpoint_path).expanduser()
            if checkpoint_path is not None
            else None
        )
        diff_file = Path(diff_path).expanduser() if diff_path is not None else None
        custody_file = (
            Path(custody_path).expanduser()
            if custody_path is not None
            else None
        )
        taxonomy_file = (
            Path(taxonomy_path).expanduser()
            if taxonomy_path is not None
            else None
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
                str(checkpoint_file)
                if checkpoint_file is not None
                else None
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
                sha256_file_if_exists(diff_file)
                if diff_file is not None
                else None
            ),
            "custody_path": (
                str(custody_file)
                if custody_file is not None
                else None
            ),
            "custody_sha256": (
                sha256_file_if_exists(custody_file)
                if custody_file is not None
                else None
            ),
            "taxonomy_path": (
                str(taxonomy_file)
                if taxonomy_file is not None
                else None
            ),
            "taxonomy_sha256": (
                sha256_file_if_exists(taxonomy_file)
                if taxonomy_file is not None
                else None
            ),
            "dataset": {
                "path": str(self._data_path),
                "sha256": sha256_file(self._data_path),
            },
            "migrations": benchmark_migration_metadata(),
            "git": _git_state(),
            "model_info": report.model_info,
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
                conversation.conversation_id
                for conversation in report.conversations
            ],
            "selection": _manifest_selection_summary(report),
            "retained_db_paths": retained_db_paths,
            "warning_counts": report.model_info.get("warning_counts", {}),
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
            failure_taxonomy_summary=failure_taxonomy_summary,
        )
        destination = _manifest_path_for_report(Path(report_path).expanduser())
        return write_json_atomic(destination, manifest)

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
    def _read_reuse_db_metadata(reuse_db: str | Path) -> dict[str, Any]:
        metadata_path = LoCoMoBenchmark._resolve_reuse_db(reuse_db).parent / _BENCHMARK_METADATA_FILENAME
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

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
        status: str,
        rebuild_result: RebuildResult | None = None,
    ) -> None:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "benchmark_name": "LoCoMo",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "data_path": str(self._data_path),
            "data_sha256": sha256_file(self._data_path),
            "conversation_id": conversation.conversation_id,
            "turn_count": len(
                conversation.turns[:max_turns]
                if max_turns is not None
                else conversation.turns
            ),
            "max_turns": max_turns,
            "assistant_mode_id": _DEFAULT_ASSISTANT_MODE_ID,
            "provider": self._llm_provider,
            "llm_model": self._llm_model,
            "judge_model": self._judge_model,
            "embedding_backend": self._embedding_backend,
            "embedding_model": self._embedding_model,
            "ablation_config": (
                ablation.model_dump(mode="json", exclude_none=True)
                if ablation is not None
                else None
            ),
            "trusted_evaluation": bool(trusted_evaluation),
            "ingest_mode": ingest_mode,
            "status": status,
            "rebuild_result": (
                rebuild_result.model_dump(mode="json")
                if rebuild_result is not None
                else None
            ),
            "manifests_dir": str(self._manifests_dir),
            "manifests_sha256": sha256_directory(self._manifests_dir),
            **_benchmark_migration_metadata(),
            "git": _git_state(),
        }
        write_json_atomic(metadata_dir / _BENCHMARK_METADATA_FILENAME, metadata)

    @staticmethod
    def _write_ingestion_progress(
        progress_path: Path | None,
        *,
        conversation: BenchmarkConversation,
        selected_turns: Sequence[Any],
        ingested_turns: int,
        status: str,
        last_turn_id: str | None,
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
        write_json_atomic(progress_path, payload)


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip("-_") or "conversation"


def _benchmark_migration_metadata() -> dict[str, object]:
    metadata = benchmark_migration_metadata()
    return {
        "migrations_dir": metadata["path"],
        "migrations_sha256": metadata["sha256"],
        "migration_versions": metadata["versions"],
        "latest_migration_version": metadata["latest_version"],
    }


def _question_result_numeric_summary(report: BenchmarkReport, field_name: str) -> dict[str, float | int | None]:
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
    conversation_inputs: Sequence[tuple[int, BenchmarkConversation, list[BenchmarkQuestion]]],
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
            conversation.conversation_id
            for _, conversation, _ in conversation_inputs
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
            conversation.conversation_id
            for conversation in report.conversations
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
