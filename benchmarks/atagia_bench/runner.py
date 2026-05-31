"""Atagia-bench v0 benchmark runner."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from benchmarks.activation_flags import benchmark_activation_flags
from benchmarks.artifact_hash import (
    sha256_directory,
    sha256_file,
    sha256_file_if_exists,
)
from benchmarks.atagia_bench.adapter import (
    AtagiaBenchAdapter,
    AtagiaBenchConversation,
    AtagiaBenchDataset,
    AtagiaBenchPersonaData,
    AtagiaBenchQuestion,
)
from benchmarks.atagia_bench.graders import GradeResult, resolve_grader
from benchmarks.custody_summary import summarize_retrieval_custody
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import (
    LLMCallRecorder,
    install_llm_call_recorder,
    merge_llm_call_summaries,
)
from benchmarks.llm_config import provider_api_key_kwargs
from benchmarks.migration_metadata import benchmark_migration_metadata
from benchmarks.numeric_summary import summarize_numeric_values
from benchmarks.retained_db_paths import validate_retained_benchmark_db_dir
from benchmarks.scorer import LLMJudgeScorer
from benchmarks.source_evidence import source_evidence_from_turns
from benchmarks.trusted_eval import (
    activate_trusted_evaluation_memories,
    trusted_evaluation_ablation,
)
from atagia import Atagia
from atagia.core.config import ANSWER_STANCE_PROMPT_VARIANTS, ANSWER_STANCES
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.models.schemas_memory import RetrievalTrace
from atagia.models.schemas_replay import AblationConfig
from atagia.services.chat_support import chat_model
from atagia.services.llm_client import (
    LLMError,
    StructuredOutputError,
)
from atagia.services.errors import LLMUnavailableError
from atagia.services.model_resolution import (
    COMPONENTS_BY_ID,
    provider_qualified_model,
)
from atagia.services.run_counters import (
    RunCounterAccumulator,
    normalize_run_counters,
    reset_run_counter_accumulator,
    set_run_counter_accumulator,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_HOLDOUT_PATH = Path(__file__).resolve().parent / "data" / "holdout_v0.json"
_DEFAULT_BENCHMARK_DB_DIR = _PROJECT_ROOT / "docs" / "tmp" / "benchmark_dbs"
_BENCHMARK_DB_FILENAME = "benchmark.db"
_BENCHMARK_DB_METADATA_FILENAME = "run_metadata.json"

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


# ---- Report models ----


class AtagiaQuestionResult(BaseModel):
    """Full output for one benchmarked question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    ground_truth: str
    prediction: str
    answer_type: str
    category_tags: list[str] = Field(default_factory=list)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    grade: GradeResult
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)
    conversation_id: str
    persona_id: str
    trace: dict[str, Any] = Field(default_factory=dict)


class CategoryStats(BaseModel):
    """Aggregated stats for one category tag."""

    model_config = ConfigDict(extra="forbid")

    category: str
    count: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    avg_score: float = Field(ge=0.0, le=1.0)


class AtagiaBenchReport(BaseModel):
    """Full benchmark report for an Atagia-bench run."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    benchmark_name: str = "atagia-bench-v0"
    timestamp: str
    run_duration_seconds: float = Field(ge=0.0)
    config: dict[str, Any] = Field(default_factory=dict)
    personas_used: list[str] = Field(default_factory=list)
    total_questions: int = Field(ge=0)
    total_passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    avg_score: float = Field(ge=0.0, le=1.0)
    priority_failure_count: int = Field(
        ge=0,
        validation_alias=AliasChoices(
            "priority_failure_count",
            "critical_error_count",
        ),
    )
    per_question: list[AtagiaQuestionResult] = Field(default_factory=list)
    per_category: list[CategoryStats] = Field(default_factory=list)

    @property
    def critical_error_count(self) -> int:
        """Legacy compatibility for reports emitted before the terminology rename."""
        return self.priority_failure_count


# ---- Runner ----


class AtagiaBenchRunner:
    """Run the Atagia-bench v0 benchmark against a live Atagia instance."""

    def __init__(
        self,
        llm_provider: str,
        llm_api_key: str | None,
        llm_model: str | None,
        judge_model: str | None = None,
        ingest_model: str | None = None,
        retrieval_model: str | None = None,
        answer_model: str | None = None,
        component_models: dict[str, str] | None = None,
        manifests_dir: str | Path | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
        data_dir: str | Path | None = None,
        answer_postcondition_guard_enabled: bool = False,
        answer_stance: str = "reactive",
        answer_stance_prompt_variant: str = "baseline",
    ) -> None:
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._base_model = provider_qualified_model(llm_provider, llm_model)
        unknown_components = set(component_models or {}).difference(COMPONENTS_BY_ID)
        if unknown_components:
            raise ValueError(
                "Unknown component model override(s): "
                f"{', '.join(sorted(unknown_components))}"
            )
        self._component_models = {
            component_id: provider_qualified_model(llm_provider, model) or model
            for component_id, model in (component_models or {}).items()
        }
        role_specific_models = any(
            (
                ingest_model,
                retrieval_model,
                answer_model,
                self._component_models,
            )
        )
        self._role_specific_models = bool(role_specific_models)
        self._forced_global_model = None if role_specific_models else self._base_model
        self._ingest_model = provider_qualified_model(
            llm_provider,
            ingest_model or (llm_model if role_specific_models else None),
        )
        self._retrieval_model = provider_qualified_model(
            llm_provider,
            retrieval_model or (llm_model if role_specific_models else None),
        )
        self._answer_model = (
            provider_qualified_model(
                llm_provider,
                answer_model or (llm_model if role_specific_models else None),
            )
            or self._base_model
        )
        self._judge_model = (
            provider_qualified_model(llm_provider, judge_model)
            if judge_model is not None
            else self._answer_model or self._forced_global_model
        )
        self._manifests_dir = (
            Path(manifests_dir).expanduser()
            if manifests_dir is not None
            else _DEFAULT_MANIFESTS_DIR
        )
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._adapter = AtagiaBenchAdapter(data_dir)
        self._answer_postcondition_guard_enabled = answer_postcondition_guard_enabled
        if answer_stance not in ANSWER_STANCES:
            raise ValueError(
                "answer_stance must be one of: reactive, proactive"
            )
        if answer_stance_prompt_variant not in ANSWER_STANCE_PROMPT_VARIANTS:
            raise ValueError(
                "answer_stance_prompt_variant must be one of: "
                f"{', '.join(ANSWER_STANCE_PROMPT_VARIANTS)}"
            )
        self._answer_stance = answer_stance
        self._answer_stance_prompt_variant = answer_stance_prompt_variant

    async def run(
        self,
        persona_ids: list[str] | None = None,
        category_tags: list[str] | None = None,
        question_ids: list[str] | None = None,
        exclude_question_ids: list[str] | None = None,
        benchmark_split: str = "all",
        holdout_question_ids: list[str] | None = None,
        ablation: AblationConfig | None = None,
        trusted_evaluation: bool = False,
        parallel_personas: int = 1,
        benchmark_db_dir: str | Path | None = None,
        requested_benchmark_db_dir: str | Path | None = None,
        allow_temp_benchmark_db_dir: bool = False,
        keep_db: bool = False,
        reuse_db: str | Path | None = None,
        evaluate_only: bool = False,
        invocation_args: list[str] | None = None,
    ) -> AtagiaBenchReport:
        """Run the benchmark and return a structured report."""
        if parallel_personas < 1:
            raise ValueError("parallel_personas must be at least 1")
        dataset = self._adapter.load(persona_ids)
        if evaluate_only and reuse_db is None:
            raise ValueError("evaluate_only requires reuse_db")
        if reuse_db is not None:
            evaluate_only = True
            if len(dataset.personas) != 1:
                raise ValueError("reuse_db requires exactly one selected persona")
        effective_benchmark_db_dir: Path | None = None
        if keep_db:
            effective_benchmark_db_dir = validate_retained_benchmark_db_dir(
                benchmark_db_dir or _DEFAULT_BENCHMARK_DB_DIR,
                allow_temp_benchmark_db_dir=allow_temp_benchmark_db_dir,
            )
        started_at = perf_counter()
        all_results: list[AtagiaQuestionResult] = []
        llm_summaries: list[dict[str, Any]] = []
        benchmark_db_entries: list[dict[str, Any]] = []
        run_counters = RunCounterAccumulator()
        category_filter = set(category_tags) if category_tags else None
        question_filter = set(question_ids) if question_ids else None
        exclude_question_filter = (
            set(exclude_question_ids) if exclude_question_ids else None
        )
        parallel_limit = min(parallel_personas, len(dataset.personas) or 1)

        if parallel_limit == 1:
            for persona_index, persona_data in enumerate(dataset.personas, start=1):
                persona_id = persona_data.persona.persona_id
                print(
                    f"Persona {persona_index}/{len(dataset.personas)}: "
                    f"{persona_data.persona.display_name} ({persona_id})",
                    flush=True,
                )
                results, llm_summary, db_entry = await self._run_persona(
                    persona_data,
                    category_filter=category_filter,
                    question_filter=question_filter,
                    exclude_question_filter=exclude_question_filter,
                    ablation=ablation,
                    trusted_evaluation=trusted_evaluation,
                    benchmark_db_dir=effective_benchmark_db_dir,
                    keep_db=keep_db,
                    reuse_db=reuse_db,
                    evaluate_only=evaluate_only,
                    run_counters=run_counters,
                )
                all_results.extend(results)
                llm_summaries.append(llm_summary)
                if db_entry:
                    benchmark_db_entries.append(db_entry)
        else:
            persona_reports = await self._run_personas_parallel(
                dataset.personas,
                parallel_limit=parallel_limit,
                category_filter=category_filter,
                question_filter=question_filter,
                exclude_question_filter=exclude_question_filter,
                ablation=ablation,
                trusted_evaluation=trusted_evaluation,
                benchmark_db_dir=effective_benchmark_db_dir,
                keep_db=keep_db,
                reuse_db=reuse_db,
                evaluate_only=evaluate_only,
                run_counters=run_counters,
            )
            for _persona_index, results, llm_summary, db_entry in persona_reports:
                all_results.extend(results)
                llm_summaries.append(llm_summary)
                if db_entry:
                    benchmark_db_entries.append(db_entry)

        duration = perf_counter() - started_at
        return self._build_report(
            all_results,
            dataset=dataset,
            duration_seconds=duration,
            persona_ids=[p.persona.persona_id for p in dataset.personas],
            category_tags=category_tags,
            question_ids=question_ids,
            exclude_question_ids=exclude_question_ids,
            benchmark_split=benchmark_split,
            holdout_question_ids=holdout_question_ids,
            ablation=ablation,
            trusted_evaluation=trusted_evaluation,
            parallel_personas=parallel_limit,
            llm_call_summary=merge_llm_call_summaries(llm_summaries),
            run_counters=run_counters.snapshot(),
            benchmark_db_entries=benchmark_db_entries,
            keep_db=keep_db,
            reuse_db=reuse_db,
            evaluate_only=evaluate_only,
            benchmark_db_dir=benchmark_db_dir,
            effective_benchmark_db_dir=effective_benchmark_db_dir,
            allow_temp_benchmark_db_dir=allow_temp_benchmark_db_dir,
            requested_benchmark_db_dir=requested_benchmark_db_dir,
            invocation_args=invocation_args,
        )

    async def _run_personas_parallel(
        self,
        personas: list[AtagiaBenchPersonaData],
        *,
        parallel_limit: int,
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        exclude_question_filter: set[str] | None,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db: str | Path | None,
        evaluate_only: bool,
        run_counters: RunCounterAccumulator,
    ) -> list[tuple[int, list[AtagiaQuestionResult], dict[str, Any], dict[str, Any]]]:
        """Run personas concurrently while preserving deterministic report order."""
        semaphore = asyncio.Semaphore(parallel_limit)

        async def run_one(
            persona_index: int,
            persona_data: AtagiaBenchPersonaData,
        ) -> tuple[int, list[AtagiaQuestionResult], dict[str, Any], dict[str, Any]]:
            async with semaphore:
                persona_id = persona_data.persona.persona_id
                print(
                    f"Persona {persona_index}/{len(personas)}: "
                    f"{persona_data.persona.display_name} ({persona_id})",
                    flush=True,
                )
                results, llm_summary, db_entry = await self._run_persona(
                    persona_data,
                    category_filter=category_filter,
                    question_filter=question_filter,
                    exclude_question_filter=exclude_question_filter,
                    ablation=ablation,
                    trusted_evaluation=trusted_evaluation,
                    benchmark_db_dir=benchmark_db_dir,
                    keep_db=keep_db,
                    reuse_db=reuse_db,
                    evaluate_only=evaluate_only,
                    run_counters=run_counters,
                )
                return persona_index, results, llm_summary, db_entry

        completed = await asyncio.gather(
            *[
                run_one(persona_index, persona_data)
                for persona_index, persona_data in enumerate(personas, start=1)
            ]
        )
        return sorted(completed, key=lambda item: item[0])

    async def _run_persona(
        self,
        persona_data: AtagiaBenchPersonaData,
        *,
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        exclude_question_filter: set[str] | None,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
        benchmark_db_dir: str | Path | None,
        keep_db: bool,
        reuse_db: str | Path | None,
        evaluate_only: bool,
        run_counters: RunCounterAccumulator,
    ) -> tuple[list[AtagiaQuestionResult], dict[str, Any], dict[str, Any]]:
        """Run all conversations and questions for one persona."""
        persona_id = persona_data.persona.persona_id

        if reuse_db is not None:
            source_db_path = self._resolve_reuse_db(reuse_db)
            with TemporaryDirectory(
                prefix=f"atagia-bench-{persona_id}-reuse-"
            ) as temp_dir:
                db_path = Path(temp_dir) / _BENCHMARK_DB_FILENAME
                self._copy_sqlite_db(source_db_path, db_path)
                return await self._run_persona_with_db(
                    persona_data,
                    category_filter=category_filter,
                    question_filter=question_filter,
                    exclude_question_filter=exclude_question_filter,
                    ablation=ablation,
                    trusted_evaluation=trusted_evaluation,
                    db_path=db_path,
                    metadata_dir=None,
                    source_reuse_db=source_db_path,
                    evaluate_only=True,
                    run_counters=run_counters,
                )

        if keep_db:
            metadata_dir = self._new_persistent_db_dir(
                benchmark_db_dir=benchmark_db_dir,
                persona_id=persona_id,
            )
            db_path = metadata_dir / _BENCHMARK_DB_FILENAME
            return await self._run_persona_with_db(
                persona_data,
                category_filter=category_filter,
                question_filter=question_filter,
                exclude_question_filter=exclude_question_filter,
                ablation=ablation,
                trusted_evaluation=trusted_evaluation,
                db_path=db_path,
                metadata_dir=metadata_dir,
                source_reuse_db=None,
                evaluate_only=evaluate_only,
                run_counters=run_counters,
            )

        with TemporaryDirectory(prefix=f"atagia-bench-{persona_id}-") as temp_dir:
            return await self._run_persona_with_db(
                persona_data,
                category_filter=category_filter,
                question_filter=question_filter,
                exclude_question_filter=exclude_question_filter,
                ablation=ablation,
                trusted_evaluation=trusted_evaluation,
                db_path=Path(temp_dir) / _BENCHMARK_DB_FILENAME,
                metadata_dir=None,
                source_reuse_db=None,
                evaluate_only=False,
                run_counters=run_counters,
            )

    async def _run_persona_with_db(
        self,
        persona_data: AtagiaBenchPersonaData,
        *,
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        exclude_question_filter: set[str] | None,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
        db_path: Path,
        metadata_dir: Path | None,
        source_reuse_db: Path | None,
        evaluate_only: bool,
        run_counters: RunCounterAccumulator,
    ) -> tuple[list[AtagiaQuestionResult], dict[str, Any], dict[str, Any]]:
        persona_id = persona_data.persona.persona_id
        llm_recorder = LLMCallRecorder()
        model_kwargs = self._atagia_model_kwargs()
        results: list[AtagiaQuestionResult] = []

        counter_token = set_run_counter_accumulator(run_counters)
        async with Atagia(
            db_path=db_path,
            manifests_dir=self._manifests_dir,
            **model_kwargs,
            **provider_api_key_kwargs(self._llm_provider, self._llm_api_key),
            embedding_backend=self._embedding_backend,
            embedding_model=self._embedding_model,
            skip_belief_revision=ablation.skip_belief_revision if ablation else False,
            skip_compaction=ablation.skip_compaction if ablation else False,
            answer_postcondition_guard_enabled=self._answer_postcondition_guard_enabled,
            answer_stance=self._answer_stance,
            answer_stance_prompt_variant=self._answer_stance_prompt_variant,
        ) as engine:
            runtime = engine.runtime
            if runtime is None:
                raise RuntimeError("Atagia runtime unavailable")
            install_llm_call_recorder(runtime.llm_client, llm_recorder)

            user_id = f"bench-{persona_id}"
            await engine.create_user(user_id)

            with llm_recorder.context(persona_id=persona_id):
                if evaluate_only:
                    print(
                        f"  Reusing ingested database for {persona_id}; skipping ingestion",
                        flush=True,
                    )
                    turn_message_ids = await self._load_turn_message_ids(
                        engine,
                        user_id=user_id,
                        persona_data=persona_data,
                    )
                else:
                    turn_message_ids = {}
                    for conv_index, conversation in enumerate(
                        persona_data.conversations, start=1
                    ):
                        print(
                            f"  Ingesting conversation {conv_index}/"
                            f"{len(persona_data.conversations)}: "
                            f"{conversation.conversation_id}",
                            flush=True,
                        )
                        await engine.create_conversation(
                            user_id,
                            conversation.conversation_id,
                            assistant_mode_id=conversation.assistant_mode_id,
                        )
                        turn_message_ids.update(
                            await self._ingest_conversation(
                                engine,
                                user_id=user_id,
                                conversation=conversation,
                                ablation=ablation,
                            )
                        )

                    drained = await engine.flush(timeout_seconds=1800.0)
                    if not drained:
                        raise RuntimeError(
                            f"Timed out draining workers for persona {persona_id}"
                        )

                trusted_activation_count = (
                    await activate_trusted_evaluation_memories(engine.runtime, user_id)
                    if trusted_evaluation and engine.runtime is not None
                    else 0
                )

                questions = self._filter_questions(
                    persona_data.questions,
                    category_filter,
                    question_filter,
                    exclude_question_filter,
                )
                for q_index, question in enumerate(questions, start=1):
                    print(
                        f"  Question {q_index}/{len(questions)}: "
                        f"{question.question_id}",
                        flush=True,
                    )
                    with llm_recorder.context(
                        persona_id=persona_id,
                        question_id=question.question_id,
                    ):
                        result = await self._run_question_on_db_snapshot(
                            base_db_path=db_path,
                            user_id=user_id,
                            persona_data=persona_data,
                            question=question,
                            ablation=(
                                trusted_evaluation_ablation(ablation)
                                if trusted_evaluation
                                else ablation
                            ),
                            turn_message_ids=turn_message_ids,
                            trusted_evaluation=trusted_evaluation,
                            trusted_activation_count=trusted_activation_count,
                            llm_recorder=llm_recorder,
                            run_counters=run_counters,
                        )
                    results.append(result)
        reset_run_counter_accumulator(counter_token)

        llm_summary = llm_recorder.summary()
        db_entry = self._benchmark_db_metadata(
            persona_id=persona_id,
            db_path=db_path,
            metadata_dir=metadata_dir,
            source_reuse_db=source_reuse_db,
            evaluate_only=evaluate_only,
            question_count=len(results),
        )
        if metadata_dir is not None and db_entry:
            write_json_atomic(
                metadata_dir / _BENCHMARK_DB_METADATA_FILENAME,
                db_entry,
            )
        return results, llm_summary, db_entry

    async def _run_question_on_db_snapshot(
        self,
        *,
        base_db_path: Path,
        user_id: str,
        persona_data: AtagiaBenchPersonaData,
        question: AtagiaBenchQuestion,
        ablation: AblationConfig | None,
        turn_message_ids: dict[str, str],
        trusted_evaluation: bool,
        trusted_activation_count: int,
        llm_recorder: LLMCallRecorder,
        run_counters: RunCounterAccumulator,
    ) -> AtagiaQuestionResult:
        """Run one benchmark question against a disposable real Atagia DB."""
        model_kwargs = self._atagia_model_kwargs()
        with TemporaryDirectory() as temp_dir:
            question_db_path = Path(temp_dir) / _BENCHMARK_DB_FILENAME
            self._copy_sqlite_db(base_db_path, question_db_path)
            counter_token = set_run_counter_accumulator(run_counters)
            try:
                async with Atagia(
                    db_path=question_db_path,
                    manifests_dir=self._manifests_dir,
                    **model_kwargs,
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
                    answer_stance=self._answer_stance,
                    answer_stance_prompt_variant=self._answer_stance_prompt_variant,
                ) as question_engine:
                    runtime = question_engine.runtime
                    if runtime is None:
                        raise RuntimeError("Atagia runtime unavailable")
                    install_llm_call_recorder(runtime.llm_client, llm_recorder)
                    judge = LLMJudgeScorer(
                        runtime.llm_client,
                        self._judge_model or chat_model(runtime.settings),
                    )
                    result = await self._run_question(
                        question_engine,
                        judge=judge,
                        user_id=user_id,
                        persona_data=persona_data,
                        question=question,
                        ablation=ablation,
                        turn_message_ids=turn_message_ids,
                        trusted_evaluation=trusted_evaluation,
                        trusted_activation_count=trusted_activation_count,
                    )
            finally:
                reset_run_counter_accumulator(counter_token)
            return result

    def _atagia_model_kwargs(self) -> dict[str, Any]:
        if self._forced_global_model is not None:
            return {"llm_forced_global_model": self._forced_global_model}
        return {
            "llm_ingest_model": self._ingest_model,
            "llm_retrieval_model": self._retrieval_model,
            "llm_chat_model": self._answer_model,
            "llm_component_models": dict(self._component_models),
        }

    async def _load_turn_message_ids(
        self,
        engine: Atagia,
        *,
        user_id: str,
        persona_data: AtagiaBenchPersonaData,
    ) -> dict[str, str]:
        """Map authored turn ids to persisted message ids in a reusable DB."""
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")
        turn_message_ids: dict[str, str] = {}
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            for conversation in persona_data.conversations:
                rows = await messages.list_messages_for_conversation(
                    conversation.conversation_id,
                    user_id,
                )
                if len(rows) != len(conversation.turns):
                    raise RuntimeError(
                        "Reusable benchmark DB turn/message mapping failed for "
                        f"{conversation.conversation_id}: expected "
                        f"{len(conversation.turns)} messages, found {len(rows)}"
                    )
                turn_message_ids.update(
                    {
                        turn.turn_id: str(message["id"])
                        for turn, message in zip(
                            conversation.turns,
                            rows,
                            strict=True,
                        )
                    }
                )
        finally:
            await connection.close()
        return turn_message_ids

    def _benchmark_db_metadata(
        self,
        *,
        persona_id: str,
        db_path: Path,
        metadata_dir: Path | None,
        source_reuse_db: Path | None,
        evaluate_only: bool,
        question_count: int,
    ) -> dict[str, Any]:
        if metadata_dir is None and source_reuse_db is None:
            return {}
        reported_db_path = source_reuse_db or db_path
        return {
            "manifest_kind": "atagia_bench_retained_db_metadata",
            "benchmark_name": "atagia-bench-v0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "persona_id": persona_id,
            "user_id": f"bench-{persona_id}",
            "db_path": str(reported_db_path),
            "db_sha256": sha256_file_if_exists(reported_db_path),
            "metadata_dir": str(metadata_dir) if metadata_dir is not None else None,
            "effective_benchmark_db_dir": (
                str(metadata_dir.parent) if metadata_dir is not None else None
            ),
            "source_reuse_db": (
                str(source_reuse_db) if source_reuse_db is not None else None
            ),
            "copied_to_temporary_db": source_reuse_db is not None,
            "evaluate_only": evaluate_only,
            "question_count": question_count,
            "models": self._model_config_summary(),
            "activation_flags": self._activation_flags(),
            "embedding_model": self._embedding_model,
            "dataset": {
                "path": str(self._adapter.data_dir),
                "sha256": sha256_directory(self._adapter.data_dir),
            },
            "manifests": {
                "path": str(self._manifests_dir),
                "sha256": sha256_directory(self._manifests_dir),
            },
        }

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
    def _new_persistent_db_dir(
        *,
        benchmark_db_dir: str | Path | None,
        persona_id: str,
    ) -> Path:
        base_dir = (
            Path(benchmark_db_dir).expanduser()
            if benchmark_db_dir is not None
            else _DEFAULT_BENCHMARK_DB_DIR
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"atagia_bench_{_safe_path_component(persona_id)}_{timestamp}"
        candidate = base_dir / stem
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base_dir / f"{stem}_{suffix}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

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

    async def _ingest_conversation(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: AtagiaBenchConversation,
        ablation: AblationConfig | None,
    ) -> dict[str, str]:
        """Ingest all turns from a single conversation."""
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")

        for turn in conversation.turns:
            await engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                role=turn.role,
                text=turn.text,
                occurred_at=turn.timestamp or None,
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
            if runtime.settings.workers_enabled:
                drained = await engine.flush(timeout_seconds=1800.0)
                if not drained:
                    raise RuntimeError(
                        f"Timed out draining workers during ingestion "
                        f"for {conversation.conversation_id}"
                    )
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation(conversation.conversation_id, user_id)
        finally:
            await connection.close()
        if len(messages) != len(conversation.turns):
            raise RuntimeError(
                "Benchmark turn/message mapping failed for "
                f"{conversation.conversation_id}: expected {len(conversation.turns)} "
                f"messages, found {len(messages)}"
            )
        return {
            turn.turn_id: str(message["id"])
            for turn, message in zip(conversation.turns, messages, strict=True)
        }

    async def _run_question(
        self,
        engine: Atagia,
        *,
        judge: LLMJudgeScorer,
        user_id: str,
        persona_data: AtagiaBenchPersonaData,
        question: AtagiaBenchQuestion,
        ablation: AblationConfig | None,
        turn_message_ids: dict[str, str],
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> AtagiaQuestionResult:
        """Retrieve context, generate answer, and grade one question."""
        # Find the conversation this question relates to (use the first
        # evidence turn's conversation, or the first conversation)
        # If the question specifies a target mode (e.g., privacy boundary tests),
        # use that mode instead of resolving from the evidence conversation.
        target_mode = (question.grader_config or {}).get("check_mode")
        if target_mode:
            conversation_id = self._resolve_conversation_by_mode(
                target_mode,
                persona_data,
            )
            assistant_mode_id = target_mode
        else:
            conversation_id = self._resolve_question_conversation(
                question,
                persona_data,
            )
            assistant_mode_id = self._resolve_conversation_mode(
                conversation_id,
                persona_data,
            )

        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")
        effective_ground_truth = self._ground_truth_for_question(question, ablation)
        chat_started_at = perf_counter()
        try:
            chat_result = await engine.chat(
                user_id=user_id,
                conversation_id=conversation_id,
                message=question.question_text,
                mode=assistant_mode_id,
                debug=True,
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
            prediction = chat_result.response_text
        except (LLMError, LLMUnavailableError, StructuredOutputError) as exc:
            retrieval_time_ms = (perf_counter() - chat_started_at) * 1000.0
            trace_payload = self._basic_question_trace(
                question=question,
                retrieval_trace=RetrievalTrace(
                    query_text=question.question_text,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    timestamp_iso=runtime.clock.now().isoformat(),
                ),
                assistant_mode_id=assistant_mode_id,
                conversation_id=conversation_id,
                turn_message_ids=turn_message_ids,
                trusted_evaluation=trusted_evaluation,
                trusted_activation_count=trusted_activation_count,
            )
            self._annotate_benchmark_privacy_mode(trace_payload, ablation)
            return self._technical_failure_result(
                question=question,
                stage="retrieval",
                exc=exc,
                prediction="",
                memories_used=0,
                retrieval_time_ms=retrieval_time_ms,
                conversation_id=conversation_id,
                persona_id=persona_data.persona.persona_id,
                trace=trace_payload,
            )
        retrieval_time_ms = (perf_counter() - chat_started_at) * 1000.0
        selected_memory_count = len(
            chat_result.composed_context.selected_memory_ids
            if chat_result.composed_context is not None
            else []
        )

        grader = resolve_grader(question.grader, llm_judge=judge)
        grader_config = self._grader_config_for_question(
            question,
            ablation,
            persona_data=persona_data,
            answer_stance=self._answer_stance,
        )
        try:
            grade = await grader.grade(
                prediction=prediction,
                ground_truth=effective_ground_truth,
                config=grader_config,
            )
        except (LLMError, StructuredOutputError) as exc:
            trace_payload = await self._build_question_trace_from_chat_result(
                engine,
                user_id=user_id,
                question=question,
                chat_result=chat_result,
                assistant_mode_id=assistant_mode_id,
                conversation_id=conversation_id,
                turn_message_ids=turn_message_ids,
                passed=False,
                trusted_evaluation=trusted_evaluation,
                trusted_activation_count=trusted_activation_count,
            )
            self._annotate_benchmark_privacy_mode(trace_payload, ablation)
            return self._technical_failure_result(
                question=question,
                stage="judge",
                exc=exc,
                prediction=prediction,
                memories_used=selected_memory_count,
                retrieval_time_ms=retrieval_time_ms,
                conversation_id=conversation_id,
                persona_id=persona_data.persona.persona_id,
                trace=trace_payload,
            )

        trace_payload = await self._build_question_trace_from_chat_result(
            engine,
            user_id=user_id,
            question=question,
            chat_result=chat_result,
            assistant_mode_id=assistant_mode_id,
            conversation_id=conversation_id,
            turn_message_ids=turn_message_ids,
            passed=grade.passed,
            trusted_evaluation=trusted_evaluation,
            trusted_activation_count=trusted_activation_count,
        )
        self._annotate_benchmark_privacy_mode(trace_payload, ablation)
        trace_payload["grade_context"] = self._grade_context_for_question(
            question,
            grader_config,
        )

        return AtagiaQuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            ground_truth=effective_ground_truth,
            prediction=prediction,
            answer_type=question.answer_type,
            category_tags=question.category_tags,
            evidence_turn_ids=question.evidence_turn_ids,
            grade=grade,
            memories_used=selected_memory_count,
            retrieval_time_ms=retrieval_time_ms,
            conversation_id=conversation_id,
            persona_id=persona_data.persona.persona_id,
            trace=trace_payload,
        )

    @staticmethod
    def _technical_failure_result(
        *,
        question: AtagiaBenchQuestion,
        stage: str,
        exc: Exception,
        prediction: str,
        memories_used: int,
        retrieval_time_ms: float,
        conversation_id: str,
        persona_id: str,
        trace: dict[str, Any] | None = None,
    ) -> AtagiaQuestionResult:
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
            "Atagia-bench question %s failed for persona_id=%s conversation_id=%s question_id=%s: %s: %s",
            stage,
            persona_id,
            conversation_id,
            question.question_id,
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
        return AtagiaQuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            ground_truth=question.ground_truth,
            prediction=prediction,
            answer_type=question.answer_type,
            category_tags=question.category_tags,
            evidence_turn_ids=question.evidence_turn_ids,
            grade=GradeResult(
                passed=False,
                score=0.0,
                reason=f"{reason_prefix}: {exc_class}: {truncated}",
                grader_name=question.grader,
            ),
            memories_used=memories_used,
            retrieval_time_ms=retrieval_time_ms,
            conversation_id=conversation_id,
            persona_id=persona_id,
            trace=trace_payload,
        )

    async def _build_question_trace_from_chat_result(
        self,
        engine: Atagia,
        *,
        user_id: str,
        question: AtagiaBenchQuestion,
        chat_result: Any,
        assistant_mode_id: str,
        conversation_id: str,
        turn_message_ids: dict[str, str],
        passed: bool,
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> dict[str, Any]:
        """Build benchmark trace data from the real Atagia chat result."""
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")

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
        evidence_turn_ids_by_message_id: dict[str, list[str]] = {}
        for turn_id in question.evidence_turn_ids:
            message_id = turn_message_ids.get(turn_id)
            if message_id is None:
                continue
            evidence_turn_ids_by_message_id.setdefault(message_id, []).append(turn_id)

        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            selected_rows = await memories.list_memory_objects_by_ids(
                user_id,
                selected_memory_ids,
            )
            evidence_rows_by_id: dict[str, dict[str, Any]] = {}
            evidence_turn_ids_by_memory_id: dict[str, list[str]] = {}
            for message_id in evidence_message_ids:
                for row in await memories.list_for_source_message(
                    user_id=user_id,
                    source_message_id=message_id,
                    statuses=None,
                ):
                    memory_id = str(row["id"])
                    evidence_rows_by_id[memory_id] = row
                    evidence_turn_ids_by_memory_id.setdefault(memory_id, []).extend(
                        evidence_turn_ids_by_message_id.get(message_id, [])
                    )
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
        gold_evidence_diagnostics = self._gold_evidence_diagnostics(
            evidence_rows=evidence_rows,
            retrieval_custody=retrieval_custody,
            selected_memory_ids=selected_memory_ids,
            evidence_turn_ids_by_message_id=evidence_turn_ids_by_message_id,
            evidence_turn_ids_by_memory_id=evidence_turn_ids_by_memory_id,
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
        context_payload = debug.get("context_view") if isinstance(debug, dict) else None
        return {
            "diagnosis_bucket": diagnosis_bucket,
            "sufficiency_diagnostic": diagnosis_bucket,
            "shadow_sufficiency_diagnostics": debug.get("retrieval_sufficiency"),
            "trusted_evaluation": trusted_evaluation,
            "trusted_activation_count": trusted_activation_count,
            "mode": assistant_mode_id,
            "retrieval_profile_id": assistant_mode_id,
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
            "gold_evidence_diagnostics": gold_evidence_diagnostics,
            "gold_evidence_diagnostic_summary": (
                self._gold_evidence_diagnostic_summary(gold_evidence_diagnostics)
            ),
            "selected_memory_ids": selected_memory_ids,
            "selected_memory_rows_found": len(selected_rows),
            "selected_evidence_memory_ids": selected_evidence_ids,
            "selected_memory_summaries": self._memory_summaries(selected_rows),
            "context": self._context_summary_from_chat_context(
                context, context_payload
            ),
            "retrieval_custody": retrieval_custody,
            "retrieval_trace": {
                "query_text": question.question_text,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "timestamp_iso": runtime.clock.now().isoformat(),
                "source": "real_chat_debug",
            },
            "answer_postcondition_guard": debug.get("answer_postcondition_guard"),
            "debug_authority": debug.get("authority"),
        }

    @staticmethod
    def _benchmark_privacy_enforcement(ablation: AblationConfig | None) -> str:
        if ablation is None:
            return "enforce"
        return ablation.privacy_enforcement

    @classmethod
    def _privacy_off_private_fact(
        cls,
        question: AtagiaBenchQuestion,
        ablation: AblationConfig | None,
    ) -> str | None:
        if cls._benchmark_privacy_enforcement(ablation) != "off":
            return None
        if question.answer_type != "privacy_check" or question.grader != "abstention":
            return None
        private_fact = str(
            (question.grader_config or {}).get("private_fact") or "",
        ).strip()
        return private_fact or None

    @classmethod
    def _ground_truth_for_question(
        cls,
        question: AtagiaBenchQuestion,
        ablation: AblationConfig | None,
    ) -> str:
        private_fact = cls._privacy_off_private_fact(question, ablation)
        if private_fact is not None:
            return private_fact
        return question.ground_truth

    @classmethod
    def _grader_config_for_question(
        cls,
        question: AtagiaBenchQuestion,
        ablation: AblationConfig | None,
        persona_data: AtagiaBenchPersonaData | None = None,
        answer_stance: str = "reactive",
    ) -> dict[str, Any]:
        config = dict(question.grader_config or {})
        config.setdefault("question_text", question.question_text)
        if persona_data is not None:
            source_evidence = cls._source_evidence_for_question(question, persona_data)
            if source_evidence:
                config["source_evidence"] = source_evidence
        config["benchmark_privacy_enforcement"] = cls._benchmark_privacy_enforcement(
            ablation,
        )
        config["answer_stance"] = answer_stance
        if cls._privacy_off_private_fact(question, ablation) is not None:
            config["privacy_off_retrieval_expected"] = True
        if question.grader == "abstention":
            config["abstention_kind"] = (
                "privacy_gated"
                if str(config.get("private_fact") or "").strip()
                else "unknown"
            )
        return config

    @staticmethod
    def _source_evidence_for_question(
        question: AtagiaBenchQuestion,
        persona_data: AtagiaBenchPersonaData,
    ) -> list[dict[str, str]]:
        evidence_items: list[dict[str, str]] = []
        for conversation in persona_data.conversations:
            evidence_items.extend(
                source_evidence_from_turns(
                    evidence_turn_ids=question.evidence_turn_ids,
                    turns=conversation.turns,
                    conversation_id=conversation.conversation_id,
                )
            )
        evidence_order = {
            turn_id: index for index, turn_id in enumerate(question.evidence_turn_ids)
        }
        evidence_items.sort(
            key=lambda item: evidence_order.get(str(item.get("turn_id") or ""), 0)
        )
        return evidence_items

    @staticmethod
    def _grade_context_for_question(
        question: AtagiaBenchQuestion,
        grader_config: dict[str, Any],
    ) -> dict[str, Any]:
        raw_source_evidence = grader_config.get("source_evidence")
        source_evidence = (
            raw_source_evidence
            if isinstance(raw_source_evidence, list)
            else []
        )
        source_turn_ids = [
            str(item.get("turn_id") or "")
            for item in source_evidence
            if isinstance(item, dict) and item.get("turn_id")
        ]
        source_timestamps = [
            str(item.get("timestamp") or "")
            for item in source_evidence
            if isinstance(item, dict) and item.get("timestamp")
        ]
        return {
            "grader": question.grader,
            "judge_mode": (
                "source_aware_llm_judge"
                if question.grader == "llm_judge" and source_evidence
                else question.grader
            ),
            "source_evidence_used": bool(source_evidence),
            "source_evidence_source": (
                "official_benchmark_dataset" if source_evidence else None
            ),
            "source_turn_ids": source_turn_ids,
            "source_timestamps": source_timestamps,
            "abstention_kind": grader_config.get("abstention_kind"),
        }

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
    def _basic_question_trace(
        *,
        question: AtagiaBenchQuestion,
        retrieval_trace: RetrievalTrace,
        assistant_mode_id: str,
        conversation_id: str,
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
            "evidence_turn_ids": list(question.evidence_turn_ids),
            "evidence_message_ids": evidence_message_ids,
            "missing_evidence_turn_ids": missing_evidence_turn_ids,
            "evidence_memory_ids": [],
            "gold_evidence_diagnostics": [],
            "gold_evidence_diagnostic_summary": (
                AtagiaBenchRunner._gold_evidence_diagnostic_summary([])
            ),
            "selected_memory_ids": [],
            "selected_evidence_memory_ids": [],
            "retrieval_custody": [],
            "retrieval_trace": retrieval_trace.model_dump(mode="json"),
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
    def _aggregate_warning_counts(
        results: list[AtagiaQuestionResult],
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
        for result in results:
            if not result.grade.passed:
                counts["failed_questions"] += 1
            diagnosis = str(result.trace.get("diagnosis_bucket") or "")
            if diagnosis in counts:
                counts[diagnosis] += 1
            retrieval_trace = result.trace.get("retrieval_trace")
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
    def _trace_field_counts(
        results: list[AtagiaQuestionResult],
        field_name: str,
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for result in results:
            trace = result.trace if isinstance(result.trace, dict) else {}
            raw_value = trace.get(field_name)
            value = str(raw_value).strip() if raw_value is not None else ""
            counts[value or "unknown"] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _aggregate_retrieval_custody_summary(
        results: list[AtagiaQuestionResult],
    ) -> dict[str, object]:
        return summarize_retrieval_custody(
            result.trace.get("retrieval_custody", [])
            if isinstance(result.trace, dict)
            else []
            for result in results
        )

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
            }
        if isinstance(context_payload, dict):
            return {
                "items_included": context_payload.get("items_included", 0),
                "items_dropped": context_payload.get("items_dropped", 0),
                "budget_tokens": context_payload.get("budget_tokens", 0),
                "total_tokens_estimate": context_payload.get(
                    "total_tokens_estimate", 0
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
        }

    @staticmethod
    def _gold_evidence_diagnostics(
        *,
        evidence_rows: list[dict[str, Any]],
        retrieval_custody: list[dict[str, Any]],
        selected_memory_ids: list[str],
        evidence_turn_ids_by_message_id: dict[str, list[str]] | None = None,
        evidence_turn_ids_by_memory_id: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Summarize where each gold evidence memory reached in retrieval."""
        custody_by_candidate_id = {
            str(record.get("candidate_id") or ""): record
            for record in retrieval_custody
            if isinstance(record, dict) and str(record.get("candidate_id") or "")
        }
        composed_ids = set(selected_memory_ids)
        turn_ids_by_message = evidence_turn_ids_by_message_id or {}
        turn_ids_by_memory = evidence_turn_ids_by_memory_id or {}
        diagnostics: list[dict[str, Any]] = []

        for row in sorted(evidence_rows, key=lambda item: str(item.get("id") or "")):
            memory_id = str(row.get("id") or "")
            custody = custody_by_candidate_id.get(memory_id)
            channels = _safe_trace_str_list(
                custody.get("channels") if custody is not None else None
            )
            source_message_id = str(row.get("source_message_id") or "")
            found_after_fusion = (
                custody is not None and custody.get("fusion_position") is not None
            )
            source_turn_ids = sorted(
                set(turn_ids_by_memory.get(memory_id, []))
                or set(turn_ids_by_message.get(source_message_id, []))
            )
            shortlisted = (
                _safe_trace_bool(custody.get("shortlisted")) if custody else False
            )
            scored = _safe_trace_bool(custody.get("scored")) if custody else False
            selected = _safe_trace_bool(custody.get("selected")) if custody else False
            composed = memory_id in composed_ids
            diagnostics.append(
                {
                    "memory_id": memory_id,
                    "source_message_id": source_message_id or None,
                    "source_turn_ids": source_turn_ids,
                    "object_type": str(row.get("object_type") or ""),
                    "scope": str(row.get("scope") or ""),
                    "status": str(row.get("status") or ""),
                    "privacy_level": int(row.get("privacy_level") or 0),
                    "candidate_record_found": custody is not None,
                    "found_before_fusion": bool(channels),
                    "found_after_fusion": found_after_fusion,
                    "channels": channels,
                    "channel_ranks": _safe_trace_rank_map(
                        custody.get("channel_ranks") if custody is not None else None
                    ),
                    "retrieval_sources": _safe_trace_str_list(
                        custody.get("retrieval_sources")
                        if custody is not None
                        else None
                    ),
                    "fusion_position": _safe_trace_int(
                        custody.get("fusion_position") if custody is not None else None
                    ),
                    "shortlisted": shortlisted,
                    "shortlist_rank": _safe_trace_int(
                        custody.get("shortlist_rank") if custody is not None else None
                    ),
                    "shortlist_status": (
                        str(custody.get("shortlist_status") or "")
                        if custody is not None
                        else "not_found"
                    ),
                    "scored": scored,
                    "score_rank": _safe_trace_int(
                        custody.get("score_rank") if custody is not None else None
                    ),
                    "score_status": (
                        str(custody.get("score_status") or "")
                        if custody is not None
                        else "not_found"
                    ),
                    "selected": selected,
                    "selection_rank": _safe_trace_int(
                        custody.get("selection_rank") if custody is not None else None
                    ),
                    "composed": composed,
                    "composer_decision": (
                        str(custody.get("composer_decision") or "")
                        if custody is not None
                        else "not_found"
                    ),
                    "filter_reason": (
                        str(custody.get("filter_reason") or "")
                        if (
                            custody is not None
                            and custody.get("filter_reason") is not None
                        )
                        else None
                    ),
                    "last_observed_stage": _gold_evidence_last_observed_stage(
                        candidate_record_found=custody is not None,
                        found_after_fusion=found_after_fusion,
                        shortlisted=shortlisted,
                        scored=scored,
                        selected=selected,
                        composed=composed,
                    ),
                }
            )
        return diagnostics

    @staticmethod
    def _gold_evidence_diagnostic_summary(
        diagnostics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate gold-evidence stage coverage for quick report triage."""
        channel_counts: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter()
        for diagnostic in diagnostics:
            stage_counts[str(diagnostic.get("last_observed_stage") or "unknown")] += 1
            for channel in diagnostic.get("channels") or []:
                channel_counts[str(channel)] += 1
        return {
            "gold_evidence_count": len(diagnostics),
            "candidate_record_found_count": sum(
                1 for item in diagnostics if item.get("candidate_record_found")
            ),
            "found_before_fusion_count": sum(
                1 for item in diagnostics if item.get("found_before_fusion")
            ),
            "found_after_fusion_count": sum(
                1 for item in diagnostics if item.get("found_after_fusion")
            ),
            "shortlisted_count": sum(
                1 for item in diagnostics if item.get("shortlisted")
            ),
            "scored_count": sum(1 for item in diagnostics if item.get("scored")),
            "selected_count": sum(1 for item in diagnostics if item.get("selected")),
            "composed_count": sum(1 for item in diagnostics if item.get("composed")),
            "channel_counts": dict(sorted(channel_counts.items())),
            "last_observed_stage_counts": dict(sorted(stage_counts.items())),
        }

    def _resolve_question_conversation(
        self,
        question: AtagiaBenchQuestion,
        persona_data: AtagiaBenchPersonaData,
    ) -> str:
        """Determine which conversation a question should be asked in.

        For cross-conversation questions, uses the LAST conversation
        (simulating a user asking in a new session about past info).
        """
        if not question.evidence_turn_ids:
            # Abstention or questions without evidence: use the last conversation
            return persona_data.conversations[-1].conversation_id

        # Build a map of turn_id -> conversation_id
        turn_to_conv: dict[str, str] = {}
        for conv in persona_data.conversations:
            for turn in conv.turns:
                turn_to_conv[turn.turn_id] = conv.conversation_id

        evidence_conversations = set()
        for turn_id in question.evidence_turn_ids:
            conv_id = turn_to_conv.get(turn_id)
            if conv_id:
                evidence_conversations.add(conv_id)

        if len(evidence_conversations) > 1:
            # Cross-conversation question: ask from the last conversation
            return persona_data.conversations[-1].conversation_id

        if evidence_conversations:
            return evidence_conversations.pop()

        return persona_data.conversations[-1].conversation_id

    def _resolve_conversation_by_mode(
        self,
        target_mode: str,
        persona_data: AtagiaBenchPersonaData,
    ) -> str:
        """Find a conversation with the given retrieval profile.

        Used for privacy boundary tests where the question must be asked
        from a specific mode context (e.g., asking from coding_debug to
        test that personal_assistant memories are not leaked).
        """
        for conv in persona_data.conversations:
            if conv.assistant_mode_id == target_mode:
                return conv.conversation_id
        # If no conversation exists in that mode, use the last conversation
        return persona_data.conversations[-1].conversation_id

    def _resolve_conversation_mode(
        self,
        conversation_id: str,
        persona_data: AtagiaBenchPersonaData,
    ) -> str:
        """Get the retrieval profile for a conversation."""
        for conv in persona_data.conversations:
            if conv.conversation_id == conversation_id:
                return conv.assistant_mode_id
        return "general_qa"

    @staticmethod
    def _filter_questions(
        questions: list[AtagiaBenchQuestion],
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        exclude_question_filter: set[str] | None = None,
    ) -> list[AtagiaBenchQuestion]:
        """Filter questions by category tags if a filter is provided."""
        if (
            category_filter is None
            and question_filter is None
            and exclude_question_filter is None
        ):
            return questions
        return [
            q
            for q in questions
            if (
                (
                    category_filter is None
                    or category_filter.intersection(q.category_tags)
                )
                and (question_filter is None or q.question_id in question_filter)
                and (
                    exclude_question_filter is None
                    or q.question_id not in exclude_question_filter
                )
            )
        ]

    def _model_config_summary(self) -> dict[str, Any]:
        if self._forced_global_model is not None:
            model_mode = "forced_global"
        elif self._role_specific_models:
            model_mode = "role_specific"
        else:
            model_mode = "defaults"
        return {
            "model_mode": model_mode,
            "base_model": self._base_model or "",
            "forced_global_model": self._forced_global_model or "",
            "ingest_model": self._ingest_model or "",
            "retrieval_model": self._retrieval_model or "",
            "answer_model": self._answer_model or "",
            "component_models": dict(sorted(self._component_models.items())),
            "answer_stance": self._answer_stance,
            "answer_stance_prompt_variant": self._answer_stance_prompt_variant,
            "judge_model": (
                self._judge_model
                or self._answer_model
                or self._forced_global_model
                or ""
            ),
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
        results: list[AtagiaQuestionResult],
        *,
        dataset: AtagiaBenchDataset,
        duration_seconds: float,
        persona_ids: list[str],
        category_tags: list[str] | None,
        question_ids: list[str] | None,
        exclude_question_ids: list[str] | None,
        benchmark_split: str,
        holdout_question_ids: list[str] | None,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
        parallel_personas: int = 1,
        llm_call_summary: dict[str, Any] | None = None,
        benchmark_db_entries: list[dict[str, Any]] | None = None,
        keep_db: bool = False,
        reuse_db: str | Path | None = None,
        evaluate_only: bool = False,
        benchmark_db_dir: str | Path | None = None,
        effective_benchmark_db_dir: str | Path | None = None,
        allow_temp_benchmark_db_dir: bool = False,
        requested_benchmark_db_dir: str | Path | None = None,
        invocation_args: list[str] | None = None,
        run_counters: dict[str, Any] | None = None,
    ) -> AtagiaBenchReport:
        """Build the aggregated benchmark report."""
        total = len(results)
        passed = sum(1 for r in results if r.grade.passed)
        total_score = sum(r.grade.score for r in results)
        pass_rate = passed / total if total > 0 else 0.0
        avg_score = total_score / total if total > 0 else 0.0

        # Priority failures are ordinary benchmark misses in high-importance tags.
        priority_failure_count = sum(
            1 for r in results if not r.grade.passed and _is_priority_failure(r)
        )

        # Per-category stats
        category_buckets: dict[str, list[AtagiaQuestionResult]] = {}
        for result in results:
            for tag in result.category_tags:
                category_buckets.setdefault(tag, []).append(result)

        per_category: list[CategoryStats] = []
        for category, category_results in sorted(category_buckets.items()):
            cat_passed = sum(1 for r in category_results if r.grade.passed)
            cat_total = len(category_results)
            cat_score = sum(r.grade.score for r in category_results)
            per_category.append(
                CategoryStats(
                    category=category,
                    count=cat_total,
                    pass_count=cat_passed,
                    pass_rate=cat_passed / cat_total if cat_total > 0 else 0.0,
                    avg_score=cat_score / cat_total if cat_total > 0 else 0.0,
                )
            )

        return AtagiaBenchReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_duration_seconds=round(duration_seconds, 2),
            config={
                "provider": self._llm_provider,
                **self._model_config_summary(),
                "activation_flags": self._activation_flags(),
                "category_filter": category_tags,
                "question_filter": question_ids,
                "exclude_question_filter": exclude_question_ids,
                "benchmark_split": benchmark_split,
                "holdout_question_ids": holdout_question_ids,
                "ablation_config": (
                    ablation.model_dump(mode="json", exclude_none=True)
                    if ablation is not None
                    else None
                ),
                "trusted_evaluation": trusted_evaluation,
                "parallel_personas": parallel_personas,
                "keep_db": keep_db,
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
                    str(Path(effective_benchmark_db_dir).expanduser())
                    if effective_benchmark_db_dir is not None
                    else None
                ),
                "allow_temp_benchmark_db_dir": allow_temp_benchmark_db_dir,
                "reuse_db": str(reuse_db) if reuse_db is not None else None,
                "evaluate_only": evaluate_only,
                "invocation_args": invocation_args or [],
                "benchmark_databases": benchmark_db_entries or [],
                "warning_counts": self._aggregate_warning_counts(results),
                "failure_stage_counts": self._failure_stage_counts(results),
                "retrieval_custody_summary": self._aggregate_retrieval_custody_summary(
                    results
                ),
                "llm_call_summary": llm_call_summary or {},
                "run_counters": normalize_run_counters(run_counters),
            },
            personas_used=persona_ids,
            total_questions=total,
            total_passed=passed,
            pass_rate=round(pass_rate, 4),
            avg_score=round(avg_score, 4),
            priority_failure_count=priority_failure_count,
            per_question=results,
            per_category=per_category,
        )

    def build_run_manifest(
        self,
        report: AtagiaBenchReport,
        *,
        report_path: str | Path,
        diff_path: str | Path | None = None,
        holdout_path: str | Path | None = None,
        custody_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        failure_taxonomy_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an auditable manifest for one saved Atagia-bench report."""
        report_file = Path(report_path).expanduser()
        diff_file = Path(diff_path).expanduser() if diff_path is not None else None
        custody_file = (
            Path(custody_path).expanduser() if custody_path is not None else None
        )
        taxonomy_file = (
            Path(taxonomy_path).expanduser() if taxonomy_path is not None else None
        )
        resolved_holdout_path = (
            Path(holdout_path).expanduser()
            if holdout_path is not None
            else _DEFAULT_HOLDOUT_PATH
        )
        return {
            "manifest_kind": "atagia_bench_run_manifest",
            "benchmark_name": report.benchmark_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_timestamp": report.timestamp,
            "report_path": str(report_file),
            "report_sha256": sha256_file_if_exists(report_file),
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
            "dataset": {
                "path": str(self._adapter.data_dir),
                "sha256": sha256_directory(self._adapter.data_dir),
                "holdout_path": str(resolved_holdout_path),
                "holdout_sha256": (
                    sha256_file(resolved_holdout_path)
                    if resolved_holdout_path.exists()
                    else ""
                ),
            },
            "manifests": {
                "path": str(self._manifests_dir),
                "sha256": sha256_directory(self._manifests_dir),
            },
            "migrations": benchmark_migration_metadata(),
            "git": _git_state(),
            "activation_flags": self._activation_flags(),
            "config": report.config,
            "run_counters": normalize_run_counters(
                report.config.get("run_counters")
            ),
            "selection": _selection_summary(report),
            "retrieval_custody_summary": report.config.get(
                "retrieval_custody_summary",
                {},
            ),
            "failure_taxonomy_summary": failure_taxonomy_summary or {},
            "failure_stage_counts": self._failure_stage_counts(report.per_question),
            "diagnosis_bucket_counts": self._trace_field_counts(
                report.per_question,
                "diagnosis_bucket",
            ),
            "sufficiency_diagnostic_counts": self._trace_field_counts(
                report.per_question,
                "sufficiency_diagnostic",
            ),
            "result_summary": {
                "total_questions": report.total_questions,
                "total_passed": report.total_passed,
                "pass_rate": report.pass_rate,
                "avg_score": report.avg_score,
                "priority_failure_count": report.priority_failure_count,
                "run_duration_seconds": report.run_duration_seconds,
                "retrieval_time_ms": _question_result_numeric_summary(
                    report.per_question,
                    "retrieval_time_ms",
                ),
                "memories_used": _question_result_numeric_summary(
                    report.per_question,
                    "memories_used",
                ),
            },
            "personas_used": report.personas_used,
            "question_ids": [result.question_id for result in report.per_question],
            "category_breakdown": [
                stats.model_dump(mode="json") for stats in report.per_category
            ],
            "benchmark_questions_persisted_as_messages": False,
        }

    def save_run_manifest(
        self,
        report: AtagiaBenchReport,
        *,
        report_path: str | Path,
        diff_path: str | Path | None = None,
        holdout_path: str | Path | None = None,
        custody_path: str | Path | None = None,
        taxonomy_path: str | Path | None = None,
        failure_taxonomy_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a benchmark run manifest beside the saved report."""
        manifest = self.build_run_manifest(
            report,
            report_path=report_path,
            diff_path=diff_path,
            holdout_path=holdout_path,
            custody_path=custody_path,
            taxonomy_path=taxonomy_path,
            failure_taxonomy_summary=failure_taxonomy_summary,
        )
        destination = _manifest_path_for_report(Path(report_path).expanduser())
        return write_json_atomic(destination, manifest)

    @staticmethod
    def save_report(report: AtagiaBenchReport, output_dir: str | Path) -> Path:
        """Persist a benchmark report as JSON and return its path."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = output_path / f"atagia-bench-report-{timestamp}.json"
        return write_json_atomic(report_path, report.model_dump(mode="json"))

    @staticmethod
    def _failure_stage_counts(results: list[AtagiaQuestionResult]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for result in results:
            trace = result.trace if isinstance(result.trace, dict) else {}
            value = str(trace.get("failure_stage") or "").strip()
            if value:
                counts[value] += 1
        return dict(sorted(counts.items()))


def load_holdout_question_ids(path: str | Path = _DEFAULT_HOLDOUT_PATH) -> list[str]:
    """Load the frozen Atagia-bench holdout question ID list."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    question_ids = payload.get("question_ids")
    if not isinstance(question_ids, list) or not all(
        isinstance(item, str) for item in question_ids
    ):
        raise ValueError("Holdout manifest must contain a string question_ids list")
    return sorted(set(question_ids))


def _question_result_numeric_summary(
    results: list[AtagiaQuestionResult],
    field_name: str,
) -> dict[str, float | int | None]:
    return summarize_numeric_values(getattr(result, field_name) for result in results)


def _selection_summary(report: AtagiaBenchReport) -> dict[str, object]:
    holdout_ids = report.config.get("holdout_question_ids")
    question_filter = report.config.get("question_filter")
    exclude_question_filter = report.config.get("exclude_question_filter")
    return {
        "benchmark_split": report.config.get("benchmark_split", "all"),
        "question_filter": (
            question_filter if isinstance(question_filter, list) else None
        ),
        "exclude_question_filter": (
            exclude_question_filter
            if isinstance(exclude_question_filter, list)
            else None
        ),
        "holdout_question_count": (
            len(holdout_ids) if isinstance(holdout_ids, list) else 0
        ),
        "selected_question_count": len(report.per_question),
    }


def _manifest_path_for_report(report_path: Path) -> Path:
    timestamp = report_path.stem.removeprefix("atagia-bench-report-")
    return report_path.with_name(f"atagia-bench-run-manifest-{timestamp}.json")


def _safe_path_component(value: str) -> str:
    cleaned: list[str] = []
    previous_was_separator = False
    for character in value.strip().lower():
        if character.isalnum():
            cleaned.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            cleaned.append("_")
            previous_was_separator = True
    return "".join(cleaned).strip("_") or "run"


def _safe_trace_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if str(item)})


def _safe_trace_rank_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    ranks: dict[str, int] = {}
    for key, raw_rank in value.items():
        rank = _safe_trace_int(raw_rank)
        if rank is not None:
            ranks[str(key)] = rank
    return ranks


def _safe_trace_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_trace_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _gold_evidence_last_observed_stage(
    *,
    candidate_record_found: bool,
    found_after_fusion: bool,
    shortlisted: bool,
    scored: bool,
    selected: bool,
    composed: bool,
) -> str:
    if composed:
        return "composed"
    if selected:
        return "selected"
    if scored:
        return "scored"
    if shortlisted:
        return "shortlisted"
    if found_after_fusion:
        return "after_fusion"
    if candidate_record_found:
        return "candidate_record_only"
    return "not_found"


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


def _is_priority_failure(result: AtagiaQuestionResult) -> bool:
    """Determine if a failed result belongs to a high-importance tag."""
    trace = result.trace if isinstance(result.trace, dict) else {}
    if (
        result.answer_type == "privacy_check"
        and trace.get("benchmark_privacy_enforcement") == "off"
    ):
        return False
    critical_tags = {
        "privacy_boundary",
        "mode_boundary",
        "hard_fail",
        "consent_gated",
        "high_risk",
    }
    return bool(critical_tags.intersection(result.category_tags))
