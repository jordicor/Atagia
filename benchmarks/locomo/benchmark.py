"""LoCoMo benchmark orchestration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from benchmarks.base import (
    DEFAULT_SCORED_CATEGORIES,
    BenchmarkConversation,
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkRunner,
    ConversationReport,
    QuestionResult,
)
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.scorer import LLMJudgeScorer
from benchmarks.trusted_eval import (
    TRUSTED_EVALUATION_PROMPT_NOTE,
    activate_trusted_evaluation_memories,
    trusted_evaluation_ablation,
)
from atagia import Atagia
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.models.schemas_memory import RetrievalTrace
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import build_system_prompt, chat_model, resolve_policy
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.retrieval_service import RetrievalService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_ASSISTANT_MODE_ID = "general_qa"


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
        self._llm_model = llm_model
        self._judge_model = judge_model or llm_model
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
    ) -> BenchmarkReport:
        """Run the benchmark and return an aggregated report."""
        dataset = self._adapter.load()
        selected_conversations = self._select_conversations(dataset, conversation_ids)
        scored_categories = categories or list(DEFAULT_SCORED_CATEGORIES)
        question_filter = set(question_ids) if question_ids else None
        started_at = perf_counter()
        checkpoint_output = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        conversation_reports: list[ConversationReport] = []

        for conversation_index, conversation in enumerate(selected_conversations, start=1):
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
            report = await self._run_conversation(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=len(selected_conversations),
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
                checkpoint_path=checkpoint_output,
                completed_conversation_reports=tuple(conversation_reports),
                run_started_at=started_at,
                trusted_evaluation=trusted_evaluation,
            )
            conversation_reports.append(report)

        report = self._build_report(
            conversation_reports,
            ablation=ablation,
            started_at=started_at,
            model_info_extra=(
                {"trusted_evaluation": True}
                if trusted_evaluation
                else None
            ),
        )
        if checkpoint_output is not None:
            self._write_report(report, checkpoint_output)
        return report

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
    ) -> ConversationReport:
        with TemporaryDirectory(prefix=f"atagia-locomo-{conversation.conversation_id}-") as temp_dir:
            db_path = Path(temp_dir) / "benchmark.db"
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
                await engine.create_user(user_id)
                await engine.create_conversation(
                    user_id,
                    conversation.conversation_id,
                    assistant_mode_id=_DEFAULT_ASSISTANT_MODE_ID,
                )
                turn_message_ids = await self._ingest_conversation(
                    engine,
                    user_id,
                    conversation,
                    max_turns=max_turns,
                )
                drained = await engine.flush(timeout_seconds=1800.0)
                if not drained:
                    raise RuntimeError(
                        f"Timed out while draining workers for {conversation.conversation_id}"
                    )
                trusted_activation_count = (
                    await activate_trusted_evaluation_memories(engine.runtime, user_id)
                    if trusted_evaluation and engine.runtime is not None
                    else 0
                )

                runtime = engine.runtime
                if runtime is None:
                    raise RuntimeError("Atagia runtime was unexpectedly unavailable")
                judge = LLMJudgeScorer(
                    runtime.llm_client,
                    self._judge_model or chat_model(runtime.settings),
                )
                question_results: list[QuestionResult] = []
                selected_questions = (
                    filtered_questions[:max_questions]
                    if max_questions is not None
                    else filtered_questions
                )
                for question_index, question in enumerate(selected_questions, start=1):
                    print(
                        f"Conversation {conversation_index}/{conversation_count} "
                        f"({conversation.conversation_id}): question "
                        f"{question_index}/{len(selected_questions)}",
                        flush=True,
                    )
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
                    question_results.append(
                        QuestionResult(
                            question=question,
                            prediction=prediction,
                            score_result=score_result,
                            memories_used=len(pipeline_result.composed_context.selected_memory_ids),
                            retrieval_time_ms=retrieval_time_ms,
                            trace=trace_payload,
                        )
                    )
                    if checkpoint_path is not None:
                        checkpoint_report = self._build_report(
                            [
                                *completed_conversation_reports,
                                self._build_conversation_report(
                                    conversation.conversation_id,
                                    question_results,
                                ),
                            ],
                            ablation=ablation,
                            started_at=run_started_at,
                            model_info_extra={
                                "checkpoint": {
                                    "partial": True,
                                    "conversation_id": conversation.conversation_id,
                                    "conversation_index": conversation_index,
                                    "conversation_count": conversation_count,
                                    "completed_questions": len(question_results),
                                    "selected_questions": len(selected_questions),
                                    "trusted_evaluation": trusted_evaluation,
                                }
                            },
                        )
                        self._write_report(checkpoint_report, checkpoint_path)

        return self._build_conversation_report(conversation.conversation_id, question_results)

    async def _ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        max_turns: int | None = None,
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
        for turn in turns:
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
        return {
            str(turn.turn_id): str(message["id"])
            for turn, message in zip(turns, messages, strict=True)
            if turn.turn_id is not None
        }

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
        return {
            "diagnosis_bucket": self._diagnosis_bucket(
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
                max_output_tokens=512,
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
    ) -> ConversationReport:
        total_correct = sum(result.score_result.score for result in question_results)
        return ConversationReport(
            conversation_id=conversation_id,
            results=list(question_results),
            accuracy=self._accuracy(total_correct, len(question_results)),
            category_breakdown=self._category_breakdown(question_results),
        )

    @staticmethod
    def _write_report(report: BenchmarkReport, report_path: str | Path) -> Path:
        output_path = Path(report_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def save_report(report: BenchmarkReport, output_dir: str | Path) -> Path:
        """Persist a benchmark report as JSON and return its path."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = output_path / f"locomo-report-{timestamp}.json"
        return LoCoMoBenchmark._write_report(report, report_path)
