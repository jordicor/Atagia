"""LoCoMo benchmark orchestration."""

from __future__ import annotations

import json
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
from atagia import Atagia
from atagia.core.repositories import ConversationRepository
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import build_system_prompt, chat_model, resolve_policy
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.retrieval_service import RetrievalService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_ASSISTANT_MODE_ID = "general_qa"


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
        self._adapter = LoCoMoAdapter(self._data_path)

    async def run(
        self,
        ablation: AblationConfig | None = None,
        conversation_ids: list[str] | None = None,
        categories: list[int] | None = None,
        max_questions: int | None = None,
        max_turns: int | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark and return an aggregated report."""
        dataset = self._adapter.load()
        selected_conversations = self._select_conversations(dataset, conversation_ids)
        scored_categories = categories or list(DEFAULT_SCORED_CATEGORIES)
        started_at = perf_counter()
        conversation_reports: list[ConversationReport] = []
        total_correct = 0
        total_questions = 0

        for conversation_index, conversation in enumerate(selected_conversations, start=1):
            filtered_questions = conversation.filtered_questions(scored_categories)
            report = await self._run_conversation(
                conversation,
                filtered_questions=filtered_questions,
                conversation_index=conversation_index,
                conversation_count=len(selected_conversations),
                ablation=ablation,
                max_questions=max_questions,
                max_turns=max_turns,
            )
            conversation_reports.append(report)
            total_questions += len(report.results)
            total_correct += sum(result.score_result.score for result in report.results)

        duration_seconds = perf_counter() - started_at
        return BenchmarkReport(
            benchmark_name="LoCoMo",
            overall_accuracy=self._accuracy(total_correct, total_questions),
            category_breakdown=self._category_breakdown(
                result
                for report in conversation_reports
                for result in report.results
            ),
            conversations=conversation_reports,
            total_questions=total_questions,
            total_correct=total_correct,
            ablation_config=(
                ablation.model_dump(mode="json", exclude_none=True)
                if ablation is not None
                else None
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_info={
                "provider": self._llm_provider,
                "answer_model": self._llm_model or "",
                "judge_model": self._judge_model or self._llm_model or "",
                "assistant_mode_id": _DEFAULT_ASSISTANT_MODE_ID,
            },
            duration_seconds=duration_seconds,
        )

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
    ) -> ConversationReport:
        with TemporaryDirectory(prefix=f"atagia-locomo-{conversation.conversation_id}-") as temp_dir:
            db_path = Path(temp_dir) / "benchmark.db"
            async with Atagia(
                db_path=db_path,
                manifests_dir=self._manifests_dir,
                llm_provider=self._llm_provider,
                llm_api_key=self._llm_api_key,
                llm_model=self._llm_model,
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
                await self._ingest_conversation(
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
                    retrieval_started_at = perf_counter()
                    pipeline_result, assistant_mode_id = await self._retrieve_question_context(
                        engine,
                        user_id=user_id,
                        conversation_id=conversation.conversation_id,
                        question_text=question.question_text,
                        ablation=ablation,
                    )
                    retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000.0
                    prediction = await self._generate_answer(
                        runtime=runtime,
                        assistant_mode_id=assistant_mode_id,
                        pipeline_result=pipeline_result,
                        question_text=question.question_text,
                    )
                    score_result = await judge.score(
                        question=question.question_text,
                        prediction=prediction,
                        ground_truth=question.ground_truth,
                    )
                    question_results.append(
                        QuestionResult(
                            question=question,
                            prediction=prediction,
                            score_result=score_result,
                            memories_used=len(pipeline_result.composed_context.selected_memory_ids),
                            retrieval_time_ms=retrieval_time_ms,
                        )
                    )

        total_correct = sum(result.score_result.score for result in question_results)
        return ConversationReport(
            conversation_id=conversation.conversation_id,
            results=question_results,
            accuracy=self._accuracy(total_correct, len(question_results)),
            category_breakdown=self._category_breakdown(question_results),
        )

    async def _ingest_conversation(
        self,
        engine: Atagia,
        user_id: str,
        conversation: BenchmarkConversation,
        *,
        max_turns: int | None = None,
    ) -> None:
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

    async def _retrieve_question_context(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation_id: str,
        question_text: str,
        ablation: AblationConfig | None,
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
        )
        return pipeline_result, assistant_mode_id

    async def _generate_answer(
        self,
        *,
        runtime: Any,
        assistant_mode_id: str,
        pipeline_result: PipelineResult,
        question_text: str,
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
    def save_report(report: BenchmarkReport, output_dir: str | Path) -> Path:
        """Persist a benchmark report as JSON and return its path."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = output_path / f"locomo-report-{timestamp}.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report_path
