"""Atagia-bench v0 benchmark runner."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.atagia_bench.adapter import (
    AtagiaBenchAdapter,
    AtagiaBenchConversation,
    AtagiaBenchDataset,
    AtagiaBenchPersonaData,
    AtagiaBenchQuestion,
)
from benchmarks.atagia_bench.graders import GradeResult, resolve_grader
from benchmarks.scorer import LLMJudgeScorer
from benchmarks.trusted_eval import (
    TRUSTED_EVALUATION_PROMPT_NOTE,
    activate_trusted_evaluation_memories,
    trusted_evaluation_ablation,
)
from atagia import Atagia
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.models.schemas_memory import RetrievalTrace
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import build_system_prompt, chat_model, resolve_policy
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"


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

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str = "atagia-bench-v0"
    timestamp: str
    run_duration_seconds: float = Field(ge=0.0)
    config: dict[str, Any] = Field(default_factory=dict)
    personas_used: list[str] = Field(default_factory=list)
    total_questions: int = Field(ge=0)
    total_passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    avg_score: float = Field(ge=0.0, le=1.0)
    critical_error_count: int = Field(ge=0)
    per_question: list[AtagiaQuestionResult] = Field(default_factory=list)
    per_category: list[CategoryStats] = Field(default_factory=list)


# ---- Runner ----


class AtagiaBenchRunner:
    """Run the Atagia-bench v0 benchmark against a live Atagia instance."""

    def __init__(
        self,
        llm_provider: str,
        llm_api_key: str | None,
        llm_model: str | None,
        judge_model: str | None = None,
        manifests_dir: str | Path | None = None,
        embedding_backend: str = "none",
        embedding_model: str | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
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
        self._adapter = AtagiaBenchAdapter(data_dir)

    async def run(
        self,
        persona_ids: list[str] | None = None,
        category_tags: list[str] | None = None,
        question_ids: list[str] | None = None,
        ablation: AblationConfig | None = None,
        trusted_evaluation: bool = False,
    ) -> AtagiaBenchReport:
        """Run the benchmark and return a structured report."""
        dataset = self._adapter.load(persona_ids)
        started_at = perf_counter()
        all_results: list[AtagiaQuestionResult] = []
        category_filter = set(category_tags) if category_tags else None
        question_filter = set(question_ids) if question_ids else None

        for persona_index, persona_data in enumerate(dataset.personas, start=1):
            persona_id = persona_data.persona.persona_id
            print(
                f"Persona {persona_index}/{len(dataset.personas)}: "
                f"{persona_data.persona.display_name} ({persona_id})",
                flush=True,
            )
            results = await self._run_persona(
                persona_data,
                category_filter=category_filter,
                question_filter=question_filter,
                ablation=ablation,
                trusted_evaluation=trusted_evaluation,
            )
            all_results.extend(results)

        duration = perf_counter() - started_at
        return self._build_report(
            all_results,
            dataset=dataset,
            duration_seconds=duration,
            persona_ids=[p.persona.persona_id for p in dataset.personas],
            category_tags=category_tags,
            question_ids=question_ids,
            trusted_evaluation=trusted_evaluation,
        )

    async def _run_persona(
        self,
        persona_data: AtagiaBenchPersonaData,
        *,
        category_filter: set[str] | None,
        question_filter: set[str] | None,
        ablation: AblationConfig | None,
        trusted_evaluation: bool,
    ) -> list[AtagiaQuestionResult]:
        """Run all conversations and questions for one persona."""
        persona_id = persona_data.persona.persona_id

        with TemporaryDirectory(prefix=f"atagia-bench-{persona_id}-") as temp_dir:
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
                user_id = f"bench-{persona_id}"
                await engine.create_user(user_id)
                turn_message_ids: dict[str, str] = {}

                # Ingest all conversations
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
                        )
                    )

                # Wait for all extraction workers to finish
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

                # Set up judge
                runtime = engine.runtime
                if runtime is None:
                    raise RuntimeError("Atagia runtime unavailable")
                judge = LLMJudgeScorer(
                    runtime.llm_client,
                    self._judge_model or chat_model(runtime.settings),
                )

                # Run questions
                questions = self._filter_questions(
                    persona_data.questions,
                    category_filter,
                    question_filter,
                )
                results: list[AtagiaQuestionResult] = []
                for q_index, question in enumerate(questions, start=1):
                    print(
                        f"  Question {q_index}/{len(questions)}: "
                        f"{question.question_id}",
                        flush=True,
                    )
                    result = await self._run_question(
                        engine,
                        judge=judge,
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
                    )
                    results.append(result)

        return results

    async def _ingest_conversation(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: AtagiaBenchConversation,
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
        retrieval_trace = RetrievalTrace(
            query_text=question.question_text,
            user_id=user_id,
            conversation_id=conversation_id,
            timestamp_iso=runtime.clock.now().isoformat(),
        )
        retrieval_started_at = perf_counter()
        pipeline_result = await self._retrieve_context(
            engine,
            user_id=user_id,
            conversation_id=conversation_id,
            question_text=question.question_text,
            assistant_mode_id=assistant_mode_id,
            ablation=ablation,
            trace=retrieval_trace,
        )
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000.0

        prediction = await self._generate_answer(
            runtime=runtime,
            assistant_mode_id=assistant_mode_id,
            pipeline_result=pipeline_result,
            question_text=question.question_text,
            question_id=question.question_id,
            current_user_display_name=persona_data.persona.display_name,
            trusted_evaluation=trusted_evaluation,
        )

        # Grade
        grader = resolve_grader(question.grader, llm_judge=judge)
        grader_config = dict(question.grader_config or {})
        if "question_text" not in grader_config:
            grader_config["question_text"] = question.question_text
        grade = await grader.grade(
            prediction=prediction,
            ground_truth=question.ground_truth,
            config=grader_config,
        )
        trace_payload = await self._build_question_trace(
            engine,
            user_id=user_id,
            question=question,
            pipeline_result=pipeline_result,
            retrieval_trace=retrieval_trace,
            assistant_mode_id=assistant_mode_id,
            conversation_id=conversation_id,
            turn_message_ids=turn_message_ids,
            grade=grade,
            trusted_evaluation=trusted_evaluation,
            trusted_activation_count=trusted_activation_count,
        )

        return AtagiaQuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            ground_truth=question.ground_truth,
            prediction=prediction,
            answer_type=question.answer_type,
            category_tags=question.category_tags,
            evidence_turn_ids=question.evidence_turn_ids,
            grade=grade,
            memories_used=len(pipeline_result.composed_context.selected_memory_ids),
            retrieval_time_ms=retrieval_time_ms,
            conversation_id=conversation_id,
            persona_id=persona_data.persona.persona_id,
            trace=trace_payload,
        )

    async def _retrieve_context(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation_id: str,
        question_text: str,
        assistant_mode_id: str,
        ablation: AblationConfig | None,
        trace: RetrievalTrace | None = None,
    ) -> PipelineResult:
        """Run the retrieval pipeline for a question."""
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")

        return await RetrievalService(runtime).retrieve(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=question_text,
            mode=assistant_mode_id,
            ablation=ablation,
            trace=trace,
        )

    async def _build_question_trace(
        self,
        engine: Atagia,
        *,
        user_id: str,
        question: AtagiaBenchQuestion,
        pipeline_result: PipelineResult,
        retrieval_trace: RetrievalTrace,
        assistant_mode_id: str,
        conversation_id: str,
        turn_message_ids: dict[str, str],
        grade: GradeResult,
        trusted_evaluation: bool,
        trusted_activation_count: int,
    ) -> dict[str, Any]:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime unavailable")

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
                passed=grade.passed,
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

    async def _generate_answer(
        self,
        *,
        runtime: Any,
        assistant_mode_id: str,
        pipeline_result: PipelineResult,
        question_text: str,
        question_id: str,
        current_user_display_name: str | None = None,
        trusted_evaluation: bool = False,
    ) -> str:
        """Generate an answer from the retrieval context."""
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
            current_user_display_name=current_user_display_name,
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
                    "purpose": "atagia_bench_answer_generation",
                    "question_id": question_id,
                },
            )
        )
        return response.output_text

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
        """Find a conversation with the given assistant mode.

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
        """Get the assistant_mode_id for a conversation."""
        for conv in persona_data.conversations:
            if conv.conversation_id == conversation_id:
                return conv.assistant_mode_id
        return "general_qa"

    @staticmethod
    def _filter_questions(
        questions: list[AtagiaBenchQuestion],
        category_filter: set[str] | None,
        question_filter: set[str] | None,
    ) -> list[AtagiaBenchQuestion]:
        """Filter questions by category tags if a filter is provided."""
        if category_filter is None and question_filter is None:
            return questions
        return [
            q for q in questions
            if (
                (category_filter is None or category_filter.intersection(q.category_tags))
                and (question_filter is None or q.question_id in question_filter)
            )
        ]

    def _build_report(
        self,
        results: list[AtagiaQuestionResult],
        *,
        dataset: AtagiaBenchDataset,
        duration_seconds: float,
        persona_ids: list[str],
        category_tags: list[str] | None,
        question_ids: list[str] | None,
        trusted_evaluation: bool,
    ) -> AtagiaBenchReport:
        """Build the aggregated benchmark report."""
        total = len(results)
        passed = sum(1 for r in results if r.grade.passed)
        total_score = sum(r.grade.score for r in results)
        pass_rate = passed / total if total > 0 else 0.0
        avg_score = total_score / total if total > 0 else 0.0

        # Critical errors: privacy violations, stale superseded answers
        critical_count = sum(
            1 for r in results
            if not r.grade.passed and _is_critical(r)
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
                "answer_model": self._llm_model or "",
                "judge_model": self._judge_model or self._llm_model or "",
                "embedding_backend": self._embedding_backend,
                "category_filter": category_tags,
                "question_filter": question_ids,
                "trusted_evaluation": trusted_evaluation,
            },
            personas_used=persona_ids,
            total_questions=total,
            total_passed=passed,
            pass_rate=round(pass_rate, 4),
            avg_score=round(avg_score, 4),
            critical_error_count=critical_count,
            per_question=results,
            per_category=per_category,
        )

    @staticmethod
    def save_report(report: AtagiaBenchReport, output_dir: str | Path) -> Path:
        """Persist a benchmark report as JSON and return its path."""
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = output_path / f"atagia-bench-report-{timestamp}.json"
        report_path.write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return report_path


def _is_critical(result: AtagiaQuestionResult) -> bool:
    """Determine if a failed result is a critical error."""
    critical_tags = {
        "privacy_boundary",
        "mode_boundary",
        "hard_fail",
        "consent_gated",
        "high_risk",
    }
    return bool(critical_tags.intersection(result.category_tags))
