"""Single-question LoCoMo replay probe against a frozen Atagia database."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

import aiosqlite

from atagia import Atagia
from atagia.core.repositories import ConversationRepository, summary_mirror_id
from atagia.memory.retrieval_planner import build_retrieval_fts_queries
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import build_system_prompt, chat_model, resolve_policy
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.retrieval_service import RetrievalService
from benchmarks.base import BenchmarkConversation, BenchmarkQuestion
from benchmarks.locomo.adapter import LoCoMoAdapter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "benchmarks" / "data" / "locomo10.json"
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, help="Path to the frozen SQLite database")
    parser.add_argument("--conversation-id", default="conv-26", help="LoCoMo conversation id")
    parser.add_argument("--user-id", default="benchmark-user", help="User id stored in the frozen DB")
    parser.add_argument("--data-path", default=str(_DEFAULT_DATA_PATH), help="Path to locomo10.json")
    parser.add_argument("--manifests-dir", default=str(_DEFAULT_MANIFESTS_DIR), help="Path to manifests directory")
    parser.add_argument("--question-index", type=int, help="1-based LoCoMo question index (for example: 7)")
    parser.add_argument("--question-id", help="Question id (for example: q7 or conv-26:q7)")
    parser.add_argument("--question-text", help="Custom question text when not using the dataset entry")
    parser.add_argument("--output", help="Artifact path. Defaults to docs/tmp/ with a timestamped filename.")
    parser.add_argument(
        "--embedding-backend",
        default=os.getenv("ATAGIA_EMBEDDING_BACKEND", "none"),
        help="Embedding backend to use during retrieval",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("ATAGIA_EMBEDDING_MODEL"),
        help="Embedding model name",
    )
    parser.add_argument("--llm-provider", default=os.getenv("ATAGIA_LLM_PROVIDER"), help="LLM provider override")
    parser.add_argument("--llm-api-key", default=os.getenv("ATAGIA_LLM_API_KEY"), help="LLM API key override")
    parser.add_argument("--llm-model", default=os.getenv("ATAGIA_LLM_CHAT_MODEL"), help="LLM model override")
    parser.add_argument("--skip-need-detection", action="store_true", help="Disable LLM need detection")
    parser.add_argument("--skip-applicability-scoring", action="store_true", help="Disable LLM applicability scoring")
    parser.add_argument("--skip-answer", action="store_true", help="Skip final answer generation")
    parser.add_argument("--summary-hit-limit", type=int, default=10, help="How many summary surface hits to log")
    args = parser.parse_args()
    if not any([args.question_index, args.question_id, args.question_text]):
        parser.error("one of --question-index, --question-id, or --question-text is required")
    return args


def _load_conversation(data_path: str | Path, conversation_id: str) -> BenchmarkConversation:
    dataset = LoCoMoAdapter(data_path).load()
    for conversation in dataset.conversations:
        if conversation.conversation_id == conversation_id:
            return conversation
    raise ValueError(f"Unknown conversation id: {conversation_id}")


def _parse_question_number(question_ref: str) -> int:
    normalized = question_ref.strip()
    if ":" in normalized:
        normalized = normalized.split(":")[-1]
    if normalized.startswith("q"):
        normalized = normalized[1:]
    question_number = int(normalized)
    if question_number <= 0:
        raise ValueError("Question number must be positive")
    return question_number


def _resolve_question(conversation: BenchmarkConversation, args: argparse.Namespace) -> tuple[str, str, str | None, list[str], int | None]:
    if args.question_text:
        return "custom", args.question_text, None, [], None
    if args.question_index is not None:
        question_number = args.question_index
    elif args.question_id is not None:
        question_number = _parse_question_number(args.question_id)
    else:
        raise ValueError("A dataset-backed question requires --question-index or --question-id")
    if question_number > len(conversation.questions):
        raise ValueError(
            f"Conversation {conversation.conversation_id} has only {len(conversation.questions)} questions"
        )
    question = conversation.questions[question_number - 1]
    return (
        question.question_id,
        question.question_text,
        question.ground_truth,
        list(question.evidence_turn_ids),
        question.category,
    )


def _default_output_path(conversation_id: str, question_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_question_id = question_id.replace(":", "_")
    return _PROJECT_ROOT / "docs" / "tmp" / f"locomo_replay_probe_{conversation_id}_{safe_question_id}_{timestamp}.json"


def _query_tokens(question_text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for query in build_retrieval_fts_queries(question_text):
        for token in _QUERY_TOKEN_PATTERN.findall(query.lower()):
            if token == "or" or len(token) <= 1 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def _surface_record(memory_object: dict[str, Any]) -> dict[str, Any]:
    payload_json = memory_object.get("payload_json") or {}
    summary_kind = None
    hierarchy_level = None
    if isinstance(payload_json, dict):
        summary_kind = payload_json.get("summary_kind")
        hierarchy_level = payload_json.get("hierarchy_level")
    if memory_object.get("object_type") == "summary_view":
        surface = "memory_objects.summary_view"
    else:
        surface = "memory_objects.atomic"
    return {
        "surface": surface,
        "summary_kind": summary_kind,
        "hierarchy_level": hierarchy_level,
    }


def _raw_candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(candidate["id"]),
        "canonical_text": str(candidate.get("canonical_text", "")),
        "object_type": str(candidate.get("object_type", "")),
        "scope": str(candidate.get("scope", "")),
        "status": str(candidate.get("status", "")),
        "privacy_level": int(candidate.get("privacy_level", 0)),
        "temporal_type": str(candidate.get("temporal_type", "")),
        "valid_from": candidate.get("valid_from"),
        "valid_to": candidate.get("valid_to"),
        "updated_at": candidate.get("updated_at"),
        "position_rank": candidate.get("position_rank"),
        "retrieval_sources": list(candidate.get("retrieval_sources", [])),
        "rrf_score": candidate.get("rrf_score"),
        "rrf_score_raw": candidate.get("rrf_score_raw"),
        "payload_json": candidate.get("payload_json"),
        **_surface_record(candidate),
    }


def _scored_candidate_record(pipeline_candidate: Any) -> dict[str, Any]:
    memory_object = pipeline_candidate.memory_object
    return {
        "memory_id": pipeline_candidate.memory_id,
        "llm_applicability": pipeline_candidate.llm_applicability,
        "retrieval_score": pipeline_candidate.retrieval_score,
        "vitality_boost": pipeline_candidate.vitality_boost,
        "confirmation_boost": pipeline_candidate.confirmation_boost,
        "need_boost": pipeline_candidate.need_boost,
        "penalty": pipeline_candidate.penalty,
        "final_score": pipeline_candidate.final_score,
        "memory_object": {
            "id": memory_object.get("id"),
            "canonical_text": memory_object.get("canonical_text"),
            "object_type": memory_object.get("object_type"),
            "scope": memory_object.get("scope"),
            "payload_json": memory_object.get("payload_json"),
            **_surface_record(memory_object),
        },
    }


async def _summary_surface_hits(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str,
    question_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    cursor = await connection.execute(
        """
        SELECT *
        FROM summary_views
        WHERE user_id = ?
          AND (conversation_id = ? OR conversation_id IS NULL)
        ORDER BY created_at DESC, id DESC
        """,
        (user_id, conversation_id),
    )
    rows = await cursor.fetchall()
    tokens = _query_tokens(question_text)
    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        payload = dict(row)
        summary_text = str(payload.get("summary_text", "")).lower()
        overlap_tokens = [token for token in tokens if token in summary_text]
        if not overlap_tokens:
            continue
        scored_rows.append(
            (
                len(overlap_tokens),
                {
                    "id": str(payload["id"]),
                    "summary_kind": str(payload.get("summary_kind", "")),
                    "hierarchy_level": int(payload.get("hierarchy_level", 0)),
                    "conversation_id": payload.get("conversation_id"),
                    "workspace_id": payload.get("workspace_id"),
                    "created_at": str(payload.get("created_at", "")),
                    "summary_text": str(payload.get("summary_text", "")),
                    "source_object_ids": json.loads(payload.get("source_object_ids_json", "[]")),
                    "overlap_tokens": overlap_tokens,
                    "mirror_id": summary_mirror_id(str(payload["id"])),
                },
            )
        )
    scored_rows.sort(
        key=lambda item: (
            -item[0],
            -datetime.fromisoformat(item[1]["created_at"]).timestamp(),
            item[1]["id"],
        )
    )
    results: list[dict[str, Any]] = []
    for _score, payload in scored_rows[:limit]:
        mirror_cursor = await connection.execute(
            """
            SELECT id
            FROM memory_objects
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, payload["mirror_id"]),
        )
        mirror_row = await mirror_cursor.fetchone()
        payload["mirror_exists"] = mirror_row is not None
        results.append(payload)
    return results


async def _generate_answer(
    *,
    runtime: Any,
    assistant_mode_id: str,
    pipeline_result: PipelineResult,
    question_text: str,
    model_override: str | None,
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
            model=model_override or chat_model(runtime.settings),
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


async def _run(args: argparse.Namespace) -> Path:
    conversation = _load_conversation(args.data_path, args.conversation_id)
    question_id, question_text, ground_truth, evidence_turn_ids, category = _resolve_question(conversation, args)
    output_path = Path(args.output).expanduser() if args.output else _default_output_path(args.conversation_id, question_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ablation = AblationConfig(
        skip_need_detection=args.skip_need_detection,
        skip_applicability_scoring=args.skip_applicability_scoring,
    )

    async with Atagia(
        db_path=args.db_path,
        manifests_dir=args.manifests_dir,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        context_cache_enabled=False,
    ) as engine:
        runtime = engine.runtime
        if runtime is None:
            raise RuntimeError("Atagia runtime was unexpectedly unavailable")

        connection = await runtime.open_connection()
        try:
            conversations = ConversationRepository(connection, runtime.clock)
            conversation_row = await conversations.get_conversation(args.conversation_id, args.user_id)
            if conversation_row is None:
                raise ValueError(
                    f"Conversation {args.conversation_id} was not found for user {args.user_id}"
                )
            assistant_mode_id = str(conversation_row["assistant_mode_id"])
            summary_hits = await _summary_surface_hits(
                connection,
                user_id=args.user_id,
                conversation_id=args.conversation_id,
                question_text=question_text,
                limit=args.summary_hit_limit,
            )
        finally:
            await connection.close()

        pipeline_result = await RetrievalService(runtime).retrieve(
            user_id=args.user_id,
            conversation_id=args.conversation_id,
            message_text=question_text,
            mode=assistant_mode_id,
            ablation=ablation,
        )

        answer: str | None = None
        answer_error: str | None = None
        if not args.skip_answer:
            try:
                answer = await _generate_answer(
                    runtime=runtime,
                    assistant_mode_id=assistant_mode_id,
                    pipeline_result=pipeline_result,
                    question_text=question_text,
                    model_override=args.llm_model,
                )
            except Exception as exc:  # pragma: no cover - best effort artifact enrichment
                logger.exception("Replay probe answer generation failed")
                answer_error = f"{type(exc).__name__}: {exc}"

    artifact = {
        "conversation_id": args.conversation_id,
        "question_id": question_id,
        "question_text": question_text,
        "ground_truth": ground_truth,
        "category": category,
        "evidence_turn_ids": evidence_turn_ids,
        "db_path": str(Path(args.db_path).expanduser()),
        "user_id": args.user_id,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "ablation": ablation.model_dump(mode="json"),
        "retrieval_plan": pipeline_result.retrieval_plan.model_dump(mode="json"),
        "stage_timings": dict(pipeline_result.stage_timings),
        "summary_surface_hits": summary_hits,
        "raw_candidate_count": len(pipeline_result.raw_candidates),
        "raw_candidates": [_raw_candidate_record(candidate) for candidate in pipeline_result.raw_candidates],
        "scored_candidate_count": len(pipeline_result.scored_candidates),
        "scored_candidates": [
            _scored_candidate_record(candidate)
            for candidate in pipeline_result.scored_candidates
        ],
        "selected_memory_ids": list(pipeline_result.composed_context.selected_memory_ids),
        "memory_block": pipeline_result.composed_context.memory_block,
        "answer": answer,
        "answer_error": answer_error,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = _parse_args()
    output_path = asyncio.run(_run(args))
    print(output_path)


if __name__ == "__main__":
    main()
