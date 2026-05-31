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
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import aiosqlite

from atagia import Atagia
from atagia.core.repositories import ConversationRepository, summary_mirror_id
from atagia.memory.retrieval_planner import build_retrieval_fts_queries
from atagia.models.schemas_replay import AblationConfig
from atagia.services.model_resolution import COMPONENTS_BY_ID, provider_qualified_model
from benchmarks.base import BenchmarkConversation
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_config import provider_api_key_kwargs
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
    parser.add_argument("--provider", default=None, help="LLM provider used to qualify bare model values")
    parser.add_argument("--api-key", default=None, help="Provider-specific LLM API key override")
    parser.add_argument(
        "--llm-model",
        default=os.getenv("ATAGIA_LLM_CHAT_MODEL"),
        help="Legacy alias for --answer-model; used as the real Atagia chat model.",
    )
    parser.add_argument(
        "--answer-model",
        help="Model override for the real Atagia chat path, e.g. openrouter/google/gemini-3.1-flash-lite,medium",
    )
    parser.add_argument("--forced-global-model", help="Force one model for every Atagia internal LLM component")
    parser.add_argument("--ingest-model", help="Model for Atagia ingest components")
    parser.add_argument("--retrieval-model", help="Model for Atagia retrieval components")
    parser.add_argument("--chat-model", help="Model for Atagia chat component fallback")
    parser.add_argument(
        "--privacy-enforcement",
        choices=("enforce", "audit_only", "off"),
        default="off",
        help="Request privacy_enforcement for the real Atagia chat call. Defaults to off.",
    )
    parser.add_argument(
        "--answer-postcondition-guard",
        action="store_true",
        help="Enable the real Atagia answer postcondition verifier.",
    )
    parser.add_argument(
        "--component-model",
        action="append",
        default=[],
        metavar="COMPONENT=MODEL",
        help="Override one Atagia LLM component model. Repeatable.",
    )
    parser.add_argument("--skip-need-detection", action="store_true", help="Disable LLM need detection")
    parser.add_argument("--skip-applicability-scoring", action="store_true", help="Disable LLM applicability scoring")
    parser.add_argument("--summary-hit-limit", type=int, default=10, help="How many summary surface hits to log")
    args = parser.parse_args()
    if not any([args.question_index, args.question_id, args.question_text]):
        parser.error("one of --question-index, --question-id, or --question-text is required")
    if args.llm_model and args.answer_model:
        parser.error("--llm-model is a legacy alias; use either --llm-model or --answer-model, not both")
    try:
        _parse_component_model_overrides(args.component_model)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _answer_model(args: argparse.Namespace) -> str | None:
    return args.answer_model or args.llm_model


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


def _qualified_model(provider: str | None, model: str | None) -> str | None:
    return provider_qualified_model(provider, model)


def _qualified_component_models(
    provider: str | None,
    component_models: dict[str, str],
) -> dict[str, str]:
    return {
        component_id: _qualified_model(provider, model) or model
        for component_id, model in component_models.items()
    }


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


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _scored_candidate_record(pipeline_candidate: Any) -> dict[str, Any]:
    if isinstance(pipeline_candidate, dict):
        memory_object = pipeline_candidate.get("memory_object") or {}
        return {
            "memory_id": pipeline_candidate.get("memory_id"),
            "llm_applicability": pipeline_candidate.get("llm_applicability"),
            "retrieval_score": pipeline_candidate.get("retrieval_score"),
            "vitality_boost": pipeline_candidate.get("vitality_boost"),
            "confirmation_boost": pipeline_candidate.get("confirmation_boost"),
            "need_boost": pipeline_candidate.get("need_boost"),
            "penalty": pipeline_candidate.get("penalty"),
            "final_score": pipeline_candidate.get("final_score"),
            "memory_object": {
                "id": memory_object.get("id"),
                "canonical_text": memory_object.get("canonical_text"),
                "object_type": memory_object.get("object_type"),
                "scope": memory_object.get("scope"),
                "payload_json": memory_object.get("payload_json"),
                **_surface_record(memory_object),
            },
        }
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
                    "character_id": payload.get("character_id"),
                    "scope_canonical": payload.get("scope_canonical"),
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


def _copy_sqlite_db(source_db_path: Path, destination_db_path: Path) -> None:
    destination_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db_path, destination_db_path)
    for suffix in ("-wal", "-shm"):
        sidecar = source_db_path.with_name(f"{source_db_path.name}{suffix}")
        if sidecar.exists():
            shutil.copy2(
                sidecar,
                destination_db_path.with_name(f"{destination_db_path.name}{suffix}"),
            )


async def _run(args: argparse.Namespace) -> Path:
    conversation = _load_conversation(args.data_path, args.conversation_id)
    question_id, question_text, ground_truth, evidence_turn_ids, category = _resolve_question(conversation, args)
    output_path = Path(args.output).expanduser() if args.output else _default_output_path(args.conversation_id, question_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ablation = AblationConfig(
        privacy_enforcement=args.privacy_enforcement,
        skip_need_detection=args.skip_need_detection,
        skip_applicability_scoring=args.skip_applicability_scoring,
    )
    source_db_path = Path(args.db_path).expanduser()
    with TemporaryDirectory(prefix="atagia-locomo-replay-probe-") as temp_dir:
        probe_db_path = Path(temp_dir) / "benchmark.db"
        _copy_sqlite_db(source_db_path, probe_db_path)
        async with Atagia(
            db_path=probe_db_path,
            manifests_dir=args.manifests_dir,
            llm_forced_global_model=_qualified_model(args.provider, args.forced_global_model),
            llm_ingest_model=_qualified_model(args.provider, args.ingest_model),
            llm_retrieval_model=_qualified_model(args.provider, args.retrieval_model),
            llm_chat_model=_qualified_model(args.provider, _answer_model(args) or args.chat_model),
            llm_component_models=_qualified_component_models(
                args.provider,
                _parse_component_model_overrides(args.component_model),
            ),
            **provider_api_key_kwargs(args.provider, args.api_key),
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            context_cache_enabled=False,
            answer_postcondition_guard_enabled=args.answer_postcondition_guard,
        ) as engine:
            runtime = engine.runtime
            if runtime is None:
                raise RuntimeError("Atagia runtime was unexpectedly unavailable")

            connection = await runtime.open_connection()
            try:
                conversations = ConversationRepository(connection, runtime.clock)
                conversation_row = await conversations.get_conversation(
                    args.conversation_id,
                    args.user_id,
                )
                if conversation_row is None:
                    raise ValueError(
                        f"Conversation {args.conversation_id} was not found for user {args.user_id}"
                    )
                retrieval_profile_id = str(
                    conversation_row["mode"]
                    or conversation_row["assistant_mode_id"]
                )
                platform_id = conversation_row["platform_id"]
                user_persona_id = conversation_row["user_persona_id"]
                character_id = conversation_row["character_id"]
                summary_hits = await _summary_surface_hits(
                    connection,
                    user_id=args.user_id,
                    conversation_id=args.conversation_id,
                    question_text=question_text,
                    limit=args.summary_hit_limit,
                )
            finally:
                await connection.close()

            chat_result = await engine.chat(
                user_id=args.user_id,
                conversation_id=args.conversation_id,
                message=question_text,
                mode=retrieval_profile_id,
                ablation=ablation,
                debug=True,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                privacy_enforcement=args.privacy_enforcement,
                authenticated_user_privilege_level=(
                    "atagia_master"
                    if args.privacy_enforcement == "off"
                    else None
                ),
                authenticated_user_is_atagia_master=(
                    args.privacy_enforcement == "off"
                ),
            )

    debug = chat_result.debug if isinstance(chat_result.debug, dict) else {}
    context = chat_result.composed_context
    scored_candidates = debug.get("scored_candidates")
    scored_candidates_list = scored_candidates if isinstance(scored_candidates, list) else []
    candidate_search_summary = debug.get("candidate_search_summary")
    raw_candidate_count = 0
    if isinstance(candidate_search_summary, dict):
        raw_candidate_count = int(candidate_search_summary.get("raw_candidate_count") or 0)

    artifact = {
        "conversation_id": args.conversation_id,
        "question_id": question_id,
        "question_text": question_text,
        "ground_truth": ground_truth,
        "category": category,
        "evidence_turn_ids": evidence_turn_ids,
        "db_path": str(source_db_path),
        "db_snapshot_isolation": "temporary_copy",
        "user_id": args.user_id,
        "platform_id": platform_id,
        "user_persona_id": user_persona_id,
        "character_id": character_id,
        "mode": retrieval_profile_id,
        "retrieval_profile_id": retrieval_profile_id,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "provider": args.provider,
        "llm_model": _answer_model(args),
        "answer_model": _qualified_model(args.provider, _answer_model(args)),
        "forced_global_model": _qualified_model(args.provider, args.forced_global_model),
        "ingest_model": _qualified_model(args.provider, args.ingest_model),
        "retrieval_model": _qualified_model(args.provider, args.retrieval_model),
        "chat_model": _qualified_model(args.provider, args.chat_model),
        "component_models": _qualified_component_models(
            args.provider,
            _parse_component_model_overrides(args.component_model),
        ),
        "ablation": ablation.model_dump(mode="json"),
        "privacy_enforcement": args.privacy_enforcement,
        "real_chat_path": True,
        "retrieval_event_id": chat_result.retrieval_event_id,
        "retrieval_plan": _jsonable(debug.get("retrieval_plan")),
        "retrieval_sufficiency": _jsonable(debug.get("retrieval_sufficiency")),
        "candidate_search_summary": _jsonable(candidate_search_summary),
        "stage_timings": {},
        "summary_surface_hits": summary_hits,
        "raw_candidate_count": raw_candidate_count,
        "raw_candidates": [],
        "scored_candidate_count": len(scored_candidates_list),
        "scored_candidates": [
            _scored_candidate_record(candidate)
            for candidate in scored_candidates_list
        ],
        "selected_memory_ids": (
            list(context.selected_memory_ids)
            if context is not None
            else list(debug.get("selected_memory_ids") or [])
        ),
        "memory_block": context.memory_block if context is not None else None,
        "answer": chat_result.response_text,
        "answer_error": None,
        "debug": _jsonable(debug),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return write_json_atomic(output_path, artifact)


def main() -> None:
    args = _parse_args()
    output_path = asyncio.run(_run(args))
    print(output_path)


if __name__ == "__main__":
    main()
