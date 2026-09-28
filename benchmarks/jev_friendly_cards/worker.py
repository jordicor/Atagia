"""Run a selected production checkout through real SQLite and retrieval paths."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any

from atagia.app import initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.topic_repository import TopicRepository
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.topic_working_set import TopicWorkingSetUpdater
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
    RetrievalTrace,
)
from atagia.services.llm_client import LLMClient
from atagia.services.retrieval_service import RetrievalService

from benchmarks.jev_friendly_cards.routes import settings_for_arm


_ATAGIA_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _atagia_role(role: str, *, origin: str) -> str:
    if origin not in {"synthetic", "aurvek_local_snapshot"}:
        raise ValueError(f"Unknown benchmark source origin: {origin!r}")
    if origin == "aurvek_local_snapshot" and role == "bot":
        return "assistant"
    if isinstance(role, str) and role in _ATAGIA_ROLES:
        return role
    raise ValueError(f"Unsupported {origin} message role: {role!r}")


def _context_messages(
    case: dict[str, Any],
) -> tuple[list[ExtractionContextMessage], list[dict[str, Any]], list[dict[str, Any]]]:
    origin = case["origin"]
    real = origin == "aurvek_local_snapshot"
    result: list[ExtractionContextMessage] = []
    omitted: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, item in enumerate(case["context_messages"], start=1):
        role = _atagia_role(item["role"], origin=origin)
        included = not real or item.get("text_available") is True
        provenance.append({
            "original_index": index,
            "original_role": item["role"],
            "atagia_role": role,
            "included": included,
        })
        if not included:
            omitted.append({"original_index": index, "role": item["role"]})
            continue
        result.append(
            ExtractionContextMessage(
                id=f"recent_{len(result) + 1}",
                role=role,
                content=item["text"] if real else item["content"],
                seq=len(result) + 1,
                occurred_at=item.get("occurred_at"),
            )
        )
    return result, omitted, provenance


async def run_full_flow(
    client: LLMClient[Any],
    *,
    case: dict[str, Any],
    fixture: dict[str, Any],
    followup_query: str,
    arm: str,
    database_path: Path,
) -> dict[str, Any]:
    """Consume one fresh client while creating this slot's state and retrieval result."""
    if not database_path.is_absolute() or database_path.exists() or not followup_query.strip():
        raise ValueError("Full flow requires a fresh SQLite path and fixed follow-up query")
    if fixture["source_message_id"] != case["case_id"]:
        raise ValueError("Case and source message identity differ")
    when = datetime.fromisoformat(fixture["occurred_at"])
    if when.tzinfo is None:
        raise ValueError("Fixed case clock needs a timezone")
    settings = replace(
        settings_for_arm(arm),
        sqlite_path=str(database_path),
        openrouter_api_key="runtime-bootstrap-no-dispatch",
        anthropic_api_key="runtime-bootstrap-no-dispatch",
        typesafe_api_key="runtime-bootstrap-no-dispatch",
    )
    started = perf_counter()
    runtime = await initialize_runtime(settings)
    try:
        bootstrap_client = runtime.llm_client
        await bootstrap_client.aclose()
        runtime.llm_client = client
        runtime.clock = FrozenClock(when)
        if (
            settings.embedding_backend != "none"
            or runtime.embedding_connection is not None
            or runtime.worker_tasks
            or any(
                getattr(runtime, name) is not None
                for name in (
                    "ingest_worker", "contract_worker", "graph_worker",
                    "revision_worker", "compaction_worker", "evaluation_worker",
                    "initial_context_package_worker", "transcript_rebuild_worker",
                    "lifecycle_worker", "durable_job_dispatcher",
                )
            )
        ):
            raise ValueError("The isolated benchmark started another model caller")
        setup_ms = (perf_counter() - started) * 1000
        connection = await runtime.open_connection()
        try:
            return await _run_with_connection(
                runtime, connection, client=client, case=case, fixture=fixture,
                followup_query=followup_query, setup_ms=setup_ms,
            )
        finally:
            await connection.close()
    finally:
        await runtime.close()


async def _run_with_connection(
    runtime: Any,
    connection: Any,
    *,
    client: LLMClient[Any],
    case: dict[str, Any],
    fixture: dict[str, Any],
    followup_query: str,
    setup_ms: float,
) -> dict[str, Any]:
    settings = runtime.settings
    user_id = fixture["user_id"]
    conversation_id = fixture["conversation_id"]
    source_id = fixture["source_message_id"]
    recent, omitted_context, context_role_provenance = _context_messages(case)
    source_role = _atagia_role(case["role"], origin=case["origin"])
    users = UserRepository(connection, runtime.clock)
    conversations = ConversationRepository(connection, runtime.clock)
    messages = MessageRepository(connection, runtime.clock)
    memories = MemoryObjectRepository(connection, runtime.clock)
    topics = TopicRepository(connection, runtime.clock)
    await users.create_user(user_id)
    await conversations.create_conversation(
        conversation_id, user_id, None, fixture["assistant_mode_id"], "Evaluation"
    )
    for item in recent:
        await messages.create_message(
            item.id, conversation_id, item.role, item.seq, item.content,
            occurred_at=item.occurred_at,
        )
    source_seq = len(recent) + 1
    await messages.create_message(
        source_id, conversation_id, source_role, source_seq, case["source_text"],
        occurred_at=fixture["occurred_at"],
    )
    prior_topics = (fixture.get("topic") or {}).get("snapshot") or {}
    for topic in (*prior_topics.get("active_topics", []), *prior_topics.get("parked_topics", [])):
        await topics.create_topic(
            user_id=user_id,
            conversation_id=conversation_id,
            topic_id=topic["id"],
            status="active" if topic in prior_topics.get("active_topics", []) else "parked",
            title=topic["title"],
            summary=topic.get("summary", ""),
            active_goal=topic.get("active_goal"),
            open_questions=topic.get("open_questions", []),
            decisions=topic.get("decisions", []),
        )
    policy = runtime.policy_resolver.resolve(
        runtime.manifests[fixture["assistant_mode_id"]], None, None
    )
    context = ExtractionConversationContext(
        user_id=user_id,
        conversation_id=conversation_id,
        source_message_id=source_id,
        assistant_mode_id=fixture["assistant_mode_id"],
        mode=fixture["mode"],
        recent_messages=recent,
        privacy_enforcement=fixture["privacy_enforcement"],
    )
    extractor = MemoryExtractor(
        llm_client=client,
        clock=runtime.clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=runtime.storage_backend,
        settings=settings,
    )
    started_extraction = perf_counter()
    details = await extractor.extract_with_persistence_and_chunk_plan(
        message_text=case["source_text"],
        role=source_role,
        conversation_context=context,
        resolved_policy=policy,
        occurred_at=fixture["occurred_at"],
    )
    extraction_ms = (perf_counter() - started_extraction) * 1000
    source_checks = []
    for item in (
        *details.result.evidences, *details.result.beliefs,
        *details.result.contract_signals, *details.result.state_updates,
    ):
        reference = item.source_reference
        if reference is None or item.source_quote != reference.quote(case["source_text"]):
            raise ValueError("Extracted memory lacks an exact source reference")
        source_checks.append(reference.model_dump(mode="json"))

    topic_result = None
    if case["primary_family"] == "topics":
        updater = TopicWorkingSetUpdater(
            llm_client=client,
            clock=runtime.clock,
            topic_repository=topics,
            message_repository=messages,
            settings=settings,
        )
        topic_result = await updater.update_from_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[{
                "id": source_id, "seq": source_seq, "role": source_role,
                "text": case["source_text"], "created_at": fixture["occurred_at"],
            }],
        )
    await messages.create_message(
        "followup_query", conversation_id, "user", source_seq + 1,
        followup_query, occurred_at=fixture["occurred_at"],
    )
    followup = await messages.get_message("followup_query", user_id)
    if followup is None or followup["text"] != followup_query:
        raise ValueError("The fixed follow-up message was not persisted")
    trace = RetrievalTrace(
        query_text=followup_query,
        user_id=user_id,
        conversation_id=conversation_id,
        timestamp_iso=fixture["occurred_at"],
        privacy_enforcement="off",
    )
    started_retrieval = perf_counter()
    retrieval = await RetrievalService(runtime).retrieve_with_connection(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        message_text=followup_query,
        mode=fixture["mode"],
        privacy_enforcement="off",
        stored_messages=[followup],
        trace=trace,
    )
    retrieval_ms = (perf_counter() - started_retrieval) * 1000
    connection.row_factory = sqlite3.Row
    rows = {}
    for table in (
        "memory_objects", "memory_support_edges", "memory_evidence_spans",
        "conversation_topics",
    ):
        cursor = await connection.execute(f"SELECT * FROM {table} ORDER BY id")
        rows[table] = [dict(row) for row in await cursor.fetchall()]
    for span in rows["memory_evidence_spans"]:
        if span["message_id"] == source_id:
            start, end = span["char_start"], span["char_end"]
            if (
                start is None or end is None
                or case["source_text"][start:end] != span["quote_text"]
            ):
                raise ValueError("Persisted source span differs from the input")
    return {
        "setup_ms": setup_ms,
        "extraction_ms": extraction_ms,
        "retrieval_ms": retrieval_ms,
        "extraction": details.result.model_dump(mode="json"),
        "persisted": details.persisted,
        "source_references": source_checks,
        "chunk_plan": asdict(details.chunk_plan),
        "topic_updates": topic_result,
        "database_rows": rows,
        "omitted_context": omitted_context,
        "role_provenance": {
            "source": {"original_role": case["role"], "atagia_role": source_role},
            "context": context_role_provenance,
        },
        "context_timestamp_basis": (
            "Unknown original context event time; SQLite insertion uses the fixed "
            "source fixture clock and is not source-date evidence."
        ),
        "retrieval_recent_message_ids": [followup["id"]],
        "retrieval": retrieval.model_dump(mode="json"),
    }
