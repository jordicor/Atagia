"""Run one LoCoMo memory extraction on a copied benchmark DB.

This is a surgical diagnostic for failures such as Sonnet output-limit or
watchdog bounded-retry loops. It never mutates the source DB unless --write-db
is explicitly used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from benchmarks.json_artifacts import write_json_atomic
from benchmarks.output_root import bench_output_root
from benchmarks.llm_config import provider_api_key_kwargs
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder
from atagia import Atagia
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import PolicyResolver
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.chat_support import recent_context
from atagia.services.model_resolution import provider_qualified_model


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS_DIR = _PROJECT_ROOT / "manifests"
_DEFAULT_OUTPUT_DIR = bench_output_root() / "locomo" / "extractor_diagnostics"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, help="Source benchmark.db path")
    parser.add_argument("--message-id", required=True, help="Message id to extract")
    parser.add_argument("--user-id", default="benchmark-user")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Extractor model override, e.g. anthropic/claude-sonnet-4-6.",
    )
    parser.add_argument("--manifests-dir", default=str(_DEFAULT_MANIFESTS_DIR))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Run against --db-path directly. By default a diagnostic copy is used.",
    )
    parser.add_argument(
        "--llm-progress-every",
        type=int,
        default=1,
        help="Log live LLM summary every N calls.",
    )
    return parser


async def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    source_db_path = Path(args.db_path).expanduser()
    if not source_db_path.is_file():
        raise ValueError(f"DB does not exist: {source_db_path}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_db_path = (
        source_db_path
        if args.write_db
        else _copy_sqlite_db_for_diagnostic(source_db_path, output_dir, args.message_id)
    )

    component_models = {}
    if args.model:
        extractor_model = provider_qualified_model(args.provider, args.model) or args.model
        component_models["extractor"] = extractor_model
        component_models["extraction_watchdog"] = extractor_model

    recorder = LLMCallRecorder(progress_interval=args.llm_progress_every)
    started_at = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "source_db_path": str(source_db_path),
        "diagnostic_db_path": str(diagnostic_db_path),
        "write_db": bool(args.write_db),
        "message_id": args.message_id,
        "user_id": args.user_id,
        "provider": args.provider,
        "extractor_model": component_models.get("extractor"),
        "extraction_watchdog_model": component_models.get("extraction_watchdog"),
        "started_at": started_at.isoformat(),
        "success": False,
    }

    try:
        async with Atagia(
            db_path=diagnostic_db_path,
            manifests_dir=args.manifests_dir,
            llm_component_models=component_models,
            **provider_api_key_kwargs(args.provider, args.api_key),
        ) as engine:
            runtime = engine.runtime
            if runtime is None:
                raise RuntimeError("Atagia runtime was unexpectedly unavailable")
            install_llm_call_recorder(runtime.llm_client, recorder)
            connection = await runtime.open_connection()
            try:
                diagnostic = await _run_single_extraction(
                    connection=connection,
                    runtime=runtime,
                    message_id=args.message_id,
                    user_id=args.user_id,
                )
                payload.update(diagnostic)
                payload["success"] = True
            finally:
                await connection.close()
    except BaseException as exc:
        payload["exception"] = {
            "class": exc.__class__.__name__,
            "message": str(exc),
        }
    finally:
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["llm_call_summary"] = recorder.summary()
        payload["llm_calls"] = recorder.records()

    output_path = output_dir / (
        f"extractor-diagnostic-{_safe_path(args.message_id)}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json_atomic(output_path, payload)
    payload["output_path"] = str(output_path)
    return payload


async def _run_single_extraction(
    *,
    connection: Any,
    runtime: Any,
    message_id: str,
    user_id: str,
) -> dict[str, Any]:
    messages = MessageRepository(connection, runtime.clock)
    conversations = ConversationRepository(connection, runtime.clock)
    users = UserRepository(connection, runtime.clock)
    memories = MemoryObjectRepository(connection, runtime.clock)

    message = await messages.get_message(message_id, user_id)
    if message is None:
        raise ValueError(f"Unknown message_id for user {user_id}: {message_id}")
    conversation = await conversations.get_conversation(
        str(message["conversation_id"]),
        user_id,
    )
    if conversation is None:
        raise ValueError(f"Message conversation is missing: {message['conversation_id']}")
    memory_preferences = await users.get_memory_preferences(user_id)
    prior_messages = await _prior_messages(
        connection,
        conversation_id=str(conversation["id"]),
        user_id=user_id,
        before_seq=int(message["seq"] or 0),
        limit=6,
    )

    assistant_mode_id = str(conversation["assistant_mode_id"])
    manifest = runtime.manifest_loader.load_all()[assistant_mode_id]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    context = ExtractionConversationContext(
        user_id=user_id,
        conversation_id=str(conversation["id"]),
        source_message_id=message_id,
        workspace_id=(
            str(conversation["workspace_id"])
            if conversation.get("workspace_id") is not None
            else None
        ),
        assistant_mode_id=assistant_mode_id,
        user_persona_id=conversation.get("user_persona_id"),
        platform_id=str(conversation.get("platform_id") or "default"),
        character_id=(
            str(conversation["character_id"])
            if conversation.get("character_id") is not None
            else (
                str(conversation["workspace_id"])
                if conversation.get("workspace_id") is not None
                else None
            )
        ),
        mode=str(conversation.get("mode") or assistant_mode_id),
        recent_messages=recent_context(prior_messages),
        remember_across_chats=bool(memory_preferences["remember_across_chats"]),
        remember_across_devices=bool(memory_preferences["remember_across_devices"]),
        memory_privacy_mode=memory_preferences["memory_privacy_mode"],
        temporary=bool(conversation.get("temporary")),
        purge_on_close=bool(conversation.get("purge_on_close")),
        isolated_mode=bool(conversation.get("isolated_mode")),
        incognito=bool(conversation.get("incognito")) or bool(conversation.get("isolated_mode")),
    )

    extractor = MemoryExtractor(
        llm_client=runtime.llm_client,
        clock=runtime.clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=runtime.storage_backend,
        embedding_index=runtime.embedding_index,
        settings=runtime.settings,
    )
    before_ids = {str(row["id"]) for row in await memories.list_for_user(user_id)}
    details = await extractor.extract_with_persistence_and_chunk_plan(
        message_text=str(message["text"]),
        role=str(message["role"]),
        conversation_context=context,
        resolved_policy=resolved_policy,
        occurred_at=message.get("occurred_at"),
    )
    after_rows = await memories.list_for_user(user_id)
    new_rows = [row for row in after_rows if str(row["id"]) not in before_ids]
    return {
        "conversation_id": str(conversation["id"]),
        "message_role": str(message["role"]),
        "message_seq": message["seq"],
        "message_text_preview": " ".join(str(message["text"]).split())[:300],
        "recent_message_count": len(prior_messages),
        "result": _json_safe(details.result),
        "chunk_plan": _json_safe(details.chunk_plan),
        "persisted_memory_ids": [str(row["id"]) for row in details.persisted],
        "new_memory_ids": [str(row["id"]) for row in new_rows],
    }


async def _prior_messages(
    connection: Any,
    *,
    conversation_id: str,
    user_id: str,
    before_seq: int,
    limit: int,
) -> list[dict[str, Any]]:
    cursor = await connection.execute(
        """
        SELECT m.*
        FROM messages AS m
        JOIN conversations AS c ON c.id = m.conversation_id
        WHERE m.conversation_id = ?
          AND c.user_id = ?
          AND m.seq < ?
        ORDER BY m.seq DESC
        LIMIT ?
        """,
        (conversation_id, user_id, before_seq, limit),
    )
    rows = [dict(row) for row in await cursor.fetchall()]
    return list(reversed(rows))


def _copy_sqlite_db_for_diagnostic(
    source_db_path: Path,
    output_dir: Path,
    message_id: str,
) -> Path:
    destination = output_dir / (
        f"diagnostic-{_safe_path(message_id)}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.db"
    )
    shutil.copy2(source_db_path, destination)
    for suffix in ("-wal", "-shm"):
        sidecar = source_db_path.with_name(f"{source_db_path.name}{suffix}")
        if sidecar.exists():
            shutil.copy2(sidecar, destination.with_name(f"{destination.name}{suffix}"))
    return destination


def _safe_path(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = asyncio.run(run_diagnostic(args))
    except Exception as exc:
        print(f"extractor diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
