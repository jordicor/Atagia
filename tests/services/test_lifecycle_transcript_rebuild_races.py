"""Cross-connection serialization between lifecycle writes and transcript rebuilds."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    ConversationStatus,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
)
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.services.lifecycle_service import (
    DELETE_CONVERSATION_CONFIRMATION,
    ERASE_ALL_DATA_CONFIRMATION,
    HARD_DELETE_MEMORY_CONFIRMATION,
    ConversationLifecycleService,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)
CLOCK = FrozenClock(datetime(2026, 7, 13, 16, 0, tzinfo=timezone.utc))
USER_ID = "usr_lifecycle_race"
CONVERSATION_ID = "cnv_lifecycle_race"
MESSAGE_ID = "msg_lifecycle_race"
MEMORY_ID = "mem_lifecycle_race"


class _NoopEmbeddingIndex:
    async def delete(self, memory_id: str) -> None:
        del memory_id


class _Runtime:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self.clock = CLOCK
        self.storage_backend = InProcessBackend()
        self.embedding_index = _NoopEmbeddingIndex()
        self.llm_client = None
        self.settings = SimpleNamespace(
            erasure_purge_streams=False,
            storage_backend="inprocess",
        )

    async def open_connection(self) -> aiosqlite.Connection:
        return await open_connection(self.database_path)


class _ObserveImmediate:
    """Expose the exact point where the lifecycle writer lock is requested."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.immediate_attempted = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
            self.immediate_attempted.set()
        return await self._connection.execute(sql, *args, **kwargs)


async def _seed(database_path: str, *, pending_deletion: bool) -> aiosqlite.Connection:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    await sync_assistant_modes(
        connection,
        ManifestLoader(MANIFESTS_DIR).load_all(),
        CLOCK,
    )
    await UserRepository(connection, CLOCK).create_user(USER_ID)
    await ConversationRepository(connection, CLOCK).create_conversation(
        CONVERSATION_ID,
        USER_ID,
        None,
        "coding_debug",
        "Lifecycle race",
    )
    message = await MessageRepository(connection, CLOCK).create_message(
        MESSAGE_ID,
        CONVERSATION_ID,
        "user",
        1,
        "Canonical source text.",
    )
    await MemoryObjectRepository(connection, CLOCK).create_memory_object(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Canonical memory text",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id=MEMORY_ID,
        payload={"source_message_ids": [str(message["id"])]},
    )
    if pending_deletion:
        await connection.execute(
            "UPDATE conversations SET status = ? WHERE id = ?",
            (ConversationStatus.PENDING_DELETION.value, CONVERSATION_ID),
        )
        await connection.commit()
    return connection


async def _install_uncommitted_blocking_selection(
    connection: aiosqlite.Connection,
    *,
    state: str,
) -> None:
    timestamp = CLOCK.now().isoformat()
    await connection.execute("BEGIN IMMEDIATE")
    await connection.execute(
        """
        INSERT INTO transcript_rebuild_workflows(
            id, operation_id, user_id, conversation_id, selection_epoch,
            transcript_hash, mutation_kind, selected_message_ids_json,
            abandoned_message_ids_json, supporting_message_ids_json,
            affected_memory_ids_json, affected_summary_ids_json,
            orchestrator_job_id, stage, start_derivation_revision,
            created_at, updated_at
        ) VALUES (
            'trb_lifecycle_race', 'op_lifecycle_race', ?, ?, 1,
            'transcript-hash', 'replace', '[]', '[]', '[]', '[]', '[]',
            'job_lifecycle_race', 'sources', 0, ?, ?
        )
        """,
        (USER_ID, CONVERSATION_ID, timestamp, timestamp),
    )
    await connection.execute(
        """
        INSERT INTO conversation_transcript_selections(
            user_id, conversation_id, selection_epoch, transcript_hash,
            current_workflow_id, state, updated_at
        ) VALUES (?, ?, 1, 'transcript-hash', 'trb_lifecycle_race', ?, ?)
        """,
        (USER_ID, CONVERSATION_ID, state, timestamp),
    )


async def _canonical_snapshot(connection: aiosqlite.Connection) -> tuple[Any, ...]:
    user = await (
        await connection.execute(
            "SELECT COUNT(*) AS total FROM users WHERE id = ?", (USER_ID,)
        )
    ).fetchone()
    conversation = await (
        await connection.execute(
            "SELECT status, closed_at FROM conversations WHERE id = ? AND user_id = ?",
            (CONVERSATION_ID, USER_ID),
        )
    ).fetchone()
    memory = await (
        await connection.execute(
            "SELECT canonical_text, status FROM memory_objects WHERE id = ? AND user_id = ?",
            (MEMORY_ID, USER_ID),
        )
    ).fetchone()
    messages = await (
        await connection.execute(
            "SELECT COUNT(*) AS total FROM messages WHERE conversation_id = ?",
            (CONVERSATION_ID,),
        )
    ).fetchone()
    tombstones = await (
        await connection.execute(
            "SELECT COUNT(*) AS total FROM deletion_tombstones",
        )
    ).fetchone()
    erasures = await (
        await connection.execute(
            "SELECT COUNT(*) AS total FROM user_erasure_cleanups",
        )
    ).fetchone()
    return (
        int(user["total"]),
        None if conversation is None else tuple(conversation),
        None if memory is None else tuple(memory),
        int(messages["total"]),
        int(tombstones["total"]),
        int(erasures["total"]),
    )


async def _run_mutation(
    mutation: str,
    service: ConversationLifecycleService,
    connection: aiosqlite.Connection,
) -> None:
    if mutation == "close_conversation":
        await service.close_conversation(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )
    elif mutation == "archive_conversation":
        await service.archive_conversation(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )
    elif mutation == "edit_memory":
        await service.edit_memory(
            connection,
            user_id=USER_ID,
            memory_id=MEMORY_ID,
            new_text="Mutated memory text",
        )
    elif mutation == "archive_memory":
        await service.delete_memory(
            connection,
            user_id=USER_ID,
            memory_id=MEMORY_ID,
        )
    elif mutation == "hard_delete_memory":
        await service.delete_memory(
            connection,
            user_id=USER_ID,
            memory_id=MEMORY_ID,
            hard=True,
            confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
        )
    elif mutation == "delete_conversation":
        await service.delete_conversation(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            confirmation=DELETE_CONVERSATION_CONFIRMATION,
        )
    elif mutation == "purge_conversation":
        assert (
            await service.purge_pending_deleted_conversations(connection, limit=1) == 1
        )
    elif mutation == "erase_user":
        await service.erase_user_data(
            connection,
            user_id=USER_ID,
            confirmation=ERASE_ALL_DATA_CONFIRMATION,
        )
    else:  # pragma: no cover - the parameter list is the exhaustive dispatch
        raise AssertionError(f"Unknown mutation: {mutation}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "close_conversation",
        "archive_conversation",
        "edit_memory",
        "archive_memory",
        "hard_delete_memory",
        "delete_conversation",
        "purge_conversation",
        "erase_user",
    ],
)
async def test_uncommitted_replacement_wins_writer_race_and_blocks_lifecycle_mutation(
    mutation: str,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / f"{mutation}.db")
    raw_lifecycle_connection = await _seed(
        database_path,
        pending_deletion=mutation == "purge_conversation",
    )
    lifecycle_connection = _ObserveImmediate(raw_lifecycle_connection)
    replacement_connection = await open_connection(database_path)
    runtime = _Runtime(database_path)
    mutation_task: asyncio.Task[None] | None = None
    try:
        before = await _canonical_snapshot(raw_lifecycle_connection)
        await _install_uncommitted_blocking_selection(
            replacement_connection,
            state="rebuilding",
        )
        mutation_task = asyncio.create_task(
            _run_mutation(
                mutation,
                ConversationLifecycleService(runtime),
                lifecycle_connection,
            )
        )
        await asyncio.wait_for(
            lifecycle_connection.immediate_attempted.wait(),
            timeout=5.0,
        )
        assert not mutation_task.done()

        await replacement_connection.commit()
        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(mutation_task, timeout=5.0)
        mutation_task = None

        assert await _canonical_snapshot(raw_lifecycle_connection) == before
    finally:
        if mutation_task is not None:
            mutation_task.cancel()
            await asyncio.gather(mutation_task, return_exceptions=True)
        if replacement_connection.in_transaction:
            await replacement_connection.rollback()
        await replacement_connection.close()
        await runtime.storage_backend.close()
        await raw_lifecycle_connection.close()


@pytest.mark.asyncio
async def test_user_erasure_is_blocked_during_transcript_remediation(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "erase-remediation.db")
    raw_lifecycle_connection = await _seed(database_path, pending_deletion=False)
    lifecycle_connection = _ObserveImmediate(raw_lifecycle_connection)
    replacement_connection = await open_connection(database_path)
    runtime = _Runtime(database_path)
    mutation_task: asyncio.Task[None] | None = None
    try:
        before = await _canonical_snapshot(raw_lifecycle_connection)
        await _install_uncommitted_blocking_selection(
            replacement_connection,
            state="remediation_required",
        )
        mutation_task = asyncio.create_task(
            _run_mutation(
                "erase_user",
                ConversationLifecycleService(runtime),
                lifecycle_connection,
            )
        )
        await asyncio.wait_for(
            lifecycle_connection.immediate_attempted.wait(),
            timeout=5.0,
        )

        await replacement_connection.commit()
        with pytest.raises(TranscriptRebuildRemediationRequiredError):
            await asyncio.wait_for(mutation_task, timeout=5.0)
        mutation_task = None

        assert await _canonical_snapshot(raw_lifecycle_connection) == before
    finally:
        if mutation_task is not None:
            mutation_task.cancel()
            await asyncio.gather(mutation_task, return_exceptions=True)
        if replacement_connection.in_transaction:
            await replacement_connection.rollback()
        await replacement_connection.close()
        await runtime.storage_backend.close()
        await raw_lifecycle_connection.close()
