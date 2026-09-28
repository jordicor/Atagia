"""Tests for conversation activity aggregation and warm-up."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.conversation_activity_repository import ConversationActivityRepository
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.storage_backend import build_recent_window_key
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.models.schemas_memory import ConversationStatus
from atagia.services.conversation_activity_service import ConversationActivityService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.errors import TranscriptRebuildInProgressError
from tests.recent_window_support import stored_recent_window
from tests.turn_telemetry_support import sample_turn_telemetry
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


class NoopProvider(LLMProvider):
    name = "noop-activity-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(
            f"LLM should not be called in activity tests: {request.metadata}"
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(
            f"Embeddings should not be called in activity tests: {request.model}"
        )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-activity.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_ingest_model="openai/test-model",
        llm_retrieval_model="openai/test-model",
        llm_component_models={"intent_classifier": "openai/test-model"},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppRuntime:
    provider = NoopProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    runtime.clock = FrozenClock(datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc))
    return runtime


async def _seed_user_conversation(
    runtime: AppRuntime,
    *,
    user_id: str,
    conversation_id: str,
    workspace_id: str | None = None,
    title: str = "Chat",
) -> None:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        workspaces = WorkspaceRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)

        if await users.get_user(user_id) is None:
            await users.create_user(user_id)
        if workspace_id is not None:
            if await workspaces.get_workspace(workspace_id, user_id) is None:
                await workspaces.create_workspace(
                    workspace_id,
                    user_id,
                    "Workspace",
                    {"timezone": "UTC"},
                )
        await conversations.create_conversation(
            conversation_id,
            user_id,
            workspace_id,
            "coding_debug",
            title,
        )
    finally:
        await connection.close()


def _install_stats_compute_barrier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reached: asyncio.Event,
    resume: asyncio.Event,
) -> None:
    original_compute = ConversationActivityService._compute_conversation_stats

    async def compute_then_pause(
        service: ConversationActivityService,
        connection: Any,
        *,
        user_id: str,
        conversation: dict[str, Any],
        as_of: str | None,
    ) -> dict[str, Any]:
        stats = await original_compute(
            service,
            connection,
            user_id=user_id,
            conversation=conversation,
            as_of=as_of,
        )
        reached.set()
        await asyncio.wait_for(resume.wait(), timeout=5.0)
        return stats

    monkeypatch.setattr(
        ConversationActivityService,
        "_compute_conversation_stats",
        compute_then_pause,
    )


async def _commit_blocking_transcript_selection(
    connection: Any,
    *,
    user_id: str,
    conversation_id: str,
) -> None:
    timestamp = "2026-03-10T12:00:00+00:00"
    workflow_id = f"trb_activity_{user_id}"
    await connection.execute("BEGIN IMMEDIATE")
    revision_cursor = await connection.execute(
        """
        UPDATE user_lifecycles
        SET derivation_revision = derivation_revision + 1,
            updated_at = ?
        WHERE user_id = ? AND state = 'active'
        RETURNING derivation_revision
        """,
        (timestamp, user_id),
    )
    revision = await revision_cursor.fetchone()
    assert revision is not None
    await connection.execute(
        """
        INSERT INTO transcript_rebuild_workflows(
            id, operation_id, user_id, conversation_id, selection_epoch,
            transcript_hash, mutation_kind, selected_message_ids_json,
            abandoned_message_ids_json, supporting_message_ids_json,
            affected_memory_ids_json, affected_summary_ids_json,
            orchestrator_job_id, stage, start_derivation_revision,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, 1, 'activity-race-hash', 'replace',
                  '[]', '[]', '[]', '[]', '[]', ?, 'sources', ?, ?, ?)
        """,
        (
            workflow_id,
            f"op_activity_{user_id}",
            user_id,
            conversation_id,
            f"job_activity_{user_id}",
            int(revision["derivation_revision"]),
            timestamp,
            timestamp,
        ),
    )
    await connection.execute(
        """
        INSERT INTO conversation_transcript_selections(
            user_id, conversation_id, selection_epoch, transcript_hash,
            current_workflow_id, state, updated_at
        ) VALUES (?, ?, 1, 'activity-race-hash', ?, 'rebuilding', ?)
        """,
        (user_id, conversation_id, workflow_id, timestamp),
    )
    await connection.commit()


async def _commit_message_source_mutation(
    connection: Any,
    runtime: AppRuntime,
    *,
    mutation: str,
    conversation_id: str,
    message_id: str,
) -> None:
    if mutation == "append":
        await MessageRepository(connection, runtime.clock).create_message(
            f"{message_id}_appended",
            conversation_id,
            "assistant",
            5,
            "The fifth message committed on the competing runtime.",
            9,
            {},
            "2026-03-10T11:05:00+00:00",
        )
        return
    await connection.execute("BEGIN IMMEDIATE")
    if mutation == "edit":
        await connection.execute(
            "UPDATE messages SET text = ? WHERE id = ?",
            ("The competing runtime edited this canonical message.", message_id),
        )
    else:
        await connection.execute(
            "DELETE FROM messages WHERE id = ?",
            (message_id,),
        )
    await connection.commit()


async def _seed_activity_messages(
    connection: Any,
    runtime: AppRuntime,
    *,
    conversation_id: str,
    message_prefix: str,
) -> None:
    messages = MessageRepository(connection, runtime.clock)
    for seq in range(1, 5):
        await messages.create_message(
            f"{message_prefix}_{seq}",
            conversation_id,
            "user" if seq % 2 else "assistant",
            seq,
            f"Canonical activity message {seq}.",
            5,
            {},
            f"2026-03-10T11:0{seq}:00+00:00",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dimension", "old_value", "new_value"),
    [
        ("workspace_id", "ws_activity_old", "ws_activity_new"),
        ("assistant_mode_id", "coding_debug", "general_qa"),
        ("user_persona_id", "persona_old", "persona_new"),
        ("platform_id", "platform_old", "platform_new"),
        ("character_id", "character_old", "character_new"),
        ("incognito", 0, 1),
    ],
)
async def test_no_refresh_activity_reads_use_live_namespace_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: str,
    old_value: object,
    new_value: object,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    user_id = f"usr_activity_namespace_{dimension}"
    conversation_id = f"cnv_activity_namespace_{dimension}"
    try:
        await _seed_user_conversation(
            runtime,
            user_id=user_id,
            conversation_id=conversation_id,
            workspace_id="ws_activity_old",
        )
        await WorkspaceRepository(connection, runtime.clock).create_workspace(
            "ws_activity_new",
            user_id,
            "New workspace",
            {"timezone": "UTC"},
        )
        await connection.execute(
            """
            UPDATE conversations
            SET user_persona_id = ?,
                platform_id = ?,
                character_id = ?,
                incognito = 0
            WHERE id = ? AND user_id = ?
            """,
            (
                "persona_old",
                "platform_old",
                "character_old",
                conversation_id,
                user_id,
            ),
        )
        await connection.commit()
        await MessageRepository(connection, runtime.clock).create_message(
            f"msg_activity_namespace_{dimension}",
            conversation_id,
            "user",
            1,
            "Namespace changes must not reuse stale membership columns.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime)
        await service.refresh_conversation_activity_stats(
            connection,
            user_id,
            conversation_id,
        )

        await connection.execute(
            f"UPDATE conversations SET {dimension} = ? WHERE id = ? AND user_id = ?",
            (new_value, conversation_id, user_id),
        )
        await connection.commit()

        old_filters: dict[str, Any] = {
            "workspace_id": "ws_activity_old",
            "assistant_mode_id": "coding_debug",
            "user_persona_id": "persona_old",
            "platform_id": "platform_old",
            "character_id": "character_old",
            "incognito": False,
        }
        new_filters = dict(old_filters)
        old_filters[dimension] = (
            bool(old_value) if dimension == "incognito" else old_value
        )
        new_filters[dimension] = (
            bool(new_value) if dimension == "incognito" else new_value
        )

        assert (
            await service.list_hot_conversations(
                connection,
                user_id,
                refresh=False,
                **old_filters,
            )
            == []
        )
        hot_rows = await service.list_hot_conversations(
            connection,
            user_id,
            refresh=False,
            **new_filters,
        )
        assert len(hot_rows) == 1
        assert hot_rows[0][dimension] == new_filters[dimension]

        old_snapshot = await service.get_activity_snapshot(
            connection,
            user_id,
            refresh=False,
            namespace_filter=True,
            **old_filters,
        )
        assert old_snapshot["conversations"] == []
        new_snapshot = await service.get_activity_snapshot(
            connection,
            user_id,
            refresh=False,
            namespace_filter=True,
            **new_filters,
        )
        assert len(new_snapshot["conversations"]) == 1
        assert new_snapshot["conversations"][0][dimension] == new_filters[dimension]
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_kind", ["snapshot", "hot_list"])
@pytest.mark.parametrize("boundary_change", ["namespace", "owner"])
async def test_no_refresh_activity_read_rejects_concurrent_membership_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_kind: str,
    boundary_change: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    read_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    read_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_boundary_{read_kind}_{boundary_change}"
    conversation_id = f"cnv_activity_boundary_{read_kind}_{boundary_change}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await read_connection.execute(
            "UPDATE conversations SET platform_id = ? WHERE id = ?",
            ("platform_before", conversation_id),
        )
        await read_connection.commit()
        await MessageRepository(read_connection, runtime_a.clock).create_message(
            f"msg_activity_boundary_{read_kind}_{boundary_change}",
            conversation_id,
            "user",
            1,
            "The row and its live membership must share one source stamp.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime_a)
        await service.refresh_conversation_activity_stats(
            read_connection,
            user_id,
            conversation_id,
        )
        original_require = service._require_current_activity_membership

        async def pause_before_source_check(*args: Any, **kwargs: Any) -> None:
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            await original_require(*args, **kwargs)

        monkeypatch.setattr(
            service,
            "_require_current_activity_membership",
            pause_before_source_check,
        )
        if read_kind == "snapshot":
            read_task = asyncio.create_task(
                service.get_activity_snapshot(
                    read_connection,
                    user_id,
                    platform_id="platform_before",
                    namespace_filter=True,
                    refresh=False,
                )
            )
        else:
            read_task = asyncio.create_task(
                service.list_hot_conversations(
                    read_connection,
                    user_id,
                    platform_id="platform_before",
                    refresh=False,
                )
            )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        if boundary_change == "namespace":
            await mutation_connection.execute(
                "UPDATE conversations SET platform_id = ? WHERE id = ?",
                ("platform_after", conversation_id),
            )
        else:
            new_user_id = f"{user_id}_new_owner"
            await UserRepository(
                mutation_connection,
                runtime_b.clock,
            ).create_user(new_user_id)
            await mutation_connection.execute(
                "UPDATE conversations SET user_id = ? WHERE id = ?",
                (new_user_id, conversation_id),
            )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(read_task, timeout=5.0)
        read_task = None
    finally:
        resume.set()
        if read_task is not None:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
        await mutation_connection.close()
        await read_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("read_kind", "membership_change"),
    [
        ("snapshot", "namespace_empty"),
        ("hot_list", "namespace_empty"),
        ("hot_list", "status_empty"),
        ("hot_list", "temporary_empty"),
        ("hot_list", "namespace_top_n"),
    ],
)
async def test_no_refresh_activity_read_rejects_membership_phantoms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_kind: str,
    membership_change: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    read_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    read_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_phantom_{read_kind}_{membership_change}"
    first_id = f"cnv_activity_phantom_{read_kind}_{membership_change}_a"
    second_id = f"cnv_activity_phantom_{read_kind}_{membership_change}_b"
    target_platform = "platform_activity_phantom_target"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=first_id,
        )
        if membership_change == "namespace_top_n":
            await _seed_user_conversation(
                runtime_a,
                user_id=user_id,
                conversation_id=second_id,
            )
        conversation_ids = (
            [first_id, second_id]
            if membership_change == "namespace_top_n"
            else [first_id]
        )
        for index, conversation_id in enumerate(conversation_ids, start=1):
            initial_platform = (
                target_platform
                if membership_change == "namespace_top_n" and index == 1
                else "platform_activity_phantom_outside"
            )
            await read_connection.execute(
                "UPDATE conversations SET platform_id = ? WHERE id = ?",
                (initial_platform, conversation_id),
            )
            await read_connection.commit()
            await MessageRepository(
                read_connection,
                runtime_a.clock,
            ).create_message(
                f"msg_activity_phantom_{read_kind}_{membership_change}_{index}",
                conversation_id,
                "user",
                1,
                f"Activity phantom candidate {index}.",
                5,
                {},
                f"2026-03-10T11:0{index}:00+00:00",
            )
            await ConversationActivityService(
                runtime_a
            ).refresh_conversation_activity_stats(
                read_connection,
                user_id,
                conversation_id,
            )

        if membership_change == "status_empty":
            await read_connection.execute(
                "UPDATE conversations SET status = ? WHERE id = ?",
                (ConversationStatus.ARCHIVED.value, first_id),
            )
            await read_connection.commit()
        elif membership_change == "temporary_empty":
            await read_connection.execute(
                "UPDATE conversations SET temporary = 1 WHERE id = ?",
                (first_id,),
            )
            await read_connection.commit()
        elif membership_change == "namespace_top_n":
            await read_connection.execute(
                """
                UPDATE conversation_activity_stats
                SET likely_soon_score = CASE conversation_id
                    WHEN ? THEN 0.1
                    WHEN ? THEN 0.9
                    ELSE likely_soon_score
                END
                WHERE user_id = ? AND conversation_id IN (?, ?)
                """,
                (first_id, second_id, user_id, first_id, second_id),
            )
            await read_connection.commit()

        service = ConversationActivityService(runtime_a)
        original_require = service._require_current_activity_membership
        pause_once = True

        async def pause_before_membership_check(
            *args: Any,
            **kwargs: Any,
        ) -> None:
            nonlocal pause_once
            if pause_once:
                pause_once = False
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
            await original_require(*args, **kwargs)

        monkeypatch.setattr(
            service,
            "_require_current_activity_membership",
            pause_before_membership_check,
        )
        if read_kind == "snapshot":
            read_task = asyncio.create_task(
                service.get_activity_snapshot(
                    read_connection,
                    user_id,
                    platform_id=target_platform,
                    namespace_filter=True,
                    refresh=False,
                )
            )
        else:
            hot_kwargs: dict[str, Any] = {
                "limit": 1,
                "refresh": False,
            }
            if membership_change.startswith("namespace_"):
                hot_kwargs["platform_id"] = target_platform
            read_task = asyncio.create_task(
                service.list_hot_conversations(
                    read_connection,
                    user_id,
                    **hot_kwargs,
                )
            )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        if membership_change.startswith("namespace_"):
            entering_id = (
                second_id if membership_change == "namespace_top_n" else first_id
            )
            await mutation_connection.execute(
                "UPDATE conversations SET platform_id = ? WHERE id = ?",
                (target_platform, entering_id),
            )
        elif membership_change == "status_empty":
            await mutation_connection.execute(
                "UPDATE conversations SET status = ? WHERE id = ?",
                (ConversationStatus.ACTIVE.value, first_id),
            )
        else:
            await mutation_connection.execute(
                "UPDATE conversations SET temporary = 0 WHERE id = ?",
                (first_id,),
            )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(read_task, timeout=5.0)
        read_task = None

        if read_kind == "snapshot":
            current = await service.get_activity_snapshot(
                read_connection,
                user_id,
                platform_id=target_platform,
                namespace_filter=True,
                refresh=False,
            )
            assert [row["conversation_id"] for row in current["conversations"]] == [
                first_id
            ]
        else:
            hot_kwargs = {"limit": 1, "refresh": False}
            if membership_change.startswith("namespace_"):
                hot_kwargs["platform_id"] = target_platform
            current = await service.list_hot_conversations(
                read_connection,
                user_id,
                **hot_kwargs,
            )
            expected_id = (
                second_id if membership_change == "namespace_top_n" else first_id
            )
            assert [row["conversation_id"] for row in current] == [expected_id]
    finally:
        resume.set()
        if read_task is not None:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
        await mutation_connection.close()
        await read_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["enter", "exit"])
@pytest.mark.parametrize(
    "boundary_kind",
    ["user_persona_id", "platform_id", "character_id", "incognito", "status", "owner"],
)
async def test_single_warmup_rejects_concurrent_eligibility_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    boundary_kind: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    warmup_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    target_user_id = f"usr_warmup_eligibility_{boundary_kind}_{direction}"
    other_user_id = f"{target_user_id}_other"
    conversation_id = f"cnv_warmup_eligibility_{boundary_kind}_{direction}"
    target_namespace: dict[str, Any] = {
        "user_persona_id": "persona_warmup_target",
        "platform_id": "platform_warmup_target",
        "character_id": "character_warmup_target",
        "incognito": True,
    }
    try:
        await UserRepository(warmup_connection, runtime_a.clock).create_user(
            target_user_id
        )
        await UserRepository(warmup_connection, runtime_a.clock).create_user(
            other_user_id
        )
        initial_owner = (
            other_user_id
            if boundary_kind == "owner" and direction == "enter"
            else target_user_id
        )
        await ConversationRepository(
            warmup_connection,
            runtime_a.clock,
        ).create_conversation(
            conversation_id,
            initial_owner,
            None,
            "coding_debug",
            "Warmup eligibility",
        )
        await MessageRepository(warmup_connection, runtime_a.clock).create_message(
            f"msg_warmup_eligibility_{boundary_kind}_{direction}",
            conversation_id,
            "user",
            1,
            "Warmup eligibility is a live boundary.",
            7,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        initial_namespace = dict(target_namespace)
        if boundary_kind in target_namespace and direction == "enter":
            initial_namespace[boundary_kind] = (
                False if boundary_kind == "incognito" else f"{boundary_kind}_outside"
            )
        initial_status = (
            ConversationStatus.ARCHIVED.value
            if boundary_kind == "status" and direction == "enter"
            else ConversationStatus.ACTIVE.value
        )
        await warmup_connection.execute(
            """
            UPDATE conversations
            SET user_persona_id = ?,
                platform_id = ?,
                character_id = ?,
                incognito = ?,
                status = ?
            WHERE id = ?
            """,
            (
                initial_namespace["user_persona_id"],
                initial_namespace["platform_id"],
                initial_namespace["character_id"],
                1 if initial_namespace["incognito"] else 0,
                initial_status,
                conversation_id,
            ),
        )
        await warmup_connection.commit()

        service = ConversationActivityService(runtime_a)
        pause_once = True
        if direction == "enter":
            original_require = service._require_current_warmup_eligibility

            async def pause_early_return(
                *args: Any,
                **kwargs: Any,
            ) -> None:
                nonlocal pause_once
                if pause_once:
                    pause_once = False
                    reached.set()
                    await asyncio.wait_for(resume.wait(), timeout=5.0)
                await original_require(*args, **kwargs)

            monkeypatch.setattr(
                service,
                "_require_current_warmup_eligibility",
                pause_early_return,
            )
        else:
            original_refresh = service._refresh_conversation_activity_stats_with_source

            async def pause_before_refresh(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                nonlocal pause_once
                if pause_once:
                    pause_once = False
                    reached.set()
                    await asyncio.wait_for(resume.wait(), timeout=5.0)
                return await original_refresh(*args, **kwargs)

            monkeypatch.setattr(
                service,
                "_refresh_conversation_activity_stats_with_source",
                pause_before_refresh,
            )

        warmup_task = asyncio.create_task(
            service.warmup_conversation(
                warmup_connection,
                target_user_id,
                conversation_id,
                **target_namespace,
            )
        )
        await asyncio.wait_for(reached.wait(), timeout=5.0)

        if boundary_kind == "owner":
            new_owner = target_user_id if direction == "enter" else other_user_id
            await mutation_connection.execute(
                "UPDATE conversations SET user_id = ? WHERE id = ?",
                (new_owner, conversation_id),
            )
        elif boundary_kind == "status":
            new_status = (
                ConversationStatus.ACTIVE.value
                if direction == "enter"
                else ConversationStatus.ARCHIVED.value
            )
            await mutation_connection.execute(
                "UPDATE conversations SET status = ? WHERE id = ?",
                (new_status, conversation_id),
            )
        else:
            new_value: Any = target_namespace[boundary_kind]
            if direction == "exit":
                new_value = (
                    False
                    if boundary_kind == "incognito"
                    else f"{boundary_kind}_outside"
                )
            await mutation_connection.execute(
                f"UPDATE conversations SET {boundary_kind} = ? WHERE id = ?",
                (
                    1 if new_value is True else 0 if new_value is False else new_value,
                    conversation_id,
                ),
            )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None

        current = await service.warmup_conversation(
            warmup_connection,
            target_user_id,
            conversation_id,
            **target_namespace,
        )
        if direction == "enter":
            assert current["recent_message_count"] == 1
            assert current["warmup_errors"] == []
        else:
            assert current["recent_message_count"] == 0
            assert current["warmup_errors"] == ["conversation_not_found"]
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await mutation_connection.close()
        await warmup_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_specific_warmup_keeps_active_temporary_conversation_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    user_id = "usr_warmup_active_temporary"
    conversation_id = "cnv_warmup_active_temporary"
    try:
        await _seed_user_conversation(
            runtime,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await connection.execute(
            "UPDATE conversations SET temporary = 1 WHERE id = ?",
            (conversation_id,),
        )
        await connection.commit()
        await MessageRepository(connection, runtime.clock).create_message(
            "msg_warmup_active_temporary",
            conversation_id,
            "user",
            1,
            "A directly requested active temporary chat remains eligible.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )

        result = await ConversationActivityService(runtime).warmup_conversation(
            connection,
            user_id,
            conversation_id,
        )
        assert result["recent_message_count"] == 1
        assert result["warmup_errors"] == []
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier_phase", ["payload", "final"])
async def test_single_warmup_rechecks_namespace_after_stats_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    barrier_phase: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    warmup_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = f"usr_warmup_namespace_recheck_{barrier_phase}"
    conversation_id = f"cnv_warmup_namespace_recheck_{barrier_phase}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await warmup_connection.execute(
            "UPDATE conversations SET platform_id = ? WHERE id = ?",
            ("platform_warmup_recheck", conversation_id),
        )
        await warmup_connection.commit()
        await MessageRepository(warmup_connection, runtime_a.clock).create_message(
            f"msg_warmup_namespace_recheck_{barrier_phase}",
            conversation_id,
            "user",
            1,
            "Namespace must remain exact through final warmup publication.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime_a)
        if barrier_phase == "payload":
            original_barrier = service._build_warmup_payload
        else:
            original_barrier = service._require_current_warmup_identity

        async def pause_at_barrier(*args: Any, **kwargs: Any) -> Any:
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return await original_barrier(*args, **kwargs)

        monkeypatch.setattr(
            service,
            (
                "_build_warmup_payload"
                if barrier_phase == "payload"
                else "_require_current_warmup_identity"
            ),
            pause_at_barrier,
        )
        warmup_task = asyncio.create_task(
            service.warmup_conversation(
                warmup_connection,
                user_id,
                conversation_id,
                platform_id="platform_warmup_recheck",
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await mutation_connection.execute(
            "UPDATE conversations SET platform_id = ? WHERE id = ?",
            ("platform_warmup_outside", conversation_id),
        )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
        assert (
            await stored_recent_window(runtime_a.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            is None
        )
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await mutation_connection.close()
        await warmup_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded_kind", ["archived", "temporary"])
async def test_refreshed_activity_sets_exclude_non_active_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    excluded_kind: str,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    user_id = f"usr_activity_excluded_{excluded_kind}"
    conversation_id = f"cnv_activity_excluded_{excluded_kind}"
    try:
        await _seed_user_conversation(
            runtime,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection, runtime.clock).create_message(
            f"msg_activity_excluded_{excluded_kind}",
            conversation_id,
            "user",
            1,
            "Only active non-temporary conversations belong to refreshed sets.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime)
        await service.refresh_conversation_activity_stats(
            connection,
            user_id,
            conversation_id,
        )
        if excluded_kind == "archived":
            await connection.execute(
                "UPDATE conversations SET status = ? WHERE id = ?",
                (ConversationStatus.ARCHIVED.value, conversation_id),
            )
        else:
            await connection.execute(
                "UPDATE conversations SET temporary = 1 WHERE id = ?",
                (conversation_id,),
            )
        await connection.commit()

        refreshed = await service.get_activity_snapshot(
            connection,
            user_id,
            refresh=True,
        )
        assert refreshed["conversations"] == []
        assert (
            await service.list_hot_conversations(
                connection,
                user_id,
                refresh=True,
            )
            == []
        )

        historical = await service.get_activity_snapshot(
            connection,
            user_id,
            refresh=False,
        )
        assert len(historical["conversations"]) == 1
        assert (
            await service.list_hot_conversations(
                connection,
                user_id,
                refresh=False,
            )
            == []
        )
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", [False, True])
async def test_activity_reads_fail_closed_across_conversation_owner_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh: bool,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    old_user_id = f"usr_activity_owner_old_{refresh}"
    new_user_id = f"usr_activity_owner_new_{refresh}"
    conversation_id = f"cnv_activity_owner_{refresh}"
    try:
        await _seed_user_conversation(
            runtime,
            user_id=old_user_id,
            conversation_id=conversation_id,
        )
        await UserRepository(connection, runtime.clock).create_user(new_user_id)
        await MessageRepository(connection, runtime.clock).create_message(
            f"msg_activity_owner_{refresh}",
            conversation_id,
            "user",
            1,
            "These metrics belong only to the current conversation owner.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime)
        await service.refresh_conversation_activity_stats(
            connection,
            old_user_id,
            conversation_id,
        )

        await connection.execute(
            "UPDATE conversations SET user_id = ? WHERE id = ?",
            (new_user_id, conversation_id),
        )
        await connection.commit()

        old_snapshot = await service.get_activity_snapshot(
            connection,
            old_user_id,
            conversation_id=conversation_id,
            refresh=refresh,
        )
        assert old_snapshot["conversations"] == []
        assert (
            await service.list_hot_conversations(
                connection,
                old_user_id,
                refresh=refresh,
            )
            == []
        )

        new_snapshot = await service.get_activity_snapshot(
            connection,
            new_user_id,
            conversation_id=conversation_id,
            refresh=refresh,
        )
        if refresh:
            assert len(new_snapshot["conversations"]) == 1
            assert new_snapshot["conversations"][0]["user_id"] == new_user_id
        else:
            assert new_snapshot["conversations"] == []
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_activity_stats_aggregate_messages_retrieval_and_histograms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(
            runtime,
            user_id="usr_1",
            conversation_id="cnv_1",
            workspace_id="wrk_1",
        )
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)

            runtime.clock = FrozenClock(datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "First weekly check-in",
                5,
                {},
                "2026-03-02T09:00:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 2, 9, 1, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_2",
                "cnv_1",
                "assistant",
                2,
                "Noted.",
                2,
                {},
                "2026-03-02T09:01:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_3",
                "cnv_1",
                "user",
                3,
                "Second weekly check-in",
                5,
                {},
                "2026-03-09T09:00:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 5, tzinfo=timezone.utc))
            await messages.create_message(
                "msg_4",
                "cnv_1",
                "assistant",
                4,
                "Still on track.",
                3,
                {},
                "2026-03-09T09:05:00+00:00",
            )
            runtime.clock = FrozenClock(datetime(2026, 3, 9, 9, 6, tzinfo=timezone.utc))
            await events.create_event(
                {
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_3",
                    "response_message_id": "msg_4",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["weekly"]},
                    "selected_memory_ids_json": [],
                    "context_view_json": {},
                    "outcome_json": {},
                },
                telemetry=sample_turn_telemetry(),
            )
            runtime.clock = FrozenClock(
                datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
            )

            service = ConversationActivityService(runtime)
            snapshot = await service.get_activity_snapshot(
                connection,
                "usr_1",
                conversation_id="cnv_1",
            )
            stats = snapshot["conversations"][0]

            assert stats["message_count"] == 4
            assert stats["user_message_count"] == 2
            assert stats["assistant_message_count"] == 2
            assert stats["retrieval_count"] == 1
            assert stats["active_day_count"] == 3
            assert stats["recent_1d_message_count"] == 0
            assert stats["recent_7d_message_count"] == 2
            assert stats["recent_30d_message_count"] == 4
            assert len(stats["weekday_histogram_json"]) == 7
            assert len(stats["hour_histogram_json"]) == 24
            assert len(stats["hour_of_week_histogram_json"]) == 168
            assert stats["median_return_interval_minutes"] == 5850.0
            assert stats["p90_return_interval_minutes"] == 10080.0
            assert stats["schedule_pattern_kind"] == "weekly"
            assert stats["main_thread_score"] > 0.0
            assert 0.0 <= stats["likely_soon_score"] <= 1.0

            stored = await ConversationActivityRepository(
                connection, runtime.clock
            ).get_activity_stats(
                user_id="usr_1",
                conversation_id="cnv_1",
            )
            assert stored is not None
            assert stored["conversation_id"] == "cnv_1"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_hot_ranking_prefers_recent_recurrent_conversations_and_respects_user_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(
            runtime, user_id="usr_a", conversation_id="cnv_hot"
        )
        await _seed_user_conversation(
            runtime, user_id="usr_a", conversation_id="cnv_cold"
        )
        await _seed_user_conversation(
            runtime, user_id="usr_b", conversation_id="cnv_other"
        )

        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)

            await messages.create_message(
                "msg_hot_1",
                "cnv_hot",
                "user",
                1,
                "Recent ping",
                2,
                {},
                "2026-03-10T11:00:00+00:00",
            )
            await messages.create_message(
                "msg_hot_2",
                "cnv_hot",
                "assistant",
                2,
                "Recent pong",
                2,
                {},
                "2026-03-10T11:05:00+00:00",
            )
            for index, day in enumerate([1, 8, 15, 22], start=1):
                await messages.create_message(
                    f"msg_cold_{index}",
                    "cnv_cold",
                    "user" if index % 2 else "assistant",
                    index,
                    f"Older message {index}",
                    2,
                    {},
                    f"2026-02-{day:02d}T09:00:00+00:00",
                )
            await messages.create_message(
                "msg_other_1",
                "cnv_other",
                "user",
                1,
                "Different user recent ping",
                2,
                {},
                "2026-03-10T11:10:00+00:00",
            )

            service = ConversationActivityService(runtime)
            hot = await service.list_hot_conversations(connection, "usr_a", limit=10)
            assert [row["conversation_id"] for row in hot][:2] == [
                "cnv_hot",
                "cnv_cold",
            ]
            assert all(row["user_id"] == "usr_a" for row in hot)
            assert "cnv_other" not in {row["conversation_id"] for row in hot}

            other_user = await service.list_hot_conversations(
                connection, "usr_b", limit=10
            )
            assert [row["conversation_id"] for row in other_user] == ["cnv_other"]
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_warmup_primes_recent_window_and_recommended_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_user_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            for seq in range(1, 7):
                await messages.create_message(
                    f"msg_{seq}",
                    "cnv_1",
                    "user" if seq % 2 else "assistant",
                    seq,
                    f"Message {seq}",
                    2,
                    {},
                    f"2026-03-10T11:{seq:02d}:00+00:00",
                )

            service = ConversationActivityService(runtime)
            single = await service.warmup_conversation(
                connection,
                "usr_1",
                "cnv_1",
                max_messages=3,
            )
            assert single["recent_window_key"] == build_recent_window_key(
                "usr_1",
                "cnv_1",
            )
            assert single["recent_message_count"] == 3
            assert single["recent_message_ids"] == ["msg_4", "msg_5", "msg_6"]
            assert single["cached_context_available"] is False
            assert (
                await stored_recent_window(runtime.storage_backend,
                    build_recent_window_key("usr_1", "cnv_1")
                )
                == single["recent_messages"]
            )

            recommended = await service.warmup_recommended_conversations(
                connection,
                "usr_1",
                limit=1,
                total_message_budget=2,
                per_conversation_message_budget=2,
            )
            assert recommended["warmed_conversation_count"] == 1
            assert recommended["warmed_message_count"] == 2
            assert recommended["hot_conversations"][0]["conversation_id"] == "cnv_1"

            bounded = await service.warmup_conversation(
                connection,
                "usr_1",
                "cnv_1",
                max_messages=-1,
            )
            assert bounded["recent_message_count"] == 1
            assert bounded["recent_message_ids"] == ["msg_6"]

            no_budget = await service.warmup_recommended_conversations(
                connection,
                "usr_1",
                limit=-10,
                total_message_budget=-1,
            )
            assert no_budget["requested_limit"] == 1
            assert no_budget["warmed_conversation_count"] == 0
            assert no_budget["warmed_message_count"] == 0
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_warmup_captured_window_cannot_publish_after_conversation_delete_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    mutation_connection = None
    reached = asyncio.Event()
    resume = asyncio.Event()
    try:
        await _seed_user_conversation(
            runtime,
            user_id="usr_window_race",
            conversation_id="cnv_window_race",
        )
        messages = MessageRepository(connection, runtime.clock)
        await messages.create_message(
            "msg_window_race",
            "cnv_window_race",
            "user",
            1,
            "Window captured before branch replacement.",
            5,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        original_finalize = ConversationActivityService._finalize_warmup_payload

        async def finalize_after_lifecycle_barrier(
            service: ConversationActivityService,
            payload: Any,
        ) -> dict[str, Any]:
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return await original_finalize(service, payload)

        monkeypatch.setattr(
            ConversationActivityService,
            "_finalize_warmup_payload",
            finalize_after_lifecycle_barrier,
        )
        service = ConversationActivityService(runtime)
        warmup = asyncio.create_task(
            service.warmup_conversation(
                connection,
                "usr_window_race",
                "cnv_window_race",
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        mutation_connection = await runtime.open_connection()
        await mutation_connection.execute(
            """
            UPDATE conversations
            SET status = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                ConversationStatus.PENDING_DELETION.value,
                runtime.clock.now().isoformat(),
                "cnv_window_race",
                "usr_window_race",
            ),
        )
        await mutation_connection.commit()
        resume.set()
        with pytest.raises(TranscriptRebuildInProgressError):
            await warmup

        assert (
            await stored_recent_window(runtime.storage_backend,
                build_recent_window_key("usr_window_race", "cnv_window_race")
            )
            is None
        )
    finally:
        resume.set()
        if mutation_connection is not None:
            await mutation_connection.close()
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_refresh_user_does_not_publish_stats_after_transcript_rebuild_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    refresh_connection = await runtime.open_connection()
    mutation_connection = await runtime.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    refresh_task: asyncio.Task[list[dict[str, Any]]] | None = None
    try:
        await _seed_user_conversation(
            runtime,
            user_id="usr_refresh_user_race",
            conversation_id="cnv_refresh_user_race",
        )
        await MessageRepository(refresh_connection, runtime.clock).create_message(
            "msg_refresh_user_race",
            "cnv_refresh_user_race",
            "user",
            1,
            "Source captured before replacement.",
            5,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        _install_stats_compute_barrier(
            monkeypatch,
            reached=reached,
            resume=resume,
        )
        refresh_task = asyncio.create_task(
            ConversationActivityService(runtime).refresh_user_activity_stats(
                refresh_connection,
                "usr_refresh_user_race",
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_blocking_transcript_selection(
            mutation_connection,
            user_id="usr_refresh_user_race",
            conversation_id="cnv_refresh_user_race",
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(refresh_task, timeout=5.0)
        refresh_task = None
        assert (
            await ConversationActivityRepository(
                refresh_connection,
                runtime.clock,
            ).get_activity_stats(
                user_id="usr_refresh_user_race",
                conversation_id="cnv_refresh_user_race",
            )
            is None
        )
    finally:
        resume.set()
        if refresh_task is not None:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
        await mutation_connection.close()
        await refresh_connection.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_refresh_conversation_does_not_publish_stats_after_revision_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    refresh_connection = await runtime.open_connection()
    mutation_connection = await runtime.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    refresh_task: asyncio.Task[dict[str, Any] | None] | None = None
    try:
        await _seed_user_conversation(
            runtime,
            user_id="usr_refresh_conversation_race",
            conversation_id="cnv_refresh_conversation_race",
        )
        await MessageRepository(refresh_connection, runtime.clock).create_message(
            "msg_refresh_conversation_race",
            "cnv_refresh_conversation_race",
            "user",
            1,
            "Source captured before revision change.",
            5,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        _install_stats_compute_barrier(
            monkeypatch,
            reached=reached,
            resume=resume,
        )
        refresh_task = asyncio.create_task(
            ConversationActivityService(runtime).refresh_conversation_activity_stats(
                refresh_connection,
                "usr_refresh_conversation_race",
                "cnv_refresh_conversation_race",
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await mutation_connection.execute("BEGIN IMMEDIATE")
        await mutation_connection.execute(
            """
            UPDATE user_lifecycles
            SET derivation_revision = derivation_revision + 1,
                updated_at = ?
            WHERE user_id = ? AND state = 'active'
            """,
            (
                "2026-03-10T12:00:00+00:00",
                "usr_refresh_conversation_race",
            ),
        )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(refresh_task, timeout=5.0)
        refresh_task = None
        assert (
            await ConversationActivityRepository(
                refresh_connection,
                runtime.clock,
            ).get_activity_stats(
                user_id="usr_refresh_conversation_race",
                conversation_id="cnv_refresh_conversation_race",
            )
            is None
        )
    finally:
        resume.set()
        if refresh_task is not None:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
        if mutation_connection.in_transaction:
            await mutation_connection.rollback()
        await mutation_connection.close()
        await refresh_connection.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_kind", ["single", "bulk"])
@pytest.mark.parametrize("mutation", ["append", "edit", "delete"])
async def test_activity_refresh_rejects_concurrent_message_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_kind: str,
    mutation: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    refresh_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    refresh_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_refresh_{refresh_kind}_{mutation}"
    conversation_id = f"cnv_activity_refresh_{refresh_kind}_{mutation}"
    message_prefix = f"msg_activity_refresh_{refresh_kind}_{mutation}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await _seed_activity_messages(
            refresh_connection,
            runtime_a,
            conversation_id=conversation_id,
            message_prefix=message_prefix,
        )
        _install_stats_compute_barrier(
            monkeypatch,
            reached=reached,
            resume=resume,
        )
        service = ConversationActivityService(runtime_a)
        if refresh_kind == "single":
            refresh_task = asyncio.create_task(
                service.refresh_conversation_activity_stats(
                    refresh_connection,
                    user_id,
                    conversation_id,
                )
            )
        else:
            refresh_task = asyncio.create_task(
                service.refresh_user_activity_stats(
                    refresh_connection,
                    user_id,
                )
            )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_message_source_mutation(
            mutation_connection,
            runtime_b,
            mutation=mutation,
            conversation_id=conversation_id,
            message_id=f"{message_prefix}_1",
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(refresh_task, timeout=5.0)
        refresh_task = None
        assert (
            await ConversationActivityRepository(
                refresh_connection,
                runtime_a.clock,
            ).get_activity_stats(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            is None
        )
    finally:
        resume.set()
        if refresh_task is not None:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
        await mutation_connection.close()
        await refresh_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("set_change", ["add", "remove"])
async def test_bulk_activity_refresh_rejects_concurrent_conversation_set_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    set_change: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    refresh_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    refresh_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_set_{set_change}"
    conversation_id = f"cnv_activity_set_{set_change}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(refresh_connection, runtime_a.clock).create_message(
            f"msg_activity_set_{set_change}",
            conversation_id,
            "user",
            1,
            "The candidate set is captured before this refresh pauses.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        _install_stats_compute_barrier(
            monkeypatch,
            reached=reached,
            resume=resume,
        )
        refresh_task = asyncio.create_task(
            ConversationActivityService(runtime_a).refresh_user_activity_stats(
                refresh_connection,
                user_id,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        if set_change == "add":
            await ConversationRepository(
                mutation_connection,
                runtime_b.clock,
            ).create_conversation(
                f"{conversation_id}_new",
                user_id,
                None,
                "coding_debug",
                "New concurrent conversation",
            )
        else:
            await mutation_connection.execute(
                "UPDATE conversations SET status = ? WHERE id = ? AND user_id = ?",
                (ConversationStatus.ARCHIVED.value, conversation_id, user_id),
            )
            await mutation_connection.commit()
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(refresh_task, timeout=5.0)
        refresh_task = None
        assert (
            await ConversationActivityRepository(
                refresh_connection,
                runtime_a.clock,
            ).get_activity_stats(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            is None
        )
    finally:
        resume.set()
        if refresh_task is not None:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
        await mutation_connection.close()
        await refresh_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consumer_kind",
    ["snapshot", "hot_list", "warmup", "recommended_warmup"],
)
@pytest.mark.parametrize("mutation", ["append", "edit", "delete"])
async def test_activity_consumers_reject_stats_from_an_older_message_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer_kind: str,
    mutation: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    read_connection = await runtime_a.open_connection()
    mutation_connection = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    read_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_consumer_{consumer_kind}_{mutation}"
    conversation_id = f"cnv_activity_consumer_{consumer_kind}_{mutation}"
    message_prefix = f"msg_activity_consumer_{consumer_kind}_{mutation}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await _seed_activity_messages(
            read_connection,
            runtime_a,
            conversation_id=conversation_id,
            message_prefix=message_prefix,
        )
        service = ConversationActivityService(runtime_a)

        if consumer_kind in {"snapshot", "hot_list"}:
            original_refresh = service._refresh_user_activity_stats_with_sources

            async def refresh_then_pause(*args: Any, **kwargs: Any) -> Any:
                result = await original_refresh(*args, **kwargs)
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
                return result

            monkeypatch.setattr(
                service,
                "_refresh_user_activity_stats_with_sources",
                refresh_then_pause,
            )
        elif consumer_kind == "warmup":
            original_single_refresh = (
                service._refresh_conversation_activity_stats_with_source
            )

            async def single_refresh_then_pause(*args: Any, **kwargs: Any) -> Any:
                result = await original_single_refresh(*args, **kwargs)
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
                return result

            monkeypatch.setattr(
                service,
                "_refresh_conversation_activity_stats_with_source",
                single_refresh_then_pause,
            )
        else:
            original_hot_list = service._list_hot_conversations_with_sources

            async def hot_list_then_pause(*args: Any, **kwargs: Any) -> Any:
                result = await original_hot_list(*args, **kwargs)
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
                return result

            monkeypatch.setattr(
                service,
                "_list_hot_conversations_with_sources",
                hot_list_then_pause,
            )

        if consumer_kind == "snapshot":
            read_task = asyncio.create_task(
                service.get_activity_snapshot(read_connection, user_id)
            )
        elif consumer_kind == "hot_list":
            read_task = asyncio.create_task(
                service.list_hot_conversations(read_connection, user_id)
            )
        elif consumer_kind == "warmup":
            read_task = asyncio.create_task(
                service.warmup_conversation(
                    read_connection,
                    user_id,
                    conversation_id,
                )
            )
        else:
            read_task = asyncio.create_task(
                service.warmup_recommended_conversations(
                    read_connection,
                    user_id,
                    limit=1,
                    total_message_budget=4,
                    per_conversation_message_budget=4,
                )
            )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_message_source_mutation(
            mutation_connection,
            runtime_b,
            mutation=mutation,
            conversation_id=conversation_id,
            message_id=f"{message_prefix}_1",
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(read_task, timeout=5.0)
        read_task = None
        if consumer_kind in {"warmup", "recommended_warmup"}:
            assert (
                await stored_recent_window(runtime_a.storage_backend,
                    build_recent_window_key(user_id, conversation_id)
                )
                is None
            )
    finally:
        resume.set()
        if read_task is not None:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
        await mutation_connection.close()
        await read_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("warmup_kind", ["single", "recommended"])
async def test_warmup_survives_transient_context_cache_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warmup_kind: str,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    user_id = f"usr_activity_cache_outage_{warmup_kind}"
    conversation_id = f"cnv_activity_cache_outage_{warmup_kind}"
    try:
        await _seed_user_conversation(
            runtime,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection, runtime.clock).create_message(
            f"msg_activity_cache_outage_{warmup_kind}",
            conversation_id,
            "user",
            1,
            "Canonical SQLite warmup must survive an optional cache outage.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )

        async def unavailable_context_cache(_key: str) -> dict[str, Any] | None:
            raise RuntimeError("transient cache outage")

        monkeypatch.setattr(
            runtime.storage_backend,
            "get_context_view",
            unavailable_context_cache,
        )
        service = ConversationActivityService(runtime)
        if warmup_kind == "single":
            result = await service.warmup_conversation(
                connection,
                user_id,
                conversation_id,
            )
            warmed = result
        else:
            result = await service.warmup_recommended_conversations(
                connection,
                user_id,
                limit=1,
                total_message_budget=1,
                per_conversation_message_budget=1,
            )
            warmed = result["warmed_conversations"][0]

        assert warmed["recent_message_count"] == 1
        assert warmed["cached_context_available"] is False
        assert "context_cache_read_failed" in warmed["warmup_errors"]
        assert (
            await stored_recent_window(runtime.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            == warmed["recent_messages"]
        )
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
async def test_warmup_rejects_context_view_after_cache_revision_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = "usr_activity_cache_revision"
    conversation_id = "cnv_activity_cache_revision"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            "msg_activity_cache_revision",
            conversation_id,
            "user",
            1,
            "The context view belongs to the captured cache revision only.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )

        async def context_view_then_pause(_key: str) -> dict[str, Any]:
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return {"items": ["captured cache revision"]}

        monkeypatch.setattr(
            runtime_a.storage_backend,
            "get_context_view",
            context_view_then_pause,
        )
        warmup_task = asyncio.create_task(
            ConversationActivityService(runtime_a).warmup_conversation(
                connection_a,
                user_id,
                conversation_id,
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        identity = await UserLifecycleRepository(
            connection_b,
            runtime_b.clock,
        ).get_active_identity(user_id)
        assert identity is not None
        assert (
            await UserLifecycleRepository(
                connection_b,
                runtime_b.clock,
            ).bump_cache_revision(
                user_id,
                expected_lifecycle_epoch=identity.lifecycle_epoch,
            )
            == identity.cache_revision + 1
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_change", ["append", "edit", "delete", "selection"])
async def test_warmup_cleans_exact_window_when_source_changes_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_change: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = f"usr_activity_post_publish_{source_change}"
    conversation_id = f"cnv_activity_post_publish_{source_change}"
    message_id = f"msg_activity_post_publish_{source_change}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            message_id,
            conversation_id,
            "user",
            1,
            "This exact window must be removed if its source changes.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        original_publish = ContextCacheService.publish_recent_window
        pause_once = True

        async def publish_then_pause(
            cache_service: ContextCacheService,
            **kwargs: Any,
        ) -> bool:
            nonlocal pause_once
            published = await original_publish(cache_service, **kwargs)
            if cache_service.runtime is runtime_a and pause_once:
                pause_once = False
                assert published
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
            return published

        monkeypatch.setattr(
            ContextCacheService,
            "publish_recent_window",
            publish_then_pause,
        )
        warmup_task = asyncio.create_task(
            ConversationActivityService(runtime_a).warmup_conversation(
                connection_a,
                user_id,
                conversation_id,
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        key = build_recent_window_key(user_id, conversation_id)
        assert await stored_recent_window(runtime_a.storage_backend, key) is not None
        if source_change == "selection":
            await _commit_blocking_transcript_selection(
                connection_b,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        else:
            await _commit_message_source_mutation(
                connection_b,
                runtime_b,
                mutation=source_change,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
        assert await stored_recent_window(runtime_a.storage_backend, key) is None
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_failed_warmup_cleanup_preserves_newer_recent_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = "usr_activity_preserve_newer_window"
    conversation_id = "cnv_activity_preserve_newer_window"
    message_id = "msg_activity_preserve_newer_window"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            message_id,
            conversation_id,
            "user",
            1,
            "Old publication.",
            3,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        original_publish = ContextCacheService.publish_recent_window
        pause_once = True

        async def publish_then_pause(
            cache_service: ContextCacheService,
            **kwargs: Any,
        ) -> bool:
            nonlocal pause_once
            published = await original_publish(cache_service, **kwargs)
            if cache_service.runtime is runtime_a and pause_once:
                pause_once = False
                assert published
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
            return published

        monkeypatch.setattr(
            ContextCacheService,
            "publish_recent_window",
            publish_then_pause,
        )
        warmup_task = asyncio.create_task(
            ConversationActivityService(runtime_a).warmup_conversation(
                connection_a,
                user_id,
                conversation_id,
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_message_source_mutation(
            connection_b,
            runtime_b,
            mutation="append",
            conversation_id=conversation_id,
            message_id=message_id,
        )
        user_identity = await UserLifecycleRepository(
            connection_b,
            runtime_b.clock,
        ).get_active_identity(user_id)
        conversation_identity = await ConversationLifecycleRepository(
            connection_b,
            runtime_b.clock,
        ).get_active_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        assert user_identity is not None
        assert conversation_identity is not None
        newer_messages = [{"id": "newer", "role": "assistant", "text": "Newer"}]
        assert await ContextCacheService(runtime_a).publish_recent_window(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=newer_messages,
            lifecycle_epoch=user_identity.lifecycle_epoch,
            lifecycle_cleanup_key=user_identity.lifecycle_cleanup_key,
            cache_revision=user_identity.cache_revision,
            derivation_revision=user_identity.derivation_revision,
            conversation_lifecycle_epoch=conversation_identity.lifecycle_epoch,
            conversation_source_revision=conversation_identity.source_revision,
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
        assert (
            await stored_recent_window(runtime_a.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            == newer_messages
        )
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_recommended_warmup_cleans_earlier_windows_when_later_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    user_id = "usr_activity_recommended_partial_cleanup"
    conversation_ids = [
        "cnv_activity_recommended_partial_a",
        "cnv_activity_recommended_partial_b",
    ]
    try:
        for index, conversation_id in enumerate(conversation_ids, start=1):
            await _seed_user_conversation(
                runtime_a,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await MessageRepository(connection_a, runtime_a.clock).create_message(
                f"msg_activity_recommended_partial_{index}",
                conversation_id,
                "user",
                1,
                f"Recommended source {index}.",
                4,
                {},
                f"2026-03-10T11:0{index}:00+00:00",
            )

        service = ConversationActivityService(runtime_a)
        original_warmup = service.warmup_conversation
        mutated = False

        async def mutate_other_after_first_warmup(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal mutated
            result = await original_warmup(*args, **kwargs)
            if not mutated:
                mutated = True
                warmed_id = str(result["conversation_id"])
                other_id = next(item for item in conversation_ids if item != warmed_id)
                await MessageRepository(
                    connection_b,
                    runtime_b.clock,
                ).create_message(
                    "msg_activity_recommended_partial_concurrent",
                    other_id,
                    "assistant",
                    2,
                    "This later source changed after the first window published.",
                    10,
                    {},
                    "2026-03-10T11:03:00+00:00",
                )
            return result

        monkeypatch.setattr(
            service,
            "warmup_conversation",
            mutate_other_after_first_warmup,
        )
        with pytest.raises(TranscriptRebuildInProgressError):
            await service.warmup_recommended_conversations(
                connection_a,
                user_id,
                limit=2,
                total_message_budget=2,
                per_conversation_message_budget=1,
            )

        for conversation_id in conversation_ids:
            assert (
                await stored_recent_window(runtime_a.storage_backend,
                    build_recent_window_key(user_id, conversation_id)
                )
                is None
            )
    finally:
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_kind", ["snapshot", "hot_list"])
async def test_activity_read_cannot_return_after_cross_runtime_selection_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_kind: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    read_task: asyncio.Task[Any] | None = None
    user_id = f"usr_activity_read_{read_kind}"
    conversation_id = f"cnv_activity_read_{read_kind}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            f"msg_activity_read_{read_kind}",
            conversation_id,
            "user",
            1,
            "This branch must not escape a later selection fence.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime_a)
        await service.refresh_conversation_activity_stats(
            connection_a,
            user_id,
            conversation_id,
        )
        original_require = service._require_current_activity_membership

        async def pause_before_final_validation(
            *args: Any,
            **kwargs: Any,
        ) -> None:
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            await original_require(*args, **kwargs)

        monkeypatch.setattr(
            service,
            "_require_current_activity_membership",
            pause_before_final_validation,
        )
        if read_kind == "snapshot":
            read_task = asyncio.create_task(
                service.get_activity_snapshot(
                    connection_a,
                    user_id,
                    conversation_id=conversation_id,
                    refresh=False,
                )
            )
        else:
            read_task = asyncio.create_task(
                service.list_hot_conversations(
                    connection_a,
                    user_id,
                    refresh=False,
                )
            )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_blocking_transcript_selection(
            connection_b,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(read_task, timeout=5.0)
        read_task = None
    finally:
        resume.set()
        if read_task is not None:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_warmup_cannot_repopulate_after_cross_runtime_selection_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = "usr_activity_warmup_cross_runtime"
    conversation_id = "cnv_activity_warmup_cross_runtime"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            "msg_activity_warmup_cross_runtime",
            conversation_id,
            "user",
            1,
            "This old branch window must be removed after selection starts.",
            9,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        backend_type = type(runtime_a.storage_backend)
        original_publish = backend_type.set_recent_window_for_lifecycle

        async def pause_after_backend_write(
            backend: Any,
            key: str,
            messages: list[dict[str, Any]],
            *,
            user_id: str,
            conversation_id: str,
            lifecycle_cleanup_key: str,
            lifecycle_epoch: str,
            cache_revision: int,
            derivation_revision: int,
            conversation_lifecycle_epoch: str,
            conversation_source_revision: int,
        ) -> bool:
            published = await original_publish(
                backend,
                key,
                messages,
                user_id=user_id,
                conversation_id=conversation_id,
                lifecycle_cleanup_key=lifecycle_cleanup_key,
                lifecycle_epoch=lifecycle_epoch,
                cache_revision=cache_revision,
                derivation_revision=derivation_revision,
                conversation_lifecycle_epoch=conversation_lifecycle_epoch,
                conversation_source_revision=conversation_source_revision,
            )
            if backend is runtime_a.storage_backend:
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
            return published

        monkeypatch.setattr(
            backend_type,
            "set_recent_window_for_lifecycle",
            pause_after_backend_write,
        )
        warmup_task = asyncio.create_task(
            ConversationActivityService(runtime_a).warmup_conversation(
                connection_a,
                user_id,
                conversation_id,
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        assert (
            await stored_recent_window(runtime_a.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            is not None
        )
        await _commit_blocking_transcript_selection(
            connection_b,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
        assert (
            await stored_recent_window(runtime_a.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            is None
        )
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["append", "edit", "delete"])
async def test_warmup_removes_published_window_after_conversation_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = f"usr_activity_source_{mutation}"
    conversation_id = f"cnv_activity_source_{mutation}"
    message_id = f"msg_activity_source_{mutation}"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            message_id,
            conversation_id,
            "user",
            1,
            "The warmup captured this canonical message.",
            7,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        original_identity = await ConversationLifecycleRepository(
            connection_a,
            runtime_a.clock,
        ).get_active_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        assert original_identity is not None
        backend_type = type(runtime_a.storage_backend)
        original_publish = backend_type.set_recent_window_for_lifecycle

        async def pause_after_backend_write(
            backend: Any,
            key: str,
            messages: list[dict[str, Any]],
            *,
            user_id: str,
            conversation_id: str,
            lifecycle_cleanup_key: str,
            lifecycle_epoch: str,
            cache_revision: int,
            derivation_revision: int,
            conversation_lifecycle_epoch: str,
            conversation_source_revision: int,
        ) -> bool:
            published = await original_publish(
                backend,
                key,
                messages,
                user_id=user_id,
                conversation_id=conversation_id,
                lifecycle_cleanup_key=lifecycle_cleanup_key,
                lifecycle_epoch=lifecycle_epoch,
                cache_revision=cache_revision,
                derivation_revision=derivation_revision,
                conversation_lifecycle_epoch=conversation_lifecycle_epoch,
                conversation_source_revision=conversation_source_revision,
            )
            if backend is runtime_a.storage_backend:
                reached.set()
                await asyncio.wait_for(resume.wait(), timeout=5.0)
            return published

        monkeypatch.setattr(
            backend_type,
            "set_recent_window_for_lifecycle",
            pause_after_backend_write,
        )
        warmup_task = asyncio.create_task(
            ConversationActivityService(runtime_a).warmup_conversation(
                connection_a,
                user_id,
                conversation_id,
                refresh_stats=False,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        assert await stored_recent_window(runtime_a.storage_backend,
            build_recent_window_key(user_id, conversation_id)
        ) == [
            {
                "id": message_id,
                "seq": 1,
                "role": "user",
                "text": "The warmup captured this canonical message.",
                "occurred_at": "2026-03-10T11:00:00+00:00",
            }
        ]
        if mutation == "append":
            await MessageRepository(connection_b, runtime_b.clock).create_message(
                f"{message_id}_new",
                conversation_id,
                "assistant",
                2,
                "A concurrent append won after the old window was written.",
                9,
                {},
                "2026-03-10T11:01:00+00:00",
            )
        elif mutation == "edit":
            await connection_b.execute(
                "UPDATE messages SET text = ? WHERE id = ?",
                ("A concurrent edit replaced the captured text.", message_id),
            )
            await connection_b.commit()
        else:
            await connection_b.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,),
            )
            await connection_b.commit()
        current_identity = await ConversationLifecycleRepository(
            connection_b,
            runtime_b.clock,
        ).get_active_identity(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        assert current_identity is not None
        assert current_identity.lifecycle_epoch == original_identity.lifecycle_epoch
        assert current_identity.source_revision > original_identity.source_revision
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
        assert (
            await stored_recent_window(runtime_a.storage_backend,
                build_recent_window_key(user_id, conversation_id)
            )
            is None
        )
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_recommended_warmup_cannot_return_after_cross_runtime_selection_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    reached = asyncio.Event()
    resume = asyncio.Event()
    warmup_task: asyncio.Task[dict[str, Any]] | None = None
    user_id = "usr_activity_recommended_cross_runtime"
    conversation_id = "cnv_activity_recommended_cross_runtime"
    try:
        await _seed_user_conversation(
            runtime_a,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        await MessageRepository(connection_a, runtime_a.clock).create_message(
            "msg_activity_recommended_cross_runtime",
            conversation_id,
            "user",
            1,
            "A recommended result must keep its outer selection fence.",
            8,
            {},
            "2026-03-10T11:00:00+00:00",
        )
        service = ConversationActivityService(runtime_a)
        original_warmup = service.warmup_conversation

        async def pause_after_nested_warmup(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            result = await original_warmup(*args, **kwargs)
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(service, "warmup_conversation", pause_after_nested_warmup)
        warmup_task = asyncio.create_task(
            service.warmup_recommended_conversations(
                connection_a,
                user_id,
                limit=1,
                total_message_budget=1,
                per_conversation_message_budget=1,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        await _commit_blocking_transcript_selection(
            connection_b,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        resume.set()

        with pytest.raises(TranscriptRebuildInProgressError):
            await asyncio.wait_for(warmup_task, timeout=5.0)
        warmup_task = None
    finally:
        resume.set()
        if warmup_task is not None:
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()
