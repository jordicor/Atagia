"""API tests for memory routes."""

from __future__ import annotations

from functools import partial

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    SpaceBoundaryMode,
)

from tests.turn_telemetry_support import sample_turn_telemetry

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-memory.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/reply-test-model",
        llm_ingest_model="openai/chat-test-model",
        llm_retrieval_model="openai/score-test-model",
        llm_component_models={"intent_classifier": "openai/classify-test-model"},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_memory_routes_support_feedback_lookup_and_contract_view(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 1, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(users.create_user, "usr_2")
            client.portal.call(
                lambda: conversations.create_conversation(
                    "cnv_1",
                    "usr_1",
                    None,
                    "coding_debug",
                    "Chat",
                    platform_id="web",
                )
            )
            client.portal.call(
                messages.create_message, "msg_1", "cnv_1", "user", 1, "Need help", 2, {}
            )
            client.portal.call(
                messages.create_message,
                "msg_2",
                "cnv_1",
                "assistant",
                2,
                "Try this",
                2,
                {},
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="User is debugging retries.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_1",
                    platform_id="web",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_2",
                    conversation_id=None,
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.GLOBAL_USER,
                    canonical_text="Other user's memory.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    memory_id="mem_2",
                    platform_id="web",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Different same-user memory.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.8,
                    privacy_level=0,
                    memory_id="mem_3",
                    platform_id="web",
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.STATE_SNAPSHOT,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="User is focused on retries.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    payload={"focus_topic": "retries", "urgency": "high"},
                    memory_id="mem_state",
                    platform_id="web",
                )
            )
            event = client.portal.call(
                partial(events.create_event, telemetry=sample_turn_telemetry()),
                {
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_1",
                    "response_message_id": "msg_2",
                    "assistant_mode_id": "coding_debug",
                    "platform_id": "web",
                    "retrieval_plan_json": {"fts_queries": ["retry"]},
                    "selected_memory_ids_json": ["mem_1"],
                    "context_view_json": {
                        "selected_memory_ids": ["mem_1"],
                        "items_included": 1,
                        "items_dropped": 0,
                    },
                    "outcome_json": {},
                },
            )

            feedback_response = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_1",
                    "feedback_type": "useful",
                    "score": 0.9,
                    "metadata": {"source": "api"},
                },
            )
            assert feedback_response.status_code == 200
            assert feedback_response.json()["memory_id"] == "mem_1"

            wrong_feedback = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_2",
                    "feedback_type": "irrelevant",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert wrong_feedback.status_code == 404

            mismatched_feedback = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_3",
                    "feedback_type": "irrelevant",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert mismatched_feedback.status_code == 409

            invalid_feedback_type = client.post(
                "/v1/memory/feedback",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "retrieval_event_id": event["id"],
                    "memory_id": "mem_1",
                    "feedback_type": "not-valid",
                    "score": 0.1,
                    "metadata": {},
                },
            )
            assert invalid_feedback_type.status_code == 422

            memory_response = client.get(
                "/v1/memory/objects/mem_1",
                params={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                },
            )
            assert memory_response.status_code == 200
            assert (
                memory_response.json()["canonical_text"] == "User is debugging retries."
            )

            missing_memory = client.get(
                "/v1/memory/objects/mem_1",
                params={
                    "user_id": "usr_2",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                },
            )
            assert missing_memory.status_code == 404

            contract_response = client.get(
                "/v1/users/usr_1/contract",
                params={"conversation_id": "cnv_1", "platform_id": "web"},
            )
            assert contract_response.status_code == 200
            assert "directness" in contract_response.json()

            scoped_state_response = client.get(
                "/v1/users/usr_1/state",
                params={"conversation_id": "cnv_1", "platform_id": "web"},
            )
            assert scoped_state_response.status_code == 200
            assert scoped_state_response.json() == {
                "focus_topic": "retries",
                "urgency": "high",
            }

            async def install_selected_transcript_blocker() -> None:
                now = runtime.clock.now().isoformat()
                await connection.execute(
                    """
                    INSERT INTO transcript_rebuild_workflows(
                        id,
                        operation_id,
                        user_id,
                        conversation_id,
                        selection_epoch,
                        transcript_hash,
                        mutation_kind,
                        selected_message_ids_json,
                        abandoned_message_ids_json,
                        supporting_message_ids_json,
                        affected_memory_ids_json,
                        affected_summary_ids_json,
                        orchestrator_job_id,
                        stage,
                        start_derivation_revision,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trw_feedback_selection_wins",
                        "op_feedback_selection_wins",
                        "usr_1",
                        "cnv_1",
                        1,
                        "hash_feedback_selection_wins",
                        "replace",
                        "[]",
                        "[]",
                        "[]",
                        "[]",
                        "[]",
                        "job_feedback_selection_wins",
                        "aggregates",
                        0,
                        now,
                        now,
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO conversation_transcript_selections(
                        user_id,
                        conversation_id,
                        selection_epoch,
                        transcript_hash,
                        current_workflow_id,
                        state,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'rebuilding', ?)
                    """,
                    (
                        "usr_1",
                        "cnv_1",
                        1,
                        "hash_feedback_selection_wins",
                        "trw_feedback_selection_wins",
                        now,
                    ),
                )
                await connection.commit()

            feedback_before = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM memory_feedback_events WHERE user_id = ?",
                    ("usr_1",),
                )
            )
            client.portal.call(connection.execute, "BEGIN IMMEDIATE")
            request_started = threading.Event()
            feedback_responses = []
            feedback_errors = []

            def submit_feedback() -> None:
                request_started.set()
                try:
                    feedback_responses.append(
                        client.post(
                            "/v1/memory/feedback",
                            json={
                                "user_id": "usr_1",
                                "conversation_id": "cnv_1",
                                "platform_id": "web",
                                "retrieval_event_id": event["id"],
                                "memory_id": "mem_1",
                                "feedback_type": "confirmed_by_user",
                                "score": 1.0,
                                "metadata": {"source": "selection_wins_test"},
                            },
                        )
                    )
                except BaseException as exc:  # pragma: no cover - diagnostic path
                    feedback_errors.append(exc)

            feedback_thread = threading.Thread(target=submit_feedback, daemon=True)
            feedback_thread.start()
            assert request_started.wait(timeout=2.0)
            feedback_thread.join(timeout=0.05)
            assert feedback_thread.is_alive()

            client.portal.call(install_selected_transcript_blocker)
            feedback_thread.join(timeout=2.0)
            assert not feedback_thread.is_alive()
            assert feedback_errors == []
            assert len(feedback_responses) == 1
            blocked_feedback = feedback_responses[0]
            assert blocked_feedback.status_code == 409
            assert (
                blocked_feedback.json()["code"]
                == "selected_transcript_rebuild_in_progress"
            )
            feedback_after = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT id FROM memory_feedback_events WHERE user_id = ?",
                    ("usr_1",),
                )
            )
            assert feedback_after == feedback_before


def test_memory_lifecycle_routes_edit_archive_and_hard_delete(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 1, 1, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                lambda: conversations.create_conversation(
                    "cnv_1",
                    "usr_1",
                    None,
                    "coding_debug",
                    "Chat",
                    platform_id="web",
                )
            )
            client.portal.call(
                messages.create_message,
                "msg_edit_source",
                "cnv_1",
                "user",
                1,
                "Original memory text.",
            )
            client.portal.call(
                messages.create_message,
                "msg_delete_source",
                "cnv_1",
                "user",
                2,
                "Delete me.",
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Original memory text.",
                    index_text="original memory text",
                    extraction_hash="hash_edit",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_edit",
                    platform_id="web",
                    payload={"source_message_ids": ["msg_edit_source"]},
                )
            )
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Delete me.",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_delete",
                    platform_id="web",
                    payload={"source_message_ids": ["msg_delete_source"]},
                )
            )

            edit_response = client.patch(
                "/v1/memories/mem_edit",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "canonical_text": "Updated memory text.",
                },
            )
            assert edit_response.status_code == 200
            assert edit_response.json()["canonical_text"] == "Updated memory text."
            edited = client.portal.call(memories.get_memory_object, "mem_edit", "usr_1")
            assert edited["index_text"] is None
            assert edited["extraction_hash"] is None
            history_count = client.portal.call(
                lambda: connection.execute_fetchall(
                    "SELECT previous_text, new_text FROM memory_edit_history WHERE memory_id = ?",
                    ("mem_edit",),
                )
            )
            assert history_count[0]["previous_text"] == "Original memory text."

            archive_response = client.post(
                "/v1/memories/mem_edit/delete",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                },
            )
            assert archive_response.status_code == 200
            assert archive_response.json()["deleted_memories"] == 1
            archived = client.portal.call(
                memories.get_memory_object, "mem_edit", "usr_1"
            )
            assert archived["status"] == "archived"

            missing_confirmation = client.post(
                "/v1/memories/mem_delete/delete",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "hard": True,
                },
            )
            assert missing_confirmation.status_code == 400

            hard_delete_response = client.post(
                "/v1/memories/mem_delete/delete",
                json={
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "platform_id": "web",
                    "hard": True,
                    "confirmation": "HARD_DELETE_MEMORY",
                },
            )
            assert hard_delete_response.status_code == 200
            assert hard_delete_response.json()["deleted_memories"] == 1
            assert (
                client.portal.call(memories.get_memory_object, "mem_delete", "usr_1")
                is None
            )


@pytest.mark.parametrize(
    ("space_id", "boundary_mode"),
    [
        ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
        ("space_severed", SpaceBoundaryMode.SEVERANCE),
    ],
)
def test_memory_routes_enforce_space_boundaries_for_broader_scope_memories(
    tmp_path: Path,
    space_id: str,
    boundary_mode: SpaceBoundaryMode,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 4, 2, 1, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            spaces = SpaceRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                lambda: spaces.resolve_space(
                    owner_user_id="usr_1",
                    space_id=space_id,
                    boundary_mode=boundary_mode,
                    display_name=space_id,
                    source_kind="explicit",
                    source_id=space_id,
                )
            )
            client.portal.call(
                lambda: conversations.create_conversation(
                    "cnv_outside",
                    "usr_1",
                    None,
                    "coding_debug",
                    "Outside",
                    platform_id="web",
                )
            )
            client.portal.call(
                lambda: conversations.create_conversation(
                    "cnv_inside",
                    "usr_1",
                    None,
                    "coding_debug",
                    "Inside",
                    platform_id="web",
                    active_space_id=space_id,
                )
            )

            memory_ids = {
                "get": f"mem_{space_id}_get",
                "edit": f"mem_{space_id}_edit",
                "delete": f"mem_{space_id}_delete",
            }
            for operation, memory_id in memory_ids.items():
                client.portal.call(
                    lambda operation=operation, memory_id=memory_id: (
                        memories.create_memory_object(
                            user_id="usr_1",
                            conversation_id=None,
                            assistant_mode_id="coding_debug",
                            object_type=MemoryObjectType.EVIDENCE,
                            scope=MemoryScope.GLOBAL_USER,
                            canonical_text=f"{operation} memory inside {space_id}.",
                            source_kind=MemorySourceKind.VERBATIM,
                            confidence=0.9,
                            privacy_level=0,
                            memory_id=memory_id,
                            platform_id="web",
                            scope_canonical=MemoryScope.USER.value,
                            space_id=space_id,
                            space_boundary_mode=boundary_mode.value,
                        )
                    )
                )

            outside_params = {
                "user_id": "usr_1",
                "conversation_id": "cnv_outside",
                "platform_id": "web",
            }
            inside_params = {
                "user_id": "usr_1",
                "conversation_id": "cnv_inside",
                "platform_id": "web",
            }

            outside_get = client.get(
                f"/v1/memory/objects/{memory_ids['get']}",
                params=outside_params,
            )
            assert outside_get.status_code == 404

            outside_edit = client.patch(
                f"/v1/memories/{memory_ids['edit']}",
                json={
                    **outside_params,
                    "canonical_text": "Outside edit must not land.",
                },
            )
            assert outside_edit.status_code == 404
            stored_edit = client.portal.call(
                memories.get_memory_object, memory_ids["edit"], "usr_1"
            )
            assert stored_edit["canonical_text"] == f"edit memory inside {space_id}."

            outside_delete = client.post(
                f"/v1/memories/{memory_ids['delete']}/delete",
                json=outside_params,
            )
            assert outside_delete.status_code == 404
            stored_delete = client.portal.call(
                memories.get_memory_object, memory_ids["delete"], "usr_1"
            )
            assert stored_delete["status"] == "active"

            inside_get = client.get(
                f"/v1/memory/objects/{memory_ids['get']}",
                params=inside_params,
            )
            assert inside_get.status_code == 200
            assert (
                inside_get.json()["canonical_text"] == f"get memory inside {space_id}."
            )

            inside_edit = client.patch(
                f"/v1/memories/{memory_ids['edit']}",
                json={
                    **inside_params,
                    "canonical_text": "Inside edit is allowed.",
                },
            )
            assert inside_edit.status_code == 200
            assert inside_edit.json()["canonical_text"] == "Inside edit is allowed."

            inside_delete = client.post(
                f"/v1/memories/{memory_ids['delete']}/delete",
                json=inside_params,
            )
            assert inside_delete.status_code == 200
            archived_delete = client.portal.call(
                memories.get_memory_object, memory_ids["delete"], "usr_1"
            )
            assert archived_delete["status"] == "archived"


def test_memory_read_rejects_stale_row_after_completed_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        with _connection(client) as connection:
            client.portal.call(
                UserRepository(connection, runtime.clock).create_user,
                "usr_read_race",
            )
            client.portal.call(
                lambda: ConversationRepository(
                    connection,
                    runtime.clock,
                ).create_conversation(
                    "cnv_read_race",
                    "usr_read_race",
                    None,
                    "general_qa",
                    "Read race",
                    platform_id="web",
                )
            )
            client.portal.call(
                lambda: MemoryObjectRepository(
                    connection,
                    runtime.clock,
                ).create_memory_object(
                    memory_id="mem_read_race",
                    user_id="usr_read_race",
                    conversation_id="cnv_read_race",
                    assistant_mode_id="general_qa",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Stale branch memory",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    platform_id="web",
                )
            )

        original_get = MemoryObjectRepository.get_visible_memory_object
        raced = False

        async def get_while_replacement_completes(
            repository: MemoryObjectRepository,
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            nonlocal raced
            row = await original_get(repository, *args, **kwargs)
            if raced:
                return row
            raced = True
            read_completed = asyncio.Event()
            replacement_completed = asyncio.Event()

            async def complete_replacement() -> None:
                await read_completed.wait()
                writer = await runtime.open_connection()
                try:
                    await writer.execute("BEGIN IMMEDIATE")
                    await writer.execute(
                        "DELETE FROM memory_objects WHERE id = ? AND user_id = ?",
                        ("mem_read_race", "usr_read_race"),
                    )
                    await writer.execute(
                        """
                        UPDATE user_lifecycles
                        SET derivation_revision = derivation_revision + 1
                        WHERE user_id = ?
                        """,
                        ("usr_read_race",),
                    )
                    await writer.commit()
                finally:
                    await writer.close()
                    replacement_completed.set()

            replacement_task = asyncio.create_task(complete_replacement())
            read_completed.set()
            await asyncio.wait_for(replacement_completed.wait(), timeout=2.0)
            await replacement_task
            return row

        monkeypatch.setattr(
            MemoryObjectRepository,
            "get_visible_memory_object",
            get_while_replacement_completes,
        )
        response = client.get(
            "/v1/memory/objects/mem_read_race",
            params={
                "user_id": "usr_read_race",
                "conversation_id": "cnv_read_race",
                "platform_id": "web",
            },
        )

        assert response.status_code == 409
        assert response.json()["code"] == "selected_transcript_rebuild_in_progress"
